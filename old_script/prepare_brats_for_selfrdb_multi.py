#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SelfRDB-style BraTS slice exporter (multi-variant, one-pass).

One command can produce **256x256 (pad/crop, geometry-preserving)** AND **64x64**
(either crop or anti-aliased resize), sharing the same Pass1 statistics.

Examples
--------
# A) Produce only 256x256 (official geometry-preserving variant)
python prepare_brats_for_selfrdb_multi.py \
  --root /path/to/brats \
  --out_base /path/to/out \
  --modalities t1,t2,t1ce,flair \
  --sizes 256 --modes crop

# B) One-pass: 256(crop) + 64(resize)
python prepare_brats_for_selfrdb_multi.py \
  --root /path/to/brats \
  --out_base /path/to/out \
  --modalities t1,t2,t1ce,flair \
  --sizes 256,64 --modes crop,resize

# C) One-pass: 256(crop) + 64(crop)  (geometry-preserving low-res)
python prepare_brats_for_selfrdb_multi.py \
  --root /path/to/brats \
  --out_base /path/to/out \
  --modalities t1,t2,t1ce,flair \
  --sizes 256,64 --modes crop,crop

Notes
-----
- "crop" means **center pad/crop without resampling** (geometry preserved):
  • 240->256 pads with zeros at borders; 256->64 takes centered crop
- "resize" uses **anti-aliased downsampling** (PIL LANCZOS) from a canonical
  256x256 representation (after intensity normalization).
- Pixel values are saved as float32 in [0,1].
- Split is stratified per subject with seed.
- Pass1 global percentiles are computed on the **train split** and cached at
  <out_base>/global_stats.json for reuse.
- Safe, atomic npy writing + resume per variant are supported.
"""

import argparse, os, json, math, time, random, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
from tempfile import NamedTemporaryFile
import concurrent.futures as futures

ALL = ["t1","t1ce","t2","flair"]
SLICE_RE = re.compile(r"slice_(\d+)\.npy$")

# ----------------------- filesystem helpers -----------------------

def safe_np_save(path: Path, arr: np.ndarray, retries: int = 3, sleep: float = 0.4):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    last = None; tmp = None
    for k in range(retries):
        try:
            with NamedTemporaryFile(dir=str(path.parent), delete=False, suffix='.tmp') as f:
                tmp = f.name
                np.save(f, arr)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except OSError as e:
            last = e
            try:
                if tmp and os.path.exists(tmp): os.unlink(tmp)
            except: pass
            time.sleep(sleep * (k+1))
    raise OSError(f"safe_np_save failed for {path}: {last}")

# ----------------------- BraTS traversal -----------------------

def find_subjects(root: Path) -> List[Path]:
    pats = []
    # standard BraTS 2021 released folders
    for pat in ["TCGA-GBM","TCGA-LGG","UPENN-GBM","CPTAC-GBM","IvyGAP","ACRIN-FMISO-Brain","UCSF-PDGM"]:
        d = root / pat
        if d.exists(): pats += [p for p in d.glob("BraTS2021_*") if p.is_dir()]
    if not pats:
        pats = [p for p in root.glob("TCGA-*/BraTS2021_*") if p.is_dir()]
    return sorted(pats)


def mod_file(sub: Path, m: str) -> Optional[Path]:
    ls = list(sub.glob(f"*_{m}.nii*"))
    return ls[0] if ls else None

# ----------------------- intensity + geometry -----------------------

def as_ras_axial(nii: nib.Nifti1Image) -> np.ndarray:
    # returns (Z,H,W) float32, canonical orientation (RAS+), axial slices
    ras = nib.as_closest_canonical(nii)
    arr = ras.get_fdata(dtype=np.float32)
    return np.transpose(arr, (2,1,0))


def center_pad_or_crop_to(arr: np.ndarray, size: int) -> np.ndarray:
    """Center pad/crop arr (H,W) to target square size without resampling."""
    H, W = arr.shape
    s = size
    # crop first if larger
    y0 = max(0, (H - s)//2); x0 = max(0, (W - s)//2)
    y1 = min(H, y0 + s); x1 = min(W, x0 + s)
    cropped = arr[y0:y1, x0:x1]
    # pad if smaller
    out = np.zeros((s, s), dtype=np.float32)
    h, w = cropped.shape
    oy = (s - h)//2; ox = (s - w)//2
    out[oy:oy+h, ox:ox+w] = cropped
    return out


def imresize01(arr01: np.ndarray, size: int) -> np.ndarray:
    # arr01: float32 [0,1]; use anti-aliased resize (LANCZOS)
    pil = Image.fromarray(arr01.astype(np.float32), mode="F")
    pil = pil.resize((size, size), Image.LANCZOS)
    out = np.asarray(pil, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)

# ----------------------- Pass1 stats -----------------------

def split_subjects(sids: List[str], ratios=(0.8,0.1,0.1), seed=42):
    assert abs(sum(ratios)-1.0) < 1e-6
    rnd = random.Random(seed); rnd.shuffle(sids)
    n = len(sids); n_tr = int(n*ratios[0]); n_va = int(n*ratios[1])
    return {"train": sids[:n_tr], "val": sids[n_tr:n_tr+n_va], "test": sids[n_tr+n_va:]}


def compute_global_percentiles(root: Path, sid2files: Dict[str, Dict[str, Path]], mods: List[str], train_sids: List[str], q_lo: float, q_hi: float, workers: int = 8, fast: int = 0, fast_subjects: int = 12) -> Tuple[Dict[str,float], Dict[str,float]]:
    # gather samples per modality; sample voxels uniformly across z for speed
    def one_subj_stats(sid: str) -> Dict[str, np.ndarray]:
        files = sid2files[sid]
        vols = {m: as_ras_axial(nib.load(str(files[m]))) for m in mods}
        Z,H,W = next(iter(vols.values())).shape
        # pick a subset of slices for speed
        if fast:
            step = max(1, Z // 8)
            zs = list(range(0, Z, step))[:8]
        else:
            zs = list(range(Z))
        data = {}
        for m in mods:
            v = vols[m][zs, :, :].astype(np.float32)
            data[m] = v.reshape(-1)
        return data

    chosen = train_sids if not fast else train_sids[:min(len(train_sids), fast_subjects)]
    # parallel load
    acc = {m: [] for m in mods}
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in tqdm(ex.map(one_subj_stats, chosen), total=len(chosen), desc="[Pass1] collecting"):
            for m in mods:
                acc[m].append(res[m])
    global_lo = {}
    global_hi = {}
    for m in mods:
        arr = np.concatenate(acc[m], axis=0)
        global_lo[m] = float(np.percentile(arr, q_lo))
        global_hi[m] = float(np.percentile(arr, q_hi))
    return global_lo, global_hi

# ----------------------- dataclasses for variants -----------------------
@dataclass
class Variant:
    size: int
    mode: str   # 'crop' or 'resize'
    name: str   # directory name under out_base

# ----------------------- main -----------------------

def parse_args():
    ap = argparse.ArgumentParser("BraTS -> SelfRDB (multi-size, one-pass)")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out_base", required=True, type=Path, help="base output folder; variants will be subfolders here")
    ap.add_argument("--modalities", default="t1,t2,t1ce,flair")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8,0.1,0.1])
    ap.add_argument("--sizes", default="256", help="comma list, e.g. 256,64")
    ap.add_argument("--modes", default="crop", help="comma list same length as sizes; each in {crop,resize}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q_lo", type=float, default=0.5)
    ap.add_argument("--q_hi", type=float, default=99.5)
    ap.add_argument("--p1_workers", type=int, default=max(os.cpu_count()//2,1))
    ap.add_argument("--fast", type=int, default=0, help="speed up Pass1 by sampling")
    ap.add_argument("--fast_subjects", type=int, default=12)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--max_slices_per_split", type=int, default=0)
    # NEW: avoid canonical 256 if you only need 64 (saves disk/time)
    ap.add_argument("--canonical256", type=int, default=1, help="1=canonicalize each slice to 256x256 in-memory first (official geometry), 0=work directly on original 240x240 for target sizes")
    ap.add_argument("--resize_from", choices=["pad256","orig"], default="pad256", help="for mode=resize: downsample from canonical 256 (pad256) or from original slice (orig)")
    return ap.parse_args()


def prepare_variants(args) -> List[Variant]:
    sizes = [int(s.strip()) for s in args.sizes.split(',') if s.strip()]
    modes = [m.strip().lower() for m in args.modes.split(',') if m.strip()]
    if len(modes) == 1 and len(sizes) > 1:
        modes = modes * len(sizes)
    assert len(sizes) == len(modes), "sizes and modes length mismatch"
    variants = []
    for s, md in zip(sizes, modes):
        assert md in ("crop","resize")
        name = f"size{s}_{md}"
        variants.append(Variant(size=s, mode=md, name=name))
    return variants


def next_start_index(out_root: Path, mods: List[str], split: str) -> int:
    i = 0
    while True:
        ok = True
        for m in mods:
            if not (out_root / m / split / f"slice_{i}.npy").exists():
                ok = False; break
        if not ok: break
        i += 1
    return i


def main():
    args = parse_args()
    mods = [m.strip().lower() for m in args.modalities.split(',') if m.strip()]
    for m in mods:
        if m not in ALL: raise SystemExit(f"unknown modality: {m}")

    variants = prepare_variants(args)
    print("[Variants]", ", ".join([f"{v.name}" for v in variants]))

    subs = find_subjects(args.root)
    if not subs: raise SystemExit("No subjects found under --root")

    sid2files = {}
    for sub in subs:
        sid = sub.name
        files = {m: mod_file(sub, m) for m in mods}
        if any(v is None for v in files.values()):
            continue
        sid2files[sid] = files

    sids = sorted(sid2files.keys())
    splits = split_subjects(sids, tuple(args.split), seed=args.seed)

    # --------- prepare output directories per variant ---------
    for v in variants:
        base = args.out_base / v.name
        for m in mods:
            for sp in ["train","val","test"]:
                (base/m/sp).mkdir(parents=True, exist_ok=True)

    # --------- Pass1: global percentiles (cached) ---------
    cache_p = args.out_base / "global_stats.json"
    if cache_p.exists():
        stats = json.loads(cache_p.read_text())
        global_lo, global_hi = stats["global_lo"], stats["global_hi"]
        print("[Pass1] Loaded cached percentiles:")
        for m in mods:
            print(f"  [{m}] lo={global_lo[m]:.6f}, hi={global_hi[m]:.6f}")
    else:
        print("[Pass1] Computing global percentiles on train split (per modality) ...")
        global_lo, global_hi = compute_global_percentiles(args.root, sid2files, mods, splits['train'], args.q_lo, args.q_hi, workers=args.p1_workers, fast=args.fast, fast_subjects=args.fast_subjects)
        cache_p.write_text(json.dumps({"global_lo": global_lo, "global_hi": global_hi}, indent=2), encoding="utf-8")
        for m in mods:
            print(f"  [{m}] lo={global_lo[m]:.6f}, hi={global_hi[m]:.6f}")

    # --------- Pass2: write slices for each variant ---------
    def norm01(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        y = (x - lo) / max(hi - lo, 1e-6)
        return np.clip(y.astype(np.float32), 0.0, 1.0)

    counters = {v.name: {"train":0, "val":0, "test":0} for v in variants}
    subj_ids = {v.name: {"train":[], "val":[], "test":[]} for v in variants}

    # resume
    if args.resume:
        for v in variants:
            base = args.out_base / v.name
            for sp in ["train","val","test"]:
                start = next_start_index(base, mods, sp)
                counters[v.name][sp] = start
                if start > 0:
                    print(f"[resume] {v.name}:{sp} start at slice_{start}")

    print("[Pass2] Writing variants …")
    for sp, sid_list in splits.items():
        for sid in tqdm(sid_list, desc=f"[Pass2] {sp}"):
            files = sid2files[sid]
            vols = {m: as_ras_axial(nib.load(str(files[m]))) for m in mods}
            Z,H,W = next(iter(vols.values())).shape
            if any(v.shape != (Z,H,W) for v in vols.values()):
                print(f"[skip] shape mismatch: {sid}"); continue

            # normalize per modality using global train percentiles
            for m in mods:
                vols[m] = norm01(vols[m], global_lo[m], global_hi[m])

            for z in range(Z):
                if args.max_slices_per_split:
                    if all(counters[v.name][sp] >= args.max_slices_per_split for v in variants):
                        break

                # choose base per modality according to flags, without writing any 256 to disk
                if args.canonical256:
                    base = {m: center_pad_or_crop_to(vols[m][z], 256) for m in mods}
                else:
                    # work directly on original normalized slice (typically 240x240)
                    base = {m: vols[m][z] for m in mods}

                for v in variants:
                    if args.max_slices_per_split and counters[v.name][sp] >= args.max_slices_per_split:
                        continue

                    imgs = {}
                    for m in mods:
                        if v.mode == 'crop':
                            imgs[m] = center_pad_or_crop_to(base[m], v.size)
                        else:  # resize
                            if args.resize_from == 'pad256' and not args.canonical256:
                                # create a 256x256 view in-memory just for resizing
                                tmp256 = center_pad_or_crop_to(base[m], 256)
                                imgs[m] = imresize01(tmp256, v.size)
                            else:
                                imgs[m] = imresize01(base[m], v.size)

                    k = counters[v.name][sp]
                    vroot = args.out_base / v.name
                    for m in mods:
                        safe_np_save(vroot / m / sp / f"slice_{k}.npy", imgs[m])
                    subj_ids[v.name][sp].append(f"{sid}|z={int(z)}")
                    counters[v.name][sp] += 1

    # write manifests per variant
    for v in variants:
        base = args.out_base / v.name
        mani = {
            "modalities": mods,
            "size": v.size,
            "size_mode": v.mode,
            "splits_counts": counters[v.name],
            "global_lo": global_lo,
            "global_hi": global_hi,
            "seed": args.seed,
            "notes": {"geometry": ("center_pad/crop to 256 then crop/resize" if args.canonical256 else "direct from original slice; crop/resize as requested"),
            "resize_from": args.resize_from,
            "canonical256": int(args.canonical256)}
        }
        (base/"manifest.json").write_text(json.dumps(mani, indent=2), encoding="utf-8")
        for sp in ["train","val","test"]:
            (base/f"subject_ids_{sp}.txt").write_text("\n".join(subj_ids[v.name][sp]), encoding="utf-8")
        print(f"[done] {base}  counts={counters[v.name]}")

if __name__ == "__main__":
    main()
