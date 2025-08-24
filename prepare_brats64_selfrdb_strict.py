#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_brats64_selfrdb_strict.py
---------------------------------

Goal
====
Strict SelfRDB-style preprocessing for BraTS → **64×64** dataset, while
preserving official geometry/FOV:
  1) Canonicalize each axial slice to **256×256** by centered pad/crop
     (no resampling) **in-memory only** (not written to disk).
  2) Intensity normalization with **global train-split percentiles**
     (default q_lo=0.5, q_hi=99.5) computed **within brain** (T1 nonzero
     as proxy mask) for all modalities ⇒ values in **[0,1]**.
  3) Downsample 256→**64** using **anti-aliased LANCZOS**.
  4) Save NumpyDataset directory tree:

      <out_root>/
        t1/{train,val,test}/slice_XXXXX.npy
        t2/{...}
        t1ce/{...}
        flair/{...}

Additionally provides a **preview mode** to dump a few sample PNG tiles
before full export so you can eyeball and stop early if something looks
wrong.

Usage
=====
# (1) Preview first (recommended):
python prepare_brats64_selfrdb_strict.py \
  --root /path/to/brats \
  --out_root /path/to/out64 \
  --modalities t1,t2,t1ce,flair \
  --preview_n 12 --preview_split val --preview_out /path/to/preview \
  --stop_after_preview 1 \
  --fast 1 --fast_subjects 24 --p1_workers 4

# (2) If preview looks good, run full export (reuses cached stats):
python prepare_brats64_selfrdb_strict.py \
  --root /path/to/brats \
  --out_root /path/to/out64 \
  --modalities t1,t2,t1ce,flair \
  --stop_after_preview 0 --p1_workers 8

Notes
=====
- **SelfRDB compliance**: global train percentiles, geometry via 256 pad/crop,
  values saved in [0,1]. Loader side `norm:true` will map to [-1,1] as usual.
- The 256 canonical step is **in-memory**, so disk usage = only 64×64 files.
- Cached stats at `<out_root>/global_stats.json` contain metadata; if you
  change q_lo/q_hi or scope, delete the cache to recompute.
- Use `--fast 1 --fast_subjects 12~24` to avoid OOM in Pass1 on modest RAM.
"""

import argparse, os, json, random, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from tempfile import NamedTemporaryFile
import concurrent.futures as futures

ALL = ["t1","t1ce","t2","flair"]

# ---------------- File helpers ----------------

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

# ---------------- BraTS traversal ----------------

def find_subjects(root: Path) -> List[Path]:
    pats=[]
    for pat in ["TCGA-GBM","TCGA-LGG","UPENN-GBM","CPTAC-GBM","IvyGAP","ACRIN-FMISO-Brain","UCSF-PDGM"]:
        d=root/pat
        if d.exists(): pats += [p for p in d.glob("BraTS2021_*") if p.is_dir()]
    if not pats:
        pats=[p for p in root.glob("TCGA-*/BraTS2021_*") if p.is_dir()]
    return sorted(pats)


def mod_file(sub: Path, m: str) -> Optional[Path]:
    ls=list(sub.glob(f"*_{m}.nii*"))
    return ls[0] if ls else None

# ---------------- Geometry & intensity ----------------

def as_ras_axial(nii: nib.Nifti1Image) -> np.ndarray:
    ras=nib.as_closest_canonical(nii)
    arr=ras.get_fdata(dtype=np.float32)
    return np.transpose(arr,(2,1,0))  # (Z,H,W)


def center_pad_or_crop_to(arr: np.ndarray, size: int) -> np.ndarray:
    H, W = arr.shape; s=size
    y0=max(0,(H-s)//2); x0=max(0,(W-s)//2)
    y1=min(H,y0+s); x1=min(W,x0+s)
    cropped=arr[y0:y1, x0:x1]
    out=np.zeros((s,s),dtype=np.float32)
    h,w=cropped.shape; oy=(s-h)//2; ox=(s-w)//2
    out[oy:oy+h, ox:ox+w]=cropped
    return out


def imresize01(arr01: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(arr01.astype(np.float32), mode="F")
    pil = pil.resize((size,size), Image.LANCZOS)
    out = np.asarray(pil, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)

# ---------------- Splitting & stats ----------------

def split_subjects(sids: List[str], ratios=(0.8,0.1,0.1), seed=42):
    assert abs(sum(ratios)-1.0) < 1e-6
    rnd=random.Random(seed); rnd.shuffle(sids)
    n=len(sids); n_tr=int(n*ratios[0]); n_va=int(n*ratios[1])
    return {"train": sids[:n_tr], "val": sids[n_tr:n_tr+n_va], "test": sids[n_tr+n_va:]}


def compute_global_percentiles_refmask(sid2files: Dict[str, Dict[str, Path]], mods: List[str], train_sids: List[str], q_lo: float, q_hi: float, workers: int = 8, fast: int = 0, fast_subjects: int = 12) -> Tuple[Dict[str,float], Dict[str,float]]:
    """Compute per-modality global percentiles **within brain** using T1>0 as mask."""
    thr = 1e-6

    def one_subj_stats(sid: str):
        files = sid2files[sid]
        vols = {m: as_ras_axial(nib.load(str(files[m]))) for m in mods}
        Z,H,W = next(iter(vols.values())).shape
        zs = list(range(Z))
        if fast:
            step = max(1, Z // 8)
            zs = list(range(0, Z, step))[:8]
        ref = 't1' if 't1' in mods else mods[0]
        refv = vols[ref][zs]
        msk = refv > thr
        out = {}
        for m in mods:
            v = vols[m][zs]
            out[m] = v[msk].reshape(-1)
        return out

    chosen = train_sids if not fast else train_sids[:min(len(train_sids), fast_subjects)]
    acc = {m: [] for m in mods}
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in tqdm(ex.map(one_subj_stats, chosen), total=len(chosen), desc="[Pass1] collecting"):
            for m in mods:
                if res[m].size:
                    acc[m].append(res[m])
    global_lo, global_hi = {}, {}
    for m in mods:
        if not acc[m]:
            global_lo[m], global_hi[m] = 0.0, 1.0
            continue
        arr = np.concatenate(acc[m], 0)
        global_lo[m] = float(np.percentile(arr, q_lo))
        global_hi[m] = float(np.percentile(arr, q_hi))
    return global_lo, global_hi

# ---------------- Preview tiles ----------------

def make_tile(imgs: Dict[str,np.ndarray], tile_size: int = 128, title: str = "") -> Image.Image:
    """Compose a 2x2 tile (t1, t2, t1ce, flair) for quick visual QC."""
    order = [m for m in ("t1","t2","t1ce","flair") if m in imgs]
    # normalize per image to [0,255] for viewing (already in [0,1])
    views = [Image.fromarray((np.clip(imgs[m],0,1)*255.0).astype(np.uint8)) for m in order]
    views = [v.resize((tile_size, tile_size), Image.NEAREST) for v in views]
    W = tile_size*2; H = tile_size*2
    canvas = Image.new('L', (W,H), 0)
    pos = [(0,0),(tile_size,0),(0,tile_size),(tile_size,tile_size)]
    for v, p in zip(views, pos):
        canvas.paste(v, p)
    if title:
        canvas = canvas.convert('RGB')
        draw = ImageDraw.Draw(canvas)
        draw.text((4,4), title, fill=(0,255,0))
    return canvas


def preview_samples(splits, sid2files, mods, global_lo, global_hi, out_dir: Path, n: int = 12, split_name: str = 'val', seed: int = 42):
    rng = random.Random(seed)
    sids = splits.get(split_name, []) or (splits['val'] if splits['val'] else splits['train'])
    sids = sids.copy(); rng.shuffle(sids)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for sid in sids:
        files = sid2files[sid]
        vols = {m: as_ras_axial(nib.load(str(files[m]))) for m in mods}
        Z,H,W = next(iter(vols.values())).shape
        zc = Z//2
        for z in [zc-1, zc, zc+1]:
            if z < 0 or z >= Z: continue
            # normalize
            imgs256 = {}
            for m in mods:
                v = (vols[m][z] - global_lo[m]) / max(global_hi[m]-global_lo[m], 1e-6)
                v = np.clip(v.astype(np.float32), 0.0, 1.0)
                v256 = center_pad_or_crop_to(v, 256)
                imgs256[m] = imresize01(v256, 64)
            title = f"{split_name}:{sid}|z={z}"
            tile = make_tile(imgs256, tile_size=128, title=title)
            tile.save(out_dir / f"preview_{count:03d}.png")
            count += 1
            if count >= n: return

# ---------------- Main ----------------

def parse_args():
    ap = argparse.ArgumentParser("BraTS -> SelfRDB strict (64x64, preview-first)")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--modalities", default="t1,t2,t1ce,flair")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8,0.1,0.1])
    ap.add_argument("--seed", type=int, default=42)
    # percentiles
    ap.add_argument("--q_lo", type=float, default=0.5)
    ap.add_argument("--q_hi", type=float, default=99.5)
    ap.add_argument("--p1_workers", type=int, default=max(os.cpu_count()//2,1))
    ap.add_argument("--fast", type=int, default=0)
    ap.add_argument("--fast_subjects", type=int, default=12)
    # preview controls
    ap.add_argument("--preview_n", type=int, default=12)
    ap.add_argument("--preview_split", choices=['train','val','test'], default='val')
    ap.add_argument("--preview_out", type=Path, default=None)
    ap.add_argument("--stop_after_preview", type=int, default=1)
    # export controls
    ap.add_argument("--max_slices_per_split", type=int, default=0)
    ap.add_argument("--resume", type=int, default=1)
    return ap.parse_args()


def next_start_index(out_root: Path, mods: List[str], split: str) -> int:
    i=0
    while True:
        ok=True
        for m in mods:
            if not (out_root/m/split/f"slice_{i}.npy").exists():
                ok=False; break
        if not ok: break
        i+=1
    return i


def main():
    args = parse_args()
    mods = [m.strip().lower() for m in args.modalities.split(',') if m.strip()]
    for m in mods:
        if m not in ALL: raise SystemExit(f"unknown modality: {m}")

    subs = find_subjects(args.root)
    if not subs: raise SystemExit("No subjects found under --root")

    sid2files = {}
    for sub in subs:
        sid=sub.name
        files={m: mod_file(sub,m) for m in mods}
        if any(v is None for v in files.values()):
            continue
        sid2files[sid]=files

    sids = sorted(sid2files.keys())
    splits = split_subjects(sids, tuple(args.split), seed=args.seed)

    # Prepare out dirs
    for m in mods:
        for sp in ["train","val","test"]:
            (args.out_root/m/sp).mkdir(parents=True, exist_ok=True)

    # Pass1: global percentiles (refmask) with cache
    cache_p = args.out_root / "global_stats.json"
    need_compute = True
    if cache_p.exists():
        try:
            stats = json.loads(cache_p.read_text())
            if stats.get('scope') == 'refmask' and float(stats.get('q_lo',args.q_lo))==args.q_lo and float(stats.get('q_hi',args.q_hi))==args.q_hi:
                global_lo, global_hi = stats["global_lo"], stats["global_hi"]
                need_compute = False
                print("[Pass1] Loaded cached percentiles (refmask):")
                for m in mods:
                    print(f"  [{m}] lo={global_lo[m]:.6f}, hi={global_hi[m]:.6f}")
        except Exception:
            need_compute = True
    if need_compute:
        print("[Pass1] Computing global percentiles on train split (refmask, per modality)…")
        global_lo, global_hi = compute_global_percentiles_refmask(sid2files, mods, splits['train'], args.q_lo, args.q_hi, workers=args.p1_workers, fast=args.fast, fast_subjects=args.fast_subjects)
        cache_p.write_text(json.dumps({"global_lo": global_lo, "global_hi": global_hi, "scope":"refmask", "q_lo": args.q_lo, "q_hi": args.q_hi}, indent=2), encoding="utf-8")
        for m in mods:
            print(f"  [{m}] lo={global_lo[m]:.6f}, hi={global_hi[m]:.6f}")

    # Preview phase
    if args.preview_n > 0:
        prev_dir = args.preview_out or (args.out_root/"_preview")
        print(f"[Preview] Saving {args.preview_n} tiles to: {prev_dir}")
        preview_samples(splits, sid2files, mods, global_lo, global_hi, prev_dir, n=args.preview_n, split_name=args.preview_split, seed=args.seed)
        print("[Preview] Done.")
        if args.stop_after_preview:
            print("[STOP] --stop_after_preview=1 ; Inspect previews and re-run with --stop_after_preview 0 to export.")
            return

    # Export Pass2 (canonicalize 256 in-memory → resize 64)
    def norm01(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)

    counters = {"train":0, "val":0, "test":0}
    if args.resume:
        for sp in ["train","val","test"]:
            counters[sp] = next_start_index(args.out_root, mods, sp)
            if counters[sp] > 0:
                print(f"[resume] {sp} start at slice_{counters[sp]}")

    print("[Pass2] Writing 64x64 slices …")
    for sp, sid_list in splits.items():
        for sid in tqdm(sid_list, desc=f"[Pass2] {sp}"):
            files = sid2files[sid]
            vols = {m: as_ras_axial(nib.load(str(files[m]))) for m in mods}
            Z,H,W = next(iter(vols.values())).shape
            if any(v.shape!=(Z,H,W) for v in vols.values()):
                print(f"[skip] shape mismatch: {sid}"); continue
            # normalize whole volume first
            for m in mods:
                vols[m] = norm01(vols[m], global_lo[m], global_hi[m])
            for z in range(Z):
                if args.max_slices_per_split and counters[sp] >= args.max_slices_per_split:
                    break
                # canonical 256 in-memory -> resize to 64
                imgs = {}
                for m in mods:
                    v256 = center_pad_or_crop_to(vols[m][z], 256)
                    imgs[m] = imresize01(v256, 64)
                k = counters[sp]
                for m in mods:
                    safe_np_save(args.out_root/m/sp/f"slice_{k}.npy", imgs[m])
                counters[sp] += 1

    # Manifest
    mani = {
        "modalities": mods,
        "image_size": 64,
        "geometry": "pad/crop to 256 (in-memory) then LANCZOS resize 64",
        "percentiles": {"q_lo": args.q_lo, "q_hi": args.q_hi, "scope": "refmask"},
        "splits_counts": counters,
        "seed": args.seed,
        "selfrdb_strict": True
    }
    (args.out_root/"manifest.json").write_text(json.dumps(mani, indent=2), encoding="utf-8")
    print("[DONE] Export complete.")

if __name__ == "__main__":
    main()
