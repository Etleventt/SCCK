#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SelfRDB-ready BraTS -> NumpyDataset (STRICT to repo requirements + 64px resize)
-----------------------------------------------------------------------------
This script prepares a SelfRDB-compatible NumpyDataset directly at 64×64
resolution by **padding to a centered square canvas** and **resizing** (no
content crop), while scaling pixel values to **[0,1]** as required in the
official SelfRDB README.

References:
- Repo & dataset format: Images should be scaled to have pixel values in [0,1].
  https://github.com/icon-lab/SelfRDB (README)

Why this script:
- Keep exactly the NumpyDataset tree layout expected by SelfRDB.
- Use robust per-SUBJECT normalization (percentiles over the whole 3D volume
  within a reference mask) to avoid overly-dark previews.
- Center the slice on a square canvas before resizing (prevents the “top-edge
  stuck to the border” artefact from asymmetric padding or bbox-cropping).
- Write an aligned "mask" modality (binary) derived from the reference modality.
- Generate preview tiles to visually confirm intensity/centering before export.

Example
-------
python prepare_brats64_selfrdb_resize_strict.py \
  --root /path/to/brats_root \
  --out_root /path/to/dataset/brats64_selfrdb \
  --modalities t1,t2,t1ce,flair \
  --ref_mod t1 \
  --split 0.8 0.1 0.1 \
  --z_lo 0.15 --z_hi 0.95 \
  --q_lo 0.5 --q_hi 99.5 \
  --export_size 64 \
  --preview_split val --preview_n 16 --preview_out /tmp/preview64 \
  --stop_after_preview 1

Notes
-----
* Output slices are **float32 in [0,1]** (SelfRDB will remap to [-1,1] at load
  time when `data.norm: true`).
* Background is zeroed via the (downsampled) mask after resizing to avoid
  interpolation leaks.
* We **do not crop** anatomies; we center-pad to a square and then resize.
* We compute percentiles **per subject & modality** across the 3D volume, but
  only inside the *reference* (e.g., T1) nonzero mask. This keeps consistency
  between slices and prevents black outputs.

Tested with: Python 3.10, nibabel, numpy, pillow, tqdm, matplotlib


python prepare_brats64_selfrdb_resize_strict.py \
  --root /home/xiaobin/Projects/DBAE/data/raw/brats \
  --out_root /home/xiaobin/Projects/SelfRDB/dataset/brats64_selfrdb \
  --modalities t1,t2,t1ce,flair \
  --ref_mod t1 \
  --split 0.8 0.1 0.1 \
  --z_lo 0.15 --z_hi 0.95 \
  --q_lo 0.5 --q_hi 99.5 \
  --export_size 64 \
  --preview_split val --preview_n 16 --preview_out /home/xiaobin/Projects/SelfRDB/preview_64 \
  --ranges_cache /home/xiaobin/Projects/SelfRDB/dataset/brats64_selfrdb/ranges_cache.json \
  --stop_after_preview 0

"""

from __future__ import annotations
import argparse, os, math, json, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ------------------------------ helpers ------------------------------ #
ALL = ["t1", "t1ce", "t2", "flair"]

SUBSETS = [
    "TCGA-GBM", "TCGA-LGG", "UPENN-GBM", "CPTAC-GBM", "IvyGAP",
    "ACRIN-FMISO-Brain", "UCSF-PDGM"
]

def find_subjects(root: Path) -> List[Path]:
    pats: List[Path] = []
    for pat in SUBSETS:
        d = root / pat
        if d.exists():
            pats += [p for p in d.glob("BraTS2021_*") if p.is_dir()]
    if not pats:
        pats = [p for p in root.glob("TCGA-*/BraTS2021_*") if p.is_dir()]
    return sorted(pats)


def mod_file(sub: Path, m: str) -> Optional[Path]:
    ls = list(sub.glob(f"*_{m}.nii*") )
    return ls[0] if ls else None


def as_ras_axial(nii_path: Path) -> np.ndarray:
    img = nib.load(str(nii_path))
    ras = nib.as_closest_canonical(img)
    arr = ras.get_fdata(dtype=np.float32)
    # Final order: (Z, H, W)
    return np.transpose(arr, (2, 1, 0))


def split_subjects(sids: List[str], ratios=(0.8, 0.1, 0.1), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6
    rnd = random.Random(seed)
    rnd.shuffle(sids)
    n = len(sids)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    return {
        "train": sids[:n_tr],
        "val": sids[n_tr:n_tr + n_va],
        "test": sids[n_tr + n_va:]
    }


def center_pad_to_square(x: np.ndarray) -> np.ndarray:
    """Pad 2D array to square canvas, centered."""
    H, W = x.shape
    S = int(max(H, W))
    out = np.zeros((S, S), dtype=x.dtype)
    sy = (S - H) // 2
    sx = (S - W) // 2
    out[sy:sy + H, sx:sx + W] = x
    return out


def pil_resize_float01(x01: np.ndarray, size: int) -> np.ndarray:
    """Resize float [0,1] image using bicubic via PIL (mode 'F')."""
    x01 = np.clip(x01, 0.0, 1.0).astype(np.float32)
    pil = Image.fromarray(x01, mode="F")
    pil = pil.resize((size, size), Image.BICUBIC)
    return np.asarray(pil, dtype=np.float32)


def pil_resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    # Use bilinear then threshold to keep smoother boundaries at 64px
    pil = pil.resize((size, size), Image.BILINEAR)
    m = (np.asarray(pil, dtype=np.float32) / 255.0) >= 0.5
    return m.astype(np.uint8)


def robust_minmax_over_volume(vol: np.ndarray, refmask_vol: np.ndarray,
                              q_lo: float, q_hi: float) -> Tuple[float, float]:
    """Percentile lo/hi INSIDE the reference nonzero mask across Z,H,W."""
    inside = vol[refmask_vol > 0]
    if inside.size == 0:
        inside = vol.reshape(-1)
    lo = float(np.percentile(inside, q_lo))
    hi = float(np.percentile(inside, q_hi))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        # Fallback
        lo = float(np.percentile(vol, 0.5))
        hi = float(np.percentile(vol, 99.5))
    return lo, hi


def safe_np_save(path: Path, arr: np.ndarray):
    """Atomic-ish save that avoids NumPy auto-appending '.npy' twice.
    Write via file handle to '<fname>.tmp' and then rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")  # e.g. slice_0.npy.tmp
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def tile_preview(samples: List[Tuple[str, int, Dict[str, np.ndarray]]],
                 out_png: Path, tile_px: int = 256):
    """Draw a grid preview; each column is one subject-slice, rows: t1,t2,t1ce,flair."""
    mods = ["t1", "t2", "t1ce", "flair"]
    cols = len(samples)
    rows = len(mods)
    W = cols * tile_px
    H = rows * tile_px
    canvas = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(canvas)

    for c, (sid, z, imgs) in enumerate(samples):
        for r, m in enumerate(mods):
            if m not in imgs:
                continue
            x01 = imgs[m]
            pil = Image.fromarray((np.clip(x01, 0, 1) * 255.0).astype(np.uint8))
            pil = pil.resize((tile_px, tile_px), Image.NEAREST)
            canvas.paste(pil, (c * tile_px, r * tile_px))
        # header
        header = f"{sid}|z={z}"
        draw.text((c * tile_px + 6, 6), header, fill=180)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)

# ------------------------------ main ------------------------------ #

def main():
    ap = argparse.ArgumentParser("BraTS -> SelfRDB NumpyDataset @64px (resize, [0,1])")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--modalities", default="t1,t2,t1ce,flair")
    ap.add_argument("--ref_mod", default="t1")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=42)

    # z-range and export size
    ap.add_argument("--z_lo", type=float, default=0.15)
    ap.add_argument("--z_hi", type=float, default=0.95)
    ap.add_argument("--export_size", type=int, default=64)

    # robust percentiles per SUBJECT (volume-wise)
    ap.add_argument("--q_lo", type=float, default=0.5)
    ap.add_argument("--q_hi", type=float, default=99.5)

    # preview options
    ap.add_argument("--preview_split", default="val", choices=["train", "val", "test"]) 
    ap.add_argument("--preview_n", type=int, default=16)
    ap.add_argument("--preview_out", type=Path, default=None)
    ap.add_argument("--stop_after_preview", type=int, default=1)

    # performance
    ap.add_argument("--max_subjects", type=int, default=0, help="limit subjects for quick runs (0=all)")

    # caching for Pass0 (per-subject percentile ranges)
    ap.add_argument("--ranges_cache", type=Path, default=None,
                    help="JSON cache file for per-subject (lo,hi) ranges; speeds up reruns")

    args = ap.parse_args()

    mods = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    for m in mods:
        if m not in ALL:
            raise SystemExit(f"Unknown modality: {m}")
    ref = args.ref_mod.lower()
    if ref not in mods:
        mods = [ref] + mods

    subs = find_subjects(args.root)
    if not subs:
        raise SystemExit("No subjects found under --root.")

    # map subject id -> file paths per modality
    sid2files: Dict[str, Dict[str, Path]] = {}
    for sub in subs:
        sid = sub.name
        files = {m: mod_file(sub, m) for m in mods}
        if any(v is None for v in files.values()):
            continue
        sid2files[sid] = files

    sids = sorted(sid2files.keys())
    if args.max_subjects > 0:
        sids = sids[:args.max_subjects]

    splits = split_subjects(sids, tuple(args.split), seed=args.seed)

    # Create folders
    for name in [*mods, "mask"]:
        for sp in ["train", "val", "test"]:
            (args.out_root / name / sp).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ---- Pass 0: collect per-subject percentile ranges (inside REF mask) ---- #
    print("[Pass0] Preparing per-subject percentile ranges (volume-wise, inside ref mask)…")
    subj_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}

    # Try to load cache
    loaded_from_cache = False
    if args.ranges_cache is not None and args.ranges_cache.exists():
        try:
            cache = json.load(open(args.ranges_cache, "r", encoding="utf-8"))
            subj_ranges = {sid: {m: (float(v[0]), float(v[1])) for m, v in mods2.items()} for sid, mods2 in cache.items()}
            loaded_from_cache = True
            print(f"[Pass0] Loaded ranges cache: {args.ranges_cache}")
        except Exception as e:
            print(f"[Pass0] Failed to load ranges cache: {e}; will recompute.")
            subj_ranges = {}

    # Determine which subjects still need ranges
    all_sids = splits["train"] + splits["val"] + splits["test"]
    need = [sid for sid in all_sids if sid not in subj_ranges]

    if need:
        print(f"[Pass0] Computing per-subject percentile ranges for {len(need)} subjects …")
        for sid in tqdm(need):
            files = sid2files[sid]
            vols = {m: as_ras_axial(files[m]) for m in mods}
            Z, H, W = next(iter(vols.values())).shape
            if any(v.shape != (Z, H, W) for v in vols.values()):
                print(f"[skip] shape mismatch: {sid}")
                continue
            refmask_vol = (vols[ref] > 0).astype(np.uint8)
            subj_ranges[sid] = {}
            for m in mods:
                lo, hi = robust_minmax_over_volume(vols[m], refmask_vol, args.q_lo, args.q_hi)
                subj_ranges[sid][m] = (lo, hi)

        # Save/merge cache
        if args.ranges_cache is not None:
            serializable = {sid: {m: [float(lo), float(hi)] for m, (lo, hi) in d.items()} for sid, d in subj_ranges.items()}
            args.ranges_cache.parent.mkdir(parents=True, exist_ok=True)
            (args.ranges_cache).write_text(json.dumps(serializable, indent=2), encoding="utf-8")
            print(f"[Pass0] Saved ranges cache: {args.ranges_cache}")
    else:
        if loaded_from_cache:
            print("[Pass0] All per-subject ranges loaded from cache; skip recompute.")
        else:
            print("[Pass0] No subjects to process.")

    # ---- Preview on selected split ---- #
    if args.preview_out is not None and args.preview_n > 0:
        sp = args.preview_split
        pool = splits[sp]
        take = min(args.preview_n, len(pool))
        chosen = rng.choice(pool, size=take, replace=False)

        samples = []
        for sid in chosen:
            files = sid2files[sid]
            vols = {m: as_ras_axial(files[m]) for m in mods}
            Z, H, W = next(iter(vols.values())).shape
            z0 = max(0, min(int(math.floor(Z * args.z_lo)), Z - 1))
            z1 = max(z0 + 1, min(int(math.ceil(Z * args.z_hi)), Z))
            z = int(rng.integers(z0, z1))

            refmask = (vols[ref][z] > 0).astype(np.uint8)
            imgs = {}
            for m in mods:
                lo, hi = subj_ranges[sid][m]
                x = vols[m][z]
                x = center_pad_to_square(x)
                # per-subject, per-modality scaling
                x01 = np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)
                x01 = pil_resize_float01(x01, args.export_size)
                msk = pil_resize_mask(center_pad_to_square(refmask), args.export_size)
                x01 = (x01 * msk.astype(np.float32))
                imgs[m] = x01.astype(np.float32)
            samples.append((sid, z, imgs))

        out_png = args.preview_out / f"preview_{args.preview_split}_{args.export_size}.png"
        tile_preview(samples, out_png)
        print(f"[Preview] wrote: {out_png}")
        if args.stop_after_preview:
            print("[STOP] --stop_after_preview=1 ; Inspect preview then re-run with 0 to export.")
            meta = {
                "modalities": mods,
                "ref_mod": ref,
                "export_size": args.export_size,
                "z_range": [args.z_lo, args.z_hi],
                "q": [args.q_lo, args.q_hi],
            }
            (args.out_root / "preview_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return

    # ---- Pass 1: export full dataset ---- #
    print("[Pass1] Exporting SelfRDB NumpyDataset at size", args.export_size)
    counters = {sp: 0 for sp in ["train", "val", "test"]}
    subject_ids = {sp: [] for sp in ["train", "val", "test"]}

    for sp, sid_list in splits.items():
        for sid in tqdm(sid_list, desc=sp):
            files = sid2files[sid]
            vols = {m: as_ras_axial(files[m]) for m in mods}
            Z, H, W = next(iter(vols.values())).shape
            if any(v.shape != (Z, H, W) for v in vols.values()):
                print(f"[skip] shape mismatch: {sid}")
                continue
            z0 = max(0, min(int(math.floor(Z * args.z_lo)), Z - 1))
            z1 = max(z0 + 1, min(int(math.ceil(Z * args.z_hi)), Z))
            refmask_vol = (vols[ref] > 0).astype(np.uint8)

            for z in range(z0, z1):
                refmask = refmask_vol[z]
                if refmask.sum() < 32:  # discard empty slices
                    continue
                m_square = center_pad_to_square(refmask)
                m_low = pil_resize_mask(m_square, args.export_size)

                for m in mods:
                    lo, hi = subj_ranges[sid][m]
                    x = vols[m][z]
                    x = center_pad_to_square(x)
                    x01 = np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)
                    x01 = pil_resize_float01(x01, args.export_size)
                    x01 = (x01 * m_low.astype(np.float32)).astype(np.float32)
                    safe_np_save(args.out_root / m / sp / f"slice_{counters[sp]}.npy", x01)

                safe_np_save(args.out_root / "mask" / sp / f"slice_{counters[sp]}.npy", m_low.astype(np.uint8))
                subject_ids[sp].append(f"{sid}|z={z}")
                counters[sp] += 1

    # Manifest
    manifest = {
        "modalities": mods,
        "ref_mod": ref,
        "export_size": args.export_size,
        "z_range": [args.z_lo, args.z_hi],
        "q": [args.q_lo, args.q_hi],
        "splits": counters,
        "note": "Images scaled to [0,1] (SelfRDB README requirement)."
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for sp in ["train", "val", "test"]:
        (args.out_root / f"subject_ids_{sp}.txt").write_text("\n".join(subject_ids[sp]), encoding="utf-8")

    print("[DONE] Wrote dataset to:", args.out_root)


if __name__ == "__main__":
    main()
