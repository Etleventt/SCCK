#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BraTS -> SelfRDB, **official-aligned** preprocessing.

What this script does (mirrors paper & repo notes):
- Loads BraTS NIfTI volumes, orients to RAS+, uses axial slices.
- Selects slices that contain brain tissue (nonzero voxels) — no tumor labels used.
- **No resize/crop**: keep native in-plane size (typically 240×240) and **zero-pad to 256×256**.
- Per *volume* normalization so the **brain-voxel mean = 1** for each modality.
- Then compute **global across-subject** robust min/max on the **train split** (per modality) and map
  intensities to **[-1, 1]**, finally remap to **[0, 1]** for saving (the repo expects [0,1]).
- Saves SelfRDB **NumpyDataset** layout (modalities/\{train,val,test\}/slice_XXXX.npy).

Usage (example):
python prepare_brats_for_selfrdb_official.py \
  --root /path/to/BraTS2021_root \
  --out_root /path/to/OUT/brats256_selfrdb \
  --modalities T1,T2,FLAIR,T1CE \
  --split 0.8 0.1 0.1 --seed 42

Notes:
- This script intentionally differs from earlier 64×64 pipelines: **no per-slice percentile scaling, no cropping, no resizing**, and evaluation/masks are not produced — matching SelfRDB.
- For **case-sensitivity**, SelfRDB examples use 'T1','T2','FLAIR','T1CE'. You can use lowercase
  but pass the same in SelfRDB `--data.source_modality` / `--data.target_modality`.
"""

import argparse, os, math, json, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib

# ----------------------- helpers -----------------------

BRATS_FOLDERS = [
    "TCGA-GBM","TCGA-LGG","UPENN-GBM","CPTAC-GBM","IvyGAP","ACRIN-FMISO-Brain","UCSF-PDGM"
]

ALL_MODS = ["T1","T1CE","T2","FLAIR","t1","t1ce","t2","flair"]


def find_subjects(root: Path) -> List[Path]:
    pats: List[Path] = []
    for pat in BRATS_FOLDERS:
        d = root / pat
        if d.exists():
            pats += [p for p in d.glob("BraTS2021_*") if p.is_dir()]
    if not pats:
        pats = [p for p in root.glob("TCGA-*/BraTS2021_*") if p.is_dir()]
    return sorted(pats)


def mod_file(sub: Path, m: str) -> Optional[Path]:
    ls = list(sub.glob(f"*_{m}.nii*"))
    return ls[0] if ls else None


def as_ras_axial(nii: nib.Nifti1Image) -> np.ndarray:
    """Return array as (Z,H,W) in RAS+ axial orientation."""
    ras = nib.as_closest_canonical(nii)
    arr = ras.get_fdata(dtype=np.float32)
    # nib canonical is (X,Y,Z) -> transpose to (Z,Y,X) == (Z,H,W)
    return np.transpose(arr, (2, 1, 0))


def pad_to_center(arr2d: np.ndarray, size: int = 256) -> np.ndarray:
    H, W = arr2d.shape
    if H == size and W == size:
        return arr2d.astype(np.float32, copy=False)
    # clip oversized (uncommon), then pad
    if H > size:
        y0 = (H - size) // 2
        arr2d = arr2d[y0:y0 + size, :]
        H = size
    if W > size:
        x0 = (W - size) // 2
        arr2d = arr2d[:, x0:x0 + size]
        W = size
    py0 = (size - H) // 2
    px0 = (size - W) // 2
    out = np.zeros((size, size), dtype=np.float32)
    out[py0:py0 + H, px0:px0 + W] = arr2d.astype(np.float32)
    return out


def brain_mask_union(vols: Dict[str, np.ndarray]) -> np.ndarray:
    """Union of nonzero voxels across available modalities. vols[m] is (Z,H,W)."""
    mask = None
    for v in vols.values():
        m = (v > 0).astype(np.uint8)
        mask = m if mask is None else (mask | m)
    return mask.astype(np.uint8)


def per_volume_mean_normalize(vol: np.ndarray, mask3d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale so that mean intensity within brain (nonzero mask) equals 1. Operates on (Z,H,W)."""
    inside = vol[mask3d > 0]
    mean = float(inside.mean()) if inside.size else float(vol.mean())
    scale = 1.0 / max(mean, eps)
    return (vol * scale).astype(np.float32)


def sample_percentiles(arr: np.ndarray, q_lo: float, q_hi: float, rng: np.random.Generator, max_samples: int = 500_000) -> Tuple[float, float]:
    """Robust global min/max via percentiles on a random subset of voxels."""
    flat = arr.reshape(-1)
    n = flat.size
    if n == 0:
        return 0.0, 1.0
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        flat = flat[idx]
    lo = float(np.percentile(flat, q_lo))
    hi = float(np.percentile(flat, q_hi))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser("BraTS -> SelfRDB (official-aligned, 256-pad, mean=1 per volume, global [-1,1] then save [0,1])")
    ap.add_argument("--root", required=True, type=Path, help="BraTS2021 root containing TCGA-*/UPENN-*/...")
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--modalities", default="T1,T2,FLAIR,T1CE", help="comma-separated (case kept for folder names)")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=256, help="final square size (pad/crop to this). Default 256")
    ap.add_argument("--nz_frac_thr", type=float, default=0.01, help="min nonzero fraction per slice to keep")
    ap.add_argument("--q_lo", type=float, default=0.1, help="train-set lower percentile for global scaling")
    ap.add_argument("--q_hi", type=float, default=99.9, help="train-set upper percentile for global scaling")
    ap.add_argument("--per_volume", type=int, default=0, help="if >0, sample this many z-slices per volume (approx)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    mods = [m.strip() for m in args.modalities.split(',') if m.strip()]
    for m in mods:
        if m not in ALL_MODS:
            raise SystemExit(f"Unknown modality: {m}")

    subs = find_subjects(args.root)
    if not subs:
        raise SystemExit("No subjects found under root.")

    # collect files per subject
    sid2files: Dict[str, Dict[str, Path]] = {}
    for sub in subs:
        sid = sub.name
        files = {m: mod_file(sub, m) for m in mods}
        if any(v is None for v in files.values()):
            continue
        sid2files[sid] = files

    sids = sorted(sid2files.keys())
    if not sids:
        raise SystemExit("No complete subjects with all requested modalities.")

    # split by subject
    ratios = tuple(float(x) for x in args.split)
    assert abs(sum(ratios) - 1.0) < 1e-6
    rnd = random.Random(args.seed)
    sids_copy = sids.copy()
    rnd.shuffle(sids_copy)
    n = len(sids_copy)
    n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
    splits = {
        "train": sids_copy[:n_tr],
        "val": sids_copy[n_tr:n_tr + n_va],
        "test": sids_copy[n_tr + n_va:]
    }

    # prepare dirs
    for m in mods:
        for sp in ["train", "val", "test"]:
            (args.out_root / m / sp).mkdir(parents=True, exist_ok=True)

    # -------- Pass 1: compute global train-set percentiles per modality (after per-volume mean=1) --------
    print("[Pass1] Computing global percentiles on train split (per modality) ...")
    global_lo: Dict[str, float] = {}
    global_hi: Dict[str, float] = {}

    for m in mods:
        samples = []
        for sid in splits["train"]:
            vol = as_ras_axial(nib.load(str(sid2files[sid][m])))  # (Z,H,W)
            # union mask uses all mods; compute once per subject to avoid cost
            if m == mods[0]:
                vols_all = {mm: as_ras_axial(nib.load(str(sid2files[sid][mm]))) for mm in mods}
                mask3d = brain_mask_union(vols_all)
            # scale mean to 1
            if 'mask3d' not in locals():
                vols_all = {mm: as_ras_axial(nib.load(str(sid2files[sid][mm]))) for mm in mods}
                mask3d = brain_mask_union(vols_all)
            vol_scaled = per_volume_mean_normalize(vol, mask3d)
            inside = vol_scaled[mask3d > 0]
            if inside.size:
                # random sub-sampling for robustness & speed
                nvox = inside.size
                take = min(nvox, 200_000)
                idx = rng.choice(nvox, size=take, replace=False)
                samples.append(inside[idx])
        if not samples:
            global_lo[m], global_hi[m] = 0.0, 2.0
        else:
            cat = np.concatenate(samples)
            lo, hi = sample_percentiles(cat, args.q_lo, args.q_hi, rng, max_samples=1_000_000)
            global_lo[m], global_hi[m] = float(lo), float(hi)
        print(f"  [{m}] lo={global_lo[m]:.6f}, hi={global_hi[m]:.6f}")

    # -------- Pass 2: write slices --------
    print("[Pass2] Writing SelfRDB NumpyDataset slices ...")

    counters = {"train": 0, "val": 0, "test": 0}
    subj_ids = {"train": [], "val": [], "test": []}

    for sp, sid_list in splits.items():
        for sid in sid_list:
            # load all requested modalities & union mask
            vols = {m: as_ras_axial(nib.load(str(sid2files[sid][m]))) for m in mods}
            mask3d = brain_mask_union(vols)
            Z, H, W = next(iter(vols.values())).shape
            if any(v.shape != (Z, H, W) for v in vols.values()):
                print(f"[skip:shape_mismatch] {sid}")
                continue

            # per-volume mean normalization for **each modality**
            vols = {m: per_volume_mean_normalize(vols[m], mask3d) for m in mods}

            # candidate z-slices where brain present
            z_indices = [z for z in range(Z) if (mask3d[z] > 0).mean() >= args.nz_frac_thr]
            if args.per_volume and args.per_volume > 0 and len(z_indices) > 0:
                # approx uniform subsample across the valid z range
                base = np.linspace(0, len(z_indices) - 1, num=min(args.per_volume, len(z_indices)))
                z_indices = [z_indices[int(round(b))] for b in base]

            for z in z_indices:
                # scale to [-1,1] globally, then save [0,1]
                for m in mods:
                    x = vols[m][z]
                    # clip with train global lo/hi for this modality
                    lo = global_lo[m]; hi = global_hi[m]
                    x = np.clip(x, lo, hi)
                    # map to [-1,1]
                    x = 2.0 * (x - lo) / (hi - lo) - 1.0
                    # then to [0,1] as expected by SelfRDB loader
                    x = 0.5 * (x + 1.0)
                    x = np.clip(x, 0.0, 1.0).astype(np.float32)
                    x = pad_to_center(x, size=args.size)
                    np.save(args.out_root / m / sp / f"slice_{counters[sp]}.npy", x)

                subj_ids[sp].append(f"{sid}|z={int(z)}")
                counters[sp] += 1

    manifest = {
        "modalities": mods,
        "splits_counts": counters,
        "size": args.size,
        "brain_slice_thr": args.nz_frac_thr,
        "q_lo": args.q_lo,
        "q_hi": args.q_hi,
        "global_lo": global_lo,
        "global_hi": global_hi,
        "seed": args.seed,
        "notes": {
            "mean_norm": "per-volume mean within union brain mask set to 1",
            "global_scale": "train-set robust percentiles per modality mapped to [-1,1], then saved in [0,1]",
            "pad": "zero-pad/crop to square size (default 256)",
            "mask": "no mask saved; SelfRDB evaluates full-slice",
        }
    }

    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for sp in ["train", "val", "test"]:
        (args.out_root / f"subject_ids_{sp}.txt").write_text("\n".join(subj_ids[sp]), encoding="utf-8")

    print("[done]", args.out_root)


if __name__ == "__main__":
    main()
