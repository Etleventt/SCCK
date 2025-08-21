#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, json, math, glob, random
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
from typing import Dict, List, Tuple, Optional

MODS_ALL = ["t1","t1ce","t2","flair"]

def subjects(root: Path) -> List[Path]:
    pats = []
    for pat in ["TCGA-GBM", "TCGA-LGG", "UPENN-GBM", "CPTAC-GBM", "IvyGAP", "ACRIN-FMISO-Brain", "UCSF-PDGM"]:
        for p in (root/pat).glob("BraTS2021_*"):
            if p.is_dir(): pats.append(p)
    # 兼容只下载了 TCGA-GBM/TCGA-LGG 的情况
    if not pats:
        pats = [p for p in root.glob("TCGA-*/BraTS2021_*") if p.is_dir()]
    return sorted(pats)

def mod_file(sub: Path, m: str) -> Optional[Path]:
    ls = list(sub.glob(f"*_{m}.nii*"))
    return ls[0] if ls else None

def as_ras_axial(nii: nib.Nifti1Image) -> np.ndarray:
    ras = nib.as_closest_canonical(nii)
    arr = ras.get_fdata(dtype=np.float32)  # (X,Y,Z)
    return np.transpose(arr, (2,1,0))      # (Z,H,W)

def percentile_minmax(img: np.ndarray, mask: np.ndarray, p_lo=0.5, p_hi=99.5, eps=1e-6) -> np.ndarray:
    inside = img[mask>0]
    if inside.size == 0: inside = img.reshape(-1)
    lo = np.percentile(inside, p_lo); hi = np.percentile(inside, p_hi)
    x = (img - lo) / (hi - lo + eps)
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def zscore_then_minmax01(img: np.ndarray, mask: np.ndarray, eps=1e-6) -> np.ndarray:
    inside = img[mask>0]
    if inside.size == 0: inside = img.reshape(-1)
    mu, sd = float(np.mean(inside)), float(np.std(inside) + eps)
    x = (img - mu) / sd
    # 将 z-score 映射到 0-1（±3σ → [0,1]）
    x = np.clip((x + 3.0) / 6.0, 0.0, 1.0)
    return x.astype(np.float32)

def resize_slice01(img01: np.ndarray, size: int) -> np.ndarray:
    if size is None: return img01
    pil = Image.fromarray((img01*255.0).astype(np.uint8), mode="L")
    pil = pil.resize((size,size), Image.BICUBIC)
    return (np.asarray(pil, dtype=np.uint8)/255.0).astype(np.float32)

def square_bbox_from_mask(mask: np.ndarray, pad: int=0) -> Tuple[slice,slice]:
    ys, xs = np.where(mask>0)
    if ys.size==0 or xs.size==0:
        H, W = mask.shape; s = min(H,W)
        y0=(H-s)//2; x0=(W-s)//2
        return (slice(y0,y0+s), slice(x0,x0+s))
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    h, w = y1-y0, x1-x0
    s = max(h, w)
    cy = (y0+y1)//2; cx = (x0+x1)//2
    y0 = max(0, cy - s//2 - pad); y1 = y0 + s + 2*pad
    x0 = max(0, cx - s//2 - pad); x1 = x0 + s + 2*pad
    return (slice(y0,y1), slice(x0,x1))

def split_subjects(sids: List[str], ratios=(0.8,0.1,0.1), seed=42):
    assert abs(sum(ratios)-1.0) < 1e-6
    rnd = random.Random(seed); rnd.shuffle(sids)
    n = len(sids); n_tr = int(n*ratios[0]); n_va = int(n*ratios[1])
    return {"train": sids[:n_tr], "val": sids[n_tr:n_tr+n_va], "test": sids[n_tr+n_va:]}

def nearly_empty(mask: np.ndarray, thr_ratio=0.01) -> bool:
    return (mask>0).mean() < thr_ratio

def main():
    ap = argparse.ArgumentParser("Prepare BraTS for SelfRDB")
    ap.add_argument("--root", required=True, type=Path, help="raw BraTS root (contains TCGA-GBM/TCGA-LGG/...)")
    ap.add_argument("--out_root", required=True, type=Path, help="output dataset root for SelfRDB")
    ap.add_argument("--modalities", default="t1,t2,flair,t1ce", help="comma list from {t1,t1ce,t2,flair}")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8,0.1,0.1])
    ap.add_argument("--size", type=int, default=None, help="optional XY size, e.g., 256; default keep original")
    ap.add_argument("--crop", type=str, default="none", choices=["none","bbox","center"], help="ROI crop")
    ap.add_argument("--per_volume", type=int, default=0, help="0=use all axial slices; >0=sample this many per volume")
    ap.add_argument("--z_lo", type=float, default=0.15, help="lower frac of Z to include when sampling")
    ap.add_argument("--z_hi", type=float, default=0.95, help="upper frac of Z to include when sampling")
    ap.add_argument("--norm", type=str, default="percentile", choices=["percentile","zscore"], help="normalization")
    ap.add_argument("--p_lo", type=float, default=0.5); ap.add_argument("--p_hi", type=float, default=99.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mods = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    for m in mods:
        if m not in MODS_ALL: raise SystemExit(f"unknown modality: {m}")

    subs = subjects(args.root)
    if not subs: raise SystemExit(f"No subjects under {args.root}")
    sid2files: Dict[str, Dict[str, Path]] = {}
    for sub in subs:
        sid = sub.name
        files = {m: mod_file(sub, m) for m in mods}
        if any(v is None for v in files.values()): continue
        sid2files[sid] = files

    sids = sorted(sid2files.keys())
    if not sids: raise SystemExit("No complete subjects with selected modalities.")
    splits = split_subjects(sids, tuple(args.split), seed=42)

    # Make dirs
    for m in mods:
        for sp in ["train","val","test"]:
            (args.out_root/m/sp).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for sp, sid_list in splits.items():
        for sid in sid_list:
            files = sid2files[sid]
            # load all selected modalities -> axial (Z,H,W)
            vols = {m: as_ras_axial(nib.load(str(files[m]))) for m in mods}
            Z,H,W = next(iter(vols.values())).shape
            if any(v.shape!=(Z,H,W) for v in vols.values()):
                print(f"[skip] shape mismatch: {sid}"); continue

            z0 = int(math.floor(Z*args.z_lo)); z1 = int(math.ceil(Z*args.z_hi))
            z0 = max(0, min(z0, Z-1)); z1 = max(z0+1, min(z1, Z))
            zs = range(z0,z1)
            if args.per_volume and args.per_volume>0:
                base = np.linspace(z0, z1-1, num=args.per_volume)
                zs = np.clip(np.round(base + rng.uniform(-1,1, size=len(base))).astype(int), z0, z1-1)

            for z in zs:
                slices = {m: vols[m][z] for m in mods}  # HxW
                # union brain mask across modalities（>0）
                union = np.zeros((H,W), dtype=np.uint8)
                for m in mods: union |= (slices[m] > 0).astype(np.uint8)
                if nearly_empty(union): continue

                # crop
                if args.crop == "bbox":
                    sy, sx = square_bbox_from_mask(union, pad=0)
                    for m in mods: slices[m] = slices[m][sy, sx]
                    union = union[sy, sx]
                elif args.crop == "center":
                    s = min(H,W); y0=(H-s)//2; x0=(W-s)//2
                    for m in mods: slices[m] = slices[m][y0:y0+s, x0:x0+s]
                    union = union[y0:y0+s, x0:x0+s]

                # normalize to [0,1]
                img01: Dict[str, np.ndarray] = {}
                for m in mods:
                    if args.norm == "percentile":
                        img01[m] = percentile_minmax(slices[m], union, args.p_lo, args.p_hi)
                    else:
                        img01[m] = zscore_then_minmax01(slices[m], union)

                    if args.size:
                        img01[m] = resize_slice01(img01[m], args.size)

                # save pairwise-consistent filenames
                for m in mods:
                    npy = (args.out_root/m/sp/f"subj-{sid}_z-{z:03d}.npy")
                    np.save(npy, img01[m])

    # manifest
    manifest = {"modalities": mods, "splits": {k: len(v) for k,v in splits.items()},
                "norm": args.norm, "percentiles":[args.p_lo,args.p_hi],
                "crop": args.crop, "size": args.size, "per_volume": args.per_volume,
                "z_range":[args.z_lo,args.z_hi]}
    (args.out_root/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[done] output root:", str(args.out_root))
if __name__ == "__main__":
    main()
