#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, glob
from pathlib import Path
import numpy as np

def fix_file(p):
    arr = np.load(p)
    need = (arr.dtype != np.float32) or np.isnan(arr).any() or np.isinf(arr).any() \
           or (arr.min() < 0.0) or (arr.max() > 1.0)
    if not need:
        return False, float(arr.min()), float(arr.max())
    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    np.save(p, arr)
    return True, float(arr.min()), float(arr.max())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--modalities", default="t1,t2,t1ce,flair")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    total = fixed = 0
    for m in mods:
        for sp in splits:
            for p in glob.glob(str(args.root / m / sp / "slice_*.npy")):
                total += 1
                changed, mn, mx = fix_file(p)
                fixed += int(changed)
    print(f"Scanned: {total} files | Fixed: {fixed} files")

if __name__ == "__main__":
    main()
