#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Downsample a SelfRDB NumpyDataset 256x256 -> 64x64 (anti-aliased resize)
import os, json, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

def imresize01(arr: np.ndarray, size: int) -> np.ndarray:
    # arr: float32 in [0,1]
    pil = Image.fromarray(arr.astype(np.float32), mode="F")
    pil = pil.resize((size, size), Image.LANCZOS)   # anti-aliased downsample
    out = np.asarray(pil, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)

def list_indices(mod_dir: Path, split: str):
    p = mod_dir / split
    if not p.exists(): return []
    return sorted(int(f[6:-4]) for f in os.listdir(p) if f.startswith("slice_") and f.endswith(".npy"))

def main():
    ap = argparse.ArgumentParser("Downsample SelfRDB dataset 256->64 by resize")
    ap.add_argument("--in_root", required=True, type=Path)   # e.g., brats256_selfrdb
    ap.add_argument("--out_root", required=True, type=Path)  # e.g., brats64_selfrdb_resize
    ap.add_argument("--modalities", default="t1,t2,t1ce,flair")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)
    # make folders
    for m in mods:
        for sp in splits:
            (args.out_root/m/sp).mkdir(parents=True, exist_ok=True)

    # copy/patch manifest if present
    man_in = args.in_root / "manifest.json"
    if man_in.exists():
        try:
            mani = json.loads(man_in.read_text())
        except Exception:
            mani = {}
    else:
        mani = {}
    mani.update({
        "source_root": str(args.in_root),
        "size": args.size,
        "notes": {"size_mode": "resize", "from": "256->64 LANCZOS"},
    })

    counts = {}
    for sp in splits:
        idxs = list_indices(args.in_root/mods[0], sp)
        counts[sp] = 0
        for i in tqdm(idxs, desc=f"[resize] {sp}"):
            for m in mods:
                src = args.in_root / m / sp / f"slice_{i}.npy"
                dst = args.out_root / m / sp / f"slice_{i}.npy"
                x = np.load(src).astype(np.float32)     # expect [0,1], 256x256
                y = imresize01(x, args.size)            # [0,1], 64x64
                np.save(dst, y.astype(np.float32))
            counts[sp] += 1

    mani["splits_counts"] = counts
    (args.out_root/"manifest.json").write_text(json.dumps(mani, indent=2), encoding="utf-8")
    print("[done]", args.out_root)

if __name__ == "__main__":
    main()
