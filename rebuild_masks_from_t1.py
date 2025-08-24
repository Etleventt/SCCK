#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

def rebuild_mask_from_t1(x: np.ndarray, thr: float = 1e-4, close_iters: int = 1) -> np.ndarray:
    m = x > thr
    if close_iters > 0:
        m = binary_closing(m, iterations=close_iters)
    m = binary_fill_holes(m)
    return m.astype(np.uint8)

def process_split(in_root: Path, out_root: Path, split: str, thr: float, close_iters: int):
    (out_root/"mask"/split).mkdir(parents=True, exist_ok=True)
    src = in_root/"t1"/split
    files = sorted(src.glob("slice_*.npy"))
    n = 0
    for p in files:
        x = np.load(p).astype(np.float32)
        m = rebuild_mask_from_t1(x, thr=thr, close_iters=close_iters)
        np.save(out_root/"mask"/split/p.name, m)
        n += 1
    return n

def main():
    ap = argparse.ArgumentParser("Rebuild 64x64 masks from t1 slices (>thr).")
    ap.add_argument("--root", required=True, type=Path, help="干净数据目录（必须含 t1/...）")
    ap.add_argument("--out_root", required=True, type=Path, help="输出目录（可与 root 相同：就地覆盖）")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--thr", type=float, default=1e-4, help="阈值（>thr 视为前景）")
    ap.add_argument("--close_iters", type=int, default=1, help="形态学闭运算迭代次数")
    args = ap.parse_args()

    summary = {}
    for sp in args.splits:
        n = process_split(args.root, args.out_root, sp, args.thr, args.close_iters)
        summary[sp] = n

    (args.out_root/"mask_manifest.json").write_text(json.dumps({
        "source": "t1>thr w/ morphological closing",
        "thr": args.thr,
        "close_iters": args.close_iters,
        "counts": summary
    }, indent=2), encoding="utf-8")
    print("[DONE] rebuilt masks:", summary)

if __name__ == "__main__":
    main()
