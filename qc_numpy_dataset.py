#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick QC for SelfRDB-style NumpyDataset (BraTS slices)
- Verifies folder structure and alignment across modalities
- Computes edge occupancy stats with configurable threshold
- Samples slices and saves modality-by-columns mosaics per split
- Writes summary.json and a CSV with per-slice stats (optional cap)

Usage example:
  python qc_numpy_dataset.py \
    --root /home/xiaobin/Projects/SelfRDB/dataset/brats64_selfrdb_clean_copy_withmask \
    --modalities t1,t2,t1ce,flair \
    --splits train,val,test \
    --n 8 --thr 1e-4 --max_stats 2000 \
    --out /home/xiaobin/Projects/SelfRDB/qc_brats64_selfrdb_clean_copy_withmask
"""
import argparse, json, os, re, random, time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

SLICE_RE = re.compile(r"slice_(\d+)\.npy$")


def parse_args():
    ap = argparse.ArgumentParser("QC for SelfRDB NumpyDataset")
    ap.add_argument("--root", required=True, type=Path, help="dataset root (contains modality folders)")
    ap.add_argument("--modalities", default="t1,t2,t1ce,flair", help="comma list of modalities present")
    ap.add_argument("--splits", default="train,val,test", help="comma list of splits to check")
    ap.add_argument("--n", type=int, default=8, help="#columns per mosaic (sampled slice indices)")
    ap.add_argument("--thr", type=float, default=1e-4, help="threshold for 'nonzero' when computing edge stats")
    ap.add_argument("--edge_band", type=int, default=8, help="#pixels from each border considered as edge band")
    ap.add_argument("--max_stats", type=int, default=2000, help="max slices per split for stats CSV (0=all)")
    ap.add_argument("--out", type=Path, default=None, help="output folder; default: <root>/../qc_<ts>")
    return ap.parse_args()


def list_indices(mod_dir: Path, split: str) -> List[int]:
    p = mod_dir / split
    if not p.exists():
        return []
    idx = []
    for name in os.listdir(p):
        m = SLICE_RE.match(name)
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def intersect_indices(root: Path, mods: List[str], split: str) -> List[int]:
    sets = []
    for m in mods:
        s = set(list_indices(root / m, split))
        sets.append(s)
    return sorted(set.intersection(*sets)) if sets else []


def load_slice(root: Path, mod: str, split: str, i: int) -> np.ndarray:
    p = root / mod / split / f"slice_{i}.npy"
    return np.load(p)


def compute_stats(arr: np.ndarray, thr: float, edge_band: int) -> Dict:
    h, w = arr.shape
    nz_frac = float((arr > thr).mean())
    top = float((arr[:edge_band, :] > thr).mean())
    bottom = float((arr[-edge_band:, :] > thr).mean())
    left = float((arr[:, :edge_band] > thr).mean())
    right = float((arr[:, -edge_band:] > thr).mean())
    ys, xs = np.where(arr > thr)
    cy = float(ys.mean()) if ys.size else None
    cx = float(xs.mean()) if xs.size else None
    return {
        "nz_frac": nz_frac,
        "edge_top": top,
        "edge_bottom": bottom,
        "edge_left": left,
        "edge_right": right,
        "center_y": cy,
        "center_x": cx,
        "H": h,
        "W": w,
    }


def save_mosaic(root: Path, out_dir: Path, mods: List[str], split: str, cols: int, sample_idx: List[int]):
    assert len(sample_idx) > 0
    # infer size from first slice
    a0 = load_slice(root, mods[0], split, sample_idx[0])
    H, W = a0.shape
    rows = len(mods)
    fig_w = max(12, cols * 2)  # generous width for readability
    fig_h = max(4, rows * 2)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    for r, m in enumerate(mods):
        for c, i in enumerate(sample_idx):
            ax = axes[r][c]
            try:
                img = load_slice(root, m, split, i)
            except Exception as e:
                img = np.zeros((H, W), dtype=np.float32)
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{split} idx={i}")
            if c == 0:
                ax.set_ylabel(m)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"mosaic_{split}.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    mods = [m.strip() for m in args.modalities.split(',') if m.strip()]
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or (args.root.parent / f"qc_{args.root.name}_{ts}")
    out.mkdir(parents=True, exist_ok=True)

    # Try to read manifest.json if present
    manifest = None
    man_p = args.root / "manifest.json"
    if man_p.exists():
        try:
            manifest = json.loads(man_p.read_text())
        except Exception:
            manifest = None

    summary = {
        "root": str(args.root),
        "modalities": mods,
        "splits": {},
        "thr": args.thr,
        "edge_band": args.edge_band,
        "has_mask_folder": (args.root / "mask").exists(),
        "has_manifest": bool(manifest is not None),
        "manifest": manifest,
    }

    # Per-split
    for split in splits:
        idx_all = intersect_indices(args.root, mods, split)
        count = len(idx_all)
        summary["splits"][split] = {
            "slice_count_common": count,
        }
        if count == 0:
            continue

        # sample for mosaic
        cols = min(args.n, count)
        sample_idx = random.sample(idx_all, k=cols)
        sample_idx.sort()
        save_mosaic(args.root, out, mods, split, cols, sample_idx)

        # stats CSV (capped)
        cap = count if args.max_stats == 0 else min(count, args.max_stats)
        step = max(1, count // cap)
        chosen = idx_all[::step][:cap]
        # write CSV by hand (avoid pandas dep)
        csv_path = out / f"edge_stats_{split}.csv"
        with csv_path.open('w', encoding='utf-8') as f:
            f.write("mod,idx,nz_frac,edge_top,edge_bottom,edge_left,edge_right,center_y,center_x,H,W\n")
            agg = []
            for i in chosen:
                for m in mods:
                    arr = load_slice(args.root, m, split, i)
                    s = compute_stats(arr, args.thr, args.edge_band)
                    agg.append(s)
                    f.write(
                        f"{m},{i},{s['nz_frac']:.6f},{s['edge_top']:.6f},{s['edge_bottom']:.6f},{s['edge_left']:.6f},{s['edge_right']:.6f},{s['center_y']},{s['center_x']},{s['H']},{s['W']}\n"
                    )
        # quick aggregate for summary
        # only use the *first* modality to estimate centering/edges shape-wise
        arrs = [load_slice(args.root, mods[0], split, i) for i in chosen]
        stats_first = [compute_stats(a, args.thr, args.edge_band) for a in arrs]
        nz_med = float(np.median([s['nz_frac'] for s in stats_first]))
        edge_t = float(np.median([s['edge_top'] for s in stats_first]))
        edge_b = float(np.median([s['edge_bottom'] for s in stats_first]))
        edge_l = float(np.median([s['edge_left'] for s in stats_first]))
        edge_r = float(np.median([s['edge_right'] for s in stats_first]))
        cy = np.array([s['center_y'] for s in stats_first if s['center_y'] is not None], dtype=float)
        cx = np.array([s['center_x'] for s in stats_first if s['center_x'] is not None], dtype=float)
        H = int(stats_first[0]['H']); W = int(stats_first[0]['W'])
        center = {
            "H": H, "W": W,
            "cy_median": float(np.median(cy)) if cy.size else None,
            "cx_median": float(np.median(cx)) if cx.size else None,
            "cy_mean": float(np.mean(cy)) if cy.size else None,
            "cx_mean": float(np.mean(cx)) if cx.size else None,
            "dist_to_center_mean": float(np.mean(np.sqrt((cy - (H-1)/2)**2 + (cx - (W-1)/2)**2))) if (cy.size and cx.size) else None,
        }
        summary["splits"][split].update({
            "nz_frac_median_first_mod": nz_med,
            "edge_median_first_mod": {
                "top": edge_t, "bottom": edge_b, "left": edge_l, "right": edge_r,
            },
            "center_summary_first_mod": center,
            "stats_csv": str(csv_path),
            "mosaic_png": str(out / f"mosaic_{split}.png"),
        })

    # write summary
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[QC DONE] Wrote:")
    print("  ", out / "summary.json")
    for split in splits:
        print("  ", out / f"mosaic_{split}.png")
        print("  ", out / f"edge_stats_{split}.csv")


if __name__ == "__main__":
    main()
