#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick scanner for [0,1] range violations in SelfRDB-ready NumpyDataset.
- Reports per (modality, split): total files, #violations, ratio,
  global min/max, pos/neg overshoot stats (max/mean/p95), NaN/Inf counts.
- Dumps a CSV with per-file min/max/overshoot and flags.
- Prints Top-K worst offenders (by positive overshoot and by negative overshoot).
"""
import argparse, os, sys, math, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

def scan_files(root: Path, modalities, splits, pattern="slice_*.npy"):
    rows = []
    for m in modalities:
        for sp in splits:
            files = sorted(glob.glob(str(root / m / sp / pattern)))
            for p in files:
                arr = np.load(p, mmap_mode=None)
                dtype = str(arr.dtype)
                a = arr.astype(np.float32, copy=False)
                nan = np.isnan(a).any()
                inf = np.isinf(a).any()
                amin = float(np.nanmin(a)) if not (nan or inf) else float('nan')
                amax = float(np.nanmax(a)) if not (nan or inf) else float('nan')
                pos_excess = max(0.0, (amax - 1.0)) if not (nan or inf) else float('nan')
                neg_excess = max(0.0, (0.0 - amin)) if not (nan or inf) else float('nan')
                bad = (dtype != 'float32') or nan or inf or (amin < 0.0) or (amax > 1.0)
                rows.append({
                    "modality": m, "split": sp, "path": p,
                    "dtype": dtype, "min": amin, "max": amax,
                    "pos_excess": pos_excess, "neg_excess": neg_excess,
                    "has_nan": nan, "has_inf": inf, "bad": bad,
                })
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame):
    out = []
    for (m, sp), g in df.groupby(["modality","split"]):
        n = len(g)
        bad = int(g["bad"].sum())
        ratio = (bad / n) if n else 0.0

        # stats ignoring NaN
        gpos = g["pos_excess"].replace([np.inf, -np.inf], np.nan).dropna()
        gneg = g["neg_excess"].replace([np.inf, -np.inf], np.nan).dropna()
        gmin = g["min"].replace([np.inf, -np.inf], np.nan)
        gmax = g["max"].replace([np.inf, -np.inf], np.nan)

        row = {
            "modality": m, "split": sp,
            "files": n, "violations": bad, "violation_ratio": round(ratio, 6),
            "global_min": round(float(np.nanmin(gmin)), 6) if len(gmin) else np.nan,
            "global_max": round(float(np.nanmax(gmax)), 6) if len(gmax) else np.nan,
            "pos_max": round(float(np.nanmax(gpos)) if len(gpos) else 0.0, 6),
            "pos_mean": round(float(np.nanmean(gpos)) if len(gpos) else 0.0, 6),
            "pos_p95": round(float(np.nanpercentile(gpos, 95)) if len(gpos) else 0.0, 6),
            "neg_max": round(float(np.nanmax(gneg)) if len(gneg) else 0.0, 6),
            "neg_mean": round(float(np.nanmean(gneg)) if len(gneg) else 0.0, 6),
            "neg_p95": round(float(np.nanpercentile(gneg, 95)) if len(gneg) else 0.0, 6),
            "nan_files": int(g["has_nan"].sum()),
            "inf_files": int(g["has_inf"].sum()),
        }
        out.append(row)
    return pd.DataFrame(out)

def print_topk(df: pd.DataFrame, topk=10):
    def show(dfk, key, title):
        dfk = dfk.sort_values(key, ascending=False).head(topk)
        if len(dfk)==0 or (dfk[key] <= 0).all():
            print(f"[Top-{topk}] {title}: none (>0)")
            return
        print(f"[Top-{topk}] {title}:")
        for _, r in dfk.iterrows():
            print(f"  {r['modality']}/{r['split']}  {key}={r[key]:.6f}  {r['path']}")
    show(df[df["pos_excess"]>0], "pos_excess", "Positive overshoot (max>1)")
    show(df[df["neg_excess"]>0], "neg_excess", "Negative overshoot (min<0)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--modalities", default="t1,t2,t1ce,flair")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--pattern", default="slice_*.npy")
    ap.add_argument("--csv_out", type=Path, default=None)
    ap.add_argument("--summary_out", type=Path, default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--warn_pos", type=float, default=0.01, help="warn if pos_excess max exceeds this (absolute on [0,1])")
    ap.add_argument("--warn_ratio", type=float, default=0.2, help="warn if violation ratio exceeds this")
    args = ap.parse_args()

    mods = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]

    df = scan_files(args.root, mods, splits, args.pattern)
    summ = summarize(df)

    # print summary
    print("\n=== Summary (per modality/split) ===")
    if len(summ):
        print(summ.to_string(index=False))
    else:
        print("No files found.")

    # save CSVs if requested
    if args.csv_out:
        df.to_csv(args.csv_out, index=False)
        print(f"\n[Saved] per-file report -> {args.csv_out}")
    if args.summary_out:
        summ.to_csv(args.summary_out, index=False)
        print(f"[Saved] summary -> {args.summary_out}")

    # show worst offenders
    print("\n=== Worst offenders ===")
    print_topk(df, args.topk)

    # quick overall decision
    global_pos_max = float(np.nanmax(df["pos_excess"])) if len(df) else 0.0
    global_ratio = float((df["bad"].sum() / len(df))) if len(df) else 0.0
    ok = (global_pos_max <= args.warn_pos) and (global_ratio <= args.warn_ratio)
    verdict = "LIKELY NEGLIGIBLE" if ok else "RECOMMEND FIX"
    print(f"\nVerdict: {verdict} | max_pos_excess={global_pos_max:.6f}, violation_ratio={global_ratio:.4f} "
          f"(thresholds: warn_pos={args.warn_pos}, warn_ratio={args.warn_ratio})")

if __name__ == "__main__":
    main()
