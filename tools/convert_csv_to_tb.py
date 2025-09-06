#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert a Lightning CSVLogger metrics.csv into TensorBoard event files.

Usage:
  python tools/convert_csv_to_tb.py \
    --csv logs/experiment_official_rerun/version_7/metrics.csv \
    --out_dir logs/experiment_official_rerun/version_7/tb \
    --align_by step \
    --include val_psnr,val_ssim,val_loss

Then launch TensorBoard:
  tensorboard --logdir logs

Notes:
  - Writes scalar time series for each selected column. Global step is taken
    from the chosen align_by column (step or epoch) if present, else row index.
  - Skips non-numeric cells and header-echo artifacts that occasionally appear
    in Lightning CSVs.
"""

import os
import argparse
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, type=str)
    ap.add_argument('--out_dir', required=False, type=str,
                    help='Directory to write TensorBoard event files. Default: <csv_dir>/tb')
    ap.add_argument('--align_by', default='step', choices=['step','epoch'])
    ap.add_argument('--include', default='', type=str,
                    help='Comma-separated columns to include. Empty=all numeric except epoch/step')
    ap.add_argument('--chunksize', default=20000, type=int)
    return ap.parse_args()


def available_columns(csv_path: str):
    with open(csv_path, 'r', encoding='utf-8') as fh:
        header = fh.readline().strip().split(',')
    return header


def main():
    args = parse_args()
    csv_path = args.csv
    out_dir = args.out_dir or os.path.join(os.path.dirname(csv_path), 'tb')
    os.makedirs(out_dir, exist_ok=True)

    header = available_columns(csv_path)
    include = [c.strip() for c in args.include.split(',') if c.strip()] if args.include else []

    # Decide which columns to log
    scalar_cols = []
    if include:
        scalar_cols = [c for c in include if c in header]
    else:
        # Heuristic: take all columns except epoch/step and lr-* that look like scalars
        for c in header:
            if c in ('epoch','step'): continue
            if c.startswith('lr-'): continue
            scalar_cols.append(c)

    usecols = [x for x in (['epoch','step'] + scalar_cols) if x in header]
    if not scalar_cols:
        print('No scalar columns found to log. Check --include or CSV header.')
        return

    writer = SummaryWriter(log_dir=out_dir)
    n_rows = 0
    n_points = 0

    for chunk in pd.read_csv(csv_path, chunksize=args.chunksize, usecols=lambda c: c in usecols):
        # Coerce step/epoch to numeric if present; filter obviously bad rows
        for c in ('epoch','step'):
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
        # Keep rows that have at least one selected scalar non-null and a valid align key if present
        mask_any = False
        for m in scalar_cols:
            if m in chunk.columns:
                mm = chunk[m].apply(lambda v: pd.notna(v))
                mask_any = mm if mask_any is False else (mask_any | mm)
        if mask_any is False:
            continue
        sub = chunk.loc[mask_any, :]
        # Log each scalar
        for _, row in sub.iterrows():
            if args.align_by in sub.columns and pd.notna(row.get(args.align_by, None)):
                step = int(float(row[args.align_by]))
            else:
                step = n_rows
            for m in scalar_cols:
                if m not in sub.columns: continue
                try:
                    v = float(row[m])
                except Exception:
                    continue
                if pd.isna(v):
                    continue
                writer.add_scalar(m, v, global_step=step)
                n_points += 1
            n_rows += 1

    writer.flush(); writer.close()
    print(f'Wrote TensorBoard events to: {out_dir} (points logged: {n_points})')


if __name__ == '__main__':
    main()

