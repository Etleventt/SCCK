#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare the ground-truth images that are fed into SSIM/PSNR under two
different dataset preparation pipelines (e.g., official create_brats_dataset.py
vs old_script/prepare_brats_for_selfrdb_official.py).

This script shows, for a handful of slices, how the GT image looks:
  - as originally stored (after each pipeline's saving), and
  - after the exact "official-aligned" evaluation preprocessing used by
    utils.compute_metrics_official (crop-to-mask -> multiply -> per-slice norm).

It optionally also shows the per-pixel absolute difference |GT_A_eval - GT_B_eval|.

Usage:
  python tools/compare_eval_gt.py \
    --dataset_a /path/to/dataset_official \
    --dataset_b /path/to/dataset_ours \
    --target_modality t2 --source_modality t1 \
    --out_dir logs/inspect/compare_eval_gt \
    --num 10 --norm mean

Notes:
  - Alignment is by slice index only. If two datasets used different slice
    selections/order, visual comparison might not be apples-to-apples.
  - Masks are taken from each dataset's mask/<split>/slice_*.npy if present.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_npy_dir(dir_path):
    files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    arrs = [np.load(os.path.join(dir_path, f)) for f in files]
    return np.asarray(arrs), files


def to01_if_needed(x):
    x = np.asarray(x)
    if np.nanmin(x) < -0.1:
        x = ((x + 1.0) / 2.0)
    return np.clip(x, 0.0, 1.0)


def center_crop(x, crop_hw):
    h, w = x.shape[-2:]
    ch, cw = crop_hw
    y0 = max(0, h // 2 - ch // 2)
    x0 = max(0, w // 2 - cw // 2)
    return x[..., y0:y0 + ch, x0:x0 + cw]


def mean_norm(x):
    x = np.abs(x)
    denom = x.mean(axis=(-1, -2), keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


def norm_01(x):
    return (x - x.min(axis=(-1, -2), keepdims=True)) / (
        x.max(axis=(-1, -2), keepdims=True) - x.min(axis=(-1, -2), keepdims=True) + 1e-8
    )


def official_preprocess_gt(gt, mask=None, norm='mean'):
    # gt: [H,W] float in [0,1]; mask: [H,W] or None
    if mask is not None:
        mh, mw = mask.shape[-2:]
        gt = center_crop(gt[None, ...], (mh, mw))[0]
        gt = gt * mask
        if norm == 'mean':
            gt = mean_norm(gt[None, ...])[0]
        else:
            gt = norm_01(gt[None, ...])[0]
    else:
        if norm == 'mean':
            gt = mean_norm(gt[None, ...])[0]
        else:
            gt = norm_01(gt[None, ...])[0]
    return gt


def pad_to_center(arr2d, size_hw):
    H, W = arr2d.shape[-2:]
    th, tw = size_hw
    if (H, W) == (th, tw):
        return arr2d
    out = np.zeros((th, tw), dtype=arr2d.dtype)
    py0 = max(0, (th - H) // 2)
    px0 = max(0, (tw - W) // 2)
    h = min(H, th)
    w = min(W, tw)
    out[py0:py0 + h, px0:px0 + w] = arr2d[:h, :w]
    return out


def match_size(a, b):
    """Center-pad both inputs to a common max(H), max(W) for visualization."""
    ha, wa = a.shape[-2:]
    hb, wb = b.shape[-2:]
    th, tw = max(ha, hb), max(wa, wb)
    return pad_to_center(a, (th, tw)), pad_to_center(b, (th, tw))


def main():
    ap = argparse.ArgumentParser('Compare GT eval inputs from two datasets')
    ap.add_argument('--dataset_a', required=True, type=str, help='Dataset root A (e.g., official)')
    ap.add_argument('--dataset_b', required=True, type=str, help='Dataset root B (e.g., ours)')
    ap.add_argument('--target_modality', default='t2', type=str)
    ap.add_argument('--source_modality', default='t1', type=str)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--out_dir', required=True, type=str)
    ap.add_argument('--num', default=10, type=int)
    ap.add_argument('--norm', default='mean', choices=['mean', '01'])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # A: load GT and mask
    tgt_a, _ = load_npy_dir(os.path.join(args.dataset_a, args.target_modality, args.split))
    mask_a = None
    mask_dir_a = os.path.join(args.dataset_a, 'mask', args.split)
    if os.path.isdir(mask_dir_a):
        try:
            mask_a, _ = load_npy_dir(mask_dir_a)
        except Exception:
            mask_a = None

    # B: load GT and mask
    tgt_b, _ = load_npy_dir(os.path.join(args.dataset_b, args.target_modality, args.split))
    mask_b = None
    mask_dir_b = os.path.join(args.dataset_b, 'mask', args.split)
    if os.path.isdir(mask_dir_b):
        try:
            mask_b, _ = load_npy_dir(mask_dir_b)
        except Exception:
            mask_b = None

    tgt_a = to01_if_needed(tgt_a)
    tgt_b = to01_if_needed(tgt_b)
    if isinstance(mask_a, np.ndarray):
        mask_a = (mask_a > 0).astype(np.uint8)
    if isinstance(mask_b, np.ndarray):
        mask_b = (mask_b > 0).astype(np.uint8)

    N = min(args.num, len(tgt_a), len(tgt_b))
    for i in range(N):
        gt_a = tgt_a[i]
        gt_b = tgt_b[i]
        mk_a = mask_a[i] if isinstance(mask_a, np.ndarray) and i < len(mask_a) else None
        mk_b = mask_b[i] if isinstance(mask_b, np.ndarray) and i < len(mask_b) else None

        gt_a_eval = official_preprocess_gt(gt_a, mask=mk_a, norm=args.norm)
        gt_b_eval = official_preprocess_gt(gt_b, mask=mk_b, norm=args.norm)

        # For a fair visual diff, match sizes via center-pad to common size
        gt_a_eval_v, gt_b_eval_v = match_size(gt_a_eval, gt_b_eval)

        fig, ax = plt.subplots(2, 4, figsize=(16, 8))
        ax[0,0].set_title('A Target (orig)')
        ax[0,0].imshow(gt_a, cmap='gray'); ax[0,0].axis('off')
        ax[0,1].set_title('B Target (orig)')
        ax[0,1].imshow(gt_b, cmap='gray'); ax[0,1].axis('off')
        ax[0,2].set_title('A Mask' if mk_a is not None else 'A Mask (None)')
        ax[0,2].imshow(mk_a if mk_a is not None else np.zeros_like(gt_a), cmap='gray'); ax[0,2].axis('off')
        ax[0,3].set_title('B Mask' if mk_b is not None else 'B Mask (None)')
        ax[0,3].imshow(mk_b if mk_b is not None else np.zeros_like(gt_b), cmap='gray'); ax[0,3].axis('off')

        ax[1,0].set_title('A Target (eval)')
        ax[1,0].imshow(gt_a_eval, cmap='gray'); ax[1,0].axis('off')
        ax[1,1].set_title('B Target (eval)')
        ax[1,1].imshow(gt_b_eval, cmap='gray'); ax[1,1].axis('off')
        ax[1,2].set_title('|A_eval - B_eval| (matched size)')
        ax[1,2].imshow(np.abs(gt_a_eval_v - gt_b_eval_v), cmap='magma'); ax[1,2].axis('off')
        ax[1,3].axis('off')

        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f'slice_{i}.png')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f'[done] Saved {N} panels to: {args.out_dir}')


if __name__ == '__main__':
    main()

