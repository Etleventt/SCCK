#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize what actually goes into SSIM/PSNR under the official-aligned evaluation,
and optionally compare against a "full-image" variant（不裁剪到掩膜尺寸，掩膜居中 pad 后整图参与统计）。

不需要预测文件也可以运行（只看预处理差异）。若提供 pred.npy，则同时展示 pred 的两种预处理效果与指标。

面板示例：
- 仅预处理对比（无 pred）：
  [row 1]  source(optional) | target(original) | mask(original)
  [row 2]  target_official  | target_full     | |official-full|
- 提供 pred.npy 时：
  [row 1]  target(original) | pred(original)  | mask(original)
  [row 2]  target_official  | pred_official   | target_full   | pred_full

Where target_eval/pred_eval are the arrays after the exact preprocessing used by
`utils.compute_metrics_official`: center-crop to mask HxW (if provided),
multiply by mask, per-slice normalization (mean or 01), then metrics use
data_range = target_eval.max().

Usage example:
  python tools/inspect_ssim_inputs.py \
    --dataset_dir /home/xiaobin/Projects/SelfRDB/dataset/brats256_selfrdb_official \
    --target_modality t2 --source_modality t1 \
    --out_dir logs/inspect/inspect_ssim_inputs \
    --num 10 --norm mean [--pred_path logs/.../pred.npy] [--compare_full]
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

try:
    # Prefer using the existing helpers for consistency
    from utils import center_crop, mean_norm, norm_01
except Exception:
    # Minimal fallbacks
    def center_crop(x, crop):
        h, w = x.shape[-2:]
        ch, cw = crop
        y0 = max(0, h//2 - ch//2)
        x0 = max(0, w//2 - cw//2)
        return x[..., y0:y0+ch, x0:x0+cw]

    def mean_norm(x):
        x = np.abs(x)
        denom = x.mean(axis=(-1,-2), keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return x / denom

    def norm_01(x):
        return (x - x.min(axis=(-1,-2), keepdims=True)) / (
            x.max(axis=(-1,-2), keepdims=True) - x.min(axis=(-1,-2), keepdims=True) + 1e-8
        )


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


def official_preprocess(gt, prd, mask=None, norm='mean'):
    # gt, prd: [H,W], in [0,1]; mask: [H,W] uint8/0-1 or None
    if mask is not None:
        mh, mw = mask.shape[-2:]
        gt = center_crop(gt[None, ...], (mh, mw))[0]
        prd = center_crop(prd[None, ...], (mh, mw))[0]
        gt = gt * mask
        prd = prd * mask
        if norm == 'mean':
            gt = mean_norm(gt[None, ...])[0]
            prd = mean_norm(prd[None, ...])[0]
        else:
            gt = norm_01(gt[None, ...])[0]
            prd = norm_01(prd[None, ...])[0]
    else:
        if norm == 'mean':
            gt = mean_norm(gt[None, ...])[0]
            prd = mean_norm(prd[None, ...])[0]
        else:
            gt = norm_01(gt[None, ...])[0]
            prd = norm_01(prd[None, ...])[0]
    return gt, prd


def pad_to_center(arr2d, size_hw):
    """Center pad 2D array to target size (H,W)."""
    H, W = arr2d.shape[-2:]
    th, tw = size_hw
    if H == th and W == tw:
        return arr2d
    out = np.zeros((th, tw), dtype=arr2d.dtype)
    py0 = max(0, (th - H) // 2)
    px0 = max(0, (tw - W) // 2)
    h = min(H, th); w = min(W, tw)
    out[py0:py0+h, px0:px0+w] = arr2d[:h, :w]
    return out


def full_preprocess(gt, prd, mask=None, norm='mean'):
    """Full-image variant: do NOT crop images to mask size; center-pad mask to image size then apply."""
    H, W = gt.shape[-2:]
    if mask is not None:
        if mask.shape[-2:] != (H, W):
            mask = pad_to_center(mask, (H, W))
        gt = gt * mask
        prd = prd * mask
    # per-slice normalization on whole image
    if norm == 'mean':
        gt = mean_norm(gt[None, ...])[0]
        prd = mean_norm(prd[None, ...])[0]
    else:
        gt = norm_01(gt[None, ...])[0]
        prd = norm_01(prd[None, ...])[0]
    return gt, prd


def main():
    ap = argparse.ArgumentParser("Inspect SSIM/PSNR evaluation inputs (official-aligned)")
    ap.add_argument('--dataset_dir', required=True, type=str)
    ap.add_argument('--source_modality', default='t1', type=str)
    ap.add_argument('--target_modality', default='t2', type=str)
    ap.add_argument('--split', default='test', type=str, choices=['train','val','test'])
    ap.add_argument('--pred_path', default='', type=str, help='Optional: path to logs/.../test_samples/pred.npy')
    ap.add_argument('--out_dir', required=True, type=str)
    ap.add_argument('--num', default=10, type=int)
    ap.add_argument('--norm', default='mean', choices=['mean','01'])
    ap.add_argument('--compare_full', action='store_true', help='Also compute/show full-image variant alongside official')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load target/source
    tgt_dir = os.path.join(args.dataset_dir, args.target_modality, args.split)
    src_dir = os.path.join(args.dataset_dir, args.source_modality, args.split)
    tgt, _ = load_npy_dir(tgt_dir)
    src = None
    if os.path.isdir(src_dir):
        try:
            src, _ = load_npy_dir(src_dir)
        except Exception:
            src = None

    # Load mask if present
    mask_dir = os.path.join(args.dataset_dir, 'mask', args.split)
    mask = None
    if os.path.isdir(mask_dir):
        try:
            mask, _ = load_npy_dir(mask_dir)
        except Exception:
            mask = None

    # Load predictions (optional)
    pred = None
    if args.pred_path:
        pred = np.load(args.pred_path)
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred[:,0]

    # bring to [0,1]
    tgt = to01_if_needed(tgt)
    if pred is not None:
        pred = to01_if_needed(pred)
    if src is not None:
        src = to01_if_needed(src)

    N = min(args.num, len(tgt), len(pred)) if pred is not None else min(args.num, len(tgt))

    for i in range(N):
        gt = tgt[i]
        mk = mask[i] if isinstance(mask, np.ndarray) and i < len(mask) else None

        if pred is not None:
            pr = pred[i]
            gt_off, pr_off = official_preprocess(gt, pr, mask=mk, norm=args.norm)
            titles_off = None
            if np.max(gt_off) > 0:
                dr = float(np.max(gt_off))
                psnr_o = _psnr(gt_off, pr_off, data_range=dr)
                ssim_o = _ssim(gt_off, pr_off, data_range=dr) * 100.0
                titles_off = f'PSNR(off) {psnr_o:.2f} | SSIM(off) {ssim_o:.2f}%'

            if args.compare_full:
                gt_full, pr_full = full_preprocess(gt, pr, mask=mk, norm=args.norm)
                drf = float(np.max(gt_full)) if np.max(gt_full) > 0 else 1.0
                psnr_f = _psnr(gt_full, pr_full, data_range=drf)
                ssim_f = _ssim(gt_full, pr_full, data_range=drf) * 100.0
                titles_full = f'PSNR(full) {psnr_f:.2f} | SSIM(full) {ssim_f:.2f}%'

                fig, ax = plt.subplots(2, 4, figsize=(16,8))
                ax[0,0].set_title('Target (orig)'); ax[0,0].imshow(gt, cmap='gray'); ax[0,0].axis('off')
                ax[0,1].set_title('Pred (orig)');   ax[0,1].imshow(pr, cmap='gray'); ax[0,1].axis('off')
                ax[0,2].set_title('Mask' if mk is not None else 'Mask (None)'); ax[0,2].imshow(mk if mk is not None else np.zeros_like(gt), cmap='gray'); ax[0,2].axis('off')
                ax[0,3].axis('off')

                ax[1,0].set_title('Target (official)'); ax[1,0].imshow(gt_off, cmap='gray'); ax[1,0].axis('off')
                ax[1,1].set_title(titles_off or 'Pred (official)'); ax[1,1].imshow(pr_off, cmap='gray'); ax[1,1].axis('off')
                ax[1,2].set_title('Target (full)'); ax[1,2].imshow(gt_full, cmap='gray'); ax[1,2].axis('off')
                ax[1,3].set_title(titles_full); ax[1,3].imshow(pr_full, cmap='gray'); ax[1,3].axis('off')
                plt.tight_layout(); out_path = os.path.join(args.out_dir, f'slice_{i}.png')
                fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
            else:
                fig, ax = plt.subplots(2, 3, figsize=(12,8))
                ax[0,0].set_title('Target (orig)'); ax[0,0].imshow(gt, cmap='gray'); ax[0,0].axis('off')
                ax[0,1].set_title('Pred (orig)');   ax[0,1].imshow(pr, cmap='gray'); ax[0,1].axis('off')
                ax[0,2].set_title('Mask' if mk is not None else 'Mask (None)'); ax[0,2].imshow(mk if mk is not None else np.zeros_like(gt), cmap='gray'); ax[0,2].axis('off')
                ax[1,0].set_title('Target (official)'); ax[1,0].imshow(gt_off, cmap='gray'); ax[1,0].axis('off')
                ax[1,1].set_title(titles_off or 'Pred (official)'); ax[1,1].imshow(pr_off, cmap='gray'); ax[1,1].axis('off')
                ax[1,2].axis('off')
                plt.tight_layout(); out_path = os.path.join(args.out_dir, f'slice_{i}.png')
                fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
        else:
            # No pred: compare target under two preprocessing choices
            gt_off, _ = official_preprocess(gt, gt, mask=mk, norm=args.norm)
            if args.compare_full:
                gt_full, _ = full_preprocess(gt, gt, mask=mk, norm=args.norm)
            else:
                gt_full = None
            fig, ax = plt.subplots(2, 3, figsize=(12,8))
            ax[0,0].set_title('Source' if src is not None else 'Source (N/A)')
            ax[0,0].imshow(src[i] if src is not None else np.zeros_like(gt), cmap='gray'); ax[0,0].axis('off')
            ax[0,1].set_title('Target (orig)'); ax[0,1].imshow(gt, cmap='gray'); ax[0,1].axis('off')
            ax[0,2].set_title('Mask' if mk is not None else 'Mask (None)'); ax[0,2].imshow(mk if mk is not None else np.zeros_like(gt), cmap='gray'); ax[0,2].axis('off')
            ax[1,0].set_title('Target (official)'); ax[1,0].imshow(gt_off, cmap='gray'); ax[1,0].axis('off')
            if args.compare_full and gt_full is not None:
                ax[1,1].set_title('Target (full)'); ax[1,1].imshow(gt_full, cmap='gray'); ax[1,1].axis('off')
                ax[1,2].set_title('|official-full|'); ax[1,2].imshow(np.abs(gt_off - gt_full), cmap='magma'); ax[1,2].axis('off')
            else:
                ax[1,1].axis('off'); ax[1,2].axis('off')
            plt.tight_layout(); out_path = os.path.join(args.out_dir, f'slice_{i}.png')
            fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)

    print(f'[done] Saved {N} panels to: {args.out_dir}')


if __name__ == '__main__':
    main()
