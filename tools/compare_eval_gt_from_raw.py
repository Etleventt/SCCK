#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从原始 NIfTI 数据直接对比“两种数据准备路径”在评测阶段送入 SSIM/PSNR 的 GT 图像：
- 官方路径（create_brats_dataset.py 风格）：按切片 min-max 到 [0,1]，T1>阈值 生成掩膜（不 pad）。
- 我们路径（old_script/prepare_brats_for_selfrdb_official.py 风格，简化版）：
  体内 union 掩膜做 per-volume 均值归一化；中心 pad 到 256×256；掩膜也 pad。

两条路径随后都按“官方评测口径”做 eval 预处理：
  裁剪到掩膜尺寸（若需要）-> 乘掩膜 -> 每切片归一化（mean/01）。

输出每个切片一张 2×4 面板：
  [row1] A Target(orig) | B Target(orig) | A Mask | B Mask
  [row2] A Target(eval) | B Target(eval) | |A_eval - B_eval| | 空

用法示例：
python tools/compare_eval_gt_from_raw.py \
  --raw_root /path/to/BraTS2021_root \
  --target_modality t2 --source_modality t1 \
  --out_dir logs/inspect/compare_eval_gt_from_raw \
  --num 10 --slice_range 27 127 --norm mean --mask_threshold 0.1
"""

import os
import re
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def find_subjects(root: Path):
    pats = []
    for p in root.glob('BraTS2021_*'):
        if p.is_dir():
            pats.append(p)
    if not pats:
        # fallback: nested train/val/test dirs
        for sp in ['train', 'val', 'test']:
            d = root / sp
            if d.exists():
                pats += [x for x in d.glob('BraTS2021_*') if x.is_dir()]
    return sorted(pats)


def mod_file(sub: Path, m: str) -> Path | None:
    # case-insensitive *_t1.nii.gz, *_T1.nii, etc.
    hits = list(sub.glob(f"*_{m}.nii*"))
    if hits:
        return hits[0]
    ml, mu = m.lower(), m.upper()
    hits = list(sub.glob(f"*_{ml}.nii*")) or list(sub.glob(f"*_{mu}.nii*"))
    if hits:
        return hits[0]
    for p in sub.glob('*.nii*'):
        if re.search(rf"_{re.escape(m)}(\.nii(\.gz)?)$", p.name, flags=re.IGNORECASE):
            return p
    return None


def load_vol_rot90(nifti_path: Path):
    # 与官方脚本一致：直接读，然后每张切片 rot90(-1)
    arr = nib.load(str(nifti_path)).get_fdata().astype(np.float32)
    # 返回 (Z,H,W) 视角，Z 为第三维
    return arr  # 不转 RAS，这样就和官方 create_brats_dataset 的切片方式一致


def slice_minmax01(x2d: np.ndarray):
    x = x2d - np.nanmin(x2d)
    den = np.nanmax(x) + 1e-8
    return (x / den).astype(np.float32)


def center_crop(x, crop_hw):
    h, w = x.shape[-2:]
    ch, cw = crop_hw
    y0 = max(0, h//2 - ch//2)
    x0 = max(0, w//2 - cw//2)
    return x[..., y0:y0+ch, x0:x0+cw]


def mean_norm(x):
    x = np.abs(x)
    den = x.mean(axis=(-1,-2), keepdims=True)
    den = np.where(den == 0, 1.0, den)
    return x / den


def norm_01(x):
    return (x - x.min(axis=(-1,-2), keepdims=True)) / (
        x.max(axis=(-1,-2), keepdims=True) - x.min(axis=(-1,-2), keepdims=True) + 1e-8
    )


def pad_to_center(arr2d, size_hw):
    H, W = arr2d.shape[-2:]
    th, tw = size_hw
    if (H, W) == (th, tw):
        return arr2d
    out = np.zeros((th, tw), dtype=arr2d.dtype)
    py0 = max(0, (th - H)//2)
    px0 = max(0, (tw - W)//2)
    h = min(H, th); w = min(W, tw)
    out[py0:py0+h, px0:px0+w] = arr2d[:h, :w]
    return out


def official_eval_gt(tgt2d: np.ndarray, t12d: np.ndarray, mask_thr: float, norm: str):
    # 官方：slice->rot90(-1)->minmax01；mask 从 T1 同样 minmax01 后阈值
    tgt = slice_minmax01(np.rot90(tgt2d, -1))
    t1s = slice_minmax01(np.rot90(t12d, -1))
    m = (t1s > mask_thr).astype(np.uint8)
    # eval 预处理：裁剪到掩膜尺寸（此处同尺寸，无需裁剪）-> 乘掩膜 -> 每切片归一化
    gt = tgt * m
    if norm == 'mean':
        gt = mean_norm(gt[None, ...])[0]
    else:
        gt = norm_01(gt[None, ...])[0]
    return tgt, m, gt


def ours_eval_gt(vols: dict[str, np.ndarray], z: int, out_size: int, norm: str):
    # vols: {mod: vol (H,W,Z)} 来自 nib.load(...).get_fdata()
    # 简化版：按切片 union 掩膜；per-volume 均值归一化（对 target）；再 pad 到 256；eval：乘掩膜+每切片归一化
    mods = list(vols.keys())
    H, W, Z = vols[mods[0]].shape
    # union 掩膜（3D）
    mask3d = None
    for v in vols.values():
        m3 = (v > 0).astype(np.uint8)
        mask3d = m3 if mask3d is None else (mask3d | m3)
    mask3d = mask3d.astype(np.uint8)
    # per-volume 均值归一化（target 模态）
    target_mod = mods[0]  # 由调用方确保第一项是 target
    v = vols[target_mod].astype(np.float32)
    inside = v[mask3d > 0]
    scale = 1.0 / max(float(inside.mean()) if inside.size else 1.0, 1e-6)
    v_norm = v * scale
    # 取第 z 张切片（未 rot），随后做 rot 与官方一致
    sl = v_norm[..., z]
    sl = np.rot90(sl, -1)
    # union 掩膜在该切片
    m2 = mask3d[..., z]
    m2 = np.rot90(m2, -1).astype(np.uint8)
    # pad 到 out_size
    tgt_pad = pad_to_center(sl, (out_size, out_size))
    m_pad = pad_to_center(m2, (out_size, out_size)).astype(np.uint8)
    # eval：乘掩膜+每切片归一化
    gt = tgt_pad * m_pad
    if norm == 'mean':
        gt = mean_norm(gt[None, ...])[0]
    else:
        gt = norm_01(gt[None, ...])[0]
    return tgt_pad, m_pad, gt


def main():
    ap = argparse.ArgumentParser('Compare GT eval inputs from raw NIfTI (official vs ours)')
    ap.add_argument('--raw_root', required=True, type=str)
    ap.add_argument('--target_modality', default='t2', type=str)
    ap.add_argument('--source_modality', default='t1', type=str)
    ap.add_argument('--extra_mods', default='flair,t1ce', type=str, help='used for union mask if present')
    ap.add_argument('--slice_range', nargs=2, type=int, default=[27, 127])
    ap.add_argument('--num', type=int, default=10)
    ap.add_argument('--out_dir', required=True, type=str)
    ap.add_argument('--out_size', type=int, default=256)
    ap.add_argument('--mask_threshold', type=float, default=0.1)
    ap.add_argument('--norm', default='mean', choices=['mean','01'])
    args = ap.parse_args()

    root = Path(args.raw_root)
    subs = find_subjects(root)
    if not subs:
        raise SystemExit('No subjects found under raw_root.')

    os.makedirs(args.out_dir, exist_ok=True)

    count = 0
    for sub in subs:
        if count >= args.num:
            break
        # gather modalities
        t1_p = mod_file(sub, args.source_modality)
        tgt_p = mod_file(sub, args.target_modality)
        if t1_p is None or tgt_p is None:
            continue
        # load volumes
        t1_v = load_vol_rot90(t1_p)
        tgt_v = load_vol_rot90(tgt_p)
        vols = {args.target_modality: tgt_v}
        for m in [x.strip() for x in args.extra_mods.split(',') if x.strip()]:
            p = mod_file(sub, m)
            if p is not None:
                vols[m] = load_vol_rot90(p)

        z_lo, z_hi = args.slice_range
        Z = tgt_v.shape[-1]
        z_lo = max(0, z_lo)
        z_hi = min(Z, z_hi)
        if z_hi <= z_lo:
            continue
        # pick one representative z roughly in the middle for this subject
        z = (z_lo + z_hi) // 2

        # --- 官方路径 ---
        tgtA_orig, mA, gtA_eval = official_eval_gt(tgt_v[..., z], t1_v[..., z], args.mask_threshold, args.norm)
        # --- 我们路径（简化） ---
        tgtB_orig, mB, gtB_eval = ours_eval_gt(vols, z, args.out_size, args.norm)

        # match display size for diff
        ha, wa = gtA_eval.shape
        hb, wb = gtB_eval.shape
        th, tw = max(ha, hb), max(wa, wb)
        A_eval_disp = pad_to_center(gtA_eval, (th, tw))
        B_eval_disp = pad_to_center(gtB_eval, (th, tw))

        fig, ax = plt.subplots(2, 4, figsize=(16, 8))
        ax[0,0].set_title('A Target (orig)'); ax[0,0].imshow(tgtA_orig, cmap='gray'); ax[0,0].axis('off')
        ax[0,1].set_title('B Target (orig)'); ax[0,1].imshow(tgtB_orig, cmap='gray'); ax[0,1].axis('off')
        ax[0,2].set_title('A Mask'); ax[0,2].imshow(mA, cmap='gray'); ax[0,2].axis('off')
        ax[0,3].set_title('B Mask'); ax[0,3].imshow(mB, cmap='gray'); ax[0,3].axis('off')

        ax[1,0].set_title('A Target (eval)'); ax[1,0].imshow(gtA_eval, cmap='gray'); ax[1,0].axis('off')
        ax[1,1].set_title('B Target (eval)'); ax[1,1].imshow(gtB_eval, cmap='gray'); ax[1,1].axis('off')
        ax[1,2].set_title('|A_eval - B_eval|'); ax[1,2].imshow(np.abs(A_eval_disp - B_eval_disp), cmap='magma'); ax[1,2].axis('off')
        ax[1,3].axis('off')

        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f'{sub.name}_z{z}.png')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        count += 1

    print(f'[done] Saved {count} panels to: {args.out_dir}')


if __name__ == '__main__':
    main()

