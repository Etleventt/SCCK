import os
import warnings
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import matplotlib.pyplot as plt

def _to01_t(x):
    # x: torch.Tensor [B, C, H, W]
    if torch.nan_to_num(x).min() < -0.1:   # 判定为 [-1,1]
        x = (x + 1.0) / 2.0
    return x.clamp(0, 1)

def save_image_pair(x0, x0_pred, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x0 = _to01_t(x0)
    x0_pred = _to01_t(x0_pred)

    n_image = min(4, x0.shape[0])
    fig, axes = plt.subplots(nrows=2, ncols=n_image, figsize=(n_image*2, 4))
    if n_image == 1:
        axes = axes[..., None]

    for i in range(n_image):
        # 单通道更稳：imshow(H,W)
        axes[0, i].imshow(x0[i, 0].detach().cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(x0_pred[i, 0].detach().cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout(pad=0.1)
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()


# -----------------------------
# Helpers (mostly unchanged)
# -----------------------------
def to_norm(x):
    x = x/2
    x = x + 0.5
    return x.clip(0, 1)

def norm_01(x):
    return (x - x.min(axis=(-1,-2), keepdims=True)) / (
        x.max(axis=(-1,-2), keepdims=True) - x.min(axis=(-1,-2), keepdims=True) + 1e-8
    )

def mean_norm(x):
    x = np.abs(x)
    denom = x.mean(axis=(-1,-2), keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom

def apply_mask_and_norm(x, mask, norm_func):
    x = x * mask
    x = norm_func(x)
    return x

def center_crop(x, crop):
    h, w = x.shape[-2:]
    ch, cw = crop
    y0 = max(0, h//2 - ch//2)
    x0 = max(0, w//2 - cw//2)
    return x[..., y0:y0+ch, x0:x0+cw]


# -----------------------------
# Robust zoomed inset (reworked to be safe for 64/128/256)
# -----------------------------
def _clip_box(x1, x2, y1, y2, H, W):
    x1 = int(max(0, min(W-1, x1)))
    x2 = int(max(0, min(W,   x2)))
    y1 = int(max(0, min(H-1, y1)))
    y2 = int(max(0, min(H,   y2)))
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return x1, x2, y1, y2

def ax_zoomed(
    ax,
    im,
    zoom_region,
    zoom_size,
    zoom_edge_color='yellow'
):
    """
    Draw main image with a zoomed inset.
    - im: 2D array in [0,1] or [0,255] (we just display as gray)
    - zoom_region: [x1,x2,y1,y2] in image coordinates; will be safely clipped
    - zoom_size:  [left,bottom,width,height] in axes fraction for inset
    """
    H, W = im.shape[-2], im.shape[-1]
    # display main
    ax.imshow(np.flip(im, axis=0), origin='lower', cmap='gray')
    ax.axis('off')

    # clip region
    x1, x2, y1, y2 = zoom_region
    x1, x2, y1, y2 = _clip_box(x1, x2, y1, y2, H, W)

    # inset
    axins = ax.inset_axes(zoom_size, xlim=(x1, x2), ylim=(y1, y2))
    axins.imshow(np.flip(im, axis=0), cmap='gray')

    # inset border
    for spine in axins.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)

    axins.set_xticks([]); axins.set_yticks([])
    ax.indicate_inset_zoom(axins, edgecolor=zoom_edge_color, linewidth=3)


# -----------------------------
# New: auto-zoom region chooser for any resolution
# -----------------------------
def _auto_zoom_region(src, tgt, prd, min_zoom=12, ratio=0.25):
    """
    Pick a zoom region centered at the max-error position inside brain.
    Fallback: image center.
    Returns [x1,x2,y1,y2].
    """
    H, W = tgt.shape
    # brain mask from target
    mask = (tgt > 0).astype(np.float32)
    # error
    err = np.abs((prd.astype(np.float32) - tgt.astype(np.float32))) * mask

    if mask.sum() >= 4 and err.max() > 0:
        cy, cx = np.unravel_index(int(np.argmax(err)), err.shape)
    else:
        cy, cx = H // 2, W // 2

    win = max(int(min(H, W) * ratio), int(min_zoom))
    half = win // 2
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half
    # let ax_zoomed clip finally, but keep roughly valid here
    return [x1, x2, y1, y2]


# -----------------------------
# save_eval_images: now auto-computes zoom for any size
# -----------------------------
def save_eval_images(
    source_images,
    target_images,
    pred_images,
    psnrs,
    ssims,
    save_path
):
    # Squeeze channel dim if exists
    source_images = source_images.squeeze()
    target_images = target_images.squeeze()
    pred_images  = pred_images.squeeze()

    # If images between [-1, 1], scale to [0, 1]
    if np.nanmin(source_images) < -0.1:
        source_images = ((source_images + 1) / 2).clip(0, 1)
    if np.nanmin(target_images) < -0.1:
        target_images = ((target_images + 1) / 2).clip(0, 1)
    if np.nanmin(pred_images) < -0.1:
        pred_images = ((pred_images + 1) / 2).clip(0, 1)

    # ensure [0,1]
    source_images = np.clip(source_images, 0, 1)
    target_images = np.clip(target_images, 0, 1)
    pred_images   = np.clip(pred_images, 0, 1)

    plt.style.use('dark_background')

    os.makedirs(os.path.join(save_path, 'sample_images'), exist_ok=True)

    # inset location (same风格): 在主图下方靠内
    zoom_size = [0, -0.4, 1, 0.47]

    for i in range(len(source_images)):
        src = source_images[i]
        tgt = target_images[i]
        prd = pred_images[i]

        # 自动选取 zoom 区域（对 64/128/256 都合适）
        zoom_region = _auto_zoom_region(src, tgt, prd, min_zoom=12, ratio=0.25)

        fig, ax = plt.subplots(1, 3, figsize=(12*1.5, 8*1.5))

        ax_zoomed(ax[0], mean_norm(src[None, ...])[0], zoom_region, zoom_size)
        ax_zoomed(ax[1], mean_norm(tgt[None, ...])[0], zoom_region, zoom_size)
        ax_zoomed(ax[2], mean_norm(prd[None, ...])[0], zoom_region, zoom_size)

        ax[0].set_title('Source')
        ax[1].set_title('Target')
        ax[2].set_title(f'PSNR: {psnrs[i]:.2f}\nSSIM: {ssims[i]:.2f}')

        path = os.path.join(save_path, 'sample_images', f'slice_{i}.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)


# -----------------------------
# save_preds (unchanged, keep normalization to [0,1])
# -----------------------------
def save_preds(preds, path):
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
    preds = ((preds + 1) / 2).clip(0, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True    )
    np.save(path, preds)


def compute_metrics(
    gt_images,
    pred_images,
    mask=None,
    norm=None,                 # 评测阶段不做 per-slice 归一化；保留参数但不使用
    subject_ids=None,
    report_path=None,
    min_valid_pixels=64,
    eps_range=1e-6
):
    import warnings
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # squeeze 到 [N,H,W]
        gt_images   = gt_images.squeeze() if gt_images.ndim == 4 else gt_images
        pred_images = pred_images.squeeze() if pred_images.ndim == 4 else pred_images
        gt_images   = gt_images[None, ...]   if gt_images.ndim == 2   else gt_images
        pred_images = pred_images[None, ...] if pred_images.ndim == 2 else pred_images
        assert gt_images.shape == pred_images.shape, "GT 和 Pred 形状必须一致"

        # to numpy + [-1,1] -> [0,1]
        if isinstance(gt_images, torch.Tensor):   gt_images = gt_images.cpu().numpy()
        if isinstance(pred_images, torch.Tensor): pred_images = pred_images.cpu().numpy()
        if np.nanmin(gt_images)   < -0.1: gt_images   = ((gt_images   + 1) / 2).clip(0, 1)
        if np.nanmin(pred_images) < -0.1: pred_images = ((pred_images + 1) / 2).clip(0, 1)
        gt_images   = np.clip(gt_images,   0.0, 1.0).astype(np.float32)
        pred_images = np.clip(pred_images, 0.0, 1.0).astype(np.float32)

        N, H, W = gt_images.shape[0], gt_images.shape[-2], gt_images.shape[-1]

        # 处理 mask -> [N,H,W] 的 bool（只乘，不再归一化/裁剪）
        if mask is not None:
            if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
            if mask.ndim == 2:
                mask = np.broadcast_to(mask[None, ...], (N, H, W))
            elif mask.ndim == 3 and mask.shape[-2:] != (H, W):
                mh, mw = mask.shape[-2:]
                y0 = max(0, mh//2 - H//2); x0 = max(0, mw//2 - W//2)
                mask = mask[:, y0:y0+H, x0:x0+W]
            mask = (mask > 0)
            gt_images   = gt_images * mask
            pred_images = pred_images * mask
        else:
            mask = np.ones((N, H, W), dtype=bool)

        psnr_values, ssim_values = [], []
        valid_ids = []

        for i in range(N):
            gt  = gt_images[i]
            prd = pred_images[i]
            m   = mask[i]

            # 有效像素太少 -> 跳过
            if int(m.sum()) < min_valid_pixels:
                continue

            # 官方风格的 data_range：取 gt.max()（注意不是 1.0）
            dr = float(gt.max())  # 之前你们就是这么传的
            if dr < eps_range:
                # 动态范围≈0，官方会导致 -inf/NaN；这里直接跳过该切片
                continue

            # ----- PSNR -----
            psnr_val = _psnr(gt, prd, data_range=dr)
            # ----- SSIM -----
            ssim_val = _ssim(gt, prd, data_range=dr) * 100.0

            if np.isfinite(psnr_val) and np.isfinite(ssim_val):
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
                valid_ids.append(i)

        psnr_values = np.asarray(psnr_values, dtype=np.float64)
        ssim_values = np.asarray(ssim_values, dtype=np.float64)

        psnr_mean = float(np.mean(psnr_values)) if psnr_values.size else float('nan')
        psnr_std  = float(np.std (psnr_values)) if psnr_values.size else float('nan')
        ssim_mean = float(np.mean(ssim_values)) if ssim_values.size else float('nan')
        ssim_std  = float(np.std (ssim_values)) if ssim_values.size else float('nan')

        if report_path is not None:
            skipped = N - len(valid_ids)
            with open(report_path, 'w') as f:
                f.write(f'PSNR: {psnr_mean:.2f} ± {psnr_std:.2f}\n')
                f.write(f'SSIM: {ssim_mean:.2f} ± {ssim_std:.2f}\n')
                f.write(f'Valid/Total slices: {len(valid_ids)}/{N} (skipped={skipped})\n')

        return {
            'psnr_mean': psnr_mean, 'psnr_std': psnr_std,
            'ssim_mean': ssim_mean, 'ssim_std': ssim_std,
            'psnrs': psnr_values, 'ssims': ssim_values,
            'subject_reports': {}
        }
        

def compute_metrics_official(
    gt_images,
    pred_images,
    mask=None,
    norm: str = 'mean',
    subject_ids=None,
    report_path=None,
):
    """Official-aligned PSNR/SSIM computation.

    Behavior matches icon-lab/SelfRDB: center-crop images to mask size (if given),
    multiply by mask, per-slice normalize (mean or 01), and compute
    PSNR/SSIM using skimage with data_range=gt.max() per-slice. SSIM is reported in %.
    """
    import warnings
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Squeeze to [N,H,W]
        gt_images   = gt_images.squeeze() if getattr(gt_images, 'ndim', 0) == 4 else gt_images
        pred_images = pred_images.squeeze() if getattr(pred_images, 'ndim', 0) == 4 else pred_images
        gt_images   = gt_images[None, ...]   if getattr(gt_images, 'ndim', 0) == 2   else gt_images
        pred_images = pred_images[None, ...] if getattr(pred_images, 'ndim', 0) == 2 else pred_images
        assert gt_images.shape == pred_images.shape, "GT 和 Pred 形状必须一致"

        # To numpy + [-1,1] -> [0,1]
        if isinstance(gt_images, torch.Tensor):   gt_images = gt_images.detach().cpu().numpy()
        if isinstance(pred_images, torch.Tensor): pred_images = pred_images.detach().cpu().numpy()
        if np.nanmin(gt_images)   < -0.1: gt_images   = ((gt_images   + 1) / 2).clip(0, 1)
        if np.nanmin(pred_images) < -0.1: pred_images = ((pred_images + 1) / 2).clip(0, 1)

        # Choose normalization
        if norm == 'mean':
            norm_func = mean_norm
        elif norm == '01':
            norm_func = norm_01
        else:
            norm_func = mean_norm

        # Apply mask and normalization, official-style: crop images to mask size first
        if mask is not None:
            if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()
            # center-crop images to mask HxW
            gt_images   = center_crop(gt_images, mask.shape[-2:])
            pred_images = center_crop(pred_images, mask.shape[-2:])
            # per-slice normalize within mask
            gt_images   = apply_mask_and_norm(gt_images, mask, norm_func)
            pred_images = apply_mask_and_norm(pred_images, mask, norm_func)
        else:
            # global per-slice normalization
            gt_images   = norm_func(gt_images)
            pred_images = norm_func(pred_images)

        psnr_values = []
        ssim_values = []

        for gt, prd in zip(gt_images, pred_images):
            gt  = gt.squeeze()
            prd = prd.squeeze()
            dr = float(np.max(gt))
            psnr_val = _psnr(gt, prd, data_range=dr)
            ssim_val = _ssim(gt, prd, data_range=dr) * 100.0
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)

        psnr_values = np.asarray(psnr_values)
        ssim_values = np.asarray(ssim_values)

        subject_reports = {}
        if subject_ids is not None:
            for i in np.unique(subject_ids):
                idx = np.where(subject_ids == i)[0]
                subject_report = {
                    'psnrs': psnr_values[idx],
                    'ssims': ssim_values[idx],
                    'psnr_mean': np.nanmean(psnr_values[idx]),
                    'ssim_mean': np.nanmean(ssim_values[idx]),
                    'psnr_std': np.nanstd(psnr_values[idx]),
                    'ssim_std': np.nanstd(ssim_values[idx])
                }
                subject_reports[i] = subject_report

        if subject_ids is not None and len(subject_reports) > 0:
            psnr_mean = float(np.nanmean([r['psnr_mean'] for r in subject_reports.values()]))
            ssim_mean = float(np.nanmean([r['ssim_mean'] for r in subject_reports.values()]))
            psnr_std  = float(np.nanstd ([r['psnr_mean'] for r in subject_reports.values()]))
            ssim_std  = float(np.nanstd ([r['ssim_mean'] for r in subject_reports.values()]))
        else:
            psnr_mean = float(np.nanmean(psnr_values))
            ssim_mean = float(np.nanmean(ssim_values))
            psnr_std  = float(np.nanstd (psnr_values))
            ssim_std  = float(np.nanstd (ssim_values))

        if report_path is not None:
            with open(report_path, 'w') as f:
                f.write(f'PSNR: {psnr_mean:.2f} ± {psnr_std:.2f}\n')
                f.write(f'SSIM: {ssim_mean:.2f} ± {ssim_std:.2f}\n')

        return {
            'psnr_mean': psnr_mean,
            'ssim_mean': ssim_mean,
            'psnr_std': psnr_std,
            'ssim_std': ssim_std,
            'psnrs': psnr_values,
            'ssims': ssim_values,
            'subject_reports': subject_reports
        }
