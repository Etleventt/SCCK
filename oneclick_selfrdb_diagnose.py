#!/usr/bin/env python3
"""
One-click pipeline for SelfRDB data diagnostics, baselines, and test-time sweeps.

What it does (sequentially unless flags disable parts):
  1) Single-GPU evaluation of a given SelfRDB checkpoint (avoids DDP test duplication).
  2) Baselines:
      - Global linear mapping (y ≈ a*x + b) on ROI.
      - Optional per-slice linear mapping.
      - Optional tiny UNet + L1 (ROI-weighted) baseline.
  3) Test-time sampling sweep for the same checkpoint by modifying config (n_steps, n_recursions, optional sampler keys if exist).
  4) Data diagnostics:
      - ROI saturation (at 0 or 1) for T2.
      - T1–T2 ROI Pearson correlation distribution.
      - Mask area ratio vs. linear-baseline PSNR scatter.
      - Top-K error heatmaps (using linear-baseline errors).
  5) (If available) Call existing repo scripts: audit_selfrdb_brats.py and qc_report_selfrdb.py with safe defaults.

Outputs are saved under --out (default: ./oneclick_out/<timestamp>/).

Requirements: numpy, pyyaml, torch, matplotlib, tqdm; optional: scikit-learn, scipy. If missing, the script will try to degrade gracefully.

Usage example:
  python oneclick_selfrdb_diagnose.py \
    --selfrdb_dir ~/Projects/SelfRDB \
    --config config_64.yaml \
    --ckpt logs/experiment/version_9/checkpoints/epoch=89-step=13680.ckpt \
    --dataset_root ~/Projects/SelfRDB/dataset/brats64_ref_t1 \
    --run_unet_baseline 0

"""
import argparse
import os
import sys
import json
import time
import math
import glob
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    print("[FATAL] PyTorch is required.")
    raise

try:
    import yaml
except Exception:
    yaml = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Optional deps
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from scipy.ndimage import uniform_filter
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -------------------------
# Utilities
# -------------------------

def now_tag():
    return time.strftime('%Y%m%d_%H%M%S')

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# PSNR for [0,1] images

def psnr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= eps:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse + eps))

# Light-weight SSIM (window-based). If SciPy unavailable, fall back to a simplified version.

def ssim_np(img1: np.ndarray, img2: np.ndarray, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    if SCIPY_OK:
        # 8x8 uniform window
        win = 8
        mu1 = uniform_filter(img1, size=win)
        mu2 = uniform_filter(img2, size=win)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = uniform_filter(img1 * img1, size=win) - mu1_sq
        sigma2_sq = uniform_filter(img2 * img2, size=win) - mu2_sq
        sigma12 = uniform_filter(img1 * img2, size=win) - mu1_mu2
    else:
        # Crude fallback (global stats)
        mu1 = img1.mean(); mu2 = img2.mean()
        sigma1_sq = img1.var(); sigma2_sq = img2.var(); sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        mu1_sq = mu1 * mu1; mu2_sq = mu2 * mu2; mu1_mu2 = mu1 * mu2
    ssim_map_num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_map_den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = ssim_map_num / (ssim_map_den + 1e-8)
    return float(np.mean(ssim_map))

@dataclass
class Paths:
    selfrdb_dir: str
    config: str
    ckpt: str
    dataset_root: str
    out_root: str

# -------------------------
# Dataset helpers
# -------------------------

class PairDataset(Dataset):
    """Loads t1/t2/mask slices from NumpyDataset structure.
       Optionally limits to a subset of indices (for speed)."""
    def __init__(self, root: str, split: str, max_items: Optional[int] = None):
        self.root = os.path.expanduser(root)
        self.t1 = sorted(glob.glob(os.path.join(self.root, f't1/{split}/slice_*.npy')))
        self.t2 = [p.replace('/t1/', '/t2/') for p in self.t1]
        self.mask = [p.replace('/t1/', '/mask/') for p in self.t1]
        assert len(self.t1) == len(self.t2) == len(self.mask), 'Modality counts mismatch.'
        if max_items:
            self.t1 = self.t1[:max_items]
            self.t2 = self.t2[:max_items]
            self.mask = self.mask[:max_items]

    def __len__(self):
        return len(self.t1)

    def __getitem__(self, idx):
        # Use memory-mapped loads to avoid heavy I/O stalls
        x = np.load(self.t1[idx], mmap_mode='r').astype(np.float32)
        y = np.load(self.t2[idx], mmap_mode='r').astype(np.float32)
        m = np.load(self.mask[idx], mmap_mode='r').astype(np.float32)
        # add channel dim
        return (
            torch.from_numpy(x)[None, ...],
            torch.from_numpy(y)[None, ...],
            torch.from_numpy(m)[None, ...],
            os.path.basename(self.t1[idx])
        )

# -------------------------
# Baselines
# -------------------------

def fit_global_linear(dataset: PairDataset, max_pixels: int = 5_000_000) -> Tuple[float, float]:
    """Fit global y ≈ a*x + b using ROI pixels (mask>0). Returns (a, b).
       Gracefully handles KeyboardInterrupt by using collected samples so far."""
    xs, ys = [], []
    picked = 0
    try:
        for i in tqdm(range(len(dataset)), desc='[LinearFit] Accum'):
            x, y, m, _ = dataset[i]
            x, y, m = x.numpy()[0], y.numpy()[0], (m.numpy()[0] > 0)
            xr = x[m]; yr = y[m]
            if xr.size == 0:
                continue
            xs.append(xr.flatten()); ys.append(yr.flatten()); picked += xr.size
            if picked >= max_pixels:
                break
    except KeyboardInterrupt:
        print('[LinearFit] Interrupted. Proceeding with collected pixels...')

    if not xs:
        return 1.0, 0.0
    X = np.concatenate(xs); Y = np.concatenate(ys)
    if SKLEARN_OK:
        reg = LinearRegression().fit(X.reshape(-1,1), Y)
        a = float(reg.coef_[0]); b = float(reg.intercept_)
    else:
        # closed-form for simple linear regression
        xm, ym = X.mean(), Y.mean()
        denom = ((X - xm)**2).sum() + 1e-8
        a = float(((X - xm)*(Y - ym)).sum() / denom)
        b = float(ym - a*xm)
    return a, b

def eval_global_linear(dataset: PairDataset, a: float, b: float, max_items: Optional[int] = None) -> Dict[str, float]:
    psnrs, ssims = [] , []
    for i in tqdm(range(len(dataset)), desc='[LinearEval]'):
        x, y, m, _ = dataset[i]
        x, y, m = x.numpy()[0], y.numpy()[0], (dataset[i][2].numpy()[0] > 0)
        yhat = a*x + b
        yhat = np.clip(yhat, 0.0, 1.0)
        psnrs.append(psnr(y, yhat))
        try:
            ssims.append(ssim_np(y[m], yhat[m]))
        except Exception:
            ssims.append(ssim_np(y, yhat))
        if max_items and (i+1) >= max_items:
            break
    return {
        'PSNR_mean': float(np.mean(psnrs)), 'PSNR_std': float(np.std(psnrs)),
        'SSIM_mean': float(np.mean(ssims)), 'SSIM_std': float(np.std(ssims)),
        'N': int(len(psnrs)), 'a': a, 'b': b,
    }

# Tiny UNet
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNetTiny(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.bott = DoubleConv(base*2, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.conv1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        b  = self.bott(self.pool2(d2))
        u2 = self.up2(b)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))
        return self.out(c1)

def train_unet_baseline(dataset_train: PairDataset, dataset_val: PairDataset, out_dir: str,
                         epochs: int = 3, batch_size: int = 64, lr: float = 1e-3,
                         roi_weight: float = 4.0, device: str = 'cuda') -> Dict[str, float]:
    model = UNetTiny().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    def loss_fn(pred, target, mask):
        # ROI weighted L1
        w = 1.0 + (roi_weight - 1.0) * (mask > 0).float()
        return (w * (pred - target).abs()).mean()
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)
    best = {'val_psnr': -1, 'epoch': -1}
    for ep in range(epochs):
        model.train();
        for x,y,m,_ in tqdm(train_loader, desc=f'[UNet] epoch {ep+1}/{epochs}'):
            x,y,m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y, m)
            loss.backward(); opt.step()
        # val
        model.eval(); psnrs=[]; ssims=[]
        with torch.no_grad():
            for x,y,m,_ in val_loader:
                x,y,m = x.to(device), y.to(device), m.to(device)
                pred = model(x)
                p = pred.clamp(0,1).cpu().numpy(); yy=y.cpu().numpy(); mm=m.cpu().numpy()>0
                for i in range(p.shape[0]):
                    psnrs.append(psnr(yy[i,0], p[i,0]))
                    try:
                        ssims.append(ssim_np(yy[i,0][mm[i,0]], p[i,0][mm[i,0]]))
                    except Exception:
                        ssims.append(ssim_np(yy[i,0], p[i,0]))
        cur = float(np.mean(psnrs))
        if cur > best['val_psnr']:
            best.update({'val_psnr': cur, 'epoch': ep+1})
            torch.save(model.state_dict(), os.path.join(out_dir, 'unet_tiny_best.pt'))
        print(f"[UNet] val PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f}; SSIM: {np.mean(ssims)*100:.2f}%")
    return {'best_val_psnr': best['val_psnr'], 'best_epoch': best['epoch']}

# -------------------------
# SelfRDB test runner & sweeps
# -------------------------

def run_selfrdb_test(paths: Paths, override_cfg: Optional[Dict]=None, tag: str='base') -> Dict[str, float]:
    """Runs `python main.py test ...` on single GPU and parses PSNR/SSIM from stdout.
       If override_cfg is provided, write a temp YAML (merging keys) and use it.
    """
    work_dir = paths.selfrdb_dir
    config_path = os.path.join(work_dir, paths.config)
    cfg_tmp_path = None
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]  # single GPU

    if override_cfg is not None:
        if yaml is None:
            print('[WARN] PyYAML not available; cannot override config.')
        else:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            # deep merge
            def deep_merge(a, b):
                for k,v in b.items():
                    if isinstance(v, dict) and isinstance(a.get(k), dict):
                        deep_merge(a[k], v)
                    else:
                        a[k] = v
                return a
            cfg = deep_merge(cfg, override_cfg)
            ensure_dir(os.path.join(paths.out_root, 'tmp_cfg'))
            cfg_tmp_path = os.path.join(paths.out_root, 'tmp_cfg', f'{tag}.yaml')
            with open(cfg_tmp_path, 'w') as f:
                yaml.safe_dump(cfg, f)
            config_path = cfg_tmp_path

    cmd = [sys.executable, 'main.py', 'test', '--config', config_path, '--ckpt_path', paths.ckpt]
    print('[CMD]', ' '.join(cmd))
    proc = subprocess.Popen(cmd, cwd=work_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stdout_lines = []
    for line in proc.stdout:
        print(line, end='')
        stdout_lines.append(line)
    ret = proc.wait()
    if ret != 0:
        print(f'[ERR] SelfRDB test failed with code {ret}')
    # parse
    out_text = '\n'.join(stdout_lines)
    import re
    m1 = re.search(r'PSNR:\s*([0-9.]+)\s*±\s*([0-9.]+)', out_text)
    m2 = re.search(r'SSIM:\s*([0-9.]+)\s*±\s*([0-9.]+)', out_text)
    res = {}
    if m1:
        res.update({'PSNR_mean': float(m1.group(1)), 'PSNR_std': float(m1.group(2))})
    if m2:
        res.update({'SSIM_mean': float(m2.group(1)), 'SSIM_std': float(m2.group(2))})
    res['tag'] = tag
    # Save to JSON
    ensure_dir(paths.out_root)
    with open(os.path.join(paths.out_root, f'selfrdb_test_{tag}.json'), 'w') as f:
        json.dump(res, f, indent=2)
    return res

# -------------------------
# Diagnostics
# -------------------------

def diagnostics(dataset_train: PairDataset, dataset_test: PairDataset, out_dir: str,
                linear_params: Tuple[float, float], topk: int = 32):
    ensure_dir(out_dir)
    # A) Saturation on T2 ROI
    sats0=[]; sats1=[]
    for i in tqdm(range(len(dataset_test)), desc='[Diag] saturation'):
        _, y, m, _ = dataset_test[i]
        y = y.numpy()[0]; m = (m.numpy()[0] > 0)
        roi = y[m]
        if roi.size == 0:
            continue
        sats0.append(np.mean(roi <= 1e-6))
        sats1.append(np.mean(roi >= 1-1e-6))
    sat_stats = {
        'sat0_mean': float(np.mean(sats0) if sats0 else 0.0),
        'sat1_mean': float(np.mean(sats1) if sats1 else 0.0),
        'N': int(len(sats0))
    }
    with open(os.path.join(out_dir, 'saturation_stats.json'), 'w') as f:
        json.dump(sat_stats, f, indent=2)

    # B) T1–T2 ROI correlation (sample up to 1000 slices)
    import random
    idxs = list(range(len(dataset_test)))
    random.shuffle(idxs)
    idxs = idxs[:min(1000, len(idxs))]
    cors=[]
    for i in tqdm(idxs, desc='[Diag] t1-t2 corr'):
        x,y,m,_ = dataset_test[i]
        x,y,m = x.numpy()[0], y.numpy()[0], (m.numpy()[0] > 0)
        if m.sum() == 0:
            continue
        xv = x[m].flatten(); yv = y[m].flatten()
        if xv.size < 10:
            continue
        c = np.corrcoef(xv, yv)[0,1]
        if np.isfinite(c):
            cors.append(float(c))
    np.save(os.path.join(out_dir, 't1t2_corr.npy'), np.array(cors))
    if plt:
        plt.figure(); plt.hist(cors, bins=40)
        plt.xlabel('Pearson corr (ROI)'); plt.ylabel('count'); plt.title('T1–T2 ROI correlation (test)')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, 't1t2_corr_hist.png'))
        plt.close()

    # C) mask area vs PSNR of linear baseline, and Top-K error heatmaps
    a,b = linear_params
    per_slice = []  # (mask_ratio, psnr, fname)
    errors = []     # (mse, x,y,yhat,fname)
    for i in tqdm(range(len(dataset_test)), desc='[Diag] linear per-slice'):
        x,y,m,fn = dataset_test[i]
        x,y,m = x.numpy()[0], y.numpy()[0], (m.numpy()[0] > 0)
        yhat = np.clip(a*x + b, 0, 1)
        ratio = float(m.mean())
        p = psnr(y, yhat)
        per_slice.append((ratio, p, fn))
        mse = float(np.mean(((y - yhat)**2)[m])) if m.sum()>0 else float(np.mean((y - yhat)**2))
        errors.append((mse, y, yhat, fn))
    per_slice_arr = np.array([[r, p] for r,p,_ in per_slice], dtype=np.float32)
    np.save(os.path.join(out_dir, 'maskratio_vs_psnr.npy'), per_slice_arr)
    with open(os.path.join(out_dir, 'maskratio_vs_psnr.csv'), 'w') as f:
        f.write('mask_ratio,psnr,fname\n')
        for (r,p,fn) in per_slice:
            f.write(f'{r:.6f},{p:.6f},{fn}\n')
    if plt:
        plt.figure()
        plt.scatter(per_slice_arr[:,0], per_slice_arr[:,1], s=6, alpha=0.5)
        plt.xlabel('mask area ratio'); plt.ylabel('PSNR (linear)')
        plt.title('Mask ratio vs PSNR (linear baseline)')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'maskratio_vs_psnr.png'))
        plt.close()

    # Top-K error heatmaps (linear baseline)
    errors.sort(key=lambda t: -t[0])
    K = min(topk, len(errors))
    grid = int(math.ceil(math.sqrt(K)))
    if plt and K>0:
        for i in range(K):
            _, y, yhat, fn = errors[i]
            err = np.abs(y - yhat)
            ymin,ymax = y.min(), y.max()
            plt.figure(figsize=(6,2))
            for j,arr,title in [(1,y,'GT'),(2,yhat,'Pred'),(3,err,'|Err|')]:
                plt.subplot(1,3,j)
                plt.imshow(arr, vmin=0, vmax=1); plt.title(title); plt.axis('off')
            plt.suptitle(f'Top-{i+1} err {fn}')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'toperr_{i+1:03d}.png'))
            plt.close()

# -------------------------
# Existing repo scripts
# -------------------------

def call_repo_script(selfrdb_dir: str, script_name: str, args_list: List[str]):
    script_path = os.path.join(selfrdb_dir, script_name)
    if not os.path.exists(script_path):
        print(f'[SKIP] {script_name} not found in repo root.')
        return
    cmd = [sys.executable, script_name] + args_list
    print('[CMD]', ' '.join(cmd))
    subprocess.call(cmd, cwd=selfrdb_dir)

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description='One-click SelfRDB diagnose & test pipeline')
    ap.add_argument('--selfrdb_dir', type=str, default='~/Projects/SelfRDB')
    ap.add_argument('--config', type=str, default='config_64.yaml')
    ap.add_argument('--ckpt', type=str, default='logs/experiment/version_9/checkpoints/epoch=89-step=13680.ckpt')
    ap.add_argument('--dataset_root', type=str, default='~/Projects/SelfRDB/dataset/brats64_ref_t1')
    ap.add_argument('--out', type=str, default=None)

    # Toggles
    ap.add_argument('--run_selfrdb_test', type=int, default=1)
    ap.add_argument('--run_sampling_sweep', type=int, default=1)
    ap.add_argument('--run_linear_baseline', type=int, default=1)
    ap.add_argument('--run_unet_baseline', type=int, default=0)
    ap.add_argument('--run_diagnostics', type=int, default=1)
    ap.add_argument('--run_repo_audit', type=int, default=1)

    # Settings
    ap.add_argument('--linear_max_pixels', type=int, default=5_000_000)
    ap.add_argument('--subset_train', type=int, default=4000)
    ap.add_argument('--subset_test', type=int, default=2000)
    ap.add_argument('--unet_epochs', type=int, default=3)
    ap.add_argument('--unet_batch', type=int, default=64)

    # Sampling sweep grid
    ap.add_argument('--steps_grid', type=str, default='10,30,50')
    ap.add_argument('--recursions_grid', type=str, default='2,4')

    args = ap.parse_args()

    selfrdb_dir = os.path.expanduser(args.selfrdb_dir)
    dataset_root = os.path.expanduser(args.dataset_root)
    out_root = args.out or os.path.join(os.getcwd(), 'oneclick_out', now_tag())
    ensure_dir(out_root)

    paths = Paths(selfrdb_dir=selfrdb_dir, config=args.config, ckpt=args.ckpt, dataset_root=dataset_root, out_root=out_root)

    # 0) Prepare datasets (limit subsets for speed where requested)
    d_train = PairDataset(dataset_root, 'train', max_items=args.subset_train)
    d_val   = PairDataset(dataset_root, 'val', max_items=min(2000, args.subset_test))
    d_test  = PairDataset(dataset_root, 'test', max_items=args.subset_test)

    summary = {}

    # 1) Single-GPU SelfRDB test
    if args.run_selfrdb_test:
        res = run_selfrdb_test(paths, override_cfg=None, tag='singleGPU')
        summary['selfrdb_test_singleGPU'] = res

    # 2) Sampling sweep
    if args.run_sampling_sweep and yaml is not None:
        # Some repos (like your SelfRDB) restrict what keys 'test' can accept.
        # 'n_recursions' appears to be rejected during `test`, so we only sweep 'n_steps'.
        steps = [int(s) for s in args.steps_grid.split(',') if s.strip()]
        sweep = []
        for s in steps:
            override = {'diffusion': {'n_steps': s}}
            tag = f'steps{s}'
            res = run_selfrdb_test(paths, override_cfg=override, tag=tag)
            sweep.append(res)
        summary['sampling_sweep'] = sweep
    elif args.run_sampling_sweep and yaml is None:
        print('[WARN] PyYAML not installed; skip sampling sweep.')
    elif args.run_sampling_sweep and yaml is None:
        print('[WARN] PyYAML not installed; skip sampling sweep.')

    # 3) Linear baselines
    linear_stats = {}
    if args.run_linear_baseline:
        a,b = fit_global_linear(d_train, max_pixels=args.linear_max_pixels)
        linear_stats['global'] = eval_global_linear(d_test, a, b)
        linear_stats['params'] = {'a': a, 'b': b}
        with open(os.path.join(out_root, 'linear_baseline.json'), 'w') as f:
            json.dump(linear_stats, f, indent=2)
        print('[LinearBaseline] ', linear_stats)

    # 4) Tiny UNet baseline (optional)
    if args.run_unet_baseline:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        unet_dir = os.path.join(out_root, 'unet_baseline'); ensure_dir(unet_dir)
        res = train_unet_baseline(d_train, d_val, unet_dir, epochs=args.unet_epochs, batch_size=args.unet_batch)
        summary['unet_baseline'] = res

    # 5) Diagnostics
    if args.run_diagnostics and args.run_linear_baseline:
        diag_dir = os.path.join(out_root, 'diagnostics'); ensure_dir(diag_dir)
        diagnostics(d_train, d_test, diag_dir, (linear_stats['params']['a'], linear_stats['params']['b']))

    # 6) Call repo scripts if present
    if args.run_repo_audit:
        # audit_selfrdb_brats.py
        call_repo_script(selfrdb_dir, 'audit_selfrdb_brats.py', [
            '--root', dataset_root,
            '--modalities', 't1', 't2', 'flair', 't1ce',
            '--target', 't2', '--image_size', '64', '--iou_thr', '0.995', '--max_samples', '1000'
        ])
        # qc_report_selfrdb.py
        call_repo_script(selfrdb_dir, 'qc_report_selfrdb.py', [
            '--root', dataset_root,
            '--modalities', 't1,t2,flair,t1ce',
            '--target', 't2', '--image_size', '64', '--iou_thr', '0.995', '--max_samples', '1000',
            '--viz_n', '16', '--out', os.path.join(out_root, 'qc_report_brats64_ref_t1')
        ])

    # Save summary
    with open(os.path.join(out_root, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n[Done] Outputs saved to:', out_root)

if __name__ == '__main__':
    main()
