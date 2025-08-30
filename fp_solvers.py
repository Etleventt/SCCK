from __future__ import annotations
import torch


def _center_clip(x: torch.Tensor, centered: bool = True):
    # Clamp to expected output domain; centered=True for [-1,1]
    if centered:
        return x.clamp_(-1.0, 1.0)
    return x.clamp_(0.0, 1.0)


@torch.no_grad()
def picard(F, x0: torch.Tensor, K: int = 2, centered: bool = True):
    """Zero-order fixed-point iteration: repeat K times x <- F(x)."""
    x = x0
    for _ in range(max(int(K), 0)):
        x = F(x)
        _center_clip(x, centered)
    return x


@torch.no_grad()
def heun2(F, x0: torch.Tensor, M: int = 1, theta: float = 0.5, centered: bool = True):
    """
    Heun-like predictor-corrector (deterministic, 2 evals per macro-step):
      1) x1 = F(x)
      2) x2 = F(x1)
      3) x  = (1-θ) * x1 + θ * x2
    """
    x = x0
    th = float(theta)
    for _ in range(max(int(M), 0)):
        x1 = F(x)
        x2 = F(x1)
        x = (1.0 - th) * x1 + th * x2
        _center_clip(x, centered)
    return x


@torch.no_grad()
def extrap(F, x0: torch.Tensor, M: int = 1, gamma: float = 0.5, centered: bool = True):
    """
    Two-step linear extrapolation (Richardson-style):
      1) x1 = F(x)
      2) x2 = F(x1)
      3) x  = x2 + γ * (x2 - x1)
    """
    x = x0
    g = float(gamma)
    for _ in range(max(int(M), 0)):
        x1 = F(x)
        x2 = F(x1)
        x = x2 + g * (x2 - x1)
        _center_clip(x, centered)
    return x

def _residual_norm(F, x, mask=None):
    r = (F(x) - x).abs()
    if mask is not None:
        # mask: [B,1,H,W] bool；仅在脑区度量残差
        r = r * mask
        denom = mask.sum(dim=(1,2,3), keepdim=True).clamp_min(1.0)
        return (r.sum(dim=(1,2,3), keepdim=True) / denom).mean()
    return r.mean()

import torch

def _clip_centered(x):
    # SelfRDB 的生成器默认 centered=True（[-1, 1]）
    return x.clamp(-1.0, 1.0)

@torch.no_grad()
def _residual_norm(F, x, mask=None):
    # 进入 F 之前先裁剪，保证与训练分布一致
    x = _clip_centered(x)
    r = (F(x) - x).abs()
    if mask is not None:
        # mask 可为 float/bool，shape [B,1,H,W]
        r = r * mask
        denom = mask.sum(dim=(1,2,3), keepdim=True).clamp_min(1.0)
        r = r.sum(dim=(1,2,3), keepdim=True) / denom
    return r.mean()

@torch.no_grad()
def extrap_safe(F, x0, gamma=0.3, centered=True, mask=None,
                max_backtracks=3, improve_ratio=0.05):
    """
    稳健外推（两次前向 + 外推 + 回溯）：
      x1 = F(x0)
      x2 = F(x1)
      x  = x2 + γ (x2 - x1)
    若残差未按比例明显下降（相对 x2 至少 5%），则减半 γ 回溯；
    仍不降则回退到 x2（等价 R=2 的“最后一次”）。
    """
    # 先确保初值在域内
    x0 = _clip_centered(x0)

    x1 = _clip_centered(F(x0))
    x2 = _clip_centered(F(x1))

    # 基线残差（在 x2 上）
    r2 = _residual_norm(F, x2, mask)

    # 初次外推候选
    g  = float(gamma)
    x  = _clip_centered(x2 + g * (x2 - x1))
    rx = _residual_norm(F, x, mask)

    # 需要达到 "相对下降≥improve_ratio" 才接受
    target = (1.0 - float(improve_ratio)) * r2
    bt = 0
    while rx > target and bt < int(max_backtracks) and g > 1e-4:
        g *= 0.5
        x  = _clip_centered(x2 + g * (x2 - x1))
        rx = _residual_norm(F, x, mask)
        bt += 1

    # 最终选择：若仍不达标，回退到 x2
    x_final = x if rx <= target else x2
    if centered:
        x_final = _clip_centered(x_final)
    return x_final