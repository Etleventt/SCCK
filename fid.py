import os
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _to01_t(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, H, W] in [-1,1] or [0,1]
    if torch.nan_to_num(x).min() < -0.1:
        x = (x + 1.0) / 2.0
    return x.clamp(0, 1)


def _prep_for_inception(x: torch.Tensor, resize_to: int = 299) -> torch.Tensor:
    """Convert tensor in [-1,1]/[0,1] to 3x299x299 normalized for Inception.

    - If C != 3, replicate the first channel to 3 channels.
    - Resize bilinearly to 299x299.
    - Normalize with ImageNet mean/std.
    """
    x = _to01_t(x)
    if x.size(1) != 3:
        x = x[:, :1, ...].repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(resize_to, resize_to), mode="bilinear", align_corners=False)
    mean = IMAGENET_MEAN.to(x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(x.device, dtype=x.dtype)
    x = (x - mean) / std
    return x


class InceptionFeatureExtractor(nn.Module):
    """
    Extract pool features (2048-D) from InceptionV3 pretrained on ImageNet.
    Uses torchvision's feature_extraction API to grab the 'avgpool' output.
    """
    def __init__(self, weights_path: Optional[str] = None, dump_dir: Optional[str] = None, dump_n: int = 0):
        super().__init__()
        net = None
        # Prefer loading from a local weights_path if provided
        if isinstance(weights_path, str) and weights_path:
            try:
                net = models.inception_v3(weights=None, aux_logits=True)
                state = torch.load(weights_path, map_location="cpu")
                # accept either full state_dict or raw weight dict
                if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                    state = state["state_dict"]
                # load non-strict to ignore aux mismatches if any
                net.load_state_dict(state, strict=False)
            except Exception as e:
                warnings.warn(f"Failed to load InceptionV3 from '{weights_path}': {e}. Falling back to torchvision weights.")
                net = None

        if net is None:
            # Use explicit weights enum; keep aux_logits=True to match official weights
            try:
                weights = models.Inception_V3_Weights.IMAGENET1K_V1
                net = models.inception_v3(weights=weights, aux_logits=True)
            except Exception:
                # Fallback: try older API
                net = models.inception_v3(pretrained=True, aux_logits=True)

        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        # Grab the 'avgpool' node
        self.fe = create_feature_extractor(net, return_nodes={'avgpool': 'pool'})
        # Dump config
        self.dump_dir = dump_dir
        self.dump_n = int(dump_n or 0)
        self._dump_counts = {}
        self.dump_prefix = 'img'

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _prep_for_inception(x)

        # Optionally dump a few preprocessed inputs for inspection
        if self.dump_dir and self.dump_n > 0:
            try:
                os.makedirs(self.dump_dir, exist_ok=True)
                prefix = getattr(self, 'dump_prefix', 'img')
                count = int(self._dump_counts.get(prefix, 0))
                remain = max(0, self.dump_n - count)
                if remain > 0:
                    n = min(remain, x.size(0))
                    # invert normalization for visualization
                    mean = IMAGENET_MEAN.to(x.device, dtype=x.dtype)
                    std = IMAGENET_STD.to(x.device, dtype=x.dtype)
                    x_vis = (x[:n] * std + mean).clamp(0, 1).detach().cpu()
                    for i in range(n):
                        idx = count + i
                        img = x_vis[i].permute(1, 2, 0).numpy()
                        out_png = os.path.join(self.dump_dir, f"{prefix}_{idx:05d}.png")
                        try:
                            plt.imsave(out_png, img)
                        except Exception:
                            pass
                        # also save the exact tensor fed to Inception (normalized)
                        out_pt = os.path.join(self.dump_dir, f"{prefix}_{idx:05d}.pt")
                        try:
                            torch.save(x[i].detach().cpu(), out_pt)
                        except Exception:
                            pass
                    self._dump_counts[prefix] = count + n
            except Exception:
                pass

        feats = self.fe(x)['pool']  # [B, 2048, 1, 1]
        feats = torch.flatten(feats, 1)  # [B, 2048]
        return feats


def _sym_matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    """Symmetric matrix square root using eigen-decomposition.
    Ensures numerical stability by clamping eigenvalues to >= 0.
    """
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    sqrt_vals = np.sqrt(vals)
    return (vecs * sqrt_vals[None, :]) @ vecs.T


def _compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def compute_fid(feats_real: np.ndarray, feats_fake: np.ndarray) -> float:
    """Compute FID from feature matrices of shape [N, D].
    Uses the symmetric form Tr(S1 + S2 - 2*sqrt(S1^{1/2} S2 S1^{1/2})).
    """
    mu1, sigma1 = _compute_stats(feats_real)
    mu2, sigma2 = _compute_stats(feats_fake)
    diff = mu1 - mu2
    covmean = _sym_matrix_sqrt(_sym_matrix_sqrt(sigma1) @ sigma2 @ _sym_matrix_sqrt(sigma1))
    trace_term = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    fid = float(diff.dot(diff) + trace_term)
    # Guard for small negative due to numeric
    if fid < 0 and fid > -1e-6:
        fid = 0.0
    return fid


@torch.inference_mode()
def compute_fid_from_numpy(
    real: np.ndarray,
    fake: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    weights_path: Optional[str] = None,
    dump_dir: Optional[str] = None,
    dump_n: int = 0,
) -> Optional[float]:
    """Compute FID given real/fake arrays shaped [N,H,W] or [N,1,H,W] in [-1,1]/[0,1].
    Returns None if Inception weights are unavailable (offline with no cache).
    """
    assert real.shape == fake.shape, "Real/Fake shapes must match"
    if real.ndim == 3:
        real = real[:, None, ...]
        fake = fake[:, None, ...]
    N = real.shape[0]
    try:
        net = InceptionFeatureExtractor(weights_path=weights_path, dump_dir=dump_dir, dump_n=dump_n).to(device)
    except Exception as e:
        warnings.warn(f"InceptionV3 weights unavailable; skipping FID. Reason: {e}")
        return None

    feats_r, feats_f = [], []
    for i in range(0, N, batch_size):
        r = torch.from_numpy(real[i:i+batch_size]).to(device).float()
        f = torch.from_numpy(fake[i:i+batch_size]).to(device).float()
        # tag prefix for dumping
        net.dump_prefix = 'real'
        fr = net(r).cpu().numpy()
        net.dump_prefix = 'fake'
        ff = net(f).cpu().numpy()
        feats_r.append(fr)
        feats_f.append(ff)
    feats_r = np.concatenate(feats_r, axis=0)
    feats_f = np.concatenate(feats_f, axis=0)
    return compute_fid(feats_r, feats_f)
