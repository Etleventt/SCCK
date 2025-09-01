#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BraTS -> SelfRDB, **official-aligned** preprocessing (speed-optimized, bug-fixed, mask export).

Key alignment to the paper/repo:
- RAS+ axial, 2D slices; **no resize**; **zero-pad / center-crop to 256×256** (no interpolation).
- Per-volume brain-ROI mean normalization to **1.0** (within multi-modal union mask).
- Train-split, per-modality **global robust percentiles** -> map to **[-1,1]**, then save **[0,1]**.
- Subject-level splits; full-slice outputs in SelfRDB NumpyDataset layout; **optional mask export**.

Quality & speed features:
- Pass1 loads each subject **once** (all modalities together), computes **union mask** once,
  samples up to `--samples_per_subject` voxels/subject per modality.
- Percentiles are **cached** to `<out_root>/global_stats.json` and reused unless `--force_recompute_stats 1`。
- Optional `--fast 1` to compute percentiles from first `--fast_subjects` training subjects。
- **Case-insensitive** modality filename matching (e.g., `T1` vs `t1`).
- **Deterministic** RNG for Pass1 sampling (stable per subject).
- **Resume** export works as expected (no accidental re-zeroing).
- **Safe save** in both CPU & CUDA branches to avoid partial writes.
- **Mask export**: `--export_mask 1` writes `mask/<split>/slice_*.npy` (uint8, 0/1)。

Usage
-----
python prepare_brats_official_aligned_256.py \
  --root /path/to/BraTS2021_root \
  --out_root /path/to/OUT/brats256_selfrdb \
  --modalities T1,T2,FLAIR,T1CE \
  --split 0.8 0.1 0.1 --seed 42 \
  --export_mask 1 --mask_kind union

Notes on masks
--------------
- `mask_kind=union`: 每个切片的掩膜取 **所有模态** 的非零并集（推荐）。
- `mask_kind=ref`:    掩膜取指定模态（`--ref_for_mask t1`）的非零区域。
- 掩膜与图像一样进行**中心裁切/零填充到 256**，保存为 **uint8 0/1**。
- 若你在评测时想按脑区计算 PSNR/SSIM，确保你的评测脚本读取 `mask/` 并传给 `compute_metrics(mask=...)`。
"""

from __future__ import annotations
import argparse, os, json, random, re, time, hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import nibabel as nib
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from concurrent.futures import ProcessPoolExecutor, as_completed

# Optional GPU path
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

BRATS_FOLDERS = [
    "TCGA-GBM","TCGA-LGG","UPENN-GBM","CPTAC-GBM","IvyGAP","ACRIN-FMISO-Brain","UCSF-PDGM"
]
ALL_MODS = ["T1","T1CE","T2","FLAIR","t1","t1ce","t2","flair"]

# ----------------------- helpers -----------------------

def find_subjects(root: Path) -> List[Path]:
    pats: List[Path] = []
    for pat in BRATS_FOLDERS:
        d = root / pat
        if d.exists():
            pats += [p for p in d.glob("BraTS2021_*") if p.is_dir()]
    if not pats:
        pats = [p for p in root.glob("TCGA-*/BraTS2021_*") if p.is_dir()]
    if not pats:
        pats = [p for p in root.glob("BraTS2021_*") if p.is_dir()]  # fallback: subjects directly under root
    return sorted(pats)


def mod_file(sub: Path, m: str) -> Path | None:
    """Case-insensitive modality file matcher like *_t1.nii.gz, *_T1.nii, etc."""
    # try exact pattern first
    hits = list(sub.glob(f"*_{m}.nii*"))
    if hits:
        return hits[0]
    # then try lower/upper
    ml, mu = m.lower(), m.upper()
    hits = list(sub.glob(f"*_{ml}.nii*")) or list(sub.glob(f"*_{mu}.nii*"))
    if hits:
        return hits[0]
    # final: regex case-insensitive over all nii files
    for p in sub.glob("*.nii*"):
        if re.search(rf"_{re.escape(m)}(\.nii(\.gz)?)$", p.name, flags=re.IGNORECASE):
            return p
    return None


def as_ras_axial(nii: nib.Nifti1Image) -> np.ndarray:
    ras = nib.as_closest_canonical(nii)
    arr = ras.get_fdata(dtype=np.float32)
    return np.transpose(arr, (2, 1, 0))  # (Z,H,W)


def pad_to_center(arr2d: np.ndarray, size: int = 256) -> np.ndarray:
    H, W = arr2d.shape
    if H == size and W == size:
        return arr2d.astype(np.float32, copy=False)
    # center-crop if larger
    if H > size:
        y0 = (H - size) // 2
        arr2d = arr2d[y0:y0 + size, :]
        H = size
    if W > size:
        x0 = (W - size) // 2
        arr2d = arr2d[:, x0:x0 + size]
        W = size
    # pad if smaller
    py0 = (size - H) // 2
    px0 = (size - W) // 2
    out = np.zeros((size, size), dtype=np.float32)
    out[py0:py0 + H, px0:px0 + W] = arr2d.astype(np.float32)
    return out


def pad_to_center_torch(arr2d: 'torch.Tensor', size: int = 256) -> 'torch.Tensor':
    H, W = int(arr2d.shape[0]), int(arr2d.shape[1])
    if H > size:
        y0 = (H - size) // 2
        arr2d = arr2d[y0:y0 + size, :]
        H = size
    if W > size:
        x0 = (W - size) // 2
        arr2d = arr2d[:, x0:x0 + size]
        W = size
    py0 = (size - H) // 2
    px0 = (size - W) // 2
    pad = (px0, size - W - px0, py0, size - H - py0)  # (left,right,top,bottom)
    return torch.nn.functional.pad(arr2d, pad, mode='constant', value=0.0)


def brain_mask_union(vols: Dict[str, np.ndarray]) -> np.ndarray:
    mask = None
    for v in vols.values():
        m = (v > 0).astype(np.uint8)
        mask = m if mask is None else (mask | m)
    return mask.astype(np.uint8)


def per_volume_mean_normalize(vol: np.ndarray, mask3d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    inside = vol[mask3d > 0]
    mean = float(inside.mean()) if inside.size else float(vol.mean())
    scale = 1.0 / max(mean, eps)
    return (vol * scale).astype(np.float32)

# ----------------------- percentiles (Pass1) -----------------------

def _sid_seed(seed: int, sid: str) -> int:
    h = hashlib.md5(sid.encode("utf-8")).hexdigest()[:8]
    return seed + int(h, 16)


def _pass1_worker(args_tuple):
    sid, mods, sid2files, samples_per_subject, seed = args_tuple
    rng = np.random.default_rng(_sid_seed(seed, sid))
    # load all requested modalities once
    vols = {}
    for m in mods:
        p = sid2files[sid][m]
        if p is None or not p.exists():
            return None
        vols[m] = as_ras_axial(nib.load(str(p)))
    mask3d = brain_mask_union(vols)
    out = {}
    for m in mods:
        v = per_volume_mean_normalize(vols[m], mask3d)
        inside = v[mask3d > 0]
        if inside.size == 0:
            continue
        nvox = inside.size
        take = min(nvox, samples_per_subject)
        idx = rng.choice(nvox, size=take, replace=False)
        out[m] = inside[idx].astype(np.float32, copy=False)
    return out


def compute_or_load_global_stats(out_root: Path, mods: List[str], splits: Dict[str, List[str]], sid2files: Dict[str, Dict[str, Path]],
                                 q_lo: float, q_hi: float, seed: int, fast: int, fast_subjects: int,
                                 samples_per_subject: int, max_samples_per_mod: int,
                                 force_recompute: int, p1_workers: int):
    stats_file = out_root / "global_stats.json"
    if stats_file.exists() and not force_recompute:
        data = json.loads(stats_file.read_text())
        lo = {k: float(v) for k, v in data["global_lo"].items()}
        hi = {k: float(v) for k, v in data["global_hi"].items()}
        print("[Pass1] Loaded cached percentiles:")
        for m in mods:
            print(f"  [{m}] lo={lo[m]:.6f}, hi={hi[m]:.6f}")
        return lo, hi

    train_sids = splits["train"]
    if fast:
        train_sids = train_sids[:min(len(train_sids), fast_subjects)]
        print(f"[Pass1-fast] Using first {len(train_sids)} training subjects for percentile estimation…")
    else:
        print("[Pass1] Computing global percentiles on full train split…")

    # per-modality sample accumulator
    samples: Dict[str, list[np.ndarray]] = {m: [] for m in mods}

    if p1_workers > 1:
        with ProcessPoolExecutor(max_workers=p1_workers) as ex:
            futures = [ex.submit(_pass1_worker, (sid, mods, sid2files, samples_per_subject, seed)) for sid in train_sids]
            for fut in tqdm(as_completed(futures), total=len(futures), desc='[Pass1] subjects(parallel)'):
                res = fut.result()
                if res is None:
                    continue
                for m, arr in res.items():
                    samples[m].append(arr)
    else:
        for sid in tqdm(train_sids, desc='[Pass1] subjects'):
            res = _pass1_worker((sid, mods, sid2files, samples_per_subject, seed))
            if res is None:
                continue
            for m, arr in res.items():
                samples[m].append(arr)

    global_lo: Dict[str, float] = {}
    global_hi: Dict[str, float] = {}
    for m in mods:
        if len(samples[m]) == 0:
            global_lo[m], global_hi[m] = 0.0, 2.0
            continue
        cat = np.concatenate(samples[m])
        if cat.size > max_samples_per_mod:
            idx = np.random.default_rng(seed+123).choice(cat.size, size=max_samples_per_mod, replace=False)
            cat = cat[idx]
        lo = float(np.percentile(cat, q_lo))
        hi = float(np.percentile(cat, q_hi))
        if hi <= lo:
            hi = lo + 1e-6
        global_lo[m], global_hi[m] = lo, hi
        print(f"  [{m}] lo={lo:.6f}, hi={hi:.6f}")

    out_root.mkdir(parents=True, exist_ok=True)
    stats_file.write_text(json.dumps({"global_lo": global_lo, "global_hi": global_hi}, indent=2), encoding="utf-8")
    return global_lo, global_hi

# ----------------------- micro-benchmark (auto device choice) -----------------------

def _preprocess_slice_np(x: np.ndarray, lo: float, hi: float, size: int) -> np.ndarray:
    x = np.clip(x, lo, hi)
    x = 2.0 * (x - lo) / (hi - lo) - 1.0
    x = 0.5 * (x + 1.0)
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    x = pad_to_center(x, size=size)
    return x


def _preprocess_slice_cuda(x: 'torch.Tensor', lo: float, hi: float, size: int) -> 'torch.Tensor':
    xt = torch.clamp(x, lo, hi)
    xt = 2.0 * (xt - lo) / (hi - lo) - 1.0
    xt = 0.5 * (xt + 1.0)
    xt = torch.clamp(xt, 0.0, 1.0)
    xt = pad_to_center_torch(xt, size=size)
    return xt


def _auto_probe_device(example_np: np.ndarray, lo: float, hi: float, size: int, repeat: int = 64) -> str:
    # CPU timing
    t0 = time.time()
    for _ in range(repeat):
        _ = _preprocess_slice_np(example_np, lo, hi, size)
    cpu_ms = (time.time() - t0) * 1000.0 / repeat

    if not (_HAS_TORCH and torch.cuda.is_available()):
        print(f"[auto-probe] cpu≈{cpu_ms:.3f} ms/slice; cuda N/A -> choose CPU")
        return 'cpu'

    # GPU timing (include H->D copy cost per slice, as in real loop)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        xt = torch.from_numpy(example_np).to('cuda', non_blocking=True)
        yt = _preprocess_slice_cuda(xt, lo, hi, size)
        _ = yt
    torch.cuda.synchronize()
    gpu_ms = (time.time() - t0) * 1000.0 / repeat

    print(f"[auto-probe] cpu≈{cpu_ms:.3f} ms/slice, cuda≈{gpu_ms:.3f} ms/slice")
    return 'cuda' if gpu_ms < cpu_ms else 'cpu'

# ----------------------- I/O helpers (safe save + resume) -----------------------

def _safe_np_save(path: Path, arr: np.ndarray, retries: int = 3, sleep: float = 0.5):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for k in range(retries):
        tmp = None
        try:
            with NamedTemporaryFile(dir=str(path.parent), delete=False, suffix='.tmp') as f:
                tmp = f.name
                np.save(f, arr)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except OSError as e:
            last_err = e
            if tmp and os.path.exists(tmp):
                try: os.unlink(tmp)
                except: pass
            time.sleep(sleep * (k+1))
    raise OSError(f"safe_np_save failed for {path} after {retries} retries: {last_err}")


def _next_start_index(out_root: Path, mods: List[str], split: str) -> int:
    """Find contiguous prefix length where *all* modalities have slice_i.npy present."""
    i = 0
    while True:
        ok = True
        for m in mods:
            if not (out_root / m / split / f"slice_{i}.npy").exists():
                ok = False; break
        if not ok:
            break
        i += 1
    return i

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser("BraTS -> SelfRDB (official-aligned, 256 pad, cached percentiles + mask)")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--modalities", default="T1,T2,FLAIR,T1CE")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nz_frac_thr", type=float, default=0.01)
    ap.add_argument("--q_lo", type=float, default=0.1)
    ap.add_argument("--q_hi", type=float, default=99.9)
    ap.add_argument("--per_volume", type=int, default=0)
    # mask export options
    ap.add_argument("--export_mask", type=int, default=0, help="1 to export mask/<split>/slice_*.npy (uint8)")
    ap.add_argument("--mask_kind", type=str, default='union', choices=['union','ref'], help="mask=union of all modalities >0, or ref modality >0")
    ap.add_argument("--ref_for_mask", type=str, default='t1', help="used when --mask_kind=ref; case-insensitive")
    # speed toggles
    ap.add_argument("--fast", type=int, default=0, help="1=estimate percentiles on a subset of train subjects")
    ap.add_argument("--fast_subjects", type=int, default=12)
    ap.add_argument("--samples_per_subject", type=int, default=100_000)
    ap.add_argument("--max_samples_per_mod", type=int, default=2_000_000)
    ap.add_argument("--force_recompute_stats", type=int, default=0)
    ap.add_argument("--resume", type=int, default=0, help="resume Pass2 by appending after existing contiguous slices")
    # parallel/GPU toggles
    ap.add_argument("--p1_workers", type=int, default=max(os.cpu_count() // 2, 1), help="workers for Pass1 percentile scan")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"], help="compute device for Pass2 ops")
    ap.add_argument("--max_slices_per_split", type=int, default=0, help="cap total slices per split (0=unlimited)")
    ap.add_argument("--auto_probe", type=int, default=1, help="when device=auto, time a few slices and pick faster path")
    ap.add_argument("--auto_probe_repeat", type=int, default=64)
    args = ap.parse_args()

    mods = [m.strip() for m in args.modalities.split(',') if m.strip()]
    for m in mods:
        if m not in ALL_MODS:
            raise SystemExit(f"Unknown modality: {m}")

    subs = find_subjects(args.root)
    if not subs:
        raise SystemExit("No subjects found under root.")

    sid2files: Dict[str, Dict[str, Path]] = {}
    for sub in subs:
        sid = sub.name
        files = {m: mod_file(sub, m) for m in mods}
        if any(v is None for v in files.values()):
            continue
        sid2files[sid] = files

    sids = sorted(sid2files.keys())
    if not sids:
        raise SystemExit("No complete subjects with all requested modalities.")

    ratios = tuple(float(x) for x in args.split)
    assert abs(sum(ratios) - 1.0) < 1e-6
    rnd = random.Random(args.seed)
    sids_copy = sids.copy(); rnd.shuffle(sids_copy)
    n = len(sids_copy)
    n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
    splits = {"train": sids_copy[:n_tr], "val": sids_copy[n_tr:n_tr + n_va], "test": sids_copy[n_tr + n_va:]}

    # prepare dirs
    for m in mods:
        for sp in ["train", "val", "test"]:
            (args.out_root / m / sp).mkdir(parents=True, exist_ok=True)
    if args.export_mask:
        for sp in ["train", "val", "test"]:
            (args.out_root / 'mask' / sp).mkdir(parents=True, exist_ok=True)

    # Pass1: global percentiles with caching / fast mode
    global_lo, global_hi = compute_or_load_global_stats(
        args.out_root, mods, splits, sid2files,
        args.q_lo, args.q_hi, args.seed,
        args.fast, args.fast_subjects,
        args.samples_per_subject, args.max_samples_per_mod,
        args.force_recompute_stats, args.p1_workers
    )

    # Pass2: write slices
    print("[Pass2] Writing SelfRDB NumpyDataset slices …")
    # counters init (support resume)
    counters = {"train": 0, "val": 0, "test": 0}
    subj_ids = {"train": [], "val": [], "test": []}
    if args.resume:
        for sp in ["train","val","test"]:
            mods_for_resume = mods + (['mask'] if args.export_mask else [])
            start = _next_start_index(args.out_root, mods_for_resume, sp)
            if start > 0:
                print(f"[resume] {sp}: starting from slice_{start}")
            counters[sp] = start

    # device resolution
    if args.device == 'auto':
        device = 'cuda' if (_HAS_TORCH and torch.cuda.is_available()) else 'cpu'
    else:
        device = args.device
    if device == 'cuda' and not _HAS_TORCH:
        print('[WARN] PyTorch not available; falling back to CPU.')
        device = 'cpu'
    if device == 'cuda':
        torch.set_float32_matmul_precision('high')
        print('[Pass2] Using CUDA for per-slice ops (I/O still on CPU).')
    else:
        print('[Pass2] Using CPU path.')

    probed = False

    ref_mask_key = args.ref_for_mask.lower()

    for sp, sid_list in splits.items():
        if args.max_slices_per_split and counters[sp] >= args.max_slices_per_split:
            print(f"[limit] {sp}: already has {counters[sp]} >= cap {args.max_slices_per_split}; skipping split")
            continue
        for sid in tqdm(sid_list, desc=f"[Pass2] {sp}"):
            if args.max_slices_per_split and counters[sp] >= args.max_slices_per_split:
                break
            vols_np = {m: as_ras_axial(nib.load(str(sid2files[sid][m]))) for m in mods}
            mask3d_np = brain_mask_union(vols_np)
            # per-volume mean normalization (CPU numpy first to avoid host<->device churn)
            for m in mods:
                vols_np[m] = per_volume_mean_normalize(vols_np[m], mask3d_np)
            Z, H, W = next(iter(vols_np.values())).shape
            z_indices = [z for z in range(Z) if (mask3d_np[z] > 0).mean() >= args.nz_frac_thr]
            if args.per_volume and args.per_volume > 0 and len(z_indices) > 0:
                base = np.linspace(0, len(z_indices) - 1, num=min(args.per_volume, len(z_indices)))
                z_indices = [z_indices[int(round(b))] for b in base]

            # Auto-probe once using the first available slice
            if not probed and len(z_indices) > 0:
                example_m = mods[0]
                z0 = z_indices[0]
                lo0, hi0 = global_lo[example_m], global_hi[example_m]
                example_np = vols_np[example_m][z0]
                if args.device == 'auto' and args.auto_probe:
                    decided = _auto_probe_device(example_np, lo0, hi0, args.size, repeat=args.auto_probe_repeat)
                    if decided != device:
                        print(f"[auto-probe] Switching Pass2 device: {device} -> {decided}")
                        device = decided
                probed = True

            if device == 'cuda':
                # move one slice at a time to GPU to keep memory bounded
                for z in z_indices:
                    if args.max_slices_per_split and counters[sp] >= args.max_slices_per_split:
                        break
                    # ---- build mask 2D before writing (on CPU) ----
                    if args.export_mask:
                        if args.mask_kind == 'union':
                            m2d = (mask3d_np[z] > 0).astype(np.uint8)
                        else:  # ref
                            k = ref_mask_key if ref_mask_key in vols_np else list(vols_np.keys())[0]
                            m2d = (vols_np[k][z] > 0).astype(np.uint8)
                        m2d = pad_to_center(m2d, size=args.size).astype(np.uint8)
                    # ---- modalities ----
                    for m in mods:
                        x = vols_np[m][z]
                        lo = global_lo[m]; hi = global_hi[m]
                        xt = torch.from_numpy(x).to('cuda', non_blocking=True)
                        xt = torch.clamp(xt, lo, hi)
                        xt = 2.0 * (xt - lo) / (hi - lo) - 1.0
                        xt = 0.5 * (xt + 1.0)
                        xt = torch.clamp(xt, 0.0, 1.0)
                        xt = pad_to_center_torch(xt, size=args.size)
                        _safe_np_save(args.out_root / m / sp / f"slice_{counters[sp]}.npy",
                                      xt.cpu().numpy().astype(np.float32))
                    if args.export_mask:
                        _safe_np_save(args.out_root / 'mask' / sp / f"slice_{counters[sp]}.npy", m2d)
                    subj_ids[sp].append(f"{sid}|z={int(z)}")
                    counters[sp] += 1
            else:
                for z in z_indices:
                    if args.max_slices_per_split and counters[sp] >= args.max_slices_per_split:
                        break
                    # ---- build mask 2D ----
                    if args.export_mask:
                        if args.mask_kind == 'union':
                            m2d = (mask3d_np[z] > 0).astype(np.uint8)
                        else:
                            k = ref_mask_key if ref_mask_key in vols_np else list(vols_np.keys())[0]
                            m2d = (vols_np[k][z] > 0).astype(np.uint8)
                        m2d = pad_to_center(m2d, size=args.size).astype(np.uint8)
                    # ---- modalities ----
                    for m in mods:
                        x = vols_np[m][z]
                        lo = global_lo[m]; hi = global_hi[m]
                        x = np.clip(x, lo, hi)
                        x = 2.0 * (x - lo) / (hi - lo) - 1.0
                        x = 0.5 * (x + 1.0)
                        x = np.clip(x, 0.0, 1.0).astype(np.float32)
                        x = pad_to_center(x, size=args.size)
                        _safe_np_save(args.out_root / m / sp / f"slice_{counters[sp]}.npy", x)
                    if args.export_mask:
                        _safe_np_save(args.out_root / 'mask' / sp / f"slice_{counters[sp]}.npy", m2d)
                    subj_ids[sp].append(f"{sid}|z={int(z)}")
                    counters[sp] += 1

    manifest = {
        "modalities": mods,
        "splits_counts": counters,
        "size": args.size,
        "brain_slice_thr": args.nz_frac_thr,
        "q_lo": args.q_lo,
        "q_hi": args.q_hi,
        "global_lo": global_lo,
        "global_hi": global_hi,
        "seed": args.seed,
        "max_slices_per_split": args.max_slices_per_split,
        "resume": bool(args.resume),
        "device_final": device,
        "pass1": {
            "fast": bool(args.fast),
            "fast_subjects": args.fast_subjects,
            "samples_per_subject": args.samples_per_subject,
            "max_samples_per_mod": args.max_samples_per_mod,
            "workers": args.p1_workers
        },
        "mask": {
            "export": bool(args.export_mask),
            "kind": args.mask_kind,
            "ref_for_mask": args.ref_for_mask
        },
        "notes": {
            "mean_norm": "per-volume mean within union brain mask set to 1",
            "global_scale": "train-set robust percentiles per modality mapped to [-1,1], saved to [0,1]",
            "pad": "zero-pad/center-crop to square size"
        }
    }

    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for sp in ["train", "val", "test"]:
        (args.out_root / f"subject_ids_{sp}.txt").write_text("\n".join(subj_ids[sp]) + "\n", encoding="utf-8")
    print("[done]", args.out_root)

if __name__ == "__main__":
    main()
