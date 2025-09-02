#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BraTS -> SelfRDB NumpyDataset (pad-or-resize, STRICT [0,1])
-----------------------------------------------------------
- 兼容 SelfRDB README：像素在 [0,1]，目录为 NumpyDataset 结构。
- 新增 --pad_mode {resize,pad}：
  * resize：与你原脚本一致（pad 到正方形后再重采样到 export_size）。
  * pad   ：不重采样，只中心 pad 到 export_size（更贴近官方 256 评测）。

建议：
- 用 pad 模式导出 256×256 微型集，搭配 config_eval256.yaml 与官方权重测试。
"""

from __future__ import annotations
import argparse, os, math, json, random, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw
from tqdm import tqdm

# ------------------------------ helpers ------------------------------ #
ALL = ["t1", "t1ce", "t2", "flair"]
SUBSETS = [
    "TCGA-GBM", "TCGA-LGG", "UPENN-GBM", "CPTAC-GBM", "IvyGAP",
    "ACRIN-FMISO-Brain", "UCSF-PDGM"
]

def find_subjects(root: Path) -> List[Path]:
    pats: List[Path] = []
    for pat in SUBSETS:
        d = root / pat
        if d.exists():
            pats += [p for p in d.glob("BraTS2021_*") if p.is_dir()]
    if not pats:
        pats = [p for p in root.glob("TCGA-*/BraTS2021_*") if p.is_dir()]
    return sorted(pats)

def mod_file(sub: Path, m: str) -> Optional[Path]:
    ls = list(sub.glob(f"*_{m}.nii*"))
    return ls[0] if ls else None

def as_ras_axial(nii_path: Path) -> np.ndarray:
    img = nib.load(str(nii_path))
    ras = nib.as_closest_canonical(img)
    arr = ras.get_fdata(dtype=np.float32)
    return np.transpose(arr, (2, 1, 0))  # (Z, H, W)

def split_subjects(sids: List[str], ratios=(0.8, 0.1, 0.1), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6
    rnd = random.Random(seed)
    rnd.shuffle(sids)
    n = len(sids)
    n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
    return {"train": sids[:n_tr], "val": sids[n_tr:n_tr + n_va], "test": sids[n_tr + n_va:]}

def center_pad_to_square(x: np.ndarray) -> np.ndarray:
    H, W = x.shape
    S = int(max(H, W))
    out = np.zeros((S, S), dtype=x.dtype)
    sy = (S - H) // 2; sx = (S - W) // 2
    out[sy:sy + H, sx:sx + W] = x
    return out

def pad_to_size_float01(x01: np.ndarray, size: int) -> np.ndarray:
    """x01 is already [0,1], 2D array; pad to (size,size) centered."""
    H, W = x01.shape
    out = np.zeros((size, size), dtype=np.float32)
    y0 = (size - H) // 2; x0 = (size - W) // 2
    out[y0:y0 + H, x0:x0 + W] = np.clip(x01, 0.0, 1.0)
    return out

def pad_to_size_mask(mask: np.ndarray, size: int) -> np.ndarray:
    H, W = mask.shape
    out = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - H) // 2; x0 = (size - W) // 2
    out[y0:y0 + H, x0:x0 + W] = (mask > 0).astype(np.uint8)
    return out

def pil_resize_float01(x01: np.ndarray, size: int) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0).astype(np.float32)
    pil = Image.fromarray(x01, mode="F")
    pil = pil.resize((size, size), Image.BICUBIC)
    out = np.asarray(pil, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0)

def pil_resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    pil = pil.resize((size, size), Image.BILINEAR)
    m = (np.asarray(pil, dtype=np.float32) / 255.0) >= 0.5
    return m.astype(np.uint8)

def robust_minmax_over_volume(vol: np.ndarray, refmask_vol: np.ndarray,
                              q_lo: float, q_hi: float) -> Tuple[float, float]:
    inside = vol[refmask_vol > 0]
    if inside.size == 0: inside = vol.reshape(-1)
    lo = float(np.percentile(inside, q_lo))
    hi = float(np.percentile(inside, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.percentile(vol, 0.5)); hi = float(np.percentile(vol, 99.5))
    return lo, hi

def safe_np_save(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f: np.save(f, arr)
    os.replace(tmp, path)

def tile_preview(samples, out_png: Path, tile_px: int = 256):
    mods = ["t1", "t2", "t1ce", "flair"]
    cols = len(samples); rows = len(mods)
    W = cols * tile_px; H = rows * tile_px
    canvas = Image.new("L", (W, H), 0); draw = ImageDraw.Draw(canvas)
    for c, (sid, z, imgs) in enumerate(samples):
        for r, m in enumerate(mods):
            if m not in imgs: continue
            x01 = imgs[m]
            pil = Image.fromarray((np.clip(x01, 0, 1) * 255.0).astype(np.uint8))
            pil = pil.resize((tile_px, tile_px), Image.NEAREST)
            canvas.paste(pil, (c * tile_px, r * tile_px))
        draw.text((c * tile_px + 6, 6), f"{sid}|z={z}", fill=180)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)

# ------------------------------ main ------------------------------ #
def main():
    ap = argparse.ArgumentParser("BraTS -> SelfRDB NumpyDataset (pad-or-resize, [0,1])")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--modalities", default="t1,t2,t1ce,flair")
    ap.add_argument("--ref_mod", default="t1")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--z_lo", type=float, default=0.15)
    ap.add_argument("--z_hi", type=float, default=0.95)
    ap.add_argument("--export_size", type=int, default=64)
    ap.add_argument("--q_lo", type=float, default=0.5)
    ap.add_argument("--q_hi", type=float, default=99.5)
    ap.add_argument("--preview_split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--preview_n", type=int, default=16)
    ap.add_argument("--preview_out", type=Path, default=None)
    ap.add_argument("--stop_after_preview", type=int, default=1)
    ap.add_argument("--max_subjects", type=int, default=0)

    ap.add_argument("--ranges_cache", type=Path, default=None,
                    help="JSON cache for per-subject (lo,hi) ranges")

    ap.add_argument("--pad_mode", choices=["resize", "pad"], default="resize",
                    help="resize: pad->resize到目标尺寸（原逻辑）；pad: 仅pad到目标尺寸，不重采样（建议 256）")

    args = ap.parse_args()

    mods = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    for m in mods:
        if m not in ALL: raise SystemExit(f"Unknown modality: {m}")
    ref = args.ref_mod.lower()
    if ref not in mods: mods = [ref] + mods

    subs = find_subjects(args.root)
    if not subs: raise SystemExit("No subjects found under --root.")

    sid2files: Dict[str, Dict[str, Path]] = {}
    for sub in subs:
        sid = sub.name
        files = {m: mod_file(sub, m) for m in mods}
        if any(v is None for v in files.values()): continue
        sid2files[sid] = files

    sids = sorted(sid2files.keys())
    if args.max_subjects > 0: sids = sids[:args.max_subjects]
    splits = split_subjects(sids, tuple(args.split), seed=args.seed)

    for name in [*mods, "mask"]:
        for sp in ["train", "val", "test"]:
            (args.out_root / name / sp).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ---- Pass 0: percentile ranges ---- #
    print("[Pass0] Computing/Loading per-subject percentile ranges …")
    subj_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    loaded_from_cache = False
    if args.ranges_cache is not None and args.ranges_cache.exists():
        try:
            cache = json.load(open(args.ranges_cache, "r", encoding="utf-8"))
            subj_ranges = {sid: {m: (float(v[0]), float(v[1])) for m, v in mods2.items()}
                           for sid, mods2 in cache.items()}
            loaded_from_cache = True
            print(f"[Pass0] Loaded ranges cache: {args.ranges_cache}")
        except Exception as e:
            print(f"[Pass0] Failed to load ranges cache: {e}; will recompute.")
            subj_ranges = {}

    all_sids = splits["train"] + splits["val"] + splits["test"]
    need = [sid for sid in all_sids if sid not in subj_ranges]
    if need:
        for sid in tqdm(need, desc="ranges"):
            files = sid2files[sid]
            vols = {m: as_ras_axial(files[m]) for m in mods}
            Z, H, W = next(iter(vols.values())).shape
            if any(v.shape != (Z, H, W) for v in vols.values()):
                print(f"[skip] shape mismatch: {sid}"); continue
            refmask_vol = (vols[ref] > 0).astype(np.uint8)
            subj_ranges[sid] = {}
            for m in mods:
                lo, hi = robust_minmax_over_volume(vols[m], refmask_vol, args.q_lo, args.q_hi)
                subj_ranges[sid][m] = (lo, hi)
        if args.ranges_cache is not None:
            serializable = {sid: {m: [float(lo), float(hi)] for m, (lo, hi) in d.items()}
                            for sid, d in subj_ranges.items()}
            args.ranges_cache.parent.mkdir(parents=True, exist_ok=True)
            args.ranges_cache.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
            print(f"[Pass0] Saved ranges cache: {args.ranges_cache}")
    else:
        if loaded_from_cache: print("[Pass0] All per-subject ranges loaded from cache.")
        else: print("[Pass0] No subjects to process.")

    # ---- Preview ---- #
    if args.preview_out is not None and args.preview_n > 0:
        sp = args.preview_split
        pool = splits[sp]
        take = min(args.preview_n, len(pool))
        chosen = rng.choice(pool, size=take, replace=False)

        samples = []
        for sid in chosen:
            files = sid2files[sid]
            vols = {m: as_ras_axial(files[m]) for m in mods}
            Z, H, W = next(iter(vols.values())).shape
            z0 = max(0, min(int(math.floor(Z * args.z_lo)), Z - 1))
            z1 = max(z0 + 1, min(int(math.ceil(Z * args.z_hi)), Z))
            z = int(rng.integers(z0, z1))

            refmask = (vols[ref][z] > 0).astype(np.uint8)
            imgs = {}
            for m in mods:
                lo, hi = subj_ranges[sid][m]
                x = center_pad_to_square(vols[m][z])
                x01 = np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)

                m_sq = center_pad_to_square(refmask)
                if args.pad_mode == "pad":
                    S = x01.shape[0]
                    if S <= args.export_size:
                        x01 = pad_to_size_float01(x01, args.export_size)
                        msk = pad_to_size_mask(m_sq, args.export_size)
                    else:
                        print(f"[warn] slice {sid}|z={z}: size {S}>{args.export_size}, fallback to resize", file=sys.stderr)
                        x01 = pil_resize_float01(x01, args.export_size)
                        msk = pil_resize_mask(m_sq, args.export_size)
                else:
                    x01 = pil_resize_float01(x01, args.export_size)
                    msk = pil_resize_mask(m_sq, args.export_size)

                x01 = (x01 * msk.astype(np.float32)).astype(np.float32)
                imgs[m] = x01
            samples.append((sid, z, imgs))

        out_png = args.preview_out / f"preview_{args.preview_split}_{args.export_size}_{args.pad_mode}.png"
        tile_preview(samples, out_png)
        print(f"[Preview] wrote: {out_png}")
        if args.stop_after_preview:
            print("[STOP] --stop_after_preview=1 ; Inspect preview then re-run with 0 to export.")
            meta = {"modalities": mods, "ref_mod": ref, "export_size": args.export_size,
                    "z_range": [args.z_lo, args.z_hi], "q": [args.q_lo, args.q_hi],
                    "pad_mode": args.pad_mode}
            (args.out_root / "preview_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return

    # ---- Export ---- #
    print(f"[Pass1] Exporting SelfRDB NumpyDataset @ {args.export_size} ({args.pad_mode})")
    counters = {sp: 0 for sp in ["train", "val", "test"]}
    subject_ids = {sp: [] for sp in ["train", "val", "test"]}

    for sp, sid_list in splits.items():
        for sid in tqdm(sid_list, desc=sp):
            files = sid2files[sid]
            vols = {m: as_ras_axial(files[m]) for m in mods}
            Z, H, W = next(iter(vols.values())).shape
            if any(v.shape != (Z, H, W) for v in vols.values()):
                print(f"[skip] shape mismatch: {sid}"); continue

            z0 = max(0, min(int(math.floor(Z * args.z_lo)), Z - 1))
            z1 = max(z0 + 1, min(int(math.ceil(Z * args.z_hi)), Z))
            refmask_vol = (vols[ref] > 0).astype(np.uint8)

            for z in range(z0, z1):
                refmask = refmask_vol[z]
                if int(refmask.sum()) < 32: continue

                m_sq = center_pad_to_square(refmask)
                if args.pad_mode == "pad":
                    S = m_sq.shape[0]
                    if S <= args.export_size:
                        m_low = pad_to_size_mask(m_sq, args.export_size)
                    else:
                        print(f"[warn] slice {sid}|z={z}: mask size {S}>{args.export_size}, fallback to resize", file=sys.stderr)
                        m_low = pil_resize_mask(m_sq, args.export_size)
                else:
                    m_low = pil_resize_mask(m_sq, args.export_size)

                for m in mods:
                    lo, hi = subj_ranges[sid][m]
                    x = center_pad_to_square(vols[m][z])
                    x01 = np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)

                    if args.pad_mode == "pad":
                        S = x01.shape[0]
                        if S <= args.export_size:
                            x01 = pad_to_size_float01(x01, args.export_size)
                        else:
                            x01 = pil_resize_float01(x01, args.export_size)
                    else:
                        x01 = pil_resize_float01(x01, args.export_size)

                    x01 = (x01 * m_low.astype(np.float32)).astype(np.float32)
                    x01 = np.clip(x01, 0.0, 1.0)
                    safe_np_save(args.out_root / m / sp / f"slice_{counters[sp]}.npy", x01)

                safe_np_save(args.out_root / "mask" / sp / f"slice_{counters[sp]}.npy", m_low.astype(np.uint8))
                subject_ids[sp].append(f"{sid}|z={z}")
                counters[sp] += 1

    manifest = {
        "modalities": mods, "ref_mod": ref, "export_size": args.export_size,
        "z_range": [args.z_lo, args.z_hi], "q": [args.q_lo, args.q_hi],
        "splits": counters, "note": "Images scaled to [0,1].", "pad_mode": args.pad_mode
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for sp in ["train", "val", "test"]:
        (args.out_root / f"subject_ids_{sp}.txt").write_text("\n".join(subject_ids[sp]), encoding="utf-8")

    print("[DONE] Wrote dataset to:", args.out_root)

if __name__ == "__main__":
    main()
