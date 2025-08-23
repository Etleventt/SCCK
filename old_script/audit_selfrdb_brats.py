#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audit SelfRDB-style BraTS dataset:
- Checks structure, filename alignment, shapes, ranges.
- Checks cross-modality alignment via IoU of nonzero masks.
- Checks evaluation mask alignment against target modality.
"""

import argparse, os, glob, sys, json, random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List

def idx_from_name(p: str) -> int:
    b = os.path.basename(p)
    try:
        return int(b.split("_")[-1].split(".")[0])
    except:
        return -1

def iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / max(int(union), 1)

def audit_split(root: str, split: str, modalities: List[str], target: str,
                image_size: int, iou_thr: float, max_samples: int, nz_thr: float):

    # ---------- gather files ----------
    per_mod_files: Dict[str, List[str]] = {}
    for m in modalities:
        d = os.path.join(root, m, split)
        files = sorted(glob.glob(os.path.join(d, "slice_*.npy")), key=idx_from_name)
        per_mod_files[m] = files

    mask_dir = os.path.join(root, "mask", split)
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "slice_*.npy")), key=idx_from_name)

    # ---------- basic presence ----------
    report = {"split": split, "counts": {}, "problems": []}
    for m, fs in per_mod_files.items():
        report["counts"][m] = len(fs)
    report["counts"]["mask"] = len(mask_files)

    # ---------- filename set equality ----------
    names_ref = set(map(os.path.basename, per_mod_files[target]))
    for m, fs in per_mod_files.items():
        names = set(map(os.path.basename, fs))
        miss = sorted(list(names_ref - names))[:5]
        extra = sorted(list(names - names_ref))[:5]
        if miss or extra:
            report["problems"].append(
                f"[FILES] {m} not matching {target}: missing={len(names_ref-names)} extra={len(names-names_ref)} "
                f"e.g. miss={miss} extra={extra}"
            )
    names_mask = set(map(os.path.basename, mask_files))
    if names_mask:
        miss_m = sorted(list(names_ref - names_mask))[:5]
        extra_m = sorted(list(names_mask - names_ref))[:5]
        if miss_m or extra_m:
            report["problems"].append(
                f"[FILES] mask not matching {target}: missing={len(names_ref-names_mask)} extra={len(names_mask-names_ref)} "
                f"e.g. miss={miss_m} extra={extra_m}"
            )
    else:
        report["problems"].append("[FILES] mask split missing or empty")

    # choose sample set
    sample_names = sorted(list(names_ref))
    if max_samples and len(sample_names) > max_samples:
        random.Random(0).shuffle(sample_names)
        sample_names = sorted(sample_names[:max_samples], key=lambda x: idx_from_name(x))

    # ---------- detailed checks ----------
    shape_bad = []
    range_bad = defaultdict(list)     # modality -> list of names
    mask_val_bad = []
    iou_bad_mod = []                  # (name, m, iou)
    iou_bad_mask = []                 # (name, iou)
    bg_noise_bad = []                 # (name, frac_outside)

    for name in sample_names:
        arrs = {}
        for m in modalities:
            p = os.path.join(root, m, split, name)
            if not os.path.exists(p): continue
            arr = np.load(p).astype(np.float32)
            arrs[m] = arr

        tgt = arrs.get(target, None)
        if tgt is None: 
            continue
        H, W = tgt.shape
        if H != image_size or W != image_size:
            shape_bad.append((name, (H,W)))

        # ranges (should be [0,1] for images)
        for m, a in arrs.items():
            vmin, vmax = float(a.min()), float(a.max())
            if vmin < -1e-4 or vmax > 1.0001:
                range_bad[m].append((name, (vmin, vmax)))

        # target nonzero mask
        tgt_mask = (tgt > 1e-6)

        # per-modality IoU vs target
        for m, a in arrs.items():
            if m == target: continue
            iou = iou_bool((a > 1e-6), tgt_mask)
            if iou < iou_thr:
                iou_bad_mod.append((name, m, iou))

        # dataset mask if exists
        mp = os.path.join(mask_dir, name)
        if os.path.exists(mp):
            m = np.load(mp)
            # mask should be binary {0,1}
            uv = np.unique(m)
            if not set(uv.tolist()).issubset({0,1}):
                mask_val_bad.append((name, uv[:8]))
            iou_m = iou_bool((m>0), tgt_mask)
            if iou_m < iou_thr:
                iou_bad_mask.append((name, iou_m))

            # background noise check: fraction >thr outside mask
            out = (~(m>0))
            frac = float((tgt[out] > 1e-6).mean())
            if frac > nz_thr:
                bg_noise_bad.append((name, frac))

    # ---------- summarize ----------
    summary_lines = []
    summary_lines.append(f"[SPLIT={split}] counts: " + ", ".join([f"{k}={v}" for k,v in report["counts"].items()]))

    if shape_bad:
        summary_lines.append(f"[SHAPE] bad shapes ({len(shape_bad)}) e.g. {shape_bad[:5]}")
    if range_bad:
        for m, L in range_bad.items():
            summary_lines.append(f"[RANGE] {m} out-of-[0,1] slices={len(L)} e.g. {L[:5]}")
    if iou_bad_mod:
        # group by modality
        bym = defaultdict(list)
        for name,m,i in iou_bad_mod: bym[m].append((name,i))
        for m, L in bym.items():
            Ls = sorted(L, key=lambda x:x[1])[:5]
            summary_lines.append(f"[ALIGN] modality-vs-{target} IoU<{iou_thr} : {m} {len(L)}/{len(sample_names)} bad, worst examples: {Ls}")
    if iou_bad_mask:
        worst = sorted(iou_bad_mask, key=lambda x:x[1])[:10]
        summary_lines.append(f"[ALIGN] mask-vs-{target} IoU<{iou_thr} : {len(iou_bad_mask)}/{len(sample_names)} bad, worst: {worst}")
    if bg_noise_bad:
        worst = sorted(bg_noise_bad, key=lambda x:x[1], reverse=True)[:5]
        summary_lines.append(f"[BG] high background nonzero outside mask: {len(bg_noise_bad)}/{len(sample_names)} bad, worst (name, frac): {worst}")

    # filename mismatch problems
    for p in report["problems"]:
        summary_lines.append(p)

    return "\n".join(summary_lines)

def main():
    ap = argparse.ArgumentParser("Audit SelfRDB BraTS dataset")
    ap.add_argument("--root", required=True, help="dataset root, e.g., /.../BraTS64")
    ap.add_argument("--modalities", nargs="+", default=["T1","T2","FLAIR","T1CE"])
    ap.add_argument("--target", default="T1CE", help="target modality used for IoU reference")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--iou_thr", type=float, default=0.995, help="alignment threshold")
    ap.add_argument("--max_samples", type=int, default=2000, help="sample up to N slices per split (0=all)")
    ap.add_argument("--nz_thr", type=float, default=0.02, help="allowed frac of nonzeros outside mask")
    args = ap.parse_args()

    if args.max_samples == 0: args.max_samples = 10**9

    print(f"[config] root={args.root} mods={args.modalities} target={args.target} size={args.image_size} iou_thr={args.iou_thr}")
    for sp in args.splits:
        print(audit_split(args.root, sp, args.modalities, args.target,
                          args.image_size, args.iou_thr, args.max_samples, args.nz_thr))
        print("-"*90)

    # extra: warn if subject leakage check is impossible
    subj_files = [os.path.join(args.root, f"subject_ids_{sp}.txt") for sp in args.splits]
    if not any(os.path.exists(f) for f in subj_files):
        print("[warn] No subject_ids_{split}.txt found. Cannot audit subject-level leakage across splits "
              "(renaming为 slice_* 会丢失受试者信息). 若需要这种审计，建议用我们的一体化导出脚本，它会写 subject_ids_*.txt。")

if __name__ == "__main__":
    main()
