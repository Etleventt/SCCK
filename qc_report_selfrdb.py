#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SelfRDB BraTS 切片数据集质检 + 可视化报告

功能：
1) 结构完整性：检查小写目录结构 t1/ t2/ flair/ t1ce/ mask/，split(train/val/test) 是否存在；
2) 计数与形状：各模态与 mask 在每个 split 的文件数量、分辨率是否一致；
3) 数值范围：像素是否在 [0,1]（允许微小浮点超界），统计越界切片；
4) 对齐度：用“非零体素”IoU 检查每个模态 vs 目标模态（默认 t1），以及 mask vs 目标模态；
5) 背景泄漏：统计掩膜外非零像素比例；
6) 受试者泄漏：若存在 subject_ids_*.txt，检查 train/val/test 之间是否有同一 subject；
7) 可视化：为每个 split 导出若干行样例，每行显示 t1 / t2 / flair / t1ce / t1+mask / t2+mask；
   另导出“最差 IoU”问题切片的拼图，方便定位。

用法：
python qc_report_selfrdb.py \
  --root /path/to/dataset/brats64_ref_t1_v2 \
  --modalities t1,t2,flair,t1ce \
  --target t1 \
  --image_size 64 \
  --iou_thr 0.995 \
  --max_samples 2000 \
  --viz_n 16 \
  --out qc_report_brats64_ref_t1_v2

备注：
- 仅依赖 numpy / matplotlib / pillow（可选安装 scikit-image 用于更丰富指标，但非必需）。
- IoU 的“非零”判定阈值默认 1e-6，可用 --nz_thr 调整。
"""

import os, argparse, json, math, csv
from glob import glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def natural_key(path):
    b = os.path.basename(path)
    try:
        return int(b.split("_")[-1].split(".")[0])
    except:
        return b

def list_slices(mod_dir):
    return sorted(glob(os.path.join(mod_dir, "slice_*.npy")), key=natural_key)

def load_slice(path):
    return np.load(path).astype(np.float32)

def iou_nonzero(a, b, thr=1e-6):
    A = (a > thr).astype(np.uint8)
    B = (b > thr).astype(np.uint8)
    inter = (A & B).sum()
    union = (A | B).sum()
    if union == 0:
        return 1.0  # 两者全零视作完美重合
    return inter / float(union)

def bg_leak_frac(img, mask, thr=1e-6):
    """掩膜外非零像素占‘掩膜外总像素’的比例"""
    outside = (mask == 0)
    if outside.sum() == 0:
        return 0.0
    return float(((img > thr) & outside).sum()) / float(outside.sum())

def check_range(arr, eps=1e-3):
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    ok = (vmin >= -eps) and (vmax <= 1.0 + eps)
    return ok, (vmin, vmax)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def overlay_mask(im, mask, alpha=0.35):
    """返回三通道可显示图像，mask 红色叠加"""
    im01 = np.clip(im, 0, 1)
    h, w = im01.shape
    rgb = np.stack([im01, im01, im01], axis=-1)
    m = (mask > 0).astype(np.float32)
    # 红色通道增强
    rgb[..., 0] = np.clip(rgb[..., 0]*(1-alpha) + alpha*m, 0, 1)
    return rgb

def grid_save(images, titles, save_path, ncols=6, figsize=(14, 10)):
    """
    images: list of HxW (gray) 或 HxW x3 (rgb)
    titles: 同长度标题
    """
    n = len(images)
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / ncols))
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        ax = plt.subplot(nrows, ncols, i+1)
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def read_subject_ids(root, split):
    p = os.path.join(root, f"subject_ids_{split}.txt")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        ids = [line.strip().split("|")[0] for line in f if line.strip()]
    return set(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="数据集根目录（含 t1/ t2/ flair/ t1ce/ mask/）")
    ap.add_argument("--modalities", default="t1,t2,flair,t1ce")
    ap.add_argument("--target", default="t1", help="对齐参考模态（任务目标，比如 t1 或 t1ce）")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--iou_thr", type=float, default=0.995)
    ap.add_argument("--nz_thr", type=float, default=1e-6, help="非零阈值（用于 IoU / 泄漏）")
    ap.add_argument("--max_samples", type=int, default=2000, help="每个 split 最多抽样多少切片做 IoU/范围统计")
    ap.add_argument("--viz_n", type=int, default=16, help="每个 split 的可视化样本数")
    ap.add_argument("--out", default="qc_report", help="报告输出目录")
    args = ap.parse_args()

    root = args.root
    out_root = args.out
    ensure_dir(out_root)

    mods = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    target = args.target.strip().lower()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # 0) 目录检查
    problems = []
    required_dirs = mods + ["mask"]
    for d in required_dirs:
        for sp in splits:
            p = os.path.join(root, d, sp)
            if not os.path.isdir(p):
                problems.append(f"[MISSING DIR] {p}")

    # 1) 计数与形状
    all_counts = defaultdict(dict)
    all_shapes_ok = True
    for sp in splits:
        shape_ref = None
        for m in required_dirs:
            d = os.path.join(root, m, sp)
            files = list_slices(d)
            all_counts[sp][m] = len(files)
            # 抽一个检查形状
            if files:
                a = load_slice(files[0])
                if a.ndim != 2 or a.shape[0] != args.image_size or a.shape[1] != args.image_size:
                    problems.append(f"[BAD SHAPE] {m}/{sp} first={a.shape}, expect ({args.image_size},{args.image_size})")
                    all_shapes_ok = False

    # 2) 受试者泄漏（如果有 subject_ids_*.txt）
    leaks = []
    sid_sets = {}
    for sp in splits:
        sids = read_subject_ids(root, sp)
        if sids is not None:
            sid_sets[sp] = sids
    if sid_sets:
        for i in range(len(splits)):
            for j in range(i+1, len(splits)):
                si, sj = splits[i], splits[j]
                inter = sid_sets[si] & sid_sets[sj]
                if inter:
                    leaks.append((si, sj, len(inter)))

    # 3) IoU / BG 泄漏 / 数值范围
    rng = np.random.default_rng(0)
    stats = { sp: { "range_bad": defaultdict(list),
                    "bg_bad": [],
                    "iou_bad": defaultdict(list),  # modality -> list[(name, iou)]
                    "mask_iou_bad": [] }
              for sp in splits }

    for sp in splits:
        # 以 target split 作为基准，和 mask split 文件名对齐
        t_dir = os.path.join(root, target, sp)
        t_files = list_slices(t_dir)
        if not t_files:
            continue

        idxs = np.arange(len(t_files))
        if len(idxs) > args.max_samples:
            idxs = rng.choice(idxs, size=args.max_samples, replace=False)
        idxs.sort()
        # 预取其它模态文件列表
        per_mod_files = { m: list_slices(os.path.join(root, m, sp)) for m in mods }
        mask_files = list_slices(os.path.join(root, "mask", sp))

        for ii in idxs:
            base = os.path.basename(t_files[ii])
            t_img = load_slice(t_files[ii])
            # mask
            if ii < len(mask_files) and os.path.basename(mask_files[ii]) == base:
                msk = load_slice(mask_files[ii]).astype(np.uint8)
            else:
                # fallback：位置不对应就按文件名搜索
                mp = os.path.join(root, "mask", sp, base)
                if os.path.exists(mp):
                    msk = load_slice(mp).astype(np.uint8)
                else:
                    msk = (t_img > args.nz_thr).astype(np.uint8)  # 尽量不为空
            # 范围检查（所有模态 + 掩膜）
            for m in mods + ["mask"]:
                if m == "mask":
                    arr = msk.astype(np.float32)
                else:
                    p = os.path.join(root, m, sp, base)
                    if not os.path.exists(p): 
                        continue
                    arr = load_slice(p)
                ok, (vmin, vmax) = check_range(arr)
                if not ok:
                    stats[sp]["range_bad"][m].append((base, vmin, vmax))

            # 背景泄漏（target 上统计最直观）
            bgf = bg_leak_frac(t_img, msk, thr=args.nz_thr)
            if bgf > 0.0:
                stats[sp]["bg_bad"].append((base, bgf))

            # IoU：各模态 vs target
            for m in mods:
                if m == target: 
                    continue
                p = os.path.join(root, m, sp, base)
                if not os.path.exists(p): 
                    continue
                x = load_slice(p)
                iou = iou_nonzero(x, t_img, thr=args.nz_thr)
                if iou < args.iou_thr:
                    stats[sp]["iou_bad"][m].append((base, iou))

            # mask vs target IoU（非零区域一致性）
            miou = iou_nonzero(msk.astype(np.float32), t_img, thr=args.nz_thr)
            if miou < args.iou_thr:
                stats[sp]["mask_iou_bad"].append((base, miou))

    # 4) 打印摘要 & 写 CSV
    summary = {
        "root": root,
        "image_size": args.image_size,
        "modalities": mods,
        "target": target,
        "splits_counts": all_counts,
        "problems": problems,
        "subject_leaks": leaks,
        "iou_thr": args.iou_thr,
        "nz_thr": args.nz_thr
    }
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("="*90)
    print(f"[SUMMARY] root={root} target={target} image_size={args.image_size}")
    for sp in splits:
        cnts = all_counts[sp]
        print(f"[SPLIT={sp}] counts: " + ", ".join([f"{k}={cnts.get(k,0)}" for k in mods+['mask']]))
    if problems:
        print("[STRUCT/SHAPE PROBLEMS]")
        for p in problems: print("  -", p)
    else:
        print("[STRUCT/SHAPE] OK")

    if leaks:
        print("[SUBJECT LEAKS] 发现跨 split 受试者重叠：")
        for si, sj, n in leaks:
            print(f"  - {si} ↔ {sj}: {n}")
    else:
        print("[SUBJECT LEAKS] 未发现（或无 subject_ids_*.txt）")

    for sp in splits:
        print("-"*90)
        print(f"[{sp}] RANGE out-of-[0,1]:")
        for m in mods + ["mask"]:
            bad = stats[sp]["range_bad"][m]
            print(f"  {m}: {len(bad)}")
            if bad[:5]:
                eg = ", ".join([f"('{b[0]}',{b[1]:.4g},{b[2]:.4g})" for b in bad[:5]])
                print("    e.g.", eg)

        print(f"[{sp}] mask-vs-{target} IoU<{args.iou_thr}: {len(stats[sp]['mask_iou_bad'])}")
        if stats[sp]['mask_iou_bad'][:5]:
            print("   worst:", sorted(stats[sp]['mask_iou_bad'], key=lambda x:x[1])[:5])

        for m in mods:
            if m == target: continue
            bad = stats[sp]["iou_bad"][m]
            print(f"[{sp}] {m}-vs-{target} IoU<{args.iou_thr}: {len(bad)}")
            if bad[:5]:
                print("   worst:", sorted(bad, key=lambda x:x[1])[:5])

        bg = sorted(stats[sp]["bg_bad"], key=lambda x: -x[1])[:5]
        print(f"[{sp}] BG nonzero outside mask > 0: {len(stats[sp]['bg_bad'])}")
        if bg:
            print("   worst (name, frac):", [(b[0], f"{b[1]:.3f}") for b in bg])

    # 写 CSV（便于快速定位具体切片）
    for sp in splits:
        sp_out = os.path.join(out_root, sp)
        ensure_dir(sp_out)
        # mask vs target
        with open(os.path.join(sp_out, "mask_vs_target_iou_bad.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["slice","iou"])
            for name, iou in sorted(stats[sp]["mask_iou_bad"], key=lambda x:x[1]):
                w.writerow([name, f"{iou:.6f}"])
        # modalities vs target
        for m in mods:
            if m == target: continue
            with open(os.path.join(sp_out, f"{m}_vs_{target}_iou_bad.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["slice","iou"])
                for name, iou in sorted(stats[sp]["iou_bad"][m], key=lambda x:x[1]):
                    w.writerow([name, f"{iou:.6f}"])
        # range
        for m in mods+["mask"]:
            with open(os.path.join(sp_out, f"{m}_range_bad.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["slice","vmin","vmax"])
                for name, vmin, vmax in stats[sp]["range_bad"][m]:
                    w.writerow([name, f"{vmin:.6f}", f"{vmax:.6f}"])
        # bg leak
        with open(os.path.join(sp_out, "bg_leak.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["slice","frac_outside_nonzero"])
            for name, frac in sorted(stats[sp]["bg_bad"], key=lambda x:-x[1]):
                w.writerow([name, f"{frac:.6f}"])

    # 5) 可视化：随机样例 + worst IoU 样例
    for sp in splits:
        sp_out = os.path.join(out_root, sp)
        ensure_dir(sp_out)

        # 随机可视化
        t_files = list_slices(os.path.join(root, target, sp))
        mask_files = list_slices(os.path.join(root, "mask", sp))
        if not t_files:
            continue
        idxs = np.arange(len(t_files))
        rng = np.random.default_rng(123+len(sp))
        if len(idxs) > args.viz_n:
            idxs = rng.choice(idxs, size=args.viz_n, replace=False)
        idxs = sorted(idxs)

        imgs, titles = [], []
        for ii in idxs:
            base = os.path.basename(t_files[ii])
            t_img = load_slice(t_files[ii])
            msk = load_slice(mask_files[ii]).astype(np.uint8) if ii<len(mask_files) else (t_img>args.nz_thr).astype(np.uint8)

            # 展示顺序：t1 / t2 / flair / t1ce / t1+mask / t2+mask（按存在性）
            for m in ["t1","t2","flair","t1ce"]:
                p = os.path.join(root, m, sp, base)
                if os.path.exists(p):
                    x = load_slice(p)
                    imgs.append(np.clip(x,0,1))
                    titles.append(f"{m}:{base}")
            imgs.append(overlay_mask(t_img, msk))
            titles.append(f"{target}+mask:{base}")

            p = os.path.join(root, "t2", sp, base)
            if os.path.exists(p):
                x2 = load_slice(p)
                imgs.append(overlay_mask(x2, msk))
                titles.append(f"t2+mask:{base}")

        grid_save(imgs, titles, os.path.join(sp_out, "random_samples.png"), ncols=6, figsize=(16, 10))

        # 最差 IoU 可视化（以 t2-vs-target 为例，若存在）
        if stats[sp]["iou_bad"].get("t2"):
            worst = sorted(stats[sp]["iou_bad"]["t2"], key=lambda x:x[1])[:min(12, len(stats[sp]['iou_bad']['t2']))]
            imgs2, titles2 = [], []
            for base, iou in worst:
                t_img = load_slice(os.path.join(root, target, sp, base))
                msk   = load_slice(os.path.join(root, "mask", sp, base)).astype(np.uint8) if os.path.exists(os.path.join(root,"mask",sp,base)) else (t_img>args.nz_thr).astype(np.uint8)
                x2    = load_slice(os.path.join(root, "t2", sp, base))
                imgs2 += [np.clip(t_img,0,1), np.clip(x2,0,1), overlay_mask(t_img, msk), overlay_mask(x2, msk)]
                titles2 += [f"{target}:{base}", f"t2:{base}", f"{target}+mask (IoU={iou:.3f})", f"t2+mask"]
            grid_save(imgs2, titles2, os.path.join(sp_out, "worst_iou_t2_vs_target.png"), ncols=4, figsize=(14, 12))

    print("="*90)
    print(f"[DONE] 质检完成，报告已写入：{out_root}")
    print("查看：summary.json / 每个 split 下的 CSV 与 PNG 拼图。")

if __name__ == "__main__":
    main()
