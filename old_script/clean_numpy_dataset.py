#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, re, os, shutil, json
from pathlib import Path
from typing import List, Dict, Set

SLICE_RE = re.compile(r"^slice_(\d+)\.npy$")

def list_indices(dirpath: Path) -> Set[int]:
    if not dirpath.exists(): return set()
    idxs=set()
    for p in dirpath.glob("slice_*.npy"):
        m=SLICE_RE.match(p.name)
        if m: idxs.add(int(m.group(1)))
    return idxs

def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hard":
        try:
            os.link(src, dst)
            return
        except OSError:
            # 跨分区无法硬链接，退化为拷贝
            shutil.copy2(src, dst)
            return
    elif mode == "symlink":
        try:
            if dst.exists(): dst.unlink()
            dst.symlink_to(src)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    else:
        shutil.copy2(src, dst)

def clean_split(root: Path, out_root: Path, split: str, modalities: List[str], require_mask: bool, link_mode: str):
    # 统计各模态索引
    sets = {}
    for m in modalities:
        sets[m] = list_indices(root / m / split)
    if require_mask:
        sets["mask"] = list_indices(root / "mask" / split)

    # 交集：只保留所有需要模态都存在的切片
    required_keys = modalities + (["mask"] if require_mask else [])
    inter = None
    for k in required_keys:
        inter = sets[k] if inter is None else (inter & sets[k])
    inter = sorted(list(inter))
    print(f"[{split}] keep={len(inter)} | " + " | ".join(f"{k}:{len(v)}" for k,v in sets.items()))

    # 输出到新目录，并重排为连续编号
    remap = []  # 记录 idx_old -> idx_new
    for new_i, old_i in enumerate(inter):
        remap.append({"old": int(old_i), "new": int(new_i)})
        for m in modalities:
            src = root / m / split / f"slice_{old_i}.npy"
            dst = out_root / m / split / f"slice_{new_i}.npy"
            link_or_copy(src, dst, link_mode)
        if require_mask:
            src = root / "mask" / split / f"slice_{old_i}.npy"
            dst = out_root / "mask" / split / f"slice_{new_i}.npy"
            link_or_copy(src, dst, link_mode)

    # 写出映射以便排查
    (out_root / f"remap_{split}.json").write_text(json.dumps(remap, indent=2), encoding="utf-8")
    return len(inter), sets

def main():
    ap = argparse.ArgumentParser("Clean SelfRDB NumpyDataset by intersecting consistent slice indices.")
    ap.add_argument("--root", required=True, type=Path, help="原始（可能脏）的 NumpyDataset 目录")
    ap.add_argument("--out_root", required=True, type=Path, help="输出干净数据目录（新目录）")
    ap.add_argument("--modalities", default="t1,t2,t1ce,flair")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--no_mask", action="store_true", help="不强制要求 mask；默认需要")
    ap.add_argument("--link", choices=["hard","symlink","copy"], default="hard", help="拷贝方式（硬链接/软链接/复制）")
    ap.add_argument("--dry_run", action="store_true", help="只统计不落盘")
    args = ap.parse_args()

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    require_mask = not args.no_mask

    # 统计 + 可选清洗
    summary = {}
    for sp in args.splits:
        if args.dry_run:
            # 仅显示将保留多少
            sets = {m: list_indices(args.root/m/sp) for m in modalities}
            if require_mask:
                sets["mask"] = list_indices(args.root/"mask"/sp)
            inter = None
            for k in modalities + (["mask"] if require_mask else []):
                inter = sets[k] if inter is None else (inter & sets[k])
            inter = sorted(list(inter))
            summary[sp] = {"keep": len(inter), "counts": {k: len(v) for k,v in sets.items()}}
            print(f"[DRY] {sp} keep={len(inter)} | " + " | ".join(f"{k}:{len(v)}" for k,v in sets.items()))
        else:
            kept, sets = clean_split(args.root, args.out_root, sp, modalities, require_mask, args.link)
            summary[sp] = {"keep": kept, "counts": {k: len(v) for k,v in sets.items()}}

    # 元信息
    meta = {
        "from_root": str(args.root),
        "out_root": str(args.out_root),
        "modalities": modalities,
        "splits": summary,
        "require_mask": require_mask,
        "link": args.link
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "manifest_clean.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[DONE] Wrote manifest:", args.out_root / "manifest_clean.json")

if __name__ == "__main__":
    main()
