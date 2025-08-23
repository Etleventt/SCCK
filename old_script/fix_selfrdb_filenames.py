#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# 路径按你的实际修改
python fix_selfrdb_filenames.py \
  --root /home/xiaobin/Projects/SelfRDB/dataset/BraTS64 \
  --modalities T1 T2 FLAIR T1CE
"""
import argparse, re
from pathlib import Path

PAT = re.compile(r"subj-(.+)_z-(\d+)\.npy$", re.IGNORECASE)

def collect_map(dirpath: Path):
    """返回 {(sid,int(z)): path} 映射"""
    m = {}
    for p in dirpath.glob("*.npy"):
        mobj = PAT.match(p.name)
        if mobj:
            sid = mobj.group(1)
            z = int(mobj.group(2))
            m[(sid, z)] = p
    return m

def main():
    ap = argparse.ArgumentParser("Rename subj-*_z-*.npy -> slice_<idx>.npy for SelfRDB")
    ap.add_argument("--root", required=True, type=Path, help="dataset root containing modality folders")
    ap.add_argument("--modalities", nargs="+", required=True, help="e.g. T1 T2 FLAIR T1CE")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # 选择第一个存在的模态作为“基准顺序”
    canon_mod = None
    for m in args.modalities:
        if (args.root/m).exists():
            canon_mod = m
            break
    if canon_mod is None:
        raise SystemExit("No modality folder found under root.")

    for sp in args.splits:
        canon_dir = args.root / canon_mod / sp
        if not canon_dir.exists():
            print(f"[skip] no split dir: {canon_dir}")
            continue

        # 如果已经是 slice_ 命名，跳过
        if any((canon_dir).glob("slice_*.npy")):
            print(f"[skip] already slice_*: {canon_dir}")
            continue

        # 基准模态的 (sid,z) -> path
        canon_map = collect_map(canon_dir)
        if not canon_map:
            print(f"[warn] no subj_*_z-*.npy in {canon_dir}, nothing to do.")
            continue

        # 以 (sid,z) 排序得到稳定顺序
        keys = sorted(canon_map.keys(), key=lambda k: (k[0], k[1]))

        # 为每个模态构建自己的 (sid,z) -> path
        per_mod_maps = {}
        for m in args.modalities:
            d = args.root / m / sp
            if not d.exists():
                print(f"[warn] modality split not found: {d}")
                continue
            per_mod_maps[m] = collect_map(d)

        # 检查对齐完整性
        for m, mmap in per_mod_maps.items():
            missing = [k for k in keys if k not in mmap]
            if missing:
                print(f"[ERROR] {m}/{sp} missing {len(missing)} slices matching baseline; e.g. {missing[:3]}")
                return

        # 执行重命名
        print(f"[info] renaming {sp}: {len(keys)} slices per modality")
        for idx, key in enumerate(keys):
            newname = f"slice_{idx}.npy"
            for m, mmap in per_mod_maps.items():
                src = mmap[key]
                dst = src.with_name(newname)
                if args.dry_run:
                    print(f"{src} -> {dst}")
                else:
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)

    print("[done] all splits processed.")

if __name__ == "__main__":
    main()
