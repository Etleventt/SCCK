#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SelfRDB-ready BraTS → 64x64 NumpyDataset (resize, no crop, robust per-slice norm)

满足 SelfRDB 官方要求：
- 目录结构：<root>/<mod>/{train,val,test}/slice_XXXX.npy
- 像素范围：每张切片都缩放到 [0,1]
- 按“受试者”分层划分 split，保证各模态对齐
- 掩膜来自 T1 非零区；插值到 64 后再清零背景，避免背景渗漏

预览用法（先看 16 张图）：
python prepare_brats64_selfrdb_resize.py \
  --root  /home/xiaobin/Projects/DBAE/data/raw/brats \
  --out_root /home/xiaobin/Projects/SelfRDB/dataset/brats64_selfrdb \
  --modalities t1,t2,t1ce,flair \
  --preview_n 16 --preview_split val \
  --preview_out /home/xiaobin/Projects/SelfRDB/preview_64 \
  --stop_after_preview 1

正式导出（去掉 stop_after_preview 或设为 0）：
python prepare_brats64_selfrdb_resize.py \
  --root  /home/xiaobin/Projects/DBAE/data/raw/brats \
  --out_root /home/xiaobin/Projects/SelfRDB/dataset/brats64_selfrdb \
  --modalities t1,t2,t1ce,flair \
  --size 64 --z_lo 0.15 --z_hi 0.95 --p_lo 0.5 --p_hi 99.5
"""
import argparse, os, math, json, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont

ALL = ["t1","t1ce","t2","flair"]

# ---------------------------- I/O & utils ----------------------------
def find_subjects(root: Path) -> List[Path]:
    pats=[]
    for pat in ["TCGA-GBM","TCGA-LGG","UPENN-GBM","CPTAC-GBM","IvyGAP","ACRIN-FMISO-Brain","UCSF-PDGM"]:
        d=root/pat
        if d.exists(): pats += [p for p in d.glob("BraTS2021_*") if p.is_dir()]
    if not pats:
        pats=[p for p in root.glob("TCGA-*/BraTS2021_*") if p.is_dir()]
    return sorted(pats)

def mod_file(sub: Path, m: str) -> Optional[Path]:
    ls=list(sub.glob(f"*_{m}.nii*"))
    return ls[0] if ls else None

def as_ras_axial(nii: nib.Nifti1Image) -> np.ndarray:
    ras=nib.as_closest_canonical(nii)
    arr=ras.get_fdata(dtype=np.float32)
    return np.transpose(arr,(2,1,0))  # (Z,H,W)

def percentile_minmax(img: np.ndarray, mask: np.ndarray,
                      p_lo=0.5, p_hi=99.5, eps=1e-6) -> np.ndarray:
    inside=img[mask>0]
    if inside.size<16:  # 过薄切片直接退化为全局
        inside=img.reshape(-1)
    lo=np.percentile(inside, p_lo)
    hi=np.percentile(inside, p_hi)
    x=(img-lo)/(hi-lo+eps)
    return np.clip(x,0.0,1.0).astype(np.float32)

def resize_float01(img01: np.ndarray, size: int) -> np.ndarray:
    # PIL 的 "F" 模式支持 float32；BICUBIC 插值后再 clip
    pil = Image.fromarray(img01.astype(np.float32), mode="F")
    pil = pil.resize((size, size), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)

def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.float32), mode="F")
    pil = pil.resize((size, size), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32)
    return (arr >= 0.5).astype(np.uint8)

def square_center(mask: np.ndarray) -> Tuple[float,float]:
    ys,xs=np.where(mask>0)
    if ys.size==0: return (mask.shape[0]/2.0, mask.shape[1]/2.0)
    return (float(ys.mean()), float(xs.mean()))

def split_subjects(sids: List[str], ratios=(0.8,0.1,0.1), seed=42):
    assert abs(sum(ratios)-1.0)<1e-6
    rnd=random.Random(seed); rnd.shuffle(sids)
    n=len(sids); n_tr=int(n*ratios[0]); n_va=int(n*ratios[1])
    return {"train": sids[:n_tr], "val": sids[n_tr:n_tr+n_va], "test": sids[n_tr+n_va:]}

# ---------------------------- preview ----------------------------
def make_preview_tile(mod2arr: Dict[str,np.ndarray], label: str, size=256) -> Image.Image:
    # 2x2: (t1, t2; t1ce, flair)
    order = ["t1","t2","t1ce","flair"]
    canv = Image.new("L", (size, size), 0)
    cell = size//2
    for idx,mod in enumerate(order):
        if mod not in mod2arr: continue
        arr = (np.clip(mod2arr[mod],0,1)*255.0).astype(np.uint8)
        im  = Image.fromarray(arr, mode="L").resize((cell,cell), Image.NEAREST)
        y = (idx//2)*cell; x=(idx%2)*cell
        canv.paste(im,(x,y))
    # 标题
    draw=ImageDraw.Draw(canv)
    try:
        font = ImageFont.load_default()
    except:
        font=None
    draw.text((5,5), label, 255, font=font)
    return canv

# ---------------------------- main ----------------------------
def main():
    ap=argparse.ArgumentParser("BraTS → SelfRDB 64x64 (resize, per-slice robust norm, bg-masked)")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--modalities", default="t1,t2,flair,t1ce")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8,0.1,0.1])
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--z_lo", type=float, default=0.15)
    ap.add_argument("--z_hi", type=float, default=0.95)
    ap.add_argument("--p_lo", type=float, default=0.5)
    ap.add_argument("--p_hi", type=float, default=99.5)
    ap.add_argument("--seed", type=int, default=42)

    # 预览相关
    ap.add_argument("--preview_n", type=int, default=16)
    ap.add_argument("--preview_split", default="val")
    ap.add_argument("--preview_out", type=Path, default=None)
    ap.add_argument("--stop_after_preview", type=int, default=0)

    args=ap.parse_args()
    mods=[m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    for m in mods:
        if m not in ALL: raise SystemExit(f"unknown modality: {m}")

    # 1) 收集病例
    subs=find_subjects(args.root)
    if not subs: raise SystemExit("No subjects found under --root.")
    sid2files={}
    for sub in subs:
        sid=sub.name
        files={m: mod_file(sub,m) for m in mods}
        if any(v is None for v in files.values()): continue
        sid2files[sid]=files
    sids=sorted(sid2files.keys())
    splits=split_subjects(sids, tuple(args.split), seed=args.seed)

    # 2) 预创建目录
    for name in [*mods, "mask"]:
        for sp in ["train","val","test"]:
            (args.out_root/name/sp).mkdir(parents=True, exist_ok=True)

    # 3) 预览：随机抽取 preview_split 的若干切片画 2x2 小图
    if args.preview_out:
        args.preview_out.mkdir(parents=True, exist_ok=True)
    if args.preview_n>0 and args.preview_out is not None:
        print(f"[Preview] Sampling {args.preview_n} tiles from split={args.preview_split}")
        rnd=random.Random(args.seed)
        pool=splits[args.preview_split]
        chosen=rnd.sample(pool, min(len(pool), args.preview_n))
        tiles=[]
        for sid in chosen:
            vols={m: as_ras_axial(nib.load(str(sid2files[sid][m]))) for m in mods}
            Z,H,W=next(iter(vols.values())).shape
            z0=max(0, min(int(math.floor(Z*args.z_lo)), Z-1))
            z1=max(z0+1, min(int(math.ceil(Z*args.z_hi)), Z))
            z = rnd.randrange(z0, z1)
            ref=vols["t1"][z]
            ref_mask=(ref>1e-6).astype(np.uint8)
            mod2={}
            for m in mods:
                x = percentile_minmax(vols[m][z], ref_mask, args.p_lo, args.p_hi)
                x = resize_float01(x, args.size)
                mod2[m]= (x*resize_mask(ref_mask,args.size)).astype(np.float32)
            lab=f"{args.preview_split}:{sid}|z={z}"
            tiles.append(make_preview_tile(mod2, lab, size=256))
        # 拼图
        n=len(tiles); cols=int(math.ceil(math.sqrt(n))); rows=int(math.ceil(n/cols))
        canv=Image.new("L",(cols*256, rows*256), 0)
        for i,t in enumerate(tiles):
            r=i//cols; c=i%cols
            canv.paste(t,(c*256, r*256))
        outp=args.preview_out/ "preview.png"
        canv.save(outp)
        print(f"[Preview] Saved: {outp}")
        if args.stop_after_preview:
            print("[STOP] --stop_after_preview=1 ; Inspect preview then re-run for export.")
            return

    # 4) 正式导出
    counters={sp:0 for sp in ["train","val","test"]}
    subject_ids={sp:[] for sp in ["train","val","test"]}

    for sp, sid_list in splits.items():
        for sid in sid_list:
            files=sid2files[sid]
            vols={m: as_ras_axial(nib.load(str(files[m]))) for m in mods}
            Z,H,W=next(iter(vols.values())).shape
            if any(v.shape!=(Z,H,W) for v in vols.values()):
                print(f"[skip] shape mismatch: {sid}"); continue

            # 有效 z 范围
            z0=max(0, min(int(math.floor(Z*args.z_lo)), Z-1))
            z1=max(z0+1, min(int(math.ceil(Z*args.z_hi)), Z))

            for z in range(z0, z1):
                ref_slice=vols["t1"][z]
                ref_mask=(ref_slice>1e-6).astype(np.uint8)
                if ref_mask.sum()<32:  # 过薄切片丢弃
                    continue

                mask64 = resize_mask(ref_mask, args.size)

                # 目标模态在掩膜内要有一定“存在感”，否则丢弃极薄/异常切片
                tgt_inside = vols["t2"][z][ref_mask>0]
                if tgt_inside.size<32 or (tgt_inside>1e-6).mean()<0.02 or tgt_inside.std()<1e-4:
                    continue

                for m in mods:
                    x = percentile_minmax(vols[m][z], ref_mask, args.p_lo, args.p_hi)
                    x = resize_float01(x, args.size)
                    x = (x * mask64.astype(np.float32)).astype(np.float32)
                    np.save(args.out_root/m/sp/f"slice_{counters[sp]}.npy", x)

                np.save(args.out_root/"mask"/sp/f"slice_{counters[sp]}.npy", mask64.astype(np.uint8))
                subject_ids[sp].append(f"{sid}|z={int(z)}")
                counters[sp]+=1

    # 5) 清单
    (args.out_root/"manifest.json").write_text(json.dumps({
        "modalities": mods,
        "ref_mod": "t1",
        "target_mod": "t2",
        "splits": counters,
        "image_size": args.size,
        "z_range":[args.z_lo,args.z_hi],
        "percentiles":[args.p_lo,args.p_hi],
        "note": "SelfRDB-ready [0,1], resize to 64, per-slice robust norm within T1 mask; bg zeroed after resize"
    }, indent=2), encoding="utf-8")
    for sp in ["train","val","test"]:
        (args.out_root/f"subject_ids_{sp}.txt").write_text("\n".join(subject_ids[sp]), encoding="utf-8")
    print("[DONE]", args.out_root, "counts:", counters)

if __name__ == "__main__":
    main()
