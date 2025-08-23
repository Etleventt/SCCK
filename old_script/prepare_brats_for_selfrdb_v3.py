#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare BraTS -> SelfRDB with *reference modality* cropping.
- One-pass load T1/T2/FLAIR/T1CE per subject
- Build bbox from REF modality (e.g., T1) nonzero mask
- Apply the *same* bbox to all modalities
- Resize to image_size, normalize to [0,1]
- Save slice_*.npy for each modality and mask/ from REF
- Write subject_ids_*.txt and manifest.json
"""
import argparse, os, math, json, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import nibabel as nib
from PIL import Image

ALL = ["t1","t1ce","t2","flair"]

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
    arr=ras.get_fdata(dtype=np.float32)         # (X,Y,Z)
    return np.transpose(arr,(2,1,0))            # (Z,H,W)

def square_bbox_from_mask(mask: np.ndarray, pad:int=0) -> Tuple[slice,slice]:
    H,W=mask.shape
    ys,xs=np.where(mask>0)
    if ys.size==0 or xs.size==0:
        s=min(H,W); y0=(H-s)//2; x0=(W-s)//2
        return slice(y0,y0+s), slice(x0,x0+s)
    y0,y1=ys.min(), ys.max()+1
    x0,x1=xs.min(), xs.max()+1
    h,w=y1-y0, x1-x0
    s=max(h,w)
    cy=(y0+y1)//2; cx=(x0+x1)//2
    y0=max(0, cy - s//2 - pad); x0=max(0, cx - s//2 - pad)
    y1=min(H, y0 + s + 2*pad); x1=min(W, x0 + s + 2*pad)
    return slice(y0,y1), slice(x0,x1)

def percentile_minmax(img: np.ndarray, mask: np.ndarray, p_lo=0.5, p_hi=99.5, eps=1e-6):
    inside=img[mask>0]
    if inside.size==0: inside=img.reshape(-1)
    lo=np.percentile(inside,p_lo); hi=np.percentile(inside,p_hi)
    x=(img-lo)/(hi-lo+eps)
    return np.clip(x,0.0,1.0).astype(np.float32)

def resize01(img01: np.ndarray, size: Optional[int]):
    if not size: return img01
    pil=Image.fromarray((img01*255.0).astype(np.uint8), mode="L")
    pil=pil.resize((size,size), Image.BICUBIC)
    return (np.asarray(pil,dtype=np.uint8)/255.0).astype(np.float32)

def resize_mask(mask: np.ndarray, size: Optional[int]):
    if not size: return (mask>0).astype(np.uint8)
    pil=Image.fromarray(((mask>0).astype(np.uint8)*255), mode="L")
    pil=pil.resize((size,size), Image.NEAREST)
    return (np.asarray(pil,dtype=np.uint8)>127).astype(np.uint8)

def split_subjects(sids: List[str], ratios=(0.8,0.1,0.1), seed=42):
    assert abs(sum(ratios)-1.0)<1e-6
    rnd=random.Random(seed); rnd.shuffle(sids)
    n=len(sids); n_tr=int(n*ratios[0]); n_va=int(n*ratios[1])
    return {"train": sids[:n_tr], "val": sids[n_tr:n_tr+n_va], "test": sids[n_tr+n_va:]}

def main():
    ap=argparse.ArgumentParser("BraTS -> SelfRDB with REF-modality cropping")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--modalities", default="t1,t2,flair,t1ce")
    ap.add_argument("--ref_mod", default="t1", help="参考模态（例如 t1）")
    ap.add_argument("--target_mod", default="t2", help="训练目标（仅写 manifest 用）")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8,0.1,0.1])
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--per_volume", type=int, default=0)  # 0=用全部切片
    ap.add_argument("--z_lo", type=float, default=0.15)
    ap.add_argument("--z_hi", type=float, default=0.95)
    ap.add_argument("--p_lo", type=float, default=0.5)
    ap.add_argument("--p_hi", type=float, default=99.5)
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args()

    mods=[m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    for m in mods:
        if m not in ALL: raise SystemExit(f"unknown modality: {m}")
    ref=args.ref_mod.lower()
    if ref not in mods: mods=[ref]+mods

    subs=find_subjects(args.root)
    if not subs: raise SystemExit("No subjects found.")
    sid2files={}
    for sub in subs:
        sid=sub.name
        files={m: mod_file(sub,m) for m in mods}
        if any(v is None for v in files.values()): continue
        sid2files[sid]=files

    sids=sorted(sid2files.keys())
    splits=split_subjects(sids, tuple(args.split), seed=42)

    for name in [*map(str.upper,mods), "mask"]:
        for sp in ["train","val","test"]:
            (args.out_root/name/sp).mkdir(parents=True, exist_ok=True)

    rng=np.random.default_rng(args.seed)
    subject_ids={sp:[] for sp in ["train","val","test"]}
    counters={sp:0 for sp in ["train","val","test"]}

    for sp, sid_list in splits.items():
        for sid in sid_list:
            files=sid2files[sid]
            vols={m: as_ras_axial(nib.load(str(files[m]))) for m in mods}  # (Z,H,W)
            Z,H,W=next(iter(vols.values())).shape
            if any(v.shape!=(Z,H,W) for v in vols.values()):
                print(f"[skip] shape mismatch: {sid}"); continue

            z0=max(0, min(int(math.floor(Z*args.z_lo)), Z-1))
            z1=max(z0+1, min(int(math.ceil(Z*args.z_hi)), Z))
            zs=range(z0,z1)
            if args.per_volume and args.per_volume>0:
                base=np.linspace(z0, z1-1, num=args.per_volume)
                zs=np.clip(np.round(base + rng.uniform(-1,1,size=len(base))).astype(int), z0, z1-1)

            for z in zs:
                # 先用 REF 的非零定义裁剪窗口
                ref_slice=vols[ref][z]
                ref_mask=(ref_slice>1e-6).astype(np.uint8)
                if ref_mask.sum()<32: continue
                sy,sx=square_bbox_from_mask(ref_mask, pad=0)

                # 同一窗口裁剪所有模态
                crops={m: vols[m][z][sy, sx] for m in mods}
                ref_mask=ref_mask[sy, sx]

                # 统一用 REF 掩膜计算分位→归一化到[0,1]，然后 resize
                imgs01={}
                for m in mods:
                    imgs01[m]=percentile_minmax(crops[m], ref_mask, args.p_lo, args.p_hi)
                    imgs01[m]=resize01(imgs01[m], args.size)
                mask01=resize_mask(ref_mask, args.size)

                idx=counters[sp]
                for m in mods:
                    np.save(args.out_root/str.upper(m)/sp/f"slice_{idx}.npy", imgs01[m])
                np.save(args.out_root/"mask"/sp/f"slice_{idx}.npy", mask01.astype(np.uint8))
                subject_ids[sp].append(f"{sid}|z={int(z)}")
                counters[sp]+=1

    (args.out_root/"manifest.json").write_text(json.dumps({
        "modalities":[m.upper() for m in mods],
        "ref_mod": ref.upper(),
        "target_mod": args.target_mod.upper(),
        "splits": counters,
        "image_size": args.size,
        "z_range":[args.z_lo,args.z_hi],
        "per_volume": args.per_volume,
        "percentiles":[args.p_lo,args.p_hi]
    }, indent=2), encoding="utf-8")
    for sp in ["train","val","test"]:
        (args.out_root/f"subject_ids_{sp}.txt").write_text("\n".join(subject_ids[sp]), encoding="utf-8")
    print("[done]", args.out_root)

if __name__ == "__main__":
    main()
