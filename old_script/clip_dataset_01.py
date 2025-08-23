# clip_dataset_01.py
import os, argparse, numpy as np
from glob import glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)  # e.g. /.../brats64_ref_t1_v2
    ap.add_argument("--modalities", default="t1,t2,flair,t1ce")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    mods=[m.strip() for m in args.modalities.split(",") if m.strip()]
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        for m in mods:
            d=os.path.join(args.root, m, sp)
            if not os.path.isdir(d): 
                print(f"[skip] {d}"); continue
            files=sorted(glob(os.path.join(d,"slice_*.npy")),
                        key=lambda p:int(os.path.basename(p).split("_")[-1].split(".")[0]))
            for p in files:
                a=np.load(p).astype(np.float32)
                a=np.clip(a, 0.0, 1.0, out=a)   # ★ 原地裁剪
                np.save(p, a)
            print(f"[ok] clipped {m}/{sp} -> [0,1]  ({len(files)} files)")
if __name__ == "__main__":
    main()
