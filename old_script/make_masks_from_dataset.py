# make_masks_from_dataset.py
# python make_masks_from_dataset.py --root /home/xiaobin/Projects/SelfRDB/dataset/BraTS64

import os, numpy as np, argparse
from glob import glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset dir, e.g., /.../BraTS64")
    ap.add_argument("--modalities", default="T1,T2,FLAIR,T1CE",
                    help="comma-separated, will take union of nonzeros")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        # 以第一个存在的模态当“索引”，拿到切片数量
        index_mod, files = None, []
        for m in mods:
            p = os.path.join(args.root, m, split)
            if os.path.isdir(p):
                files = sorted([f for f in os.listdir(p) if f.endswith(".npy")],
                               key=lambda x: int(x.split("_")[-1].split(".")[0]))
                if files:
                    index_mod = m; break
        if not files:
            print(f"[skip] no slices found for split={split}")
            continue

        out_dir = os.path.join(args.root, "mask", split)
        os.makedirs(out_dir, exist_ok=True)

        for f in files:
            union = None
            for m in mods:
                p = os.path.join(args.root, m, split, f)
                if os.path.exists(p):
                    arr = np.load(p)
                    mask = (arr > 0).astype(np.uint8)
                    union = mask if union is None else (union | mask)
            if union is None:
                continue
            np.save(os.path.join(out_dir, f), union.astype(np.uint8))
        print(f"[ok] wrote masks -> {out_dir}")

if __name__ == "__main__":
    main()
