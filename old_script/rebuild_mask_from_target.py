# rebuild_mask_from_target.py
# 用法示例：
# python rebuild_mask_from_target.py --root /home/xiaobin/Projects/SelfRDB/dataset/BraTS64 --target T1CE
import os, argparse, numpy as np
from glob import glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--target", required=True, help="目标模态文件夹名，如 T1CE")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--thr", type=float, default=1e-6)
    args = ap.parse_args()

    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        tgt_dir = os.path.join(args.root, args.target, sp)
        if not os.path.isdir(tgt_dir):
            print(f"[skip] no dir: {tgt_dir}"); continue
        out_dir = os.path.join(args.root, "mask", sp)
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(glob(os.path.join(tgt_dir, "slice_*.npy")),
                       key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
        for f in files:
            arr = np.load(f).astype(np.float32)
            m = (arr > args.thr).astype(np.uint8)
            np.save(os.path.join(out_dir, os.path.basename(f)), m)
        print(f"[ok] wrote masks from {args.target}/{sp} -> {out_dir}")

if __name__ == "__main__":
    main()
