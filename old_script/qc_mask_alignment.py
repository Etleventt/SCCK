# qc_mask_alignment.py
import os, glob, numpy as np, sys
tgt_dir = sys.argv[1]  # /.../T1CE/test
msk_dir = sys.argv[2]  # /.../mask/test
bad=[]
for g in sorted(glob.glob(os.path.join(tgt_dir, "slice_*.npy"))):
    b=os.path.basename(g); m=os.path.join(msk_dir,b)
    if not os.path.exists(m): bad.append((b,"missing")); continue
    G=(np.load(g)>1e-6); M=(np.load(m)>0)
    inter=(G&M).sum(); union=(G|M).sum()
    iou = inter/max(union,1)
    if iou<0.98: bad.append((b, f"{iou:.3f}"))
print("misaligned:",len(bad))
for x in bad[:10]: print(x)
