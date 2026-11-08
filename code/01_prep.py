"""Crop the fundus disc and resize IDRiD images (and lesion masks) to a fixed size.
Usage: python -I code/01_prep.py <idrid_root> <out_dir> [size]
<idrid_root> holds "A. Segmentation" and "B. Disease Grading" from the IDRiD release (CC BY 4.0).
Writes grading.npz (train/test images, DR grade, DME grade, ids) and lesions.npz (images, masks for MA, HE, EX, SE, OD)."""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root, out = Path(sys.argv[1]), Path(sys.argv[2])
S = int(sys.argv[3]) if len(sys.argv) > 3 else 384
out.mkdir(parents=True, exist_ok=True)


def bbox(img):
    """Bounding box of the circular fundus area (pixels brighter than a dark threshold)."""
    g = np.asarray(img.convert("L").reduce(8))
    ys, xs = np.where(g > 15)
    f = 8
    return xs.min() * f, ys.min() * f, (xs.max() + 1) * f, (ys.max() + 1) * f


def square(img, box, resample):
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    crop = img.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2))  # pads with black outside
    return crop.resize((S, S), resample)


gdir = root / "B. Disease Grading"
data = {}
for split, sub, lab in [("train", "a. Training Set", "a. IDRiD_Disease Grading_Training Labels.csv"),
                        ("test", "b. Testing Set", "b. IDRiD_Disease Grading_Testing Labels.csv")]:
    rows = [r for r in list(csv.reader(open(gdir / "2. Groundtruths" / lab)))[1:] if r and r[0]]
    X, y, dme, ids = [], [], [], []
    for r in rows:
        im = Image.open(gdir / "1. Original Images" / sub / f"{r[0]}.jpg").convert("RGB")
        X.append(np.asarray(square(im, bbox(im), Image.BILINEAR), np.uint8))
        y.append(int(r[1])); dme.append(int(r[2])); ids.append(r[0])
    data[f"X_{split}"], data[f"y_{split}"] = np.stack(X), np.array(y)
    data[f"dme_{split}"], data[f"id_{split}"] = np.array(dme), np.array(ids)
    print(split, data[f"X_{split}"].shape, np.bincount(data[f"y_{split}"]))
np.savez_compressed(out / "grading.npz", **data)

sdir = root / "A. Segmentation"
LES = [("1. Microaneurysms", "MA"), ("2. Haemorrhages", "HE"), ("3. Hard Exudates", "EX"), ("4. Soft Exudates", "SE"), ("5. Optic Disc", "OD")]
X, M, ids = [], [], []
for sub in ["a. Training Set", "b. Testing Set"]:
    for p in sorted((sdir / "1. Original Images" / sub).glob("*.jpg")):
        im = Image.open(p).convert("RGB"); box = bbox(im)
        X.append(np.asarray(square(im, box, Image.BILINEAR), np.uint8))
        masks = []
        for folder, code in LES:
            mp = sdir / "2. All Segmentation Groundtruths" / sub / folder / f"{p.stem}_{code}.tif"
            if mp.exists():
                m = Image.open(mp).convert("L")
                masks.append(np.asarray(square(m, box, Image.NEAREST)) > 0)
            else:
                masks.append(np.zeros((S, S), bool))
        M.append(np.stack(masks)); ids.append(f"{sub[0]}_{p.stem}")
np.savez_compressed(out / "lesions.npz", X=np.stack(X), M=np.stack(M), ids=np.array(ids), names=np.array([c for _, c in LES]))
print("lesion images", len(X), "with any lesion mask:", int((np.stack(M)[:, :4].reshape(len(M), -1).any(1)).sum()))
