"""Do Grad-CAM explanations point at real lesions? Usage: python -I code/04_gradcam.py <lesions.npz> <results_dir> [head ...]
For each saved seed-0 model, computes Grad-CAM (ResNet-18 layer4) on the 81 IDRiD segmentation images and reports:
- lesion energy: share of CAM mass inside the union of MA, HE, EX and SE masks, against the share of the fundus those
  masks cover (what a random map would score);
- optic-disc energy: share of CAM mass on the optic disc, against its area share (a known shortcut);
- pointing game: whether the CAM peak falls within 8 px of any lesion pixel.
Target: CE / evidential = output of the predicted grade; ordinal = sum of the four threshold logits (overall severity).
Note: segmentation images may also appear among the grading training images, so this checks where the model looks,
not held-out accuracy."""
import sys
from pathlib import Path

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from scipy import ndimage

L = np.load(sys.argv[1]); X, M = L["X"], L["M"]
res = Path(sys.argv[2]); heads = sys.argv[3:] or ["ce", "ordinal", "evidential"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
fundus = (X.mean(-1) > 15)
lesion = M[:, :4].any(1)
lesion_near = np.stack([ndimage.binary_dilation(m, iterations=8) for m in lesion])
od = M[:, 4]
print(f"lesion area share of fundus: {np.mean([lesion[i][fundus[i]].mean() for i in range(len(X))]):.3%}; "
      f"optic disc share: {np.mean([od[i][fundus[i]].mean() for i in range(len(X))]):.3%}")
for h in heads:
    path = res / f"model_{h}_s0.pt"
    if not path.exists(): print(h, "no model"); continue
    net = torchvision.models.resnet18(); net.fc = nn.Linear(512, 4 if h == "ordinal" else 5)
    net.load_state_dict(torch.load(path)); net.eval()
    feats = {}
    net.layer4.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    le, oe, hit, base_le, base_oe = [], [], [], [], []
    for i in range(len(X)):
        x = (torch.from_numpy(X[i:i + 1]).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD
        out = net(x); a = feats["a"]; a.retain_grad()
        target = out.sum() if h == "ordinal" else out[0, out[0].argmax()]
        net.zero_grad(); target.backward()
        w = a.grad.mean((2, 3), keepdim=True); cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=X.shape[1:3], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
        cam = cam * fundus[i]; tot = cam.sum() + 1e-12
        le.append(cam[lesion[i]].sum() / tot); oe.append(cam[od[i]].sum() / tot)
        base_le.append(lesion[i][fundus[i]].mean()); base_oe.append(od[i][fundus[i]].mean())
        yx = np.unravel_index(cam.argmax(), cam.shape); hit.append(bool(lesion_near[i][yx]))
    le, oe, base_le = np.array(le), np.array(oe), np.array(base_le)
    print(f"{h}: lesion energy {le.mean():.3f} (random {base_le.mean():.3f}, ratio {le.mean() / base_le.mean():.1f}x, "
          f"higher than random in {(le > base_le).mean():.0%} of images) | optic disc energy {np.mean(oe):.3f} (random {np.mean(base_oe):.3f}) | "
          f"pointing game {np.mean(hit):.0%}")
