"""Train a ResNet-18 DR grader (0-4) on IDRiD with one of three heads and evaluate on the official test set,
clean and under simulated low-quality capture.
Usage: python -I code/02_train.py <grading.npz> <out_dir> --head ce|ordinal|evidential --seed 0 [--epochs 15]
Heads: ce = softmax cross-entropy; ordinal = K-1 cumulative binary thresholds (P(grade > k));
evidential = Dirichlet evidence with the Bayes-risk loss of Sensoy et al. (2018)."""
import argparse, io, json, math, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from PIL import Image, ImageFilter
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--head", required=True, choices=["ce", "ordinal", "evidential"])
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--save_model", type=int, default=0)
ap.add_argument("--limit", type=int, default=0, help="debug: use only this many training images")
A = ap.parse_args()
torch.set_num_threads(4); torch.manual_seed(A.seed); np.random.seed(A.seed)
out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
K = 5
D = np.load(A.data)
Xtr_all, ytr_all, Xte, yte, id_te = D["X_train"], D["y_train"], D["X_test"], D["y_test"], D["id_test"]
tr_idx, va_idx = train_test_split(np.arange(len(ytr_all)), test_size=0.15, stratify=ytr_all, random_state=A.seed)
if A.limit: tr_idx = tr_idx[:A.limit]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(batch_u8):
    x = torch.from_numpy(np.ascontiguousarray(batch_u8)).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD


def augment(img):
    if np.random.rand() < 0.5: img = img[:, ::-1]
    if np.random.rand() < 0.5: img = img[::-1, :]
    img = np.rot90(img, np.random.randint(4))
    x = img.astype(np.float32) * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20)  # brightness / contrast jitter
    return np.clip(x, 0, 255).astype(np.uint8)


# ---- simulated low-quality capture (smartphone / low-cost camera style); applied only at test time ----
def degrade(img, kind):
    if kind == "clean": return img
    pil = Image.fromarray(img); S = img.shape[0]
    if kind == "lowres": return np.asarray(pil.resize((S // 4, S // 4), Image.BILINEAR).resize((S, S), Image.BILINEAR))
    if kind == "blur": return np.asarray(pil.filter(ImageFilter.GaussianBlur(3)))
    if kind == "dark":
        yy, xx = np.mgrid[:S, :S]; r = np.sqrt((yy - S / 2) ** 2 + (xx - S / 2) ** 2) / (S / 2)
        v = 1 - 0.5 * np.clip(r, 0, 1) ** 2
        x = 255 * (img / 255.0) ** 1.8 * v[..., None]
        return np.clip(x, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        buf = io.BytesIO(); pil.save(buf, "JPEG", quality=15); return np.asarray(Image.open(buf).convert("RGB"))
    raise ValueError(kind)


CONDITIONS = ["clean", "lowres", "blur", "dark", "jpeg"]

# ---- model ----
net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
n_out = K - 1 if A.head == "ordinal" else K
net.fc = nn.Linear(net.fc.in_features, n_out)


def class_probs(logits):
    """Return (class probabilities, uncertainty in [0,1]) for every head."""
    if A.head == "ce":
        p = logits.softmax(-1); u = 1 - p.max(-1).values
    elif A.head == "ordinal":
        cum = torch.sigmoid(logits)                                   # P(grade > k), k = 0..3
        cum = torch.cummin(cum, dim=-1).values                        # enforce monotone thresholds
        ext = torch.cat([torch.ones_like(cum[:, :1]), cum, torch.zeros_like(cum[:, :1])], -1)
        p = (ext[:, :-1] - ext[:, 1:]).clamp_min(0); p = p / p.sum(-1, keepdim=True)
        u = 1 - p.max(-1).values
    else:
        alpha = F.softplus(logits) + 1; S = alpha.sum(-1, keepdim=True)
        p = alpha / S; u = (K / S).squeeze(-1)
    return p, u


def loss_fn(logits, y, epoch):
    if A.head == "ce":
        return F.cross_entropy(logits, y)
    if A.head == "ordinal":
        t = (y.unsqueeze(1) > torch.arange(K - 1).unsqueeze(0)).float()
        return F.binary_cross_entropy_with_logits(logits, t)
    alpha = F.softplus(logits) + 1; S = alpha.sum(-1, keepdim=True); yo = F.one_hot(y, K).float()
    p = alpha / S
    risk = ((yo - p) ** 2 + p * (1 - p) / (S + 1)).sum(-1)              # Bayes risk of squared error
    at = yo + (1 - yo) * alpha                                         # remove evidence of the true class
    St = at.sum(-1, keepdim=True)
    kl = (torch.lgamma(St.squeeze(-1)) - math.lgamma(K) - torch.lgamma(at).sum(-1)
          + ((at - 1) * (torch.digamma(at) - torch.digamma(St))).sum(-1))
    return (risk + min(1.0, epoch / 10) * kl).mean()


def predict(X, kind="clean", bs=32):
    net.eval(); P, U = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = np.stack([degrade(x, kind) for x in X[i:i + bs]])
            p, u = class_probs(net(to_tensor(xb))); P.append(p); U.append(u)
    return torch.cat(P).numpy(), torch.cat(U).numpy()


params = [{"params": [p for n, p in net.named_parameters() if not n.startswith("fc.")], "lr": 2e-4},
          {"params": net.fc.parameters(), "lr": 1e-3}]
opt = torch.optim.AdamW(params, weight_decay=1e-4)
steps = A.epochs * math.ceil(len(tr_idx) / A.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-4, 1e-3], total_steps=steps, pct_start=0.15)
best, best_state, log, t0 = -1, None, [], time.time()
for ep in range(A.epochs):
    net.train(); perm = np.random.permutation(tr_idx); tot = 0
    for i in range(0, len(perm), A.bs):
        b = perm[i:i + A.bs]
        xb = to_tensor(np.stack([augment(Xtr_all[j]) for j in b])); yb = torch.from_numpy(ytr_all[b]).long()
        loss = loss_fn(net(xb), yb, ep)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss) * len(b)
    pv, _ = predict(Xtr_all[va_idx]); qwk = cohen_kappa_score(ytr_all[va_idx], pv.argmax(1), weights="quadratic")
    log.append(dict(epoch=ep, loss=tot / len(perm), val_qwk=qwk, sec=time.time() - t0))
    print(f"[{A.head} s{A.seed}] epoch {ep} loss {tot / len(perm):.4f} val_qwk {qwk:.3f} ({time.time() - t0:.0f}s)", flush=True)
    if qwk > best: best, best_state = qwk, {k: v.clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state)

rows = []
for kind in CONDITIONS:
    p, u = predict(Xte, kind)
    for i in range(len(yte)):
        rows.append(dict(head=A.head, seed=A.seed, condition=kind, id=str(id_te[i]), y=int(yte[i]), pred=int(p[i].argmax()),
                         unc=float(u[i]), **{f"p{k}": float(p[i, k]) for k in range(K)}))
    print(kind, "QWK", round(cohen_kappa_score(yte, p.argmax(1), weights="quadratic"), 3), "acc", round(float((p.argmax(1) == yte).mean()), 3))
import pandas as pd
pd.DataFrame(rows).to_csv(out / f"pred_{A.head}_s{A.seed}.csv", index=False)
json.dump(dict(args=vars(A), best_val_qwk=best, log=log), open(out / f"log_{A.head}_s{A.seed}.json", "w"), indent=1)
if A.save_model: torch.save(best_state, out / f"model_{A.head}_s{A.seed}.pt")
