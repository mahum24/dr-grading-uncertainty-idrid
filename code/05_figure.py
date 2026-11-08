"""One summary figure for the IDRiD study. Usage: python -I code/05_figure.py <results_dir> <out.png>
(a) QWK by capture condition, (b) change in mean uncertainty relative to clean images, (c) QWK on kept cases
as the most uncertain cases are referred. Mean over seeds, error bars = SD over seeds."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

res, out = Path(sys.argv[1]), sys.argv[2]
d = pd.concat([pd.read_csv(p) for p in sorted(res.glob("pred_*.csv"))], ignore_index=True)
heads, conds = ["ce", "ordinal", "evidential"], ["clean", "lowres", "blur", "dark", "jpeg"]
names = {"ce": "softmax (CE)", "ordinal": "ordinal", "evidential": "evidential"}
qwk = lambda g: cohen_kappa_score(g.y, g.pred, weights="quadratic")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
w = 0.26
for i, h in enumerate(heads):
    q = np.array([[qwk(d[(d["head"] == h) & (d.seed == s) & (d.condition == c)]) for c in conds] for s in sorted(d.seed.unique())])
    ax[0].bar(np.arange(len(conds)) + (i - 1) * w, q.mean(0), w, yerr=q.std(0), capsize=2, label=names[h])
    u = np.array([[d[(d["head"] == h) & (d.seed == s) & (d.condition == c)].unc.mean() for c in conds] for s in sorted(d.seed.unique())])
    rel = 100 * (u / u[:, [0]] - 1)
    ax[1].bar(np.arange(1, len(conds)) + (i - 1) * w, rel[:, 1:].mean(0), w, yerr=rel[:, 1:].std(0), capsize=2, label=names[h])
    fr = np.linspace(0, 0.5, 11); curve = []
    for s in sorted(d.seed.unique()):
        g = d[(d["head"] == h) & (d.seed == s) & (d.condition == "clean")].sort_values("unc")
        curve.append([qwk(g.iloc[: int(round(len(g) * (1 - f)))]) for f in fr])
    curve = np.array(curve)
    ax[2].plot(100 * fr, curve.mean(0), marker="o", ms=3, label=names[h])
    ax[2].fill_between(100 * fr, curve.mean(0) - curve.std(0), curve.mean(0) + curve.std(0), alpha=0.15)
ax[0].set_xticks(range(len(conds)), conds); ax[0].set_ylabel("QWK"); ax[0].set_title("(a) Grading agreement by capture condition")
ax[1].set_xticks(range(1, len(conds)), conds[1:]); ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_ylabel("change in mean uncertainty vs clean (%)"); ax[1].set_title("(b) Does uncertainty rise when quality falls?")
ax[2].set_xlabel("most uncertain cases referred (%)"); ax[2].set_ylabel("QWK on kept cases"); ax[2].set_title("(c) Referral on clean images")
for a in ax: a.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(out, dpi=150)
print("saved", out)
