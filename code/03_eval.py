"""Summarise IDRiD test results across heads and seeds. Usage: python -I code/03_eval.py <results_dir>  (e.g. results/predictions)
Clean: QWK, accuracy, referable DR (grade >= 2) AUC / sensitivity / specificity, top-label ECE.
Referral: QWK and referable sensitivity on the cases kept after deferring the most uncertain 10/20/30%.
Low-quality capture: QWK and mean uncertainty per simulated condition.
Paired test: per-image absolute grade error (mean over seeds), Wilcoxon signed-rank between heads."""
import sys
from pathlib import Path

import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score

d = pd.concat([pd.read_csv(p) for p in sorted(Path(sys.argv[1]).glob("pred_*.csv"))], ignore_index=True)
P = d[[f"p{k}" for k in range(5)]].to_numpy()
d["p_ref"] = P[:, 2:].sum(1); d["conf"] = P.max(1); d["ref_true"] = (d.y >= 2).astype(int)


def ece(g, bins=10):
    conf, acc = g.conf.to_numpy(), (g.pred == g.y).to_numpy()
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(abs(acc[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def metrics(g):
    pr = (g.p_ref >= 0.5).astype(int)
    return pd.Series(dict(qwk=cohen_kappa_score(g.y, g.pred, weights="quadratic"), acc=(g.pred == g.y).mean(),
                          ref_auc=roc_auc_score(g.ref_true, g.p_ref), sens=pr[g.ref_true == 1].mean(),
                          spec=1 - pr[g.ref_true == 0].mean(), ece=ece(g), mean_unc=g.unc.mean()))


def fmt(t): return t.groupby(level=0).agg(["mean", "std"]).round(3)

clean = d[d.condition == "clean"]
per = clean.groupby(["head", "seed"]).apply(metrics, include_groups=False)
print("== clean test set (103 images), mean and sd over seeds"); print(per.groupby("head").agg(["mean", "std"]).round(3).T.to_string())

rows = []
for (h, s), g in clean.groupby(["head", "seed"]):
    g = g.sort_values("unc")
    for frac in [0, 0.1, 0.2, 0.3]:
        keep = g.iloc[: int(round(len(g) * (1 - frac)))]
        rows.append(dict(head=h, seed=s, referred=frac, qwk=cohen_kappa_score(keep.y, keep.pred, weights="quadratic"),
                         acc=(keep.pred == keep.y).mean(), sens=((keep.p_ref >= 0.5) & (keep.ref_true == 1)).sum() / max(1, keep.ref_true.sum())))
r = pd.DataFrame(rows)
print("\n== referral: metrics on kept cases after deferring the most uncertain fraction (mean over seeds)")
print(r.groupby(["head", "referred"])[["qwk", "acc", "sens"]].mean().round(3).unstack("referred").to_string())

rob = d.groupby(["head", "condition", "seed"]).apply(lambda g: pd.Series(dict(
    qwk=cohen_kappa_score(g.y, g.pred, weights="quadratic"), unc=g.unc.mean(),
    unc_err_auc=roc_auc_score((g.pred != g.y).astype(int), g.unc) if 0 < (g.pred != g.y).mean() < 1 else np.nan)), include_groups=False)
rob = rob.groupby(["head", "condition"]).mean().round(3)
print("\n== simulated low-quality capture: QWK, mean uncertainty, AUROC of uncertainty for spotting errors (mean over seeds)")
print(rob.unstack("head").to_string())

err = clean.assign(ae=(clean.pred - clean.y).abs()).groupby(["head", "id"]).ae.mean().unstack("head")
print("\n== paired Wilcoxon on per-image |grade error| (mean over seeds)")
heads = list(err.columns)
for i in range(len(heads)):
    for j in range(i + 1, len(heads)):
        a, b = err[heads[i]], err[heads[j]]
        p = stats.wilcoxon(a, b).pvalue if (a != b).any() else 1.0
        print(f"{heads[i]} vs {heads[j]}: mean |err| {a.mean():.3f} vs {b.mean():.3f}, p = {p:.3g}")
