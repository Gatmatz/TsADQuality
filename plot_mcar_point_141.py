# -*- coding: utf-8 -*-
"""MCAR Point Missing - επίδραση ανά μετρική.
Καμπύλες: missing_mnar (mechanism=mcar). Baseline: φιλτραρισμένη στα 141 αρχεία του πειράματος.
Παλέτα: IForest=μπλε, LOF=πορτοκαλί, MP=πράσινο, AE=κόκκινο."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.abspath(".")
SUMMARY = os.path.join(ROOT, "results", "experiments", "missing_mnar", "summary.csv")
CHECKPOINT = os.path.join(ROOT, "results", "experiments", "missing_mnar", "checkpoint.csv")
BASELINE = os.path.join(ROOT, "results", "tables", "baseline_final_subset.csv")

MODELS = ["IForest", "LOF", "MP", "AE"]
COLORS = {"IForest": "#1f77b4", "LOF": "#ff7f0e", "MP": "#2ca02c", "AE": "#d62728"}
MARKERS = {"IForest": "o", "LOF": "s", "MP": "^", "AE": "D"}
METRICS = [("mean_AUC_ROC", "AUC_ROC", "AUC-ROC"),
           ("mean_AUC_PR", "AUC_PR", "AUC-PR"),
           ("mean_Recall", "Recall", "Recall")]

s = pd.read_csv(SUMMARY)
s = s[s["mechanism"] == "mcar"]
s["model"] = s["model"].replace({"Autoencoder": "AE", "Matrix Profile": "MP", "ME": "MP"})
summ = s.groupby(["fraction", "model"], as_index=False)[["mean_AUC_ROC", "mean_AUC_PR", "mean_Recall"]].mean()

bl = pd.read_csv(BASELINE)
bl["model"] = bl["model"].replace({"Autoencoder": "AE", "Matrix Profile": "MP", "ME": "MP"})
ck = pd.read_csv(CHECKPOINT, low_memory=False)
exp = set(ck[ck["error"].isna()]["file"].unique())
blf = bl[bl["file"].isin(exp) & bl["model"].isin(MODELS)]
base = {m[1]: blf.groupby("model")[m[1]].mean().to_dict() for m in METRICS}
print("baseline files:", blf["file"].nunique())

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linestyle": "-"})
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.1), sharex=True)
for ax, (sc, bc, title) in zip(axes, METRICS):
    for m in MODELS:
        sub = summ[summ["model"] == m].sort_values("fraction")
        if sub.empty: continue
        ax.plot(sub["fraction"], sub[sc], color=COLORS[m], marker=MARKERS[m],
                linewidth=2.2, markersize=4.8, label=m, zorder=3)
    for m in MODELS:
        v = base[bc].get(m)
        if v is not None and pd.notna(v):
            ax.axhline(v, color=COLORS[m], linestyle=(0, (5, 3)), linewidth=2.6, alpha=0.95, zorder=4)
    fr = sorted(summ["fraction"].dropna().unique())
    ax.set_xticks(fr); ax.set_xticklabels([f"{f:.0%}" for f in fr], fontsize=8)
    ax.set_xlabel("Fraction missing"); ax.set_ylabel(title); ax.set_title(title)
fig.suptitle("MCAR Point Missing — επίδραση ανά μετρική", y=1.03, fontsize=12, fontweight="bold")
h, l = axes[2].get_legend_handles_labels()
h.append(Line2D([0], [0], color="#222222", linestyle=(0, (5, 3)), linewidth=2.6, alpha=0.95)); l.append("clean baseline")
axes[2].legend(handles=h, labels=l, loc="lower left", frameon=True)
fig.tight_layout()
out = os.path.join(ROOT, "plot_mcar_point_141.png")
fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("Saved:", out)
