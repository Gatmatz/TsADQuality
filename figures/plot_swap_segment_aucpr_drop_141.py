# -*- coding: utf-8 -*-
"""Segment Swap: AUC-PR drop from clean baseline (%).
4 μοντέλα, num_swaps × fraction, κοινό colorbar (YlOrRd). Baseline στα 141 αρχεία.
Πιστή αναπαραγωγή σε υψηλή ανάλυση."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [("IForest", "IForest"), ("LOF", "LOF"), ("MP", "Matrix Profile"), ("AE", "Autoencoder")]
ALIAS = {"Autoencoder": "AE", "Matrix Profile": "MP", "ME": "MP"}
nsw = [1, 3, 5, 10, 20]
fr = [0.01, 0.05, 0.1, 0.2, 0.3]

s = pd.read_csv(os.path.join(ROOT, "results", "experiments", "swap_segment", "summary.csv"))
s["model"] = s["model"].replace(ALIAS)
agg = s.groupby(["model", "num_swaps", "fraction"], as_index=False)["mean_AUC_PR"].mean()

# baseline AUC-PR στα 141 αρχεία
bl = pd.read_csv(os.path.join(ROOT, "results", "tables", "baseline_final_subset.csv"))
bl["model"] = bl["model"].replace(ALIAS)
ck = pd.read_csv(os.path.join(ROOT, "results", "experiments", "swap_segment", "checkpoint.csv"), low_memory=False)
exp = set(ck[ck["error"].isna()]["file"].unique())
base = bl[bl["file"].isin(exp)].groupby("model")["AUC_PR"].mean().to_dict()

def matrix(m):
    M = np.full((len(nsw), len(fr)), np.nan)
    for i, ns in enumerate(nsw):
        for j, f in enumerate(fr):
            r = agg[(agg["model"] == m) & (agg["num_swaps"] == ns) & (np.isclose(agg["fraction"], f))]
            if not r.empty:
                M[i, j] = (base[m] - r["mean_AUC_PR"].mean()) / base[m] * 100.0
    return M

mats = {m: matrix(m) for m, _ in MODELS}
vmax = max(np.nanmax(M) for M in mats.values())

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ims = []
for ax, (m, title) in zip(axes.ravel(), MODELS):
    M = mats[m]
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ims.append(im)
    ax.set_xticks(range(len(fr))); ax.set_xticklabels([f"{f:.0%}" for f in fr])
    ax.set_yticks(range(len(nsw))); ax.set_yticklabels([str(n) for n in nsw])
    ax.set_xlabel("fraction"); ax.set_ylabel("num_swaps")
    ax.set_title(title, fontsize=12, fontweight="bold")
    for i in range(len(nsw)):
        for j in range(len(fr)):
            v = M[i, j]
            if not np.isnan(v):
                color = "white" if v / vmax > 0.6 else "black"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9, color=color, fontweight="bold")
fig.suptitle("Segment Swap: AUC-PR drop from clean baseline (%)", fontsize=13, fontweight="bold")
fig.subplots_adjust(left=0.07, right=0.88, bottom=0.08, top=0.91, wspace=0.18, hspace=0.32)
cax = fig.add_axes([0.90, 0.20, 0.018, 0.58])
cb = fig.colorbar(ims[0], cax=cax); cb.set_label("AUC-PR drop (%)")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_swap_segment_aucpr_drop_141.png")
fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("baseline files:", bl[bl['file'].isin(exp)]['file'].nunique())
print("IForest matrix:"); print(np.round(mats["IForest"]).astype(int))
print("Saved:", out)
