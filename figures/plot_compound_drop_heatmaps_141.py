# -*- coding: utf-8 -*-
"""Compound corruptions: Πτώση AUC-PR (%) και Recall (%) ανά συνδυασμό × μοντέλο.
Πιστή αναπαραγωγή της αρχικής εικόνας: 2 panels, ξεχωριστό colorbar ανά panel,
cmap YlOrRd, τιμές με 1 δεκαδικό. Υψηλή ανάλυση."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["IForest", "LOF", "MP", "AE"]
ALIAS = {"Autoencoder": "AE", "Matrix Profile": "MP", "ME": "MP"}

s = pd.read_csv(os.path.join(ROOT, "results", "experiments", "compound_corruptions", "summary.csv"))
s["model"] = s["model"].replace(ALIAS)
s = s[s["model"].isin(MODELS)]
s = s.groupby(["condition", "combination_name", "model"], as_index=False)[["mean_AUC_PR", "mean_Recall"]].mean()

# baseline = clean condition per model
clean = s[s["condition"] == "clean"]
def base(col):
    return clean.groupby("model")[col].mean().to_dict()

# σειρά γραμμών όπως στην εικόνα
COMBOS = [
    ("single", "single"),
    ("missing_freeze", "missing + freeze"),
    ("noise_missing", "noise + missing"),
    ("noise_spikes", "noise + spikes"),
    ("spikes_missing", "spikes + missing"),
    ("noise_ge_missing", "noise + G-E missing"),
    ("noise_spikes_missing", "noise + spikes + missing"),
]
METRICS = [("mean_AUC_PR", "Πτώση AUC-PR (%)"), ("mean_Recall", "Πτώση Recall (%)")]

def matrix(col):
    b = base(col)
    agg = s.groupby(["combination_name", "model"], as_index=False)[col].mean()
    M = np.full((len(COMBOS), len(MODELS)), np.nan)
    for i, (key, _) in enumerate(COMBOS):
        for j, m in enumerate(MODELS):
            r = agg[(agg["combination_name"] == key) & (agg["model"] == m)]
            if not r.empty and m in b:
                M[i, j] = (b[m] - r[col].mean()) / b[m] * 100.0
    return M

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
for ax, (col, title) in zip(axes, METRICS):
    M = matrix(col)
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto", vmin=0, vmax=np.nanmax(M))
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS)
    ax.set_yticks(range(len(COMBOS))); ax.set_yticklabels([lbl for _, lbl in COMBOS])
    ax.set_title(title, fontsize=12)
    for i in range(len(COMBOS)):
        for j in range(len(MODELS)):
            v = M[i, j]
            if not np.isnan(v):
                nrm = v / (np.nanmax(M) + 1e-9)
                color = "white" if nrm > 0.6 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_compound_drop_heatmaps_141.png")
fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
# verify
print("AUC-PR drop (rows=combos, cols=IForest/LOF/MP/AE):")
print(np.round(matrix("mean_AUC_PR"), 1))
print("Saved:", out)
