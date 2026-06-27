# -*- coding: utf-8 -*-
"""Gilbert-Elliott: τριπλή μετρική ανά (α, β) και μοντέλο.
Πιστή αναπαραγωγή του αρχικού layout: 4 μοντέλα (σειρές) × 3 μετρικές (στήλες),
κάθε panel 3×3 (α γραμμές × β στήλες), με colorbar ανά μετρική (σωστή κλίμακα ανά μετρική).
Υψηλή ανάλυση για καθαρότητα."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = pd.read_csv(os.path.join(ROOT, "results", "experiments", "gilbert_elliott_true_impact", "summary.csv"))

MODELS = ["IForest", "LOF", "MP", "AE"]
METRICS = [("mean_AUC_ROC", "AUC-ROC"), ("mean_AUC_PR", "AUC-PR"), ("mean_Recall", "Recall")]
alphas = [0.01, 0.05, 0.1]
betas = [0.1, 0.25, 0.5]

def matrix(model, col):
    M = np.full((len(alphas), len(betas)), np.nan)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            r = s[(s.model == model) & (np.isclose(s.alpha, a)) & (np.isclose(s.beta, b))]
            if not r.empty:
                M[i, j] = r[col].mean()
    return M

# per-metric color scale (ίδια όρια με την αρχική εικόνα)
vrange = {"mean_AUC_ROC": (0.30, 0.70), "mean_AUC_PR": (0.0, 0.25), "mean_Recall": (0.0, 0.25)}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})
fig, axes = plt.subplots(len(MODELS), len(METRICS), figsize=(12.5, 14.2))

for r, model in enumerate(MODELS):
    for c, (col, mlabel) in enumerate(METRICS):
        ax = axes[r, c]
        M = matrix(model, col)
        vmin, vmax = vrange[col]
        im = ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(betas))); ax.set_xticklabels([f"β={b}" for b in betas])
        ax.set_yticks(range(len(alphas))); ax.set_yticklabels([f"α={a}" for a in alphas])
        for i in range(len(alphas)):
            for j in range(len(betas)):
                v = M[i, j]
                if not np.isnan(v):
                    nrm = (v - vmin) / (vmax - vmin + 1e-9)
                    color = "white" if (nrm < 0.28 or nrm > 0.82) else "black"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color=color)
        if r == 0:
            ax.set_title(mlabel, fontsize=12, fontweight="bold", pad=10)
        if c == 0:
            ax.set_ylabel(model, fontsize=13, fontweight="bold", labelpad=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Gilbert-Elliott: τριπλή μετρική ανά (α, β) και μοντέλο", y=0.995, fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_gilbert_elliott_grid_141.png")
fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("Saved:", out)
