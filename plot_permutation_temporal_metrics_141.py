"""
Recreates the 'Permutation: temporal-tolerant metrics retain signal' plot
with the correct color palette and NO dashed baseline lines.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV  = os.path.join(PROJECT_ROOT, "results", "experiments", "swap_permutation", "summary.csv")

MODELS = ["IForest", "LOF", "MP", "AE"]

COLORS  = {
    "IForest": "#1f77b4",
    "LOF":     "#ff7f0e",
    "MP":      "#2ca02c",
    "AE":      "#d62728",
}
MARKERS = {"IForest": "o", "LOF": "s", "MP": "^", "AE": "D"}

DISPLAY = {
    "IForest": "IForest",
    "LOF":     "LOF",
    "MP":      "Matrix Profile",
    "AE":      "Autoencoder",
}

METRICS = [
    ("mean_R_AUC_ROC",           "R-AUC-ROC",           "Mean R-AUC-ROC"),
    ("mean_VUS_ROC",             "VUS-ROC",              "Mean VUS-ROC"),
    ("mean_Affiliation_Recall",  "Affiliation Recall",   "Mean Affiliation Recall"),
]

# ── Load data ────────────────────────────────────────────────────────────────
summary = pd.read_csv(SUMMARY_CSV)

# ── Build plot ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      130,
    "savefig.dpi":     300,
    "font.size":       9,
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "legend.fontsize": 8,
    "axes.grid":       True,
    "grid.alpha":      0.22,
    "grid.linestyle":  "-",
    "axes.spines.top": True,
    "axes.spines.right": True,
})

fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.1), sharex=True, sharey=False)

for ax, (sum_col, title, ylabel) in zip(axes, METRICS):
    for model in MODELS:
        sub = summary[summary["model"] == model].sort_values("n_segments")
        if sub.empty:
            continue
        
        ax.plot(
            range(len(sub)),
            sub[sum_col].values,
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=2.2,
            markersize=5.8,
            label=DISPLAY[model],
            zorder=3,
        )

    n_segments = sorted(summary["n_segments"].dropna().unique())
    ax.set_xticks(range(len(n_segments)))
    ax.set_xticklabels([str(int(n)) for n in n_segments], fontsize=9)

    ax.set_xlabel("N segments")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

fig.suptitle("Permutation: temporal-tolerant metrics retain signal", y=1.03, fontsize=12, fontweight="bold")

axes[0].legend(loc="lower left", frameon=True)

fig.tight_layout()
out_path = os.path.join(PROJECT_ROOT, "plot_permutation_temporal_metrics_141.png")
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
