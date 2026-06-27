"""
Recreates the 'Επίδραση num_bursts στο MNAR_extreme burst @ 10% missing' plot.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_CSV  = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_mnar_burst", "summary.csv")

MODELS = ["IForest", "LOF", "MP", "AE"]

COLORS  = {
    "IForest": "#1f77b4",
    "LOF":     "#ff7f0e",
    "MP":      "#2ca02c",
    "AE":      "#d62728",
}
MARKERS = {"IForest": "o", "LOF": "s", "MP": "^", "AE": "D"}

DISPLAY = {"IForest": "IForest", "LOF": "LOF", "MP": "MP", "AE": "AE"}

METRICS = [
    ("mean_AUC_ROC", "AUC-ROC", "AUC-ROC"),
    ("mean_AUC_PR",  "AUC-PR",  "AUC-PR"),
    ("mean_Recall",  "Recall",  "Recall"),
]

summary = pd.read_csv(SUMMARY_CSV)
summary = summary[(summary["mechanism"] == "mnar_extreme_burst") & (summary["fraction"] == 0.10)]

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
        sub = summary[summary["model"] == model].sort_values("num_bursts")
        if sub.empty: continue
        
        ax.plot(
            range(len(sub)),
            sub[sum_col],
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=2.2,
            markersize=5.8,
            label=DISPLAY[model],
            zorder=3,
        )

    num_bursts = sorted(summary["num_bursts"].dropna().unique())
    ax.set_xticks(range(len(num_bursts)))
    ax.set_xticklabels([f"{int(b)}" for b in num_bursts], fontsize=9)

    ax.set_xlabel("num_bursts")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

fig.suptitle("Επίδραση num_bursts στο MNAR_extreme burst @ 10% missing", y=1.03, fontsize=12, fontweight="bold")

axes[2].legend(loc="lower right", frameon=True)

fig.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_mnar_extreme_num_bursts_clean.png")
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
