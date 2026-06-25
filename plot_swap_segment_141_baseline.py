"""
Segment Swap: mean over num_swaps — με clean baselines φιλτραρισμένες
ΜΟΝΟ στα αρχεία που χρησιμοποίησε το πείραμα swap_segment (το 141-subset).
Παλέτα: IForest=μπλε, LOF=πορτοκαλί, MP=πράσινο, AE=κόκκινο.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(ROOT, "results", "experiments", "swap_segment", "summary.csv")
CHECKPOINT = os.path.join(ROOT, "results", "experiments", "swap_segment", "checkpoint.csv")
BASELINE = os.path.join(ROOT, "results", "tables", "baseline_final_subset.csv")

MODELS = ["IForest", "LOF", "MP", "AE"]
COLORS = {"IForest": "#1f77b4", "LOF": "#ff7f0e", "MP": "#2ca02c", "AE": "#d62728"}
MARKERS = {"IForest": "o", "LOF": "s", "MP": "^", "AE": "D"}
METRICS = [
    ("mean_AUC_ROC", "AUC_ROC", "AUC-ROC"),
    ("mean_AUC_PR",  "AUC_PR",  "AUC-PR"),
    ("mean_Recall",  "Recall",  "Recall"),
]

# ── δεδομένα πειράματος (καμπύλες) ───────────────────────────────────────────
summary = pd.read_csv(SUMMARY)
summary = summary.groupby(["fraction", "model"], as_index=False)[
    ["mean_AUC_ROC", "mean_AUC_PR", "mean_Recall"]
].mean()

# ── baseline ΜΟΝΟ στα 141 αρχεία του πειράματος ──────────────────────────────
baseline = pd.read_csv(BASELINE)
baseline["model"] = baseline["model"].replace(
    {"Autoencoder": "AE", "Matrix Profile": "MP", "ME": "MP"}
)
ck = pd.read_csv(CHECKPOINT, low_memory=False)
exp_files = set(ck[ck["error"].isna()]["file"].unique())
bl = baseline[baseline["file"].isin(exp_files) & baseline["model"].isin(MODELS)]
bl_means = {m[1]: bl.groupby("model")[m[1]].mean().to_dict() for m in METRICS}
print("baseline αρχεία:", bl["file"].nunique(), "| Recall baseline:", bl_means["Recall"])

# ── plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linestyle": "-",
})
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.1), sharex=True)

for ax, (sum_col, bl_col, title) in zip(axes, METRICS):
    for model in MODELS:
        sub = summary[summary["model"] == model].sort_values("fraction")
        if sub.empty:
            continue
        ax.plot(sub["fraction"], sub[sum_col], color=COLORS[model],
                marker=MARKERS[model], linewidth=2.2, markersize=4.8,
                label=model, zorder=3)
    for model in MODELS:
        val = bl_means[bl_col].get(model)
        if pd.notna(val):
            ax.axhline(val, color=COLORS[model], linestyle=(0, (5, 3)),
                       linewidth=2.6, alpha=0.95, zorder=4)
    fr = sorted(summary["fraction"].dropna().unique())
    ax.set_xticks(fr)
    ax.set_xticklabels([f"{f:.0%}" for f in fr], fontsize=8)
    ax.set_xlabel("Fraction")
    ax.set_ylabel(title)
    ax.set_title(title)

fig.suptitle("Segment swap: mean over num_swaps", y=1.03, fontsize=12, fontweight="bold")
handles, labels = axes[2].get_legend_handles_labels()
handles.append(Line2D([0], [0], color="#222222", linestyle=(0, (5, 3)),
                      linewidth=2.6, alpha=0.95))
labels.append("clean baseline")
axes[2].legend(handles=handles, labels=labels, loc="lower left", frameon=True)
fig.tight_layout()

out = os.path.join(ROOT, "plot_swap_segment_141_baseline.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved:", out)
