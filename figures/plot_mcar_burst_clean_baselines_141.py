"""
Recreates the 'MCAR Burst Missing — επίδραση ανά μετρική (μέσος όρος ανά num_bursts)' plot
using ONLY the 141 files present in the missing_true_impact experiment.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_CSV  = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_true_impact", "summary.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_true_impact", "checkpoint.csv")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")

# ── Model config ────────────────────────────────────────────────────────────
MODELS = ["IForest", "LOF", "MP", "AE"]

COLORS  = {
    "IForest": "#1f77b4",   # blue
    "LOF":     "#ff7f0e",   # orange
    "MP":      "#2ca02c",   # green
    "AE":      "#d62728",   # red
}
MARKERS = {"IForest": "o", "LOF": "s", "MP": "^", "AE": "D"}

DISPLAY = {
    "IForest": "IForest",
    "LOF":     "LOF",
    "MP":      "MP",
    "AE":      "AE",
}

# ── Metrics to plot ──────────────────────────────────────────────────────────
METRICS = [
    ("mean_AUC_ROC", "AUC_ROC", "AUC-ROC", "AUC-ROC"),
    ("mean_AUC_PR",  "AUC_PR",  "AUC-PR",  "AUC-PR"),
    ("mean_Recall",  "Recall",  "Recall",  "Recall"),
]

# ── Load data ────────────────────────────────────────────────────────────────
summary = pd.read_csv(SUMMARY_CSV)
summary = summary[summary["missing_type"] == "burst"]
# Aggregate over num_bursts by grouping on fraction and model
summary = summary.groupby(["fraction", "model"], as_index=False)[
    ["mean_AUC_ROC", "mean_AUC_PR", "mean_Recall"]
].mean()

baseline = pd.read_csv(BASELINE_CSV)

# Harmonise Autoencoder name
baseline["model"] = baseline["model"].replace({"Autoencoder": "AE", "Matrix Profile": "MP", "ME": "MP"})

# Filter to ONLY the 141 files in the experiment
checkpoint = pd.read_csv(CHECKPOINT_CSV)
experiment_files = set(checkpoint[checkpoint["error"].isna()]["file"].unique())

baseline_141 = baseline[baseline["file"].isin(experiment_files) & baseline["model"].isin(MODELS)]

# Compute baseline means
bl_means = {
    metric[1]: baseline_141.groupby("model")[metric[1]].mean().to_dict()
    for metric in METRICS
}

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

for ax, (sum_col, bl_col, title, ylabel) in zip(axes, METRICS):
    
    for model in MODELS:
        sub = summary[summary["model"] == model].sort_values("fraction")
        if sub.empty: continue
        
        ax.plot(
            sub["fraction"],
            sub[sum_col],
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=2.2,
            markersize=4.8,
            label=DISPLAY[model],
            zorder=3,
        )
        
    for model in MODELS:
        val = bl_means[bl_col].get(model)
        if pd.notna(val):
            ax.axhline(
                val,
                color=COLORS[model],
                linestyle=(0, (5, 3)),
                linewidth=2.6,
                alpha=0.95,
                zorder=4,
            )

    fractions = sorted(summary["fraction"].dropna().unique())
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{f:.0%}" for f in fractions], fontsize=8)

    ax.set_xlabel("Fraction missing")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

fig.suptitle("MCAR Burst Missing — επίδραση ανά μετρική (μέσος όρος ανά num_bursts)", y=1.03, fontsize=12, fontweight="bold")

handles, labels = axes[2].get_legend_handles_labels()
baseline_line = Line2D([0], [0], color="#222222", linestyle=(0, (5, 3)), linewidth=2.6, alpha=0.95, label="clean baseline")
handles.append(baseline_line)
labels.append("clean baseline")
axes[2].legend(handles=handles, labels=labels, loc="lower left", frameon=True)

fig.tight_layout()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_mcar_burst_clean_baselines_141.png")
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
