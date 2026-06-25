"""
Recreates the 'Επίδραση του extra white-noise στην πτώση AUC-PR / Recall' plot
using ONLY the 141 experiment files for baselines.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV  = os.path.join(PROJECT_ROOT, "results", "experiments", "compound_corruptions", "summary.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "experiments", "compound_corruptions", "checkpoint.csv")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")

MODELS = ["IForest", "LOF", "MP", "AE"]

COLORS  = {
    "IForest": "#1f77b4",
    "LOF":     "#ff7f0e",
    "MP":      "#2ca02c",
    "AE":      "#d62728",
}
MARKERS = {"IForest": "o", "LOF": "s", "MP": "^", "AE": "D"}

# ── Load data ────────────────────────────────────────────────────────────────
summary = pd.read_csv(SUMMARY_CSV, low_memory=False)
summary["model"] = summary["model"].replace({"Autoencoder": "AE", "Matrix Profile": "MP"})
summary = summary[summary["model"].isin(MODELS)].copy()

numeric_cols = ["mean_AUC_ROC", "mean_AUC_PR", "mean_Recall"]
summary = summary.groupby(["condition", "combination_name", "condition_type", "model"], as_index=False)[numeric_cols].mean()

# ── Baselines from the 141 experiment files ──────────────────────────────────
baseline = pd.read_csv(BASELINE_CSV)
baseline["model"] = baseline["model"].replace({"Autoencoder": "AE", "Matrix Profile": "MP", "ME": "MP"})

# Use the clean condition from compound_corruptions itself as baselines
# (this is what the original script does)
clean = summary[summary["condition"] == "clean"]
if not clean.empty:
    base_aucpr = clean.groupby("model")["mean_AUC_PR"].mean().to_dict()
    base_recall = clean.groupby("model")["mean_Recall"].mean().to_dict()
else:
    # Fallback: compute from the 141 baseline files
    checkpoint = pd.read_csv(CHECKPOINT_CSV, low_memory=False)
    experiment_files = set(checkpoint[checkpoint["error"].isna()]["file"].unique())
    baseline_filt = baseline[baseline["file"].isin(experiment_files) & baseline["model"].isin(MODELS)]
    base_aucpr = baseline_filt.groupby("model")["AUC_PR"].mean().to_dict()
    base_recall = baseline_filt.groupby("model")["Recall"].mean().to_dict()

# ── Filter to compound noise conditions ──────────────────────────────────────
df = summary[summary["condition_type"].eq("compound") & summary["condition"].str.contains("noise_", na=False)].copy()
df["noise_severity"] = df["condition"].str.extract(r"noise_(low|med|high|extreme)")[0]
severity_order = ["low", "med", "high", "extreme"]

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

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=False)

for ax, col, metric, base in [
    (axes[0], "mean_AUC_PR", "AUC-PR", base_aucpr),
    (axes[1], "mean_Recall", "Recall", base_recall),
]:
    for model in MODELS:
        vals = []
        for sev in severity_order:
            row = df[df["model"].eq(model) & df["noise_severity"].eq(sev)]
            val = row[col].mean() if not row.empty else np.nan
            vals.append((base.get(model, np.nan) - val) / base.get(model, np.nan) * 100.0)
        ax.plot(
            severity_order,
            vals,
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=2.3,
            markersize=5.2,
            label=model,
        )
    ax.set_title(f"Επίδραση του extra white-noise στην πτώση {metric}")
    ax.set_xlabel("Ένταση θορύβου")
    ax.set_ylabel(f"{metric} drop (%)")

axes[-1].legend(frameon=True)

fig.tight_layout()
out_path = os.path.join(PROJECT_ROOT, "plot_compound_noise_profile_141.png")
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
