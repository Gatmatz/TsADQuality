"""
Recreates the 'AUC-ROC vs. SNR with clean baselines' plot
with more prominent dashed baseline lines.

Changes vs original:
- Dashed baseline lines: alpha 0.3→0.85, linewidth 1→2.5
- Colors match image: IForest=#1f77b4, LOF=#2ca02c, MP=#d62728, AE=#9467bd
- Full model names in legend: IForest, LOF, Matrix Profile, Autoencoder
- Baselines computed on the exact 141 files used in the SNR experiment
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV  = os.path.join(PROJECT_ROOT, "results", "experiments",
                             "white_noise_snr", "summary.csv")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables",
                             "baseline_final_subset.csv")

# ── Model config ────────────────────────────────────────────────────────────
MODELS = ["IForest", "LOF", "MP", "AE"]

COLORS  = {
    "IForest": "#1f77b4",   # μπλε
    "LOF":     "#ff7f0e",   # πορτοκαλί
    "MP":      "#2ca02c",   # πράσινο
    "AE":      "#d62728",   # κόκκινο
}
MARKERS = {"IForest": "o", "LOF": "s", "MP": "^", "AE": "D"}

# Display names for the legend
DISPLAY = {
    "IForest": "IForest",
    "LOF":     "LOF",
    "MP":      "Matrix Profile",
    "AE":      "Autoencoder",
}

# ── Load data ────────────────────────────────────────────────────────────────
summary  = pd.read_csv(SUMMARY_CSV)
baseline = pd.read_csv(BASELINE_CSV)

# Baseline file uses "Autoencoder" — harmonise to short codes
alias = {"Autoencoder": "AE"}
baseline["model"] = baseline["model"].replace(alias)

# Use only the 141 files that appear in the SNR experiment
checkpoint = pd.read_csv(os.path.join(
    PROJECT_ROOT, "results", "experiments", "white_noise_snr", "checkpoint.csv"
))
experiment_files = set(checkpoint[checkpoint["error"].isna()]["file"].unique())

bl_mean = (
    baseline[
        baseline["model"].isin(MODELS)
        & baseline["file"].isin(experiment_files)
    ]
    .groupby("model")["AUC_ROC"]
    .mean()
    .to_dict()
)

# ── Build plot ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        130,
    "savefig.dpi":       300,
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linestyle":    "-",
    "axes.spines.top":   True,
    "axes.spines.right": True,
})

fig, ax = plt.subplots(figsize=(8.2, 5.0))

# ── Plot model lines ─────────────────────────────────────────────────────────
for model in MODELS:
    sub = (
        summary[summary["model"] == model]
        .sort_values("snr_db")
    )
    ax.plot(
        sub["snr_db"],
        sub["mean_AUC_ROC"],
        color=COLORS[model],
        marker=MARKERS[model],
        linewidth=2.4,
        markersize=5.5,
        label=DISPLAY[model],
        zorder=3,
    )

# ── Plot clean baselines (dashed) — NOW MUCH MORE VISIBLE ────────────────────
for model in MODELS:
    val = bl_mean.get(model)
    if val is not None:
        ax.axhline(
            val,
            color=COLORS[model],
            linestyle="--",          # clean dashes
            linewidth=2.5,           # was 1 → now 2.5
            alpha=0.85,              # was 0.3 → now 0.85
            zorder=4,
        )

# ── Axes / labels ────────────────────────────────────────────────────────────
ax.invert_xaxis()
ax.set_xlabel("SNR (dB) - higher = cleaner")
ax.set_ylabel("Mean AUC-ROC")
ax.set_title("AUC-ROC vs. SNR with clean baselines")

# ── Legend (models + baseline indicator) ────────────────────────────────────
model_handles = [
    Line2D([0], [0],
           color=COLORS[m],
           marker=MARKERS[m],
           linewidth=2.2,
           markersize=5.5,
           label=DISPLAY[m])
    for m in MODELS
]
ax.legend(handles=model_handles, loc="lower left", frameon=True)

fig.tight_layout()

out_path = os.path.join(PROJECT_ROOT, "plot_snr_clean_baselines_v2.png")
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
