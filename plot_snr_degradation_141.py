"""
Recreates the 'Performance Degradation under White Noise' plot
using ONLY the 141 files for baselines, with the correct color palette.
Shows AUC-ROC drop (baseline - corrupted) vs SNR.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV    = os.path.join(PROJECT_ROOT, "results", "experiments", "white_noise_snr", "summary.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "experiments", "white_noise_snr", "checkpoint.csv")
BASELINE_CSV   = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")

MODELS = ["IForest", "LOF", "MP", "AE"]

COLORS  = {
    "IForest": "#1f77b4",
    "LOF":     "#ff7f0e",
    "MP":      "#2ca02c",
    "AE":      "#d62728",
}
MARKERS = {"IForest": "o", "LOF": "s", "MP": "^", "AE": "D"}

# ── Load data ────────────────────────────────────────────────────────────────
summary = pd.read_csv(SUMMARY_CSV)

baseline = pd.read_csv(BASELINE_CSV)
baseline["model"] = baseline["model"].replace({"Autoencoder": "AE", "Matrix Profile": "MP", "ME": "MP"})

checkpoint = pd.read_csv(CHECKPOINT_CSV)
experiment_files = set(checkpoint[checkpoint["error"].isna()]["file"].unique())

baseline_141 = baseline[baseline["file"].isin(experiment_files) & baseline["model"].isin(MODELS)]
bl_aucroc = baseline_141.groupby("model")["AUC_ROC"].mean().to_dict()

# ── Build plot ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      130,
    "savefig.dpi":     300,
    "font.size":       9,
    "axes.titlesize":  12,
    "axes.labelsize":  10,
    "legend.fontsize": 9,
    "axes.grid":       True,
    "grid.alpha":      0.22,
    "grid.linestyle":  "-",
    "axes.spines.top": True,
    "axes.spines.right": True,
})

fig, ax = plt.subplots(figsize=(8.2, 5.0))

for model in MODELS:
    sub = summary[summary["model"] == model].sort_values("snr_db")
    if sub.empty:
        continue
    base = bl_aucroc.get(model)
    if pd.isna(base):
        continue
    drop = base - sub["mean_AUC_ROC"]
    ax.plot(
        sub["snr_db"],
        drop,
        color=COLORS[model],
        marker=MARKERS[model],
        linewidth=2.4,
        markersize=5.5,
        label=model,
        zorder=3,
    )

# Zero baseline (dashed)
ax.axhline(0, color="#111111", linestyle=(0, (5, 3)), linewidth=2.4, alpha=0.9)

ax.invert_xaxis()
ax.set_xlabel("SNR (dB) - Higher is cleaner")
ax.set_ylabel("AUC-ROC Drop (Baseline - Corrupted)")
ax.set_title("Performance Degradation under White Noise", fontweight="bold")
ax.legend(frameon=True)

fig.tight_layout()
out_path = os.path.join(PROJECT_ROOT, "plot_snr_degradation_141.png")
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
