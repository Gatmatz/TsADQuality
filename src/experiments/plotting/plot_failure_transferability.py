"""
Plot Failure Transferability Analysis
======================================
Visualizes how model failure sets overlap under corruption.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "results" / "analysis" / "failure_transferability"
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "legend.fontsize": 9, "figure.dpi": 150,
})

EXPERIMENTS = ["White Noise", "Sensor Freeze", "Spikes", "Segment Swap", "Missing Data"]
MODELS = ["IForest", "LOF", "MP", "AE"]

COLORS = {
    "White Noise": "#D62728", "Sensor Freeze": "#1F77B4",
    "Spikes": "#FF7F0E", "Segment Swap": "#2CA02C", "Missing Data": "#9467BD",
}
MARKERS = {
    "White Noise": "s", "Sensor Freeze": "D",
    "Spikes": "^", "Segment Swap": "o", "Missing Data": "v",
}
# White Noise: higher severity = lower SNR, so sort descending
INVERT = {"White Noise": True}


def load_data():
    jaccard = pd.read_csv(DATA_DIR / "pairwise_jaccard_all.csv")
    mean_j = pd.read_csv(DATA_DIR / "mean_jaccard_by_severity.csv")
    baseline = pd.read_csv(DATA_DIR / "baseline_jaccard.csv")
    shared_count = None
    if (DATA_DIR / "shared_failure_count.csv").exists():
        shared_count = pd.read_csv(DATA_DIR / "shared_failure_count.csv")
    return jaccard, mean_j, baseline, shared_count


def plot_heatmaps(jaccard_df, baseline_df):
    """2×5 heatmap grid: FN-failure (top), FP-failure (bottom)."""
    # Pick representative severity per experiment
    rep_severity = {
        "White Noise": 0.0, "Sensor Freeze": 0.10,
        "Spikes": 0.10, "Segment Swap": 0.10, "Missing Data": 0.10,
    }

    fig, axes = plt.subplots(2, 5, figsize=(24, 9))

    for row, ftype in enumerate(["fn", "fp"]):
        for col, exp in enumerate(EXPERIMENTS):
            ax = axes[row, col]

            sev = rep_severity[exp]
            sub = jaccard_df[
                (jaccard_df["experiment"] == exp) &
                (jaccard_df["severity"] == sev) &
                (jaccard_df["failure_type"] == ftype)
            ]

            # Build 4×4 matrix
            mat = np.eye(4)  # diagonal = 1
            for _, r in sub.iterrows():
                i = MODELS.index(r["model_a"])
                j = MODELS.index(r["model_b"])
                mat[i, j] = r["jaccard"]
                mat[j, i] = r["jaccard"]

            # If multiple conditions at same severity, average
            if sub.empty:
                # Try closest severity
                exp_data = jaccard_df[
                    (jaccard_df["experiment"] == exp) &
                    (jaccard_df["failure_type"] == ftype)
                ]
                if not exp_data.empty:
                    sevs = exp_data["severity"].unique()
                    closest = min(sevs, key=lambda s: abs(s - sev))
                    sub = exp_data[exp_data["severity"] == closest]
                    # Rebuild with averaged values
                    mat = np.eye(4)
                    avg = sub.groupby(["model_a", "model_b"])["jaccard"].mean()
                    for (ma, mb), jval in avg.items():
                        i = MODELS.index(ma)
                        j = MODELS.index(mb)
                        mat[i, j] = jval
                        mat[j, i] = jval
            else:
                # Average across sub-parameters at same severity
                avg = sub.groupby(["model_a", "model_b"])["jaccard"].mean()
                mat = np.eye(4)
                for (ma, mb), jval in avg.items():
                    i = MODELS.index(ma)
                    j = MODELS.index(mb)
                    mat[i, j] = jval
                    mat[j, i] = jval

            im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(MODELS, fontsize=9)
            ax.set_yticklabels(MODELS, fontsize=9)

            # Annotate
            for i in range(4):
                for j in range(4):
                    val = mat[i, j]
                    if not np.isnan(val):
                        color = "white" if val > 0.65 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=9, color=color, fontweight="bold")

            if row == 0:
                ax.set_title(exp, fontsize=12, fontweight="bold")

    axes[0, 0].set_ylabel("FN-failure\n(Recall < 0.05)", fontsize=11)
    axes[1, 0].set_ylabel("FP-failure\n(Precision < 0.10)", fontsize=11)

    fig.suptitle("Failure Transferability: Pairwise Jaccard at 10% Corruption",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    # Colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Jaccard Similarity")

    out = PLOTS_DIR / "jaccard_heatmaps.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_severity_trend(mean_j_df, baseline_df):
    """1×5: mean Jaccard vs severity, FN and FP lines."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

    # Baseline mean per experiment × failure_type
    bl_means = baseline_df.groupby(["experiment", "failure_type"])["jaccard"].mean()

    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx]
        invert = INVERT.get(exp, False)

        for ftype, style, label in [("fn", "-", "FN-failure"), ("fp", "--", "FP-failure")]:
            sub = mean_j_df[
                (mean_j_df["experiment"] == exp) &
                (mean_j_df["failure_type"] == ftype)
            ].sort_values("severity")

            if sub.empty:
                continue

            ax.plot(sub["severity"], sub["mean"],
                    marker=MARKERS[exp], color=COLORS[exp],
                    linewidth=2, markersize=6, linestyle=style, label=label)

            # Baseline reference
            bl_key = (exp, ftype)
            if bl_key in bl_means.index:
                bl_val = bl_means[bl_key]
                ax.axhline(y=bl_val, color="gray", linestyle=":",
                           alpha=0.85, linewidth=2.0, zorder=5)

        ax.set_title(exp, fontsize=12, fontweight="bold")
        ax.set_xlabel("SNR (dB)" if exp == "White Noise" else "Corruption Fraction")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        if invert:
            ax.invert_xaxis()
        if idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    axes[0].set_ylabel("Mean Jaccard", fontsize=12)
    fig.suptitle("Failure Transferability vs Corruption Severity",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    out = PLOTS_DIR / "jaccard_vs_severity.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_thesis_figure(mean_j_df, baseline_df, shared_count_df):
    """Combined 2-row thesis figure."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey="row")

    bl_means = baseline_df.groupby(["experiment", "failure_type"])["jaccard"].mean()

    for idx, exp in enumerate(EXPERIMENTS):
        invert = INVERT.get(exp, False)
        xlabel = "SNR (dB)" if exp == "White Noise" else "Corruption Fraction"

        # ── Top: Jaccard vs severity ──
        ax_top = axes[0, idx]
        for ftype, style, label in [("fn", "-", "FN-fail"), ("fp", "--", "FP-fail")]:
            sub = mean_j_df[
                (mean_j_df["experiment"] == exp) &
                (mean_j_df["failure_type"] == ftype)
            ].sort_values("severity")
            if sub.empty:
                continue
            ax_top.plot(sub["severity"], sub["mean"],
                        marker=MARKERS[exp], color=COLORS[exp],
                        linewidth=2, markersize=5, linestyle=style, label=label)

            bl_key = (exp, ftype)
            if bl_key in bl_means.index:
                ax_top.axhline(y=bl_means[bl_key], color="gray",
                               linestyle=":", alpha=0.4)

        ax_top.set_title(exp, fontsize=12, fontweight="bold")
        ax_top.grid(True, alpha=0.3)
        ax_top.set_ylim(0, 1.05)
        if invert:
            ax_top.invert_xaxis()
        if idx == 0:
            ax_top.legend(fontsize=8, loc="lower left")

        # ── Bottom: Shared failure count ──
        ax_bot = axes[1, idx]
        if shared_count_df is not None:
            for ftype, style, label in [("fn", "-", "FN-fail"), ("fp", "--", "FP-fail")]:
                sub = shared_count_df[
                    (shared_count_df["experiment"] == exp) &
                    (shared_count_df["failure_type"] == ftype)
                ].sort_values("severity")
                if sub.empty:
                    continue
                # Average across sub-parameters
                agg = sub.groupby("severity")["n_shared_files"].mean().reset_index()
                ax_bot.plot(agg["severity"], agg["n_shared_files"],
                            marker=MARKERS[exp], color=COLORS[exp],
                            linewidth=2, markersize=5, linestyle=style, label=label)

        ax_bot.set_xlabel(xlabel)
        ax_bot.grid(True, alpha=0.3)
        if invert:
            ax_bot.invert_xaxis()

    axes[0, 0].set_ylabel("Mean Jaccard", fontsize=11)
    axes[1, 0].set_ylabel("Shared Failures\n(all 4 models)", fontsize=11)

    axes[0, 0].annotate("(a) Failure Transferability", xy=(-0.3, 0.5),
                         xycoords="axes fraction", fontsize=12,
                         fontweight="bold", rotation=90, va="center")
    axes[1, 0].annotate("(b) Universal Failures", xy=(-0.3, 0.5),
                         xycoords="axes fraction", fontsize=12,
                         fontweight="bold", rotation=90, va="center")

    fig.tight_layout()
    out = PLOTS_DIR / "thesis_failure_transferability.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    print(f"Plotting failure transferability")
    print(f"Output: {PLOTS_DIR}\n")

    jaccard_df, mean_j_df, baseline_df, shared_count_df = load_data()

    plot_heatmaps(jaccard_df, baseline_df)
    plot_severity_trend(mean_j_df, baseline_df)
    plot_thesis_figure(mean_j_df, baseline_df, shared_count_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
