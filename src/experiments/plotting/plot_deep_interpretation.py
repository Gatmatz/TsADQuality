"""
Plots for Deep Interpretation Analysis
=======================================
1. Distribution of AUC drops (violin/box plots)
2. Vulnerability factors (correlation heatmap)
3. Breaking points (threshold chart)
4. Difficulty paradox (grouped bar)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "results" / "analysis" / "deep_interpretation"
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "legend.fontsize": 9, "figure.dpi": 150,
})

EXPERIMENTS = ["White Noise", "Sensor Freeze", "Spikes", "Segment Swap", "Missing Data"]
MODELS = ["IForest", "LOF", "MP", "AE"]

COLORS_MODEL = {
    "IForest": "#1F77B4", "LOF": "#FF7F0E", "MP": "#2CA02C", "AE": "#D62728",
}
COLORS_EXP = {
    "White Noise": "#D62728", "Sensor Freeze": "#1F77B4",
    "Spikes": "#FF7F0E", "Segment Swap": "#2CA02C", "Missing Data": "#9467BD",
}


def load_data():
    drops = pd.read_csv(DATA_DIR / "per_file_drops.csv")
    distributions = pd.read_csv(DATA_DIR / "drop_distributions.csv")
    correlations = pd.read_csv(DATA_DIR / "vulnerability_correlations.csv")
    categories = pd.read_csv(DATA_DIR / "vulnerability_by_category.csv")
    breaking = pd.read_csv(DATA_DIR / "breaking_points.csv")
    return drops, distributions, correlations, categories, breaking


# ═══════════════════════════════════════════════════════════════
# 1. DISTRIBUTION OF AUC DROPS — Box plots per experiment
# ═══════════════════════════════════════════════════════════════

def plot_drop_distributions(drops_df, distributions_df):
    """Box plots of AUC drop distribution at max severity per experiment."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx]
        exp_drops = drops_df[drops_df["experiment"] == exp]
        if exp_drops.empty:
            continue

        # Use max severity (most corruption)
        if exp == "White Noise":
            max_sev = exp_drops["severity"].min()  # lowest SNR = most corruption
        else:
            max_sev = exp_drops["severity"].max()

        sev_data = exp_drops[exp_drops["severity"] == max_sev]

        box_data = []
        labels = []
        for model in MODELS:
            model_drops = sev_data[sev_data["model"] == model]["auc_drop"]
            if not model_drops.empty:
                box_data.append(model_drops.values)
                labels.append(model)

        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                           medianprops=dict(color="black", linewidth=2),
                           whiskerprops=dict(linewidth=1.5),
                           flierprops=dict(markersize=3, alpha=0.5))
            for patch, model in zip(bp["boxes"], labels):
                patch.set_facecolor(COLORS_MODEL[model])
                patch.set_alpha(0.7)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.85, linewidth=2.0, zorder=5)
        ax.axhline(y=0.2, color="red", linestyle=":", alpha=0.4, linewidth=2.0, zorder=5,
                   label=">20% drop" if idx == 0 else None)
        ax.set_title(f"{exp}\n(sev={max_sev})", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    axes[0].set_ylabel("AUC Drop (baseline - corrupted)", fontsize=12)
    fig.suptitle("Distribution of Per-File AUC Drops at Maximum Corruption",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    out = PLOTS_DIR / "drop_distributions_boxplot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_catastrophic_vs_improved(distributions_df):
    """Stacked bar: % catastrophic vs % improved per model x experiment."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx]
        exp_data = distributions_df[distributions_df["experiment"] == exp]
        if exp_data.empty:
            continue

        # Max severity
        if exp == "White Noise":
            max_sev = exp_data["severity"].min()
        else:
            max_sev = exp_data["severity"].max()

        sev_data = exp_data[exp_data["severity"] == max_sev]

        x = np.arange(len(MODELS))
        width = 0.6

        catastrophic = []
        improved = []
        for model in MODELS:
            row = sev_data[sev_data["model"] == model]
            if not row.empty:
                catastrophic.append(row["pct_catastrophic"].values[0])
                improved.append(row["pct_improved"].values[0])
            else:
                catastrophic.append(0)
                improved.append(0)

        ax.bar(x, catastrophic, width, color="#D62728", alpha=0.8, label="Catastrophic (>20%)")
        ax.bar(x, [-v for v in improved], width, color="#2CA02C", alpha=0.8, label="Improved (<0%)")

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_title(f"{exp}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

        if idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("% of Files", fontsize=12)
    fig.suptitle("Catastrophic Degradation vs Accidental Improvement at Max Corruption",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    out = PLOTS_DIR / "catastrophic_vs_improved.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# 2. VULNERABILITY FACTORS — Correlation heatmap
# ═══════════════════════════════════════════════════════════════

def plot_vulnerability_heatmap(correlations_df):
    """Heatmap: mean Spearman rho (across severities) for each model x feature x experiment."""
    features = ["baseline_auc", "data_len", "ratio"]
    feature_labels = ["Baseline AUC", "Series Length", "Anomaly Ratio"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx]
        exp_corr = correlations_df[correlations_df["experiment"] == exp]
        if exp_corr.empty:
            ax.set_title(exp, fontsize=11, fontweight="bold")
            continue

        # Average across severities
        avg = exp_corr.groupby(["model", "feature"])["spearman_rho"].mean().reset_index()

        # Build matrix: models x features
        mat = np.full((len(MODELS), len(features)), np.nan)
        for i, model in enumerate(MODELS):
            for j, feat in enumerate(features):
                row = avg[(avg["model"] == model) & (avg["feature"] == feat)]
                if not row.empty:
                    mat[i, j] = row["spearman_rho"].values[0]

        im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(feature_labels, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(MODELS if idx == 0 else MODELS, fontsize=10)

        # Annotate
        for i in range(len(MODELS)):
            for j in range(len(features)):
                val = mat[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=9, color=color, fontweight="bold")

        ax.set_title(exp, fontsize=11, fontweight="bold")

    fig.suptitle("Vulnerability Factors: Spearman Correlation with AUC Drop",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    # Colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Spearman rho", fontsize=11)

    out = PLOTS_DIR / "vulnerability_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# 3. BREAKING POINTS — Threshold chart
# ═══════════════════════════════════════════════════════════════

def plot_breaking_points(breaking_df):
    """Bar chart: breaking severity per model x experiment (10% drop threshold)."""
    bp_10 = breaking_df[breaking_df["threshold"] == "10pct_drop"].copy()
    if bp_10.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize severity to [0,1] for comparability across experiments
    # White Noise: SNR values -> use as-is but note scale difference
    x = np.arange(len(EXPERIMENTS))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    for i, model in enumerate(MODELS):
        vals = []
        for exp in EXPERIMENTS:
            row = bp_10[(bp_10["experiment"] == exp) & (bp_10["model"] == model)]
            if not row.empty and not pd.isna(row["breaking_severity"].values[0]):
                vals.append(row["breaking_severity"].values[0])
            else:
                vals.append(np.nan)

        # For display: convert White Noise SNR to positive fraction-like scale
        display_vals = []
        labels_list = []
        for j, (v, exp) in enumerate(zip(vals, EXPERIMENTS)):
            if np.isnan(v):
                display_vals.append(0)
                labels_list.append("N/A")
            else:
                display_vals.append(1)  # uniform height, label with value
                if exp == "White Noise":
                    labels_list.append(f"{v:.0f}dB")
                else:
                    labels_list.append(f"{v:.0%}")

        bars = ax.bar(x + offsets[i], display_vals, width,
                      color=COLORS_MODEL[model], alpha=0.8, label=model)

        # Label each bar
        for bar, label in zip(bars, labels_list):
            if label != "N/A":
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                        label, ha="center", va="bottom", fontsize=9, fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                        "N/A", ha="center", va="bottom", fontsize=8,
                        color="gray", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(EXPERIMENTS, fontsize=11)
    ax.set_ylabel("Breaks at 10% threshold", fontsize=12)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([])
    ax.legend(fontsize=10)
    ax.set_title("Breaking Points: Minimum Severity for 10% AUC Degradation",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    out = PLOTS_DIR / "breaking_points.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# 4. DIFFICULTY PARADOX — Grouped bar chart
# ═══════════════════════════════════════════════════════════════

def plot_difficulty_paradox(categories_df):
    """AUC drop by difficulty level at max severity."""
    if categories_df.empty:
        return

    diff_data = categories_df[categories_df["feature"] == "difficulty"].copy()
    if diff_data.empty:
        return

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

    difficulty_colors = {"Easy": "#FF6B6B", "Medium": "#FFD93D", "Hard": "#6BCB77"}

    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx]
        exp_data = diff_data[diff_data["experiment"] == exp]
        if exp_data.empty:
            continue

        # Max severity
        if exp == "White Noise":
            max_sev = exp_data["severity"].min()
        else:
            max_sev = exp_data["severity"].max()

        sev_data = exp_data[exp_data["severity"] == max_sev]

        x = np.arange(len(MODELS))
        width = 0.25
        difficulties = ["Easy", "Medium", "Hard"]
        offsets = [-width, 0, width]

        for d_idx, diff in enumerate(difficulties):
            vals = []
            for model in MODELS:
                row = sev_data[(sev_data["model"] == model) & (sev_data["value"] == diff)]
                if not row.empty:
                    vals.append(row["mean_drop"].values[0])
                else:
                    vals.append(0)

            ax.bar(x + offsets[d_idx], vals, width,
                   color=difficulty_colors[diff], alpha=0.8,
                   label=diff if idx == 0 else None)

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_title(exp, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    axes[0].set_ylabel("Mean AUC Drop", fontsize=12)
    axes[0].legend(fontsize=10, title="Difficulty")
    fig.suptitle("AUC Drop by Dataset Difficulty at Maximum Corruption",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    out = PLOTS_DIR / "difficulty_paradox.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# 5. THESIS COMBINED FIGURE
# ═══════════════════════════════════════════════════════════════

def plot_thesis_figure(drops_df, correlations_df, breaking_df):
    """2-row thesis figure: (a) vulnerability heatmap, (b) breaking points timeline."""
    fig = plt.figure(figsize=(22, 10))

    # ── Top row: Baseline AUC vs AUC drop scatter (1×5) ──
    gs = fig.add_gridspec(2, 5, hspace=0.35)

    for idx, exp in enumerate(EXPERIMENTS):
        ax = fig.add_subplot(gs[0, idx])
        exp_drops = drops_df[drops_df["experiment"] == exp]
        if exp_drops.empty:
            continue

        # Max severity
        if exp == "White Noise":
            max_sev = exp_drops["severity"].min()
        else:
            max_sev = exp_drops["severity"].max()

        sev_data = exp_drops[exp_drops["severity"] == max_sev]

        for model in MODELS:
            m_data = sev_data[sev_data["model"] == model]
            ax.scatter(m_data["baseline_auc"], m_data["auc_drop"],
                       alpha=0.4, s=15, color=COLORS_MODEL[model],
                       label=model if idx == 0 else None)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(exp, fontsize=11, fontweight="bold")
        ax.set_xlabel("Baseline AUC")
        ax.grid(True, alpha=0.2)

        if idx == 0:
            ax.set_ylabel("AUC Drop")
            ax.legend(fontsize=7, markerscale=2)

    # ── Bottom row: Breaking points per experiment (1×5) ──
    bp_thresholds = ["10pct_drop", "below_0.6", "below_random"]
    thresh_colors = {"10pct_drop": "#FF7F0E", "below_0.6": "#D62728", "below_random": "#8C564B"}
    thresh_labels = {"10pct_drop": "10% drop", "below_0.6": "AUC<0.6", "below_random": "AUC<0.5"}

    for idx, exp in enumerate(EXPERIMENTS):
        ax = fig.add_subplot(gs[1, idx])
        exp_bp = breaking_df[breaking_df["experiment"] == exp]

        x = np.arange(len(MODELS))
        width = 0.25
        offsets = [-width, 0, width]

        for t_idx, thresh in enumerate(bp_thresholds):
            vals = []
            for model in MODELS:
                row = exp_bp[(exp_bp["model"] == model) & (exp_bp["threshold"] == thresh)]
                if not row.empty and not pd.isna(row["breaking_severity"].values[0]):
                    vals.append(row["breaking_severity"].values[0])
                else:
                    vals.append(np.nan)

            finite_vals = [v if not np.isnan(v) else 0 for v in vals]
            bars = ax.bar(x + offsets[t_idx], finite_vals, width,
                          color=thresh_colors[thresh], alpha=0.7,
                          label=thresh_labels[thresh] if idx == 0 else None)

            # Mark NaN bars
            for bar_idx, v in enumerate(vals):
                if np.isnan(v):
                    ax.text(x[bar_idx] + offsets[t_idx], 0.01,
                            "N/A", ha="center", va="bottom",
                            fontsize=7, color="gray", fontstyle="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.set_title(exp, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

        if exp == "White Noise":
            ax.set_ylabel("SNR (dB)")
        else:
            ax.set_ylabel("Corruption Fraction" if idx == 0 else "")

        if idx == 0:
            ax.legend(fontsize=8)

    fig.text(0.01, 0.75, "(a) Baseline AUC vs Degradation", fontsize=13,
             fontweight="bold", rotation=90, va="center")
    fig.text(0.01, 0.25, "(b) Breaking Points", fontsize=13,
             fontweight="bold", rotation=90, va="center")

    out = PLOTS_DIR / "thesis_deep_interpretation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    print(f"Plotting deep interpretation results")
    print(f"Output: {PLOTS_DIR}\n")

    drops, distributions, correlations, categories, breaking = load_data()

    plot_drop_distributions(drops, distributions)
    plot_catastrophic_vs_improved(distributions)
    plot_vulnerability_heatmap(correlations)
    plot_breaking_points(breaking)
    plot_difficulty_paradox(categories)
    plot_thesis_figure(drops, correlations, breaking)

    print("\nDone.")


if __name__ == "__main__":
    main()
