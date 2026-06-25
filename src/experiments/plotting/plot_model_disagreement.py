"""
Plot Model Disagreement Analysis
=================================
Two main figures:
  1. Kendall's tau (rank stability) vs corruption severity — per corruption type
  2. std(AUC_ROC) across models vs corruption severity — convergence effect
  3. Combined 2-panel figure for the thesis
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DISAGREE_DIR = PROJECT_ROOT / "results" / "analysis" / "model_disagreement"
PLOTS_DIR = DISAGREE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ── Experiment configs ──
# (dir_name, display_name, severity_extractor, severity_label, invert_x)
# severity_extractor: function condition_string -> numeric severity
EXPERIMENTS = [
    (
        "white_noise_snr",
        "White Noise",
        lambda c: float(re.search(r"snr_([-\d]+)dB", c).group(1)),
        "SNR (dB)",
        True,  # higher SNR = less corruption, so invert
    ),
    (
        "freeze",
        "Sensor Freeze",
        lambda c: float(re.search(r"frac_([\d.]+)_ns_(\d+)", c).group(1)),
        "Corruption Fraction",
        False),
    (
        "spikes_normal_only",
        "Spikes",
        lambda c: float(re.search(r"frac_([\d.]+)_mult", c).group(1)),
        "Corruption Fraction",
        False),
    (
        "swap_segment",
        "Segment Swap",
        lambda c: float(re.search(r"frac_([\d.]+)_ns", c).group(1)),
        "Corruption Fraction",
        False),
    (
        "missing_true_impact",
        "Missing Data",
        lambda c: float(re.search(r"frac_([\d.]+)", c).group(1)),
        "Corruption Fraction",
        False),
]

COLORS = {
    "White Noise": "#D62728",
    "Sensor Freeze": "#1F77B4",
    "Spikes": "#FF7F0E",
    "Segment Swap": "#2CA02C",
    "Missing Data": "#9467BD",
}

MARKERS = {
    "White Noise": "s",
    "Sensor Freeze": "D",
    "Spikes": "^",
    "Segment Swap": "o",
    "Missing Data": "v",
}


def load_rank_stability(exp_dir, severity_extractor):
    """Load rank_stability.csv and extract numeric severity."""
    path = DISAGREE_DIR / exp_dir / "rank_stability.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["kendall_tau_mean"])
    try:
        df["severity"] = df["condition"].apply(severity_extractor)
    except Exception:
        return None
    return df


def load_disagreement_summary(exp_dir, severity_extractor):
    """Load disagreement_summary.csv and extract numeric severity."""
    path = DISAGREE_DIR / exp_dir / "disagreement_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Separate baseline
    baseline = df[df["condition"] == "baseline_clean"]
    corrupted = df[df["condition"] != "baseline_clean"].copy()
    try:
        corrupted["severity"] = corrupted["condition"].apply(severity_extractor)
    except Exception:
        return None, None
    return baseline, corrupted


def plot_kendall_tau_combined():
    """All corruption types on one plot — Kendall's tau vs severity."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp_dir, name, extractor, xlabel, invert in EXPERIMENTS:
        df = load_rank_stability(exp_dir, extractor)
        if df is None or df.empty:
            continue

        # For experiments with sub-parameters (e.g. freeze has fraction + num_stucks),
        # aggregate: take the mean tau per severity level
        agg = df.groupby("severity").agg(
            tau_mean=("kendall_tau_mean", "mean"),
            tau_std=("kendall_tau_mean", "std")).reset_index().sort_values("severity", ascending=not invert)

        x = agg["severity"]
        y = agg["tau_mean"]

        ax.plot(x, y, marker=MARKERS[name], color=COLORS[name],
                linewidth=2, markersize=7, label=name)
        ax.fill_between(x, y - agg["tau_std"], y + agg["tau_std"],
                         alpha=0.1, color=COLORS[name])

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect agreement")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5, label="Random ranking")
    ax.set_xlabel("Corruption Severity (fraction or SNR in dB)", fontsize=12)
    ax.set_ylabel("Kendall's $\\tau$ (vs clean baseline)", fontsize=12)
    ax.set_title("Model Ranking Stability Under Corruption", fontsize=13)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_ylim(-0.15, 1.15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = PLOTS_DIR / "kendall_tau_combined.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_kendall_tau_per_experiment():
    """One subplot per corruption type for cleaner severity axis."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

    for idx, (exp_dir, name, extractor, xlabel, invert) in enumerate(EXPERIMENTS):
        ax = axes[idx]
        df = load_rank_stability(exp_dir, extractor)
        if df is None or df.empty:
            continue

        agg = df.groupby("severity").agg(
            tau_mean=("kendall_tau_mean", "mean"),
            tau_std=("kendall_tau_mean", "std")).reset_index().sort_values("severity")

        x = agg["severity"]
        y = agg["tau_mean"]

        ax.plot(x, y, marker=MARKERS[name], color=COLORS[name],
                linewidth=2, markersize=7)
        ax.fill_between(x, y - agg["tau_std"], y + agg["tau_std"],
                         alpha=0.15, color=COLORS[name])

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
        ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.4)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if invert:
            ax.invert_xaxis()

    axes[0].set_ylabel("Kendall's $\\tau$", fontsize=12)
    fig.suptitle("Model Ranking Stability Under Corruption", fontsize=14, y=1.02)
    fig.tight_layout()

    out = PLOTS_DIR / "kendall_tau_per_experiment.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_std_convergence():
    """Show that models converge (std decreases) under severe corruption."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

    for idx, (exp_dir, name, extractor, xlabel, invert) in enumerate(EXPERIMENTS):
        ax = axes[idx]
        baseline, corrupted = load_disagreement_summary(exp_dir, extractor)
        if corrupted is None or corrupted.empty:
            continue

        agg = corrupted.groupby("severity").agg(
            std_mean=("auc_std_mean", "mean"),
            auc_mean=("auc_mean_mean", "mean")).reset_index().sort_values("severity")

        x = agg["severity"]

        # Plot std
        ax.plot(x, agg["std_mean"], marker=MARKERS[name], color=COLORS[name],
                linewidth=2, markersize=7, label="std(AUC) across models")

        # Baseline reference
        if baseline is not None and not baseline.empty:
            bl_std = baseline["auc_std_mean"].values[0]
            ax.axhline(y=bl_std, color="gray", linestyle="--", alpha=0.6,
                       label=f"Baseline std = {bl_std:.3f}")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if invert:
            ax.invert_xaxis()

        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("std(AUC_ROC) across models", fontsize=11)
    fig.suptitle("Model Disagreement (std) Under Corruption", fontsize=14, y=1.02)
    fig.tight_layout()

    out = PLOTS_DIR / "std_convergence_per_experiment.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_thesis_figure():
    """
    Combined 2-row figure for the thesis:
      Top row: Kendall's tau (rank stability)
      Bottom row: std(AUC) + mean(AUC) on twin axis
    """
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey="row")

    for idx, (exp_dir, name, extractor, xlabel, invert) in enumerate(EXPERIMENTS):
        # ── Top: Kendall's tau ──
        ax_top = axes[0, idx]
        df_rank = load_rank_stability(exp_dir, extractor)
        if df_rank is not None and not df_rank.empty:
            agg = df_rank.groupby("severity").agg(
                tau_mean=("kendall_tau_mean", "mean"),
                tau_std=("kendall_tau_mean", "std")).reset_index().sort_values("severity")

            x = agg["severity"]
            y = agg["tau_mean"]
            ax_top.plot(x, y, marker=MARKERS[name], color=COLORS[name],
                        linewidth=2, markersize=6)
            ax_top.fill_between(x, y - agg["tau_std"], y + agg["tau_std"],
                                 alpha=0.12, color=COLORS[name])
            ax_top.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
            ax_top.axhline(y=0.0, color="gray", linestyle=":", alpha=0.4)

        ax_top.set_title(name, fontsize=12, fontweight="bold")
        ax_top.grid(True, alpha=0.3)
        if invert:
            ax_top.invert_xaxis()

        # ── Bottom: std + mean AUC ──
        ax_bot = axes[1, idx]
        baseline, corrupted = load_disagreement_summary(exp_dir, extractor)
        if corrupted is not None and not corrupted.empty:
            agg2 = corrupted.groupby("severity").agg(
                std_mean=("auc_std_mean", "mean"),
                auc_mean=("auc_mean_mean", "mean")).reset_index().sort_values("severity")

            x2 = agg2["severity"]
            ax_bot.plot(x2, agg2["std_mean"], marker=MARKERS[name],
                        color=COLORS[name], linewidth=2, markersize=6,
                        label="std(AUC)")

            # Mean AUC on twin axis
            ax_twin = ax_bot.twinx()
            ax_twin.plot(x2, agg2["auc_mean"], marker="x", color="gray",
                         linewidth=1.5, markersize=5, alpha=0.6, linestyle="--",
                         label="mean(AUC)")
            ax_twin.set_ylim(0.2, 0.85)
            if idx == 4:
                ax_twin.set_ylabel("mean(AUC_ROC)", fontsize=10, color="gray")
            else:
                ax_twin.set_yticklabels([])

            if baseline is not None and not baseline.empty:
                bl_std = baseline["auc_std_mean"].values[0]
                ax_bot.axhline(y=bl_std, color="gray", linestyle="--", alpha=0.4)

        ax_bot.set_xlabel(xlabel, fontsize=11)
        ax_bot.grid(True, alpha=0.3)
        if invert:
            ax_bot.invert_xaxis()

    axes[0, 0].set_ylabel("Kendall's $\\tau$", fontsize=12)
    axes[1, 0].set_ylabel("std(AUC_ROC)", fontsize=11)

    # Row labels
    axes[0, 0].annotate("(a) Rank Stability", xy=(-0.3, 0.5),
                         xycoords="axes fraction", fontsize=12,
                         fontweight="bold", rotation=90, va="center")
    axes[1, 0].annotate("(b) Metric Disagreement", xy=(-0.3, 0.5),
                         xycoords="axes fraction", fontsize=12,
                         fontweight="bold", rotation=90, va="center")

    fig.tight_layout()

    out = PLOTS_DIR / "thesis_disagreement_figure.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    print(f"Plotting model disagreement analysis")
    print(f"Output: {PLOTS_DIR}\n")

    plot_kendall_tau_combined()
    plot_kendall_tau_per_experiment()
    plot_std_convergence()
    plot_thesis_figure()

    print("\nDone.")


if __name__ == "__main__":
    main()
