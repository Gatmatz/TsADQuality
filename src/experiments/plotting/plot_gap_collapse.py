"""
Gap Collapse Mechanism Under White Noise
==========================================
Shows WHY each model degrades: the internal separation between
anomaly scores and normal scores collapses at different rates.

Thesis-quality figure with theoretical overlay.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "results" / "experiments" / "internal_analysis"
PLOTS_DIR = PROJECT_ROOT / "results" / "analysis" / "deep_interpretation" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14,
    "axes.labelsize": 13, "legend.fontsize": 10, "figure.dpi": 150,
    "font.family": "serif",
})

COLORS = {
    "IForest": "#1F77B4", "LOF": "#FF7F0E", "MP": "#2CA02C", "AE": "#D62728",
}
MARKERS = {"IForest": "s", "LOF": "D", "MP": "^", "AE": "o"}


def load_noise_internals():
    df = pd.read_csv(DATA_DIR / "aggregate_summary.csv")
    noise = df[df["corruption"].str.startswith("noise_snr")].copy()
    noise["snr"] = noise["corruption"].str.extract(r"noise_snr(-?\d+)").astype(int)
    return noise.sort_values(["model", "snr"])


def plot_gap_collapse_unified(noise_df):
    """
    Single plot: normalized gap retention (%) vs SNR for all 4 models.
    This is the key thesis figure showing WHY IForest is more robust.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    snr_order = sorted(noise_df["snr"].unique())

    for model in ["IForest", "LOF", "MP", "AE"]:
        m = noise_df[noise_df["model"] == model].sort_values("snr")

        if model == "IForest":
            gap_clean = m["separation_gap_clean_mean"].values
            gap_corr = m["separation_gap_corrupted_mean"].values
        elif model == "LOF":
            gap_clean = m["kdist_gap_clean_mean"].values
            gap_corr = m["kdist_gap_corrupted_mean"].values
        elif model == "MP":
            gap_clean = m["nn_dist_gap_clean_mean"].values
            gap_corr = m["nn_dist_gap_corrupted_mean"].values
        elif model == "AE":
            # Use error ratio as the "gap"
            # error_ratio = error_anomaly / error_normal; higher = better separation
            err_anom_clean = m["recon_error_anomaly_clean_mean"].values
            err_norm_clean = m["recon_error_normal_clean_mean"].values
            err_anom_corr = m["recon_error_anomaly_corrupted_mean"].values
            err_norm_corr = m["recon_error_normal_corrupted_mean"].values
            # Gap = error_anomaly - error_normal (positive = good)
            gap_clean = err_anom_clean - err_norm_clean
            gap_corr = err_anom_corr - err_norm_corr

        # Normalize: gap_retained = gap_corrupted / gap_clean * 100
        retention = gap_corr / gap_clean * 100

        ax.plot(m["snr"].values, retention,
                marker=MARKERS[model], color=COLORS[model],
                linewidth=2.5, markersize=8, label=model, zorder=3)

    # Reference lines
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4, linewidth=2.0, zorder=5)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=2.0, zorder=5)

    # Shade danger zones
    ax.axhspan(-50, 0, alpha=0.06, color="red")
    ax.text(42, -12, "Inverted gap\n(worse than random)", fontsize=9,
            color="#D62728", fontstyle="italic", ha="right")
    ax.text(42, 105, "Clean performance", fontsize=9,
            color="gray", fontstyle="italic", ha="right")

    ax.set_xlabel("SNR (dB)  [higher = less noise]", fontsize=13)
    ax.set_ylabel("Gap Retention (%)", fontsize=13)
    ax.set_title("Internal Separation Gap Under White Noise", fontsize=15, fontweight="bold")
    ax.set_xlim(-22, 42)
    ax.set_ylim(-50, 115)
    ax.legend(fontsize=11, loc="center left")
    ax.grid(True, alpha=0.2)
    ax.invert_xaxis()

    fig.tight_layout()
    out = PLOTS_DIR / "gap_collapse_unified.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_mechanism_4panel(noise_df):
    """
    4-panel figure: each panel shows the raw internal metric for anomaly vs normal
    windows, revealing the MECHANISM of gap collapse per model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_configs = [
        ("IForest", "path_anomaly", "path_normal",
         "Mean Path Length", "Shorter path = more anomalous"),
        ("LOF", "kdist_anomaly", "kdist_normal",
         "Mean k-Distance", "Higher k-dist = more anomalous"),
        ("MP", "nn_dist_anomaly", "nn_dist_normal",
         "Mean NN Distance", "Higher NN-dist = more anomalous"),
        ("AE", "recon_error_anomaly", "recon_error_normal",
         "Mean Reconstruction Error", "Higher error = more anomalous"),
    ]

    for idx, (model, anom_prefix, norm_prefix, ylabel, note) in enumerate(model_configs):
        ax = axes[idx]
        m = noise_df[noise_df["model"] == model].sort_values("snr")
        snr = m["snr"].values

        anom_clean = m[f"{anom_prefix}_clean_mean"].values
        anom_corr = m[f"{anom_prefix}_corrupted_mean"].values
        norm_clean = m[f"{norm_prefix}_clean_mean"].values
        norm_corr = m[f"{norm_prefix}_corrupted_mean"].values

        # Plot clean baselines as dashed horizontal lines
        ax.axhline(y=anom_clean[0], color="#D62728", linestyle=":", alpha=0.4, linewidth=2.0, zorder=5)
        ax.axhline(y=norm_clean[0], color="#1F77B4", linestyle=":", alpha=0.4, linewidth=2.0, zorder=5)

        # Plot corrupted values
        ax.plot(snr, anom_corr, "o-", color="#D62728", linewidth=2.5, markersize=7,
                label="Anomaly windows", zorder=3)
        ax.plot(snr, norm_corr, "s-", color="#1F77B4", linewidth=2.5, markersize=7,
                label="Normal windows", zorder=3)

        # Shade the gap
        ax.fill_between(snr, anom_corr, norm_corr, alpha=0.12,
                         color="green" if model != "IForest" else "orange")

        # Arrow showing gap direction
        mid_idx = len(snr) // 2
        gap = anom_corr[mid_idx] - norm_corr[mid_idx]
        if model == "IForest":
            # IForest: anomaly has SHORTER path (lower value = more anomalous)
            ax.annotate("gap", xy=(snr[mid_idx], (anom_corr[mid_idx] + norm_corr[mid_idx])/2),
                        fontsize=9, ha="center", color="gray")

        ax.set_title(f"{model}", fontsize=14, fontweight="bold", color=COLORS[model])
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.2)
        ax.invert_xaxis()

        # Add note
        ax.text(0.02, 0.02, note, transform=ax.transAxes,
                fontsize=8, fontstyle="italic", color="gray")

    fig.suptitle("How White Noise Destroys Internal Model Separation",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = PLOTS_DIR / "gap_mechanism_4panel.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_distance_inflation_comparison(noise_df):
    """
    Bar chart: % inflation of normal vs anomaly distances at 0 dB.
    Shows the ASYMMETRIC inflation that kills distance-based methods.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ["IForest", "LOF", "MP", "AE"]
    labels_map = {
        "IForest": "Path Length",
        "LOF": "k-Distance",
        "MP": "NN Distance",
        "AE": "Recon. Error",
    }

    # Get 0 dB data
    snr0 = noise_df[noise_df["snr"] == 0]

    anom_inflate = []
    norm_inflate = []

    for model in models:
        m = snr0[snr0["model"] == model]
        if model == "IForest":
            a_c, a_n = m["path_anomaly_clean_mean"].values[0], m["path_anomaly_corrupted_mean"].values[0]
            n_c, n_n = m["path_normal_clean_mean"].values[0], m["path_normal_corrupted_mean"].values[0]
        elif model == "LOF":
            a_c, a_n = m["kdist_anomaly_clean_mean"].values[0], m["kdist_anomaly_corrupted_mean"].values[0]
            n_c, n_n = m["kdist_normal_clean_mean"].values[0], m["kdist_normal_corrupted_mean"].values[0]
        elif model == "MP":
            a_c, a_n = m["nn_dist_anomaly_clean_mean"].values[0], m["nn_dist_anomaly_corrupted_mean"].values[0]
            n_c, n_n = m["nn_dist_normal_clean_mean"].values[0], m["nn_dist_normal_corrupted_mean"].values[0]
        elif model == "AE":
            a_c, a_n = m["recon_error_anomaly_clean_mean"].values[0], m["recon_error_anomaly_corrupted_mean"].values[0]
            n_c, n_n = m["recon_error_normal_clean_mean"].values[0], m["recon_error_normal_corrupted_mean"].values[0]

        anom_inflate.append((a_n - a_c) / a_c * 100)
        norm_inflate.append((n_n - n_c) / n_c * 100)

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, norm_inflate, width, color="#1F77B4", alpha=0.8,
                   label="Normal windows", edgecolor="white", linewidth=1)
    bars2 = ax.bar(x + width/2, anom_inflate, width, color="#D62728", alpha=0.8,
                   label="Anomaly windows", edgecolor="white", linewidth=1)

    # Label bars
    for bar, val in zip(bars1, norm_inflate):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{val:+.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#1F77B4")
    for bar, val in zip(bars2, anom_inflate):
        y_pos = bar.get_height() + 3 if val >= 0 else bar.get_height() - 12
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{val:+.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#D62728")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({labels_map[m]})" for m in models], fontsize=11)
    ax.set_ylabel("Change from Clean (%)", fontsize=13)
    ax.set_title("Asymmetric Impact of Noise (0 dB) on Internal Metrics",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    out = PLOTS_DIR / "asymmetric_inflation_0dB.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_thesis_combined(noise_df):
    """
    Combined 3-row thesis figure:
    (a) Gap retention curves
    (b) 4-panel mechanism
    (c) Asymmetric inflation bars
    """
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3,
                          height_ratios=[1, 1.2, 0.9])

    # ── Row 1: Gap retention (single wide plot) ──
    ax_gap = fig.add_subplot(gs[0, :])
    snr_order = sorted(noise_df["snr"].unique())

    for model in ["IForest", "LOF", "MP", "AE"]:
        m = noise_df[noise_df["model"] == model].sort_values("snr")
        snr = m["snr"].values

        if model == "IForest":
            g_c = m["separation_gap_clean_mean"].values
            g_n = m["separation_gap_corrupted_mean"].values
        elif model == "LOF":
            g_c = m["kdist_gap_clean_mean"].values
            g_n = m["kdist_gap_corrupted_mean"].values
        elif model == "MP":
            g_c = m["nn_dist_gap_clean_mean"].values
            g_n = m["nn_dist_gap_corrupted_mean"].values
        elif model == "AE":
            ea_c = m["recon_error_anomaly_clean_mean"].values
            en_c = m["recon_error_normal_clean_mean"].values
            ea_n = m["recon_error_anomaly_corrupted_mean"].values
            en_n = m["recon_error_normal_corrupted_mean"].values
            g_c = ea_c - en_c
            g_n = ea_n - en_n

        retention = g_n / g_c * 100
        ax_gap.plot(snr, retention, marker=MARKERS[model], color=COLORS[model],
                    linewidth=2.5, markersize=8, label=model, zorder=3)

    ax_gap.axhline(y=100, color="gray", linestyle="--", alpha=0.4)
    ax_gap.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax_gap.axhspan(-50, 0, alpha=0.06, color="red")
    ax_gap.set_xlabel("SNR (dB)")
    ax_gap.set_ylabel("Gap Retention (%)")
    ax_gap.set_title("(a) Internal Separation Gap Retention Under White Noise",
                     fontsize=14, fontweight="bold")
    ax_gap.legend(fontsize=11, loc="center left")
    ax_gap.grid(True, alpha=0.2)
    ax_gap.invert_xaxis()
    ax_gap.set_ylim(-50, 115)

    # ── Row 2: 4-panel mechanism ──
    configs = [
        ("IForest", "path_anomaly", "path_normal", "Path Length"),
        ("LOF", "kdist_anomaly", "kdist_normal", "k-Distance"),
        ("MP", "nn_dist_anomaly", "nn_dist_normal", "NN Distance"),
        ("AE", "recon_error_anomaly", "recon_error_normal", "Recon. Error"),
    ]

    for idx, (model, anom_p, norm_p, ylabel) in enumerate(configs):
        ax = fig.add_subplot(gs[1, idx])
        m = noise_df[noise_df["model"] == model].sort_values("snr")
        snr = m["snr"].values

        ax.axhline(y=m[f"{anom_p}_clean_mean"].values[0], color="#D62728",
                   linestyle=":", alpha=0.4, linewidth=2.0, zorder=5)
        ax.axhline(y=m[f"{norm_p}_clean_mean"].values[0], color="#1F77B4",
                   linestyle=":", alpha=0.4, linewidth=2.0, zorder=5)

        ax.plot(snr, m[f"{anom_p}_corrupted_mean"].values, "o-",
                color="#D62728", linewidth=2, markersize=5, label="Anomaly")
        ax.plot(snr, m[f"{norm_p}_corrupted_mean"].values, "s-",
                color="#1F77B4", linewidth=2, markersize=5, label="Normal")

        ax.fill_between(snr, m[f"{anom_p}_corrupted_mean"].values,
                        m[f"{norm_p}_corrupted_mean"].values, alpha=0.1, color="green")

        ax.set_title(model, fontsize=13, fontweight="bold", color=COLORS[model])
        ax.set_xlabel("SNR (dB)")
        if idx == 0:
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.invert_xaxis()

    # ── Row 3: Asymmetric inflation at 0dB ──
    ax_bar = fig.add_subplot(gs[2, :])
    snr0 = noise_df[noise_df["snr"] == 0]
    models_list = ["IForest", "LOF", "MP", "AE"]
    labels_map = {"IForest": "Path Length", "LOF": "k-Distance",
                  "MP": "NN Distance", "AE": "Recon. Error"}

    anom_inf, norm_inf = [], []
    for model in models_list:
        m = snr0[snr0["model"] == model]
        if model == "IForest":
            ac, an = m["path_anomaly_clean_mean"].values[0], m["path_anomaly_corrupted_mean"].values[0]
            nc, nn_ = m["path_normal_clean_mean"].values[0], m["path_normal_corrupted_mean"].values[0]
        elif model == "LOF":
            ac, an = m["kdist_anomaly_clean_mean"].values[0], m["kdist_anomaly_corrupted_mean"].values[0]
            nc, nn_ = m["kdist_normal_clean_mean"].values[0], m["kdist_normal_corrupted_mean"].values[0]
        elif model == "MP":
            ac, an = m["nn_dist_anomaly_clean_mean"].values[0], m["nn_dist_anomaly_corrupted_mean"].values[0]
            nc, nn_ = m["nn_dist_normal_clean_mean"].values[0], m["nn_dist_normal_corrupted_mean"].values[0]
        elif model == "AE":
            ac, an = m["recon_error_anomaly_clean_mean"].values[0], m["recon_error_anomaly_corrupted_mean"].values[0]
            nc, nn_ = m["recon_error_normal_clean_mean"].values[0], m["recon_error_normal_corrupted_mean"].values[0]
        anom_inf.append((an - ac) / ac * 100)
        norm_inf.append((nn_ - nc) / nc * 100)

    x = np.arange(len(models_list))
    width = 0.3
    bars1 = ax_bar.bar(x - width/2, norm_inf, width, color="#1F77B4", alpha=0.8, label="Normal windows")
    bars2 = ax_bar.bar(x + width/2, anom_inf, width, color="#D62728", alpha=0.8, label="Anomaly windows")

    for bar, val in zip(bars1, norm_inf):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"{val:+.0f}%", ha="center", fontsize=10, fontweight="bold", color="#1F77B4")
    for bar, val in zip(bars2, anom_inf):
        y_pos = bar.get_height() + 5 if val >= 0 else bar.get_height() - 15
        ax_bar.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f"{val:+.0f}%", ha="center", fontsize=10, fontweight="bold", color="#D62728")

    ax_bar.axhline(y=0, color="black", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{m}\n({labels_map[m]})" for m in models_list], fontsize=11)
    ax_bar.set_ylabel("Change from Clean (%)")
    ax_bar.set_title("(c) Asymmetric Impact at 0 dB: Normal Inflate More Than Anomaly",
                     fontsize=14, fontweight="bold")
    ax_bar.legend(fontsize=11)
    ax_bar.grid(True, alpha=0.2, axis="y")

    out = PLOTS_DIR / "thesis_gap_collapse_combined.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    print("Plotting Gap Collapse Mechanism")
    print(f"Output: {PLOTS_DIR}\n")

    noise_df = load_noise_internals()

    plot_gap_collapse_unified(noise_df)
    plot_mechanism_4panel(noise_df)
    plot_distance_inflation_comparison(noise_df)
    plot_thesis_combined(noise_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
