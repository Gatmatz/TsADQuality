"""Plot missing_positional experiment results for IForest.

Positional targeting: how does the LOCATION of missing data matter?
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_positional")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}

TARGET_COLORS = {
    'near_anomaly': '#4daf4a',
    'overlapping_anomaly': '#e41a1c',
    'far_from_anomaly': '#377eb8',
}
TARGET_MARKERS = {
    'near_anomaly': '^',
    'overlapping_anomaly': 'X',
    'far_from_anomaly': 'o',
}
TARGET_LABELS = {
    'near_anomaly': 'Near Anomaly',
    'overlapping_anomaly': 'Overlapping Anomaly',
    'far_from_anomaly': 'Far from Anomaly',
}


def load_data():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]
    return df


def load_baseline():
    return pd.read_csv(BASELINE_CSV)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data()
    baseline = load_baseline()
    baseline_auc = baseline['AUC_ROC'].mean()

    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))

    print(f"Total results: {len(df)} rows")
    print(f"Baseline AUC-ROC: {baseline_auc:.4f}")
    print()

    # Summary table
    print(f"{'target':>22} {'hdl':>8} {'frac':>6} {'n':>5} {'lost':>8} {'AUC-ROC':>8} {'VUS-ROC':>8}")
    print("-" * 75)
    for _, row in summary.sort_values(['target', 'handling', 'fraction']).iterrows():
        print(f"{row['target']:>22} {row['handling']:>8} {row['fraction']:>6.2f} "
              f"{int(row['n_runs']):>5} {row['mean_n_lost_anomalies']:>8.1f} "
              f"{row['mean_AUC_ROC']:>8.4f} {row['mean_VUS_ROC']:>8.4f}")
    print()

    targets = sorted(df['target'].unique())
    handlings = sorted(df['handling'].unique())

    # ================================================================
    # Plot 1: AUC-ROC vs fraction — one line per target
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for tgt in targets:
        sub = df[df['target'] == tgt]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=TARGET_MARKERS.get(tgt, 'o'),
                color=TARGET_COLORS.get(tgt, 'gray'),
                label=TARGET_LABELS.get(tgt, tgt),
                linewidth=2, markersize=8)

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0,
               alpha=0.85, zorder=5, label=f'Baseline ({baseline_auc:.3f})')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, zorder=5, alpha=0.4)
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Positional Impact of Missing Data', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'positional_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: positional_AUC_ROC.png")

    # ================================================================
    # Plot 2: Multi-metric (AUC-ROC, VUS-ROC, AUC-PR)
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, metric in zip(axes, METRICS):
        for tgt in targets:
            sub = df[df['target'] == tgt]
            means = sub.groupby('fraction')[metric].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means[metric],
                    marker=TARGET_MARKERS.get(tgt, 'o'),
                    color=TARGET_COLORS.get(tgt, 'gray'),
                    label=TARGET_LABELS.get(tgt, tgt),
                    linewidth=2, markersize=7)

        bl_val = baseline[metric].mean()
        ax.axhline(y=bl_val, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
        ax.set_xlabel('Fraction', fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(True, alpha=0.3)
        if ax == axes[-1]:
            ax.legend(fontsize=8, loc='best')

    fig.suptitle('IForest: Positional Impact — Multiple Metrics', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'positional_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: positional_multi_metric.png")

    # ================================================================
    # Plot 3: Heatmap — target × fraction
    # ================================================================
    fractions = sorted(df['fraction'].unique())
    heatmap_targets = targets

    matrix = np.full((len(heatmap_targets), len(fractions)), np.nan)

    for i, tgt in enumerate(heatmap_targets):
        for j, frac in enumerate(fractions):
            sub = df[(df['target'] == tgt) & (df['fraction'] == frac)]
            if not sub.empty:
                matrix[i, j] = sub['AUC_ROC'].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                   vmin=0.0, vmax=baseline_auc + 0.05)
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_yticks(range(len(heatmap_targets)))
    ax.set_yticklabels([TARGET_LABELS.get(t, t) for t in heatmap_targets])
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_title('IForest: Mean AUC-ROC by Position', fontsize=13)
    for i in range(len(heatmap_targets)):
        for j in range(len(fractions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.4 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=11, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'positional_heatmap.png'), dpi=300)
    plt.close(fig)
    print("Saved: positional_heatmap.png")

    # ================================================================
    # Plot 4: Lost anomalies per target
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for tgt in targets:
        sub = df[df['target'] == tgt]
        means = sub.groupby('fraction')['n_lost_anomalies'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['n_lost_anomalies'],
                marker=TARGET_MARKERS.get(tgt, 'o'),
                color=TARGET_COLORS.get(tgt, 'gray'),
                label=TARGET_LABELS.get(tgt, tgt),
                linewidth=2, markersize=8)

    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('Mean Lost Anomalies', fontsize=12)
    ax.set_title('Anomalies Lost by Position Target', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'positional_lost_anomalies.png'), dpi=300)
    plt.close(fig)
    print("Saved: positional_lost_anomalies.png")

    # ================================================================
    # Plot 5: AUC-ROC vs lost anomalies (scatter)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for tgt in targets:
        sub = df[df['target'] == tgt]
        ax.scatter(sub['n_lost_anomalies'], sub['AUC_ROC'],
                   alpha=0.4, s=25,
                   color=TARGET_COLORS.get(tgt, 'gray'),
                   label=TARGET_LABELS.get(tgt, tgt))

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, zorder=5, alpha=0.4)
    ax.set_xlabel('Number of Lost Anomalies', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: AUC-ROC vs Lost Anomalies by Position', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'positional_auc_vs_lost.png'), dpi=300)
    plt.close(fig)
    print("Saved: positional_auc_vs_lost.png")

    # ================================================================
    # Plot 6: Bar chart at fraction=20% — side by side
    # ================================================================
    f20 = summary[summary['fraction'] == 0.20]

    if not f20.empty:
        fig, ax1 = plt.subplots(figsize=(8, 5))

        bar_targets = []
        bar_auc = []
        bar_lost = []
        bar_colors = []

        for tgt in targets:
            row = f20[f20['target'] == tgt]
            if not row.empty:
                bar_targets.append(TARGET_LABELS.get(tgt, tgt))
                bar_auc.append(row['mean_AUC_ROC'].values[0])
                bar_lost.append(row['mean_n_lost_anomalies'].values[0])
                bar_colors.append(TARGET_COLORS.get(tgt, 'gray'))

        x = np.arange(len(bar_targets))
        bars = ax1.bar(x, bar_auc, 0.5, color=bar_colors, alpha=0.8)
        ax1.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5,
                    label=f'Baseline ({baseline_auc:.3f})')
        ax1.set_ylabel('AUC-ROC', fontsize=12)
        ax1.set_ylim(0, max(bar_auc) * 1.15 if bar_auc else 1)

        for bar, val in zip(bars, bar_auc):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax2 = ax1.twinx()
        ax2.plot(x, bar_lost, marker='D', color='black', linewidth=2, markersize=8,
                 label='Lost anomalies')
        ax2.set_ylabel('Mean Lost Anomalies', fontsize=12)

        ax1.set_xticks(x)
        ax1.set_xticklabels(bar_targets, fontsize=11)
        ax1.set_title('IForest: Positional Impact at Fraction=20%', fontsize=13)
        ax1.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, 'positional_bar_f20.png'), dpi=300)
        plt.close(fig)
        print("Saved: positional_bar_f20.png")

    # ================================================================
    # Plot 7: Degradation from baseline (% change)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for tgt in targets:
        sub = df[df['target'] == tgt]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        pct_change = ((means['AUC_ROC'] - baseline_auc) / baseline_auc) * 100
        ax.plot(means['fraction'], pct_change,
                marker=TARGET_MARKERS.get(tgt, 'o'),
                color=TARGET_COLORS.get(tgt, 'gray'),
                label=TARGET_LABELS.get(tgt, tgt),
                linewidth=2, markersize=8)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('AUC-ROC Change from Baseline (%)', fontsize=12)
    ax.set_title('IForest: Degradation by Position (% change)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'positional_degradation_pct.png'), dpi=300)
    plt.close(fig)
    print("Saved: positional_degradation_pct.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
