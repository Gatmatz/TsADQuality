"""Plot swap segment experiment results — dynamic swap_length."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "swap_segment")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}

NS_COLORS = {1: '#e41a1c', 3: '#377eb8', 5: '#4daf4a', 10: '#984ea3', 20: '#ff7f00'}
NS_MARKERS = {1: 's', 3: 'o', 5: '^', 10: 'D', 20: 'v'}


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]
    baseline = pd.read_csv(BASELINE_CSV)
    baseline_auc = baseline['AUC_ROC'].mean()

    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))

    print(f"Baseline: {baseline_auc:.4f}, {len(df)} rows")

    # Plot 1: AUC-ROC vs fraction per num_swaps
    fig, ax = plt.subplots(figsize=(8, 5))
    for ns in sorted(df['num_swaps'].unique()):
        sub = df[df['num_swaps'] == ns]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=NS_MARKERS.get(ns, 'o'), color=NS_COLORS.get(ns, 'gray'),
                label=f'num_swaps={ns}', linewidth=2, markersize=7)

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5,
               label=f'Baseline ({baseline_auc:.3f})')
    ax.set_xlabel('Fraction of Swapped Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Segment Swap — AUC-ROC vs Fraction', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'swap_segment_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: swap_segment_AUC_ROC.png")

    # Plot 2: Multi-metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, METRICS):
        for ns in sorted(df['num_swaps'].unique()):
            sub = df[df['num_swaps'] == ns]
            means = sub.groupby('fraction')[metric].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means[metric],
                    marker=NS_MARKERS.get(ns, 'o'), color=NS_COLORS.get(ns, 'gray'),
                    label=f'ns={ns}', linewidth=2, markersize=5)
        bl = baseline[metric].mean()
        ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
        ax.set_xlabel('Fraction', fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(True, alpha=0.3)
        if ax == axes[-1]:
            ax.legend(fontsize=8)
    fig.suptitle('IForest: Segment Swap — Multiple Metrics', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'swap_segment_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: swap_segment_multi_metric.png")

    # Plot 3: Heatmap
    fractions = sorted(df['fraction'].unique())
    ns_list = sorted(df['num_swaps'].unique())
    matrix = np.full((len(ns_list), len(fractions)), np.nan)
    n_matrix = np.full((len(ns_list), len(fractions)), 0, dtype=int)
    for i, ns in enumerate(ns_list):
        for j, frac in enumerate(fractions):
            sub = df[(df['num_swaps'] == ns) & (df['fraction'] == frac)]
            if not sub.empty:
                matrix[i, j] = sub['AUC_ROC'].mean()
                n_matrix[i, j] = len(sub)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_yticks(range(len(ns_list)))
    ax.set_yticklabels([f'ns={ns}' for ns in ns_list])
    ax.set_xlabel('Fraction', fontsize=12)
    ax.set_title('IForest: Segment Swap — Mean AUC-ROC', fontsize=13)
    for i in range(len(ns_list)):
        for j in range(len(fractions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}\n(n={n_matrix[i,j]})', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'swap_segment_heatmap.png'), dpi=300)
    plt.close(fig)
    print("Saved: swap_segment_heatmap.png")

    # Plot 4: High vs low baseline
    high_files = set(baseline[baseline['AUC_ROC'] >= 0.5]['file'].tolist())
    low_files = set(baseline[baseline['AUC_ROC'] < 0.5]['file'].tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, files, title in [
        (ax1, high_files, f'{len(high_files)} files (baseline $\\geq$ 0.5)'),
        (ax2, low_files, f'{len(low_files)} files (baseline < 0.5)'),
    ]:
        sub_df = df[df['file'].isin(files)]
        for ns in sorted(sub_df['num_swaps'].unique()):
            sub = sub_df[sub_df['num_swaps'] == ns]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=NS_MARKERS.get(ns, 'o'), color=NS_COLORS.get(ns, 'gray'),
                    label=f'ns={ns}', linewidth=2, markersize=5)
        ax.set_xlabel('Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('IForest: Segment Swap — High vs Low Baseline', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'swap_segment_high_vs_low.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: swap_segment_high_vs_low.png")

    # Plot 5: Degradation %
    fig, ax = plt.subplots(figsize=(8, 5))
    for ns in sorted(df['num_swaps'].unique()):
        sub = df[df['num_swaps'] == ns]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        pct = ((means['AUC_ROC'] - baseline_auc) / baseline_auc) * 100
        ax.plot(means['fraction'], pct,
                marker=NS_MARKERS.get(ns, 'o'), color=NS_COLORS.get(ns, 'gray'),
                label=f'ns={ns}', linewidth=2, markersize=7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xlabel('Fraction', fontsize=12)
    ax.set_ylabel('AUC-ROC Change from Baseline (%)', fontsize=12)
    ax.set_title('IForest: Segment Swap — Degradation', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'swap_segment_degradation.png'), dpi=300)
    plt.close(fig)
    print("Saved: swap_segment_degradation.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
