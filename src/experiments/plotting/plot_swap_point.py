"""Plot swap point experiment results."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "swap_point")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]
    baseline = pd.read_csv(BASELINE_CSV)
    baseline_auc = baseline['AUC_ROC'].mean()

    print(f"Baseline: {baseline_auc:.4f}, {len(df)} rows")

    # Plot 1: AUC-ROC vs fraction (line plot like other experiments)
    fig, ax = plt.subplots(figsize=(8, 5))
    means = df.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
    ax.plot(means['fraction'], means['AUC_ROC'],
            marker='o', color='#e41a1c', linewidth=2, markersize=7)
    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5,
               label=f'Baseline ({baseline_auc:.3f})')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xlabel('Fraction of Swapped Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Point Swap — AUC-ROC', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'swap_point_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: swap_point_AUC_ROC.png")

    # Plot 2: Multi-metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, METRICS):
        m = df.groupby('fraction')[metric].mean().reset_index().sort_values('fraction')
        ax.plot(m['fraction'], m[metric], marker='o', linewidth=2, markersize=7, color='#e41a1c')
        bl = baseline[metric].mean()
        ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
        ax.set_xlabel('Fraction', fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(True, alpha=0.3)
    fig.suptitle('IForest: Point Swap — Multiple Metrics', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'swap_point_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: swap_point_multi_metric.png")

    # Plot 3: Degradation %
    fig, ax = plt.subplots(figsize=(8, 5))
    means = df.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
    pct = ((means['AUC_ROC'] - baseline_auc) / baseline_auc) * 100
    ax.plot(means['fraction'], pct, marker='o', color='#e41a1c', linewidth=2, markersize=7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xlabel('Fraction of Swapped Points', fontsize=12)
    ax.set_ylabel('AUC-ROC Change from Baseline (%)', fontsize=12)
    ax.set_title('IForest: Point Swap — Degradation', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'swap_point_degradation.png'), dpi=300)
    plt.close(fig)
    print("Saved: swap_point_degradation.png")

    # Plot 4: High vs low baseline
    high_files = set(baseline[baseline['AUC_ROC'] >= 0.5]['file'].tolist())
    low_files = set(baseline[baseline['AUC_ROC'] < 0.5]['file'].tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, files, title in [
        (ax1, high_files, f'{len(high_files)} files (baseline $\\geq$ 0.5)'),
        (ax2, low_files, f'{len(low_files)} files (baseline < 0.5)'),
    ]:
        sub = df[df['file'].isin(files)]
        m = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(m['fraction'], m['AUC_ROC'], marker='o', linewidth=2, markersize=7, color='#e41a1c')
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, alpha=0.85, zorder=5)
        ax.set_xlabel('Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(True, alpha=0.3)
    fig.suptitle('IForest: Point Swap — High vs Low Baseline', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'swap_point_high_vs_low.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: swap_point_high_vs_low.png")

    # Plot 5: Inverted detection rate
    baseline_inv = (baseline['AUC_ROC'] < 0.5).mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    inv = df.groupby('fraction')['AUC_ROC'].apply(
        lambda g: (g < 0.5).mean() * 100
    ).reset_index(name='pct_inverted').sort_values('fraction')
    ax.plot(inv['fraction'], inv['pct_inverted'],
            marker='o', color='#e41a1c', linewidth=2, markersize=7)
    ax.axhline(y=baseline_inv, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5,
               label=f'Baseline ({baseline_inv:.0f}%)')
    ax.set_xlabel('Fraction of Swapped Points', fontsize=12)
    ax.set_ylabel('Files with AUC-ROC < 0.5 (%)', fontsize=12)
    ax.set_title('IForest: Point Swap — Inverted Detection Rate', fontsize=13)
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'swap_point_inverted_rate.png'), dpi=300)
    plt.close(fig)
    print("Saved: swap_point_inverted_rate.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
