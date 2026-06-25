"""Plot freeze stuck length experiment results for IForest.

Stuck length as % of series length — many short freezes vs few long freezes.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "freeze_stuck_length")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}

SR_COLORS = {0.001: '#e41a1c', 0.005: '#4daf4a', 0.01: '#984ea3', 0.05: '#ff7f00'}
SR_MARKERS = {0.001: 's', 0.005: '^', 0.01: 'D', 0.05: 'v'}


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
    print(f"{'frac':>6} {'sr':>8} {'stuck_len':>10} {'n_stucks':>9} {'n':>5} {'AUC-ROC':>8}")
    print("-" * 55)
    for _, row in summary.sort_values(['stuck_ratio', 'fraction']).iterrows():
        print(f"{row['fraction']:>6.2f} {row['stuck_ratio']:>8.3f} {row['mean_stuck_length']:>10.0f} "
              f"{row['mean_num_stucks']:>9.0f} {int(row['n_runs']):>5} {row['mean_AUC_ROC']:>8.4f}")
    print()

    # ================================================================
    # Plot 1: AUC-ROC vs fraction — one line per stuck_ratio
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for sr in sorted(df['stuck_ratio'].unique()):
        sub = df[df['stuck_ratio'] == sr]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=SR_MARKERS.get(sr, 'o'), color=SR_COLORS.get(sr, 'gray'),
                label=f'stuck_ratio={sr}', linewidth=2, markersize=8)

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0,
               alpha=0.85, zorder=5, label=f'Baseline ({baseline_auc:.3f})')
    ax.set_xlabel('Fraction of Frozen Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Stuck Length Effect — AUC-ROC vs Fraction', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'stuck_length_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: stuck_length_AUC_ROC.png")

    # ================================================================
    # Plot 2: Multi-metric
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, metric in zip(axes, METRICS):
        for sr in sorted(df['stuck_ratio'].unique()):
            sub = df[df['stuck_ratio'] == sr]
            means = sub.groupby('fraction')[metric].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means[metric],
                    marker=SR_MARKERS.get(sr, 'o'), color=SR_COLORS.get(sr, 'gray'),
                    label=f'sr={sr}', linewidth=2, markersize=6)

        bl_val = baseline[metric].mean()
        ax.axhline(y=bl_val, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
        ax.set_xlabel('Fraction', fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(True, alpha=0.3)
        if ax == axes[-1]:
            ax.legend(fontsize=8, loc='best')

    fig.suptitle('IForest: Stuck Length Effect — Multiple Metrics', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'stuck_length_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: stuck_length_multi_metric.png")

    # ================================================================
    # Plot 3: Heatmap — stuck_ratio x fraction
    # ================================================================
    fractions = sorted(df['fraction'].unique())
    sr_list = sorted(df['stuck_ratio'].unique())

    matrix = np.full((len(sr_list), len(fractions)), np.nan)
    for i, sr in enumerate(sr_list):
        for j, frac in enumerate(fractions):
            sub = df[(df['stuck_ratio'] == sr) & (df['fraction'] == frac)]
            if not sub.empty:
                matrix[i, j] = sub['AUC_ROC'].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_yticks(range(len(sr_list)))
    ax.set_yticklabels([f'sr={sr}' for sr in sr_list])
    ax.set_xlabel('Fraction of Frozen Points', fontsize=12)
    ax.set_title('IForest: Mean AUC-ROC — Stuck Length Effect', fontsize=13)
    for i in range(len(sr_list)):
        for j in range(len(fractions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'stuck_length_heatmap.png'), dpi=300)
    plt.close(fig)
    print("Saved: stuck_length_heatmap.png")

    # ================================================================
    # Plot 4: Mean stuck length per condition
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for sr in sorted(df['stuck_ratio'].unique()):
        sub = df[df['stuck_ratio'] == sr]
        means = sub.groupby('fraction')['stuck_length'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['stuck_length'],
                marker=SR_MARKERS.get(sr, 'o'), color=SR_COLORS.get(sr, 'gray'),
                label=f'sr={sr}', linewidth=2, markersize=8)

    ax.set_xlabel('Fraction of Frozen Points', fontsize=12)
    ax.set_ylabel('Mean Stuck Length', fontsize=12)
    ax.set_title('Dynamic Stuck Length per Condition', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'stuck_length_lengths.png'), dpi=300)
    plt.close(fig)
    print("Saved: stuck_length_lengths.png")

    # ================================================================
    # Plot 5: High vs Low baseline
    # ================================================================
    high_files = set(baseline[baseline['AUC_ROC'] >= 0.5]['file'].tolist())
    low_files = set(baseline[baseline['AUC_ROC'] < 0.5]['file'].tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, files, title in [
        (ax1, high_files, f'{len(high_files)} files (baseline AUC-ROC $\\geq$ 0.5)'),
        (ax2, low_files, f'{len(low_files)} files (baseline AUC-ROC < 0.5)'),
    ]:
        sub_df = df[df['file'].isin(files)]
        for sr in sorted(sub_df['stuck_ratio'].unique()):
            sub = sub_df[sub_df['stuck_ratio'] == sr]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=SR_MARKERS.get(sr, 'o'), color=SR_COLORS.get(sr, 'gray'),
                    label=f'sr={sr}', linewidth=2, markersize=6)

        ax.set_xlabel('Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('IForest: Stuck Length — High vs Low Baseline', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'stuck_length_high_vs_low.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: stuck_length_high_vs_low.png")

    # ================================================================
    # Plot 6: Degradation from baseline (%)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for sr in sorted(df['stuck_ratio'].unique()):
        sub = df[df['stuck_ratio'] == sr]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        pct_change = ((means['AUC_ROC'] - baseline_auc) / baseline_auc) * 100
        ax.plot(means['fraction'], pct_change,
                marker=SR_MARKERS.get(sr, 'o'), color=SR_COLORS.get(sr, 'gray'),
                label=f'sr={sr}', linewidth=2, markersize=8)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xlabel('Fraction of Frozen Points', fontsize=12)
    ax.set_ylabel('AUC-ROC Change from Baseline (%)', fontsize=12)
    ax.set_title('IForest: Stuck Length — Degradation from Baseline', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'stuck_length_degradation_pct.png'), dpi=300)
    plt.close(fig)
    print("Saved: stuck_length_degradation_pct.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
