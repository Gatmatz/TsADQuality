"""Plot swap permutation experiment results for IForest.

Literature approach: split series into N segments and shuffle order.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "swap_permutation")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}


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
    print(f"{'n_seg':>6} {'seg_len':>8} {'n':>5} {'AUC-ROC':>8} {'VUS-ROC':>8} {'AUC-PR':>8}")
    print("-" * 50)
    for _, row in summary.sort_values('n_segments').iterrows():
        print(f"{int(row['n_segments']):>6} {row['mean_segment_length']:>8.0f} "
              f"{int(row['n_runs']):>5} {row['mean_AUC_ROC']:>8.4f} "
              f"{row['mean_VUS_ROC']:>8.4f} {row['mean_AUC_PR']:>8.4f}")
    print()

    n_segments = sorted(df['n_segments'].unique())

    # ================================================================
    # Plot 1: AUC-ROC vs n_segments (bar chart)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    means = df.groupby('n_segments')['AUC_ROC'].mean().reset_index().sort_values('n_segments')
    stds = df.groupby('n_segments')['AUC_ROC'].std().reset_index().sort_values('n_segments')

    x = np.arange(len(means))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(means)))

    bars = ax.bar(x, means['AUC_ROC'], yerr=stds['AUC_ROC'], capsize=4,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0,
               alpha=0.85, zorder=5, label=f'Baseline ({baseline_auc:.3f})')

    for bar, val in zip(bars, means['AUC_ROC']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'N={int(n)}' for n in means['n_segments']], fontsize=11)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Segment Permutation — AUC-ROC vs N Segments', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'permutation_AUC_ROC_bar.png'), dpi=300)
    plt.close(fig)
    print("Saved: permutation_AUC_ROC_bar.png")

    # ================================================================
    # Plot 2: AUC-ROC vs n_segments (line)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(means['n_segments'], means['AUC_ROC'],
            marker='o', color='#e41a1c', linewidth=2, markersize=8)
    ax.fill_between(means['n_segments'],
                    means['AUC_ROC'] - stds['AUC_ROC'],
                    means['AUC_ROC'] + stds['AUC_ROC'],
                    alpha=0.2, color='#e41a1c')

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0,
               alpha=0.85, zorder=5, label=f'Baseline ({baseline_auc:.3f})')
    ax.set_xlabel('Number of Segments', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Segment Permutation — AUC-ROC Degradation', fontsize=13)
    ax.set_xticks(means['n_segments'].astype(int))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'permutation_AUC_ROC_line.png'), dpi=300)
    plt.close(fig)
    print("Saved: permutation_AUC_ROC_line.png")

    # ================================================================
    # Plot 3: Multi-metric
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, metric in zip(axes, METRICS):
        m = df.groupby('n_segments')[metric].mean().reset_index().sort_values('n_segments')
        s = df.groupby('n_segments')[metric].std().reset_index().sort_values('n_segments')

        ax.plot(m['n_segments'], m[metric], marker='o', color='#e41a1c',
                linewidth=2, markersize=6)
        ax.fill_between(m['n_segments'], m[metric] - s[metric],
                        m[metric] + s[metric], alpha=0.2, color='#e41a1c')

        bl_val = baseline[metric].mean()
        ax.axhline(y=bl_val, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
        ax.set_xlabel('N Segments', fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.set_xticks(m['n_segments'].astype(int))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(True, alpha=0.3)

    fig.suptitle('IForest: Segment Permutation — Multiple Metrics', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'permutation_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: permutation_multi_metric.png")

    # ================================================================
    # Plot 4: Degradation from baseline (%)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    pct_change = ((means['AUC_ROC'] - baseline_auc) / baseline_auc) * 100
    ax.bar(x, pct_change, color=['#4daf4a' if p >= 0 else '#e41a1c' for p in pct_change],
           alpha=0.8, edgecolor='black', linewidth=0.5)

    for i, (xi, val) in enumerate(zip(x, pct_change)):
        ax.text(xi, val + (0.5 if val >= 0 else -1.5),
                f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={int(n)}' for n in means['n_segments']], fontsize=11)
    ax.set_ylabel('AUC-ROC Change from Baseline (%)', fontsize=12)
    ax.set_title('IForest: Permutation — Degradation from Baseline', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'permutation_degradation_pct.png'), dpi=300)
    plt.close(fig)
    print("Saved: permutation_degradation_pct.png")

    # ================================================================
    # Plot 5: Distribution boxplot per n_segments
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    data_for_box = [df[df['n_segments'] == n]['AUC_ROC'].values for n in n_segments]
    bp = ax.boxplot(data_for_box, labels=[f'N={int(n)}' for n in n_segments],
                    patch_artist=True, showfliers=True)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(n_segments)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0,
               alpha=0.85, zorder=5, label=f'Baseline ({baseline_auc:.3f})')
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: AUC-ROC Distribution by N Segments', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'permutation_boxplot.png'), dpi=300)
    plt.close(fig)
    print("Saved: permutation_boxplot.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
