"""Plot freeze (sensor stuck) experiment results for IForest.

Uses dynamic num_stucks with stuck_length = fraction * n / num_stucks.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "freeze")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}

NS_COLORS = {1: '#e41a1c', 3: '#4daf4a', 5: '#984ea3', 10: '#ff7f00'}
NS_MARKERS = {1: 's', 3: '^', 5: 'D', 10: 'v'}


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
    baseline_auc_corr = baseline['AUC_ROC'].apply(lambda x: max(x, 1 - x)).mean()

    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))

    print(f"Total results: {len(df)} rows")
    print(f"Baseline AUC-ROC: {baseline_auc:.4f}")
    print()

    # Summary table
    print(f"{'frac':>6} {'ns':>4} {'mean_sl':>8} {'n':>5} {'AUC-ROC':>8} {'VUS-ROC':>8} {'AUC-PR':>8}")
    print("-" * 55)
    for _, row in summary.sort_values(['num_stucks', 'fraction']).iterrows():
        print(f"{row['fraction']:>6.2f} {int(row['num_stucks']):>4} {row['mean_stuck_length']:>8.0f} "
              f"{int(row['n_runs']):>5} {row['mean_AUC_ROC']:>8.4f} {row['mean_VUS_ROC']:>8.4f} {row['mean_AUC_PR']:>8.4f}")
    print()

    # ================================================================
    # Plot 1: AUC-ROC vs fraction — one line per num_stucks
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for ns in sorted(df['num_stucks'].unique()):
        ns = int(ns)
        sub = df[df['num_stucks'] == ns]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=NS_MARKERS.get(ns, 'o'), color=NS_COLORS.get(ns, 'gray'),
                label=f'ns={ns}', linewidth=2, markersize=8)

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0,
               alpha=0.85, zorder=5, label=f'Baseline ({baseline_auc:.3f})')
    ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Freeze (Sensor Stuck) — AUC-ROC vs Fraction', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'freeze_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: freeze_AUC_ROC.png")

    # ================================================================
    # Plot 2: Multi-metric (AUC-ROC, VUS-ROC, AUC-PR)
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, metric in zip(axes, METRICS):
        for ns in sorted(df['num_stucks'].unique()):
            ns = int(ns)
            sub = df[df['num_stucks'] == ns]
            means = sub.groupby('fraction')[metric].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means[metric],
                    marker=NS_MARKERS.get(ns, 'o'), color=NS_COLORS.get(ns, 'gray'),
                    label=f'ns={ns}', linewidth=2, markersize=6)

        bl_val = baseline[metric].mean()
        ax.axhline(y=bl_val, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
        ax.set_xlabel('Fraction', fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(True, alpha=0.3)
        if ax == axes[-1]:
            ax.legend(fontsize=8, loc='best')

    fig.suptitle('IForest: Freeze — Multiple Metrics', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'freeze_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: freeze_multi_metric.png")

    # ================================================================
    # Plot 3: Heatmap — num_stucks × fraction
    # ================================================================
    fractions = sorted(df['fraction'].unique())
    ns_list = sorted(df['num_stucks'].unique())
    row_labels = [f'ns={int(ns)}' for ns in ns_list]

    matrix = np.full((len(ns_list), len(fractions)), np.nan)
    for i, ns in enumerate(ns_list):
        for j, frac in enumerate(fractions):
            sub = df[(df['num_stucks'] == ns) & (df['fraction'] == frac)]
            if not sub.empty:
                matrix[i, j] = sub['AUC_ROC'].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax.set_title('IForest: Mean AUC-ROC — Freeze', fontsize=13)
    for i in range(len(ns_list)):
        for j in range(len(fractions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'freeze_heatmap.png'), dpi=300)
    plt.close(fig)
    print("Saved: freeze_heatmap.png")

    # ================================================================
    # Plot 4: High vs Low baseline
    # ================================================================
    high_files = set(baseline[baseline['AUC_ROC'] >= 0.5]['file'].tolist())
    low_files = set(baseline[baseline['AUC_ROC'] < 0.5]['file'].tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, files, title in [
        (ax1, high_files, f'{len(high_files)} files (baseline AUC-ROC $\\geq$ 0.5)'),
        (ax2, low_files, f'{len(low_files)} files (baseline AUC-ROC < 0.5)'),
    ]:
        sub_df = df[df['file'].isin(files)]
        for ns in sorted(sub_df['num_stucks'].unique()):
            ns = int(ns)
            sub = sub_df[sub_df['num_stucks'] == ns]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=NS_MARKERS.get(ns, 'o'), color=NS_COLORS.get(ns, 'gray'),
                    label=f'ns={ns}', linewidth=2, markersize=6)

        ax.set_xlabel('Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, zorder=5, alpha=0.7)
    fig.suptitle('IForest: Freeze — High vs Low Baseline', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'freeze_high_vs_low.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: freeze_high_vs_low.png")

    # ================================================================
    # Plot 5: Standard vs Corrected AUC-ROC
    # ================================================================
    df_corr = df.copy()
    df_corr['AUC_ROC_corr'] = df_corr['AUC_ROC'].apply(lambda x: max(x, 1 - x))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, base_val, title in [
        (ax1, 'AUC_ROC', baseline_auc, 'Standard AUC-ROC'),
        (ax2, 'AUC_ROC_corr', baseline_auc_corr, 'Corrected max(AUC, 1-AUC)'),
    ]:
        for ns in sorted(df_corr['num_stucks'].unique()):
            ns = int(ns)
            sub = df_corr[df_corr['num_stucks'] == ns]
            means = sub.groupby('fraction')[col].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means[col],
                    marker=NS_MARKERS.get(ns, 'o'), color=NS_COLORS.get(ns, 'gray'),
                    label=f'ns={ns}', linewidth=2, markersize=6)

        ax.axhline(y=base_val, color='black', linestyle='--', linewidth=2.0,
                   alpha=0.85, zorder=5, label=f'Baseline ({base_val:.3f})')
        ax.set_xlabel('Fraction', fontsize=12)
        ax.set_ylabel(col.replace('_', '-'), fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('IForest: Standard vs Corrected AUC-ROC — Freeze', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'freeze_corrected_auc.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: freeze_corrected_auc.png")

    # ================================================================
    # Plot 6: Mean stuck_length per condition
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for ns in sorted(df['num_stucks'].unique()):
        ns = int(ns)
        sub = df[df['num_stucks'] == ns]
        means = sub.groupby('fraction')['stuck_length'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['stuck_length'],
                marker=NS_MARKERS.get(ns, 'o'), color=NS_COLORS.get(ns, 'gray'),
                label=f'ns={ns}', linewidth=2, markersize=8)

    ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax.set_ylabel('Mean Stuck Length', fontsize=12)
    ax.set_title('Dynamic Stuck Length per Condition', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'freeze_stuck_lengths.png'), dpi=300)
    plt.close(fig)
    print("Saved: freeze_stuck_lengths.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
