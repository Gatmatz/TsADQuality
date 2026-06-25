"""Plot missing_masking experiment results for IForest.

Masking approach: remove missing points entirely instead of imputation.
Uses num_bursts (dynamic burst_length) parameter grid.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_masking")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}

NB_COLORS = {0: '#377eb8', 1: '#e41a1c', 3: '#4daf4a', 5: '#984ea3', 10: '#ff7f00'}
NB_MARKERS = {0: 'o', 1: 's', 3: '^', 5: 'D', 10: 'v'}


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

    df_point = df[df['missing_type'] == 'point']
    df_burst = df[df['missing_type'] == 'burst']

    print(f"Total results: {len(df)} rows")
    print(f"Point: {len(df_point)}, Burst: {len(df_burst)}")
    print(f"Baseline AUC-ROC: {baseline_auc:.4f}")
    print()

    # --- Summary table ---
    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    print(f"{'type':>6} {'frac':>6} {'nb':>4} {'mean_bl':>8} {'n':>5} {'AUC-ROC':>8} {'VUS-ROC':>8} {'AUC-PR':>8}")
    print("-" * 65)
    for _, row in summary.sort_values(['missing_type', 'num_bursts', 'fraction']).iterrows():
        nb = int(row['num_bursts']) if row['missing_type'] == 'burst' else '-'
        mbl = f"{row['mean_burst_length']:.0f}" if row['missing_type'] == 'burst' else '-'
        print(f"{row['missing_type']:>6} {row['fraction']:>6.2f} {str(nb):>4} {mbl:>8} "
              f"{int(row['n_runs']):>5} {row['mean_AUC_ROC']:>8.4f} {row['mean_VUS_ROC']:>8.4f} {row['mean_AUC_PR']:>8.4f}")
    print()

    # ================================================================
    # Plot 1: AUC-ROC vs fraction — point + burst (per num_bursts)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    means_pt = df_point.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
    ax.plot(means_pt['fraction'], means_pt['AUC_ROC'],
            marker='o', color=NB_COLORS[0], label='Point', linewidth=2, markersize=8)

    for nb in sorted(df_burst['num_bursts'].unique()):
        nb = int(nb)
        sub = df_burst[df_burst['num_bursts'] == nb]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=NB_MARKERS.get(nb, 'x'), color=NB_COLORS.get(nb, 'gray'),
                label=f'Burst (nb={nb})', linewidth=2, markersize=7)

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0,
               alpha=0.85, zorder=5, label=f'Baseline ({baseline_auc:.3f})')
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Missing (Masking) — AUC-ROC vs Fraction', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'masking_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: masking_AUC_ROC.png")

    # ================================================================
    # Plot 2: Multi-metric (AUC-ROC, VUS-ROC, AUC-PR) — point + burst
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, metric in zip(axes, METRICS):
        means_pt = df_point.groupby('fraction')[metric].mean().reset_index().sort_values('fraction')
        ax.plot(means_pt['fraction'], means_pt[metric],
                marker='o', color=NB_COLORS[0], label='Point', linewidth=2, markersize=7)

        for nb in sorted(df_burst['num_bursts'].unique()):
            nb = int(nb)
            sub = df_burst[df_burst['num_bursts'] == nb]
            means = sub.groupby('fraction')[metric].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means[metric],
                    marker=NB_MARKERS.get(nb, 'x'), color=NB_COLORS.get(nb, 'gray'),
                    label=f'Burst (nb={nb})', linewidth=2, markersize=6)

        bl_val = baseline[metric].mean()
        ax.axhline(y=bl_val, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
        ax.set_xlabel('Fraction', fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(True, alpha=0.3)
        if ax == axes[-1]:
            ax.legend(fontsize=8, loc='best')

    fig.suptitle('IForest: Missing (Masking) — Multiple Metrics', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'masking_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: masking_multi_metric.png")

    # ================================================================
    # Plot 3: Heatmap — num_bursts × fraction
    # ================================================================
    fractions = sorted(df['fraction'].unique())
    num_bursts_list = sorted(df_burst['num_bursts'].unique())
    row_labels = ['Point'] + [f'Burst (nb={int(nb)})' for nb in num_bursts_list]

    matrix = np.full((len(row_labels), len(fractions)), np.nan)

    for j, frac in enumerate(fractions):
        sub = df_point[df_point['fraction'] == frac]
        if not sub.empty:
            matrix[0, j] = sub['AUC_ROC'].mean()

    for i, nb in enumerate(num_bursts_list):
        for j, frac in enumerate(fractions):
            sub = df_burst[(df_burst['num_bursts'] == nb) & (df_burst['fraction'] == frac)]
            if not sub.empty:
                matrix[i + 1, j] = sub['AUC_ROC'].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_title('IForest: Mean AUC-ROC — Missing (Masking)', fontsize=13)
    for i in range(len(row_labels)):
        for j in range(len(fractions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.65 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'masking_heatmap.png'), dpi=300)
    plt.close(fig)
    print("Saved: masking_heatmap.png")

    # ================================================================
    # Plot 4: High vs Low baseline split
    # ================================================================
    high_files = set(baseline[baseline['AUC_ROC'] >= 0.5]['file'].tolist())
    low_files = set(baseline[baseline['AUC_ROC'] < 0.5]['file'].tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, files, title in [
        (ax1, high_files, f'{len(high_files)} files (baseline AUC-ROC $\\geq$ 0.5)'),
        (ax2, low_files, f'{len(low_files)} files (baseline AUC-ROC < 0.5)'),
    ]:
        sub_point = df_point[df_point['file'].isin(files)]
        sub_burst = df_burst[df_burst['file'].isin(files)]

        means_pt = sub_point.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        if not means_pt.empty:
            ax.plot(means_pt['fraction'], means_pt['AUC_ROC'],
                    marker='o', color=NB_COLORS[0], label='Point', linewidth=2, markersize=7)

        for nb in sorted(sub_burst['num_bursts'].unique()):
            nb = int(nb)
            bl_data = sub_burst[sub_burst['num_bursts'] == nb]
            means = bl_data.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=NB_MARKERS.get(nb, 'x'), color=NB_COLORS.get(nb, 'gray'),
                    label=f'Burst (nb={nb})', linewidth=2, markersize=6)

        ax.set_xlabel('Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, zorder=5, alpha=0.7)
    fig.suptitle('IForest: Missing (Masking) — High vs Low Baseline', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'masking_high_vs_low.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: masking_high_vs_low.png")

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
        point_sub = df_corr[df_corr['missing_type'] == 'point']
        burst_sub = df_corr[df_corr['missing_type'] == 'burst']

        means_pt = point_sub.groupby('fraction')[col].mean().reset_index().sort_values('fraction')
        ax.plot(means_pt['fraction'], means_pt[col],
                marker='o', color=NB_COLORS[0], label='Point', linewidth=2, markersize=7)

        for nb in sorted(burst_sub['num_bursts'].unique()):
            nb = int(nb)
            sub = burst_sub[burst_sub['num_bursts'] == nb]
            means = sub.groupby('fraction')[col].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means[col],
                    marker=NB_MARKERS.get(nb, 'x'), color=NB_COLORS.get(nb, 'gray'),
                    label=f'Burst (nb={nb})', linewidth=2, markersize=6)

        ax.axhline(y=base_val, color='black', linestyle='--', linewidth=2.0,
                   alpha=0.85, zorder=5, label=f'Baseline ({base_val:.3f})')
        ax.set_xlabel('Fraction', fontsize=12)
        ax.set_ylabel(col.replace('_', '-'), fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('IForest: Standard vs Corrected AUC-ROC — Missing (Masking)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'masking_corrected_auc.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: masking_corrected_auc.png")

    # ================================================================
    # Plot 6: Skipped files — how many files per condition
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    means_pt = df_point.groupby('fraction').size().reset_index(name='n_files').sort_values('fraction')
    ax.plot(means_pt['fraction'], means_pt['n_files'],
            marker='o', color=NB_COLORS[0], label='Point', linewidth=2, markersize=8)

    for nb in sorted(df_burst['num_bursts'].unique()):
        nb = int(nb)
        sub = df_burst[df_burst['num_bursts'] == nb]
        counts = sub.groupby('fraction').size().reset_index(name='n_files').sort_values('fraction')
        ax.plot(counts['fraction'], counts['n_files'],
                marker=NB_MARKERS.get(nb, 'x'), color=NB_COLORS.get(nb, 'gray'),
                label=f'Burst (nb={nb})', linewidth=2, markersize=7)

    ax.axhline(y=141, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5, label='Total (141)')
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('Number of Files Evaluated', fontsize=12)
    ax.set_title('IForest: Files per Condition (Masking)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'masking_files_per_condition.png'), dpi=300)
    plt.close(fig)
    print("Saved: masking_files_per_condition.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
