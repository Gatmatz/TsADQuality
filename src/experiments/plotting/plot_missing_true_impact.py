"""Plot missing_true_impact experiment results for IForest.

True impact: masking + score=0 for lost anomalies + exclude masked normals.
Compare with masking (only_normal) and imputation approaches.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_true_impact")
MASKING_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_masking")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}

NB_COLORS = {0: '#377eb8', 1: '#e41a1c', 3: '#4daf4a', 5: '#984ea3', 10: '#ff7f00'}
NB_MARKERS = {0: 'o', 1: 's', 3: '^', 5: 'D', 10: 'v'}


def load_data(results_dir):
    df = pd.read_csv(os.path.join(results_dir, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]
    return df


def load_baseline():
    return pd.read_csv(BASELINE_CSV)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data(RESULTS_DIR)
    baseline = load_baseline()
    baseline_auc = baseline['AUC_ROC'].mean()

    df_point = df[df['missing_type'] == 'point']
    df_burst = df[df['missing_type'] == 'burst']

    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))

    print(f"Total results: {len(df)} rows")
    print(f"Baseline AUC-ROC: {baseline_auc:.4f}")
    print()

    # Summary table
    print(f"{'type':>6} {'frac':>6} {'nb':>4} {'n':>5} {'lost_anom':>10} {'AUC-ROC':>8} {'VUS-ROC':>8} {'AUC-PR':>8}")
    print("-" * 70)
    for _, row in summary.sort_values(['missing_type', 'num_bursts', 'fraction']).iterrows():
        nb = int(row['num_bursts']) if row['missing_type'] == 'burst' else '-'
        print(f"{row['missing_type']:>6} {row['fraction']:>6.2f} {str(nb):>4} "
              f"{int(row['n_runs']):>5} {row['mean_n_lost_anomalies']:>10.1f} "
              f"{row['mean_AUC_ROC']:>8.4f} {row['mean_VUS_ROC']:>8.4f} {row['mean_AUC_PR']:>8.4f}")
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
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, alpha=0.85, zorder=5, label='Random (0.5)')
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: True Impact of Missing Data — AUC-ROC', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'true_impact_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: true_impact_AUC_ROC.png")

    # ================================================================
    # Plot 2: Multi-metric (AUC-ROC, VUS-ROC, AUC-PR)
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

    fig.suptitle('IForest: True Impact — Multiple Metrics', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'true_impact_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: true_impact_multi_metric.png")

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
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                   vmin=0.5, vmax=baseline_auc + 0.02)
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_title('IForest: Mean AUC-ROC — True Impact', fontsize=13)
    for i in range(len(row_labels)):
        for j in range(len(fractions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'true_impact_heatmap.png'), dpi=300)
    plt.close(fig)
    print("Saved: true_impact_heatmap.png")

    # ================================================================
    # Plot 4: Lost anomalies vs fraction
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    means_pt = df_point.groupby('fraction')['n_lost_anomalies'].mean().reset_index().sort_values('fraction')
    ax.plot(means_pt['fraction'], means_pt['n_lost_anomalies'],
            marker='o', color=NB_COLORS[0], label='Point', linewidth=2, markersize=8)

    for nb in sorted(df_burst['num_bursts'].unique()):
        nb = int(nb)
        sub = df_burst[df_burst['num_bursts'] == nb]
        means = sub.groupby('fraction')['n_lost_anomalies'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['n_lost_anomalies'],
                marker=NB_MARKERS.get(nb, 'x'), color=NB_COLORS.get(nb, 'gray'),
                label=f'Burst (nb={nb})', linewidth=2, markersize=7)

    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('Mean Lost Anomalies', fontsize=12)
    ax.set_title('IForest: Anomalies Lost per Condition', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'true_impact_lost_anomalies.png'), dpi=300)
    plt.close(fig)
    print("Saved: true_impact_lost_anomalies.png")

    # ================================================================
    # Plot 5: AUC-ROC vs lost anomalies (scatter)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df_point['n_lost_anomalies'], df_point['AUC_ROC'],
               alpha=0.3, s=20, color=NB_COLORS[0], label='Point')
    ax.scatter(df_burst['n_lost_anomalies'], df_burst['AUC_ROC'],
               alpha=0.3, s=20, color=NB_COLORS[1], label='Burst')

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xlabel('Number of Lost Anomalies', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: AUC-ROC vs Lost Anomalies', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'true_impact_auc_vs_lost.png'), dpi=300)
    plt.close(fig)
    print("Saved: true_impact_auc_vs_lost.png")

    # ================================================================
    # Plot 6: True Impact vs Masking (only_normal) — comparison
    # ================================================================
    masking_summary_path = os.path.join(MASKING_DIR, "summary.csv")
    if os.path.exists(masking_summary_path):
        masking_summary = pd.read_csv(masking_summary_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # Point missing comparison
        ti_pt = summary[(summary['missing_type'] == 'point')].sort_values('fraction')
        mk_pt = masking_summary[(masking_summary['missing_type'] == 'point')].sort_values('fraction')

        ax1.plot(ti_pt['fraction'], ti_pt['mean_AUC_ROC'],
                 marker='s', color='#e41a1c', label='True Impact', linewidth=2, markersize=8)
        ax1.plot(mk_pt['fraction'], mk_pt['mean_AUC_ROC'],
                 marker='o', color='#377eb8', label='Masking (only normal)', linewidth=2, markersize=8)
        ax1.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0,
                    alpha=0.85, zorder=5, label=f'Baseline ({baseline_auc:.3f})')
        ax1.set_xlabel('Fraction', fontsize=12)
        ax1.set_ylabel('AUC-ROC', fontsize=12)
        ax1.set_title('Point Missing', fontsize=13)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Burst missing comparison (nb=3)
        ti_burst = summary[(summary['missing_type'] == 'burst') & (summary['num_bursts'] == 3)].sort_values('fraction')
        mk_burst = masking_summary[(masking_summary['missing_type'] == 'burst') & (masking_summary['num_bursts'] == 3)].sort_values('fraction')

        ax2.plot(ti_burst['fraction'], ti_burst['mean_AUC_ROC'],
                 marker='s', color='#e41a1c', label='True Impact', linewidth=2, markersize=8)
        if not mk_burst.empty:
            ax2.plot(mk_burst['fraction'], mk_burst['mean_AUC_ROC'],
                     marker='o', color='#377eb8', label='Masking (only normal)', linewidth=2, markersize=8)
        ax2.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0,
                    alpha=0.85, zorder=5, label=f'Baseline ({baseline_auc:.3f})')
        ax2.set_xlabel('Fraction', fontsize=12)
        ax2.set_title('Burst Missing (nb=3)', fontsize=13)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle('True Impact vs Masking (only normal) — AUC-ROC Comparison', fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, 'true_impact_vs_masking.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved: true_impact_vs_masking.png")
    else:
        print("Skipping comparison plot: masking summary not found")

    # ================================================================
    # Plot 7: Degradation bar chart at fraction=20%
    # ================================================================
    f20 = summary[summary['fraction'] == 0.20]

    conditions = []
    auc_vals = []
    lost_vals = []

    pt_row = f20[f20['missing_type'] == 'point']
    if not pt_row.empty:
        conditions.append('Point')
        auc_vals.append(pt_row['mean_AUC_ROC'].values[0])
        lost_vals.append(pt_row['mean_n_lost_anomalies'].values[0])

    for nb in sorted(f20[f20['missing_type'] == 'burst']['num_bursts'].unique()):
        nb = int(nb)
        row = f20[(f20['missing_type'] == 'burst') & (f20['num_bursts'] == nb)]
        if not row.empty:
            conditions.append(f'Burst\n(nb={nb})')
            auc_vals.append(row['mean_AUC_ROC'].values[0])
            lost_vals.append(row['mean_n_lost_anomalies'].values[0])

    if conditions:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        x = np.arange(len(conditions))

        bars = ax1.bar(x, auc_vals, 0.5, color='#377eb8', alpha=0.8)
        ax1.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
        ax1.set_ylabel('AUC-ROC', fontsize=12, color='#377eb8')
        ax1.set_ylim(0.4, 0.75)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # Add value labels on bars
        for bar, val in zip(bars, auc_vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax2 = ax1.twinx()
        ax2.plot(x, lost_vals, marker='D', color='#e41a1c', linewidth=2, markersize=8)
        ax2.set_ylabel('Mean Lost Anomalies', fontsize=12, color='#e41a1c')

        ax1.set_xticks(x)
        ax1.set_xticklabels(conditions, fontsize=11)
        ax1.set_title('IForest: True Impact at Fraction=20%', fontsize=13)
        ax1.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, 'true_impact_bar_f20.png'), dpi=300)
        plt.close(fig)
        print("Saved: true_impact_bar_f20.png")

    # ================================================================
    # Plot 8: High vs Low baseline split
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

        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, zorder=5, alpha=0.7)
        ax.set_xlabel('Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('IForest: True Impact — High vs Low Baseline', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'true_impact_high_vs_low.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: true_impact_high_vs_low.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
