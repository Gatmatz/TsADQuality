"""Plot missing burst length experiment results for IForest.

Burst length as % of series length — many short gaps vs few long gaps.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_burst_length")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}

BR_COLORS = {0.001: '#e41a1c', 0.005: '#4daf4a', 0.01: '#984ea3', 0.05: '#ff7f00'}
BR_MARKERS = {0.001: 's', 0.005: '^', 0.01: 'D', 0.05: 'v'}


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
    print(f"{'frac':>6} {'br':>8} {'burst_len':>10} {'n_bursts':>9} {'act_rate':>9} {'lost':>6} {'AUC-ROC':>8}")
    print("-" * 65)
    for _, row in summary.sort_values(['burst_ratio', 'fraction']).iterrows():
        print(f"{row['fraction']:>6.2f} {row['burst_ratio']:>8.3f} {row['mean_burst_length']:>10.0f} "
              f"{row['mean_num_bursts']:>9.0f} {row['mean_actual_missing_rate']:>9.4f} "
              f"{row['mean_n_lost_anomalies']:>6.0f} {row['mean_AUC_ROC']:>8.4f}")
    print()

    # ================================================================
    # Plot 1: AUC-ROC vs fraction — one line per burst_ratio
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for br in sorted(df['burst_ratio'].unique()):
        sub = df[df['burst_ratio'] == br]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=BR_MARKERS.get(br, 'o'), color=BR_COLORS.get(br, 'gray'),
                label=f'br={br}', linewidth=2, markersize=8)

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0,
               alpha=0.85, zorder=5, label=f'Baseline ({baseline_auc:.3f})')
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Burst Length Effect — AUC-ROC vs Fraction', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'burst_length_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: burst_length_AUC_ROC.png")

    # ================================================================
    # Plot 2: Multi-metric
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, metric in zip(axes, METRICS):
        for br in sorted(df['burst_ratio'].unique()):
            sub = df[df['burst_ratio'] == br]
            means = sub.groupby('fraction')[metric].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means[metric],
                    marker=BR_MARKERS.get(br, 'o'), color=BR_COLORS.get(br, 'gray'),
                    label=f'br={br}', linewidth=2, markersize=6)

        bl_val = baseline[metric].mean()
        ax.axhline(y=bl_val, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
        ax.set_xlabel('Fraction', fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(True, alpha=0.3)
        if ax == axes[-1]:
            ax.legend(fontsize=8, loc='best')

    fig.suptitle('IForest: Burst Length Effect — Multiple Metrics', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'burst_length_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: burst_length_multi_metric.png")

    # ================================================================
    # Plot 3: Heatmap — burst_ratio × fraction
    # ================================================================
    fractions = sorted(df['fraction'].unique())
    br_list = sorted(df['burst_ratio'].unique())

    matrix = np.full((len(br_list), len(fractions)), np.nan)
    for i, br in enumerate(br_list):
        for j, frac in enumerate(fractions):
            sub = df[(df['burst_ratio'] == br) & (df['fraction'] == frac)]
            if not sub.empty:
                matrix[i, j] = sub['AUC_ROC'].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_yticks(range(len(br_list)))
    ax.set_yticklabels([f'br={br}' for br in br_list])
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_title('IForest: Mean AUC-ROC — Burst Length Effect', fontsize=13)
    for i in range(len(br_list)):
        for j in range(len(fractions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'burst_length_heatmap.png'), dpi=300)
    plt.close(fig)
    print("Saved: burst_length_heatmap.png")

    # ================================================================
    # Plot 4: Lost anomalies per burst_ratio
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for br in sorted(df['burst_ratio'].unique()):
        sub = df[df['burst_ratio'] == br]
        means = sub.groupby('fraction')['n_lost_anomalies'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['n_lost_anomalies'],
                marker=BR_MARKERS.get(br, 'o'), color=BR_COLORS.get(br, 'gray'),
                label=f'br={br}', linewidth=2, markersize=8)

    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('Mean Lost Anomalies', fontsize=12)
    ax.set_title('Lost Anomalies by Burst Ratio', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'burst_length_lost_anomalies.png'), dpi=300)
    plt.close(fig)
    print("Saved: burst_length_lost_anomalies.png")

    # ================================================================
    # Plot 5: AUC-ROC vs lost anomalies (scatter)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for br in sorted(df['burst_ratio'].unique()):
        sub = df[df['burst_ratio'] == br]
        ax.scatter(sub['n_lost_anomalies'], sub['AUC_ROC'],
                   alpha=0.4, s=25, color=BR_COLORS.get(br, 'gray'),
                   label=f'br={br}')

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, zorder=5, alpha=0.4)
    ax.set_xlabel('Number of Lost Anomalies', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: AUC-ROC vs Lost Anomalies (Burst Length)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'burst_length_auc_vs_lost.png'), dpi=300)
    plt.close(fig)
    print("Saved: burst_length_auc_vs_lost.png")

    # ================================================================
    # Plot 6: Mean burst length per condition
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for br in sorted(df['burst_ratio'].unique()):
        sub = df[df['burst_ratio'] == br]
        means = sub.groupby('fraction')['burst_length'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['burst_length'],
                marker=BR_MARKERS.get(br, 'o'), color=BR_COLORS.get(br, 'gray'),
                label=f'br={br}', linewidth=2, markersize=8)

    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('Mean Burst Length', fontsize=12)
    ax.set_title('Dynamic Burst Length per Condition', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'burst_length_lengths.png'), dpi=300)
    plt.close(fig)
    print("Saved: burst_length_lengths.png")

    # ================================================================
    # Plot 7: High vs Low baseline
    # ================================================================
    high_files = set(baseline[baseline['AUC_ROC'] >= 0.5]['file'].tolist())
    low_files = set(baseline[baseline['AUC_ROC'] < 0.5]['file'].tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, files, title in [
        (ax1, high_files, f'{len(high_files)} files (baseline AUC-ROC $\\geq$ 0.5)'),
        (ax2, low_files, f'{len(low_files)} files (baseline AUC-ROC < 0.5)'),
    ]:
        sub_df = df[df['file'].isin(files)]
        for br in sorted(sub_df['burst_ratio'].unique()):
            sub = sub_df[sub_df['burst_ratio'] == br]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=BR_MARKERS.get(br, 'o'), color=BR_COLORS.get(br, 'gray'),
                    label=f'br={br}', linewidth=2, markersize=6)

        ax.set_xlabel('Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('IForest: Burst Length — High vs Low Baseline', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'burst_length_high_vs_low.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: burst_length_high_vs_low.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
