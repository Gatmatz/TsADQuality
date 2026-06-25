"""Plot Gilbert-Elliott true impact experiment results for IForest."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "gilbert_elliott_true_impact")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}

BETA_COLORS = {0.10: '#e41a1c', 0.25: '#4daf4a', 0.50: '#377eb8'}
BETA_MARKERS = {0.10: 's', 0.25: '^', 0.50: 'o'}


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
    print(f"{'alpha':>8} {'beta':>8} {'exp_rate':>10} {'burst_len':>10} {'act_rate':>10} {'lost_anom':>10} {'AUC-ROC':>8}")
    print("-" * 75)
    for _, row in summary.sort_values(['alpha', 'beta']).iterrows():
        print(f"{row['alpha']:>8.2f} {row['beta']:>8.2f} {row['expected_rate']:>10.2%} "
              f"{row['expected_burst_len']:>10.1f} {row['mean_actual_missing_rate']:>10.4f} "
              f"{row['mean_n_lost_anomalies']:>10.1f} {row['mean_AUC_ROC']:>8.4f}")
    print()

    alphas = sorted(df['alpha'].unique())
    betas = sorted(df['beta'].unique())

    # ================================================================
    # Plot 1: AUC-ROC vs alpha — one line per beta
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for beta in betas:
        sub = df[df['beta'] == beta]
        means = sub.groupby('alpha')['AUC_ROC'].mean().reset_index().sort_values('alpha')
        ax.plot(means['alpha'], means['AUC_ROC'],
                marker=BETA_MARKERS.get(beta, 'o'), color=BETA_COLORS.get(beta, 'gray'),
                label=f'beta={beta} (burst~{1/beta:.0f})', linewidth=2, markersize=8)

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0,
               alpha=0.85, zorder=5, label=f'Baseline ({baseline_auc:.3f})')
    ax.set_xlabel('Alpha (p Good→Bad)', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Gilbert-Elliott True Impact — AUC-ROC vs Alpha', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'ge_true_impact_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: ge_true_impact_AUC_ROC.png")

    # ================================================================
    # Plot 2: Multi-metric
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, metric in zip(axes, METRICS):
        for beta in betas:
            sub = df[df['beta'] == beta]
            means = sub.groupby('alpha')[metric].mean().reset_index().sort_values('alpha')
            ax.plot(means['alpha'], means[metric],
                    marker=BETA_MARKERS.get(beta, 'o'), color=BETA_COLORS.get(beta, 'gray'),
                    label=f'beta={beta}', linewidth=2, markersize=6)

        bl_val = baseline[metric].mean()
        ax.axhline(y=bl_val, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
        ax.set_xlabel('Alpha', fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(True, alpha=0.3)
        if ax == axes[-1]:
            ax.legend(fontsize=8, loc='best')

    fig.suptitle('IForest: Gilbert-Elliott True Impact — Multiple Metrics', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'ge_true_impact_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: ge_true_impact_multi_metric.png")

    # ================================================================
    # Plot 3: Heatmap — alpha × beta
    # ================================================================
    matrix = np.full((len(betas), len(alphas)), np.nan)
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            sub = df[(df['alpha'] == alpha) & (df['beta'] == beta)]
            if not sub.empty:
                matrix[i, j] = sub['AUC_ROC'].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'{a}' for a in alphas])
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f'beta={b}\n(burst~{1/b:.0f})' for b in betas])
    ax.set_xlabel('Alpha (burst frequency)', fontsize=12)
    ax.set_title('IForest: Mean AUC-ROC — Gilbert-Elliott True Impact', fontsize=13)
    for i in range(len(betas)):
        for j in range(len(alphas)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=11, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'ge_true_impact_heatmap.png'), dpi=300)
    plt.close(fig)
    print("Saved: ge_true_impact_heatmap.png")

    # ================================================================
    # Plot 4: Lost anomalies vs actual missing rate
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for beta in betas:
        sub = df[df['beta'] == beta]
        means = sub.groupby('alpha').agg({
            'actual_missing_rate': 'mean',
            'n_lost_anomalies': 'mean'
        }).reset_index().sort_values('alpha')
        ax.plot(means['actual_missing_rate'], means['n_lost_anomalies'],
                marker=BETA_MARKERS.get(beta, 'o'), color=BETA_COLORS.get(beta, 'gray'),
                label=f'beta={beta}', linewidth=2, markersize=8)

    ax.set_xlabel('Actual Missing Rate', fontsize=12)
    ax.set_ylabel('Mean Lost Anomalies', fontsize=12)
    ax.set_title('Lost Anomalies vs Missing Rate', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'ge_true_impact_lost_anomalies.png'), dpi=300)
    plt.close(fig)
    print("Saved: ge_true_impact_lost_anomalies.png")

    # ================================================================
    # Plot 5: AUC-ROC vs actual missing rate (scatter)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for beta in betas:
        sub = df[df['beta'] == beta]
        ax.scatter(sub['actual_missing_rate'], sub['AUC_ROC'],
                   alpha=0.4, s=25, color=BETA_COLORS.get(beta, 'gray'),
                   label=f'beta={beta}')

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, zorder=5, alpha=0.4)
    ax.set_xlabel('Actual Missing Rate', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: AUC-ROC vs Missing Rate (Gilbert-Elliott)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'ge_true_impact_auc_vs_rate.png'), dpi=300)
    plt.close(fig)
    print("Saved: ge_true_impact_auc_vs_rate.png")

    # ================================================================
    # Plot 6: Expected vs actual missing rate
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for beta in betas:
        sub_summary = summary[summary['beta'] == beta]
        ax.plot(sub_summary['expected_rate'], sub_summary['mean_actual_missing_rate'],
                marker=BETA_MARKERS.get(beta, 'o'), color=BETA_COLORS.get(beta, 'gray'),
                label=f'beta={beta}', linewidth=2, markersize=8)

    # Perfect line
    lims = [0, max(summary['expected_rate'].max(), summary['mean_actual_missing_rate'].max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Perfect match')
    ax.set_xlabel('Expected Missing Rate (alpha/(alpha+beta))', fontsize=12)
    ax.set_ylabel('Actual Missing Rate', fontsize=12)
    ax.set_title('Expected vs Actual Missing Rate', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'ge_true_impact_expected_vs_actual.png'), dpi=300)
    plt.close(fig)
    print("Saved: ge_true_impact_expected_vs_actual.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
