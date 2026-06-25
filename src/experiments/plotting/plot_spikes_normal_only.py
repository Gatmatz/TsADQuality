"""Plot spikes (normal only) experiment results."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "spikes_normal_only")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}

MULT_COLORS = {3.0: '#377eb8', 5.0: '#4daf4a', 10.0: '#e41a1c'}
MULT_MARKERS = {3.0: 'o', 5.0: '^', 10.0: 's'}


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]
    baseline = pd.read_csv(BASELINE_CSV)
    baseline_auc = baseline['AUC_ROC'].mean()

    print(f"Baseline: {baseline_auc:.4f}, {len(df)} rows")

    # Plot 1: AUC-ROC vs fraction per multiplier
    fig, ax = plt.subplots(figsize=(8, 5))
    for mult in sorted(df['multiplier'].unique()):
        sub = df[df['multiplier'] == mult]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=MULT_MARKERS.get(mult, 'o'), color=MULT_COLORS.get(mult, 'gray'),
                label=f'mult={int(mult)}x', linewidth=2, markersize=7)

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5,
               label=f'Baseline ({baseline_auc:.3f})')
    ax.set_xlabel('Fraction of Normal Points with Spikes', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Spikes (Normal Only) — AUC-ROC vs Fraction', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'spikes_normal_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: spikes_normal_AUC_ROC.png")

    # Plot 2: Multi-metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, METRICS):
        for mult in sorted(df['multiplier'].unique()):
            sub = df[df['multiplier'] == mult]
            means = sub.groupby('fraction')[metric].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means[metric],
                    marker=MULT_MARKERS.get(mult, 'o'), color=MULT_COLORS.get(mult, 'gray'),
                    label=f'mult={int(mult)}x', linewidth=2, markersize=5)
        bl = baseline[metric].mean()
        ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
        ax.set_xlabel('Fraction', fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(True, alpha=0.3)
        if ax == axes[-1]:
            ax.legend(fontsize=8)
    fig.suptitle('IForest: Spikes (Normal Only) — Multiple Metrics', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'spikes_normal_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: spikes_normal_multi_metric.png")

    # Plot 3: Heatmap
    fractions = sorted(df['fraction'].unique())
    mults = sorted(df['multiplier'].unique())
    matrix = np.full((len(mults), len(fractions)), np.nan)
    n_matrix = np.full((len(mults), len(fractions)), 0, dtype=int)
    for i, mult in enumerate(mults):
        for j, frac in enumerate(fractions):
            sub = df[(df['multiplier'] == mult) & (df['fraction'] == frac)]
            if not sub.empty:
                matrix[i, j] = sub['AUC_ROC'].mean()
                n_matrix[i, j] = len(sub)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_yticks(range(len(mults)))
    ax.set_yticklabels([f'{int(m)}x' for m in mults])
    ax.set_xlabel('Fraction', fontsize=12)
    ax.set_ylabel('Multiplier', fontsize=12)
    ax.set_title('IForest: Spikes (Normal Only) — Mean AUC-ROC', fontsize=13)
    for i in range(len(mults)):
        for j in range(len(fractions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}\n(n={n_matrix[i,j]})', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'spikes_normal_heatmap.png'), dpi=300)
    plt.close(fig)
    print("Saved: spikes_normal_heatmap.png")

    # Plot 4: High vs low baseline
    high_files = set(baseline[baseline['AUC_ROC'] >= 0.5]['file'].tolist())
    low_files = set(baseline[baseline['AUC_ROC'] < 0.5]['file'].tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, files, title in [
        (ax1, high_files, f'{len(high_files)} files (baseline $\\geq$ 0.5)'),
        (ax2, low_files, f'{len(low_files)} files (baseline < 0.5)'),
    ]:
        sub_df = df[df['file'].isin(files)]
        for mult in sorted(sub_df['multiplier'].unique()):
            sub = sub_df[sub_df['multiplier'] == mult]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=MULT_MARKERS.get(mult, 'o'), color=MULT_COLORS.get(mult, 'gray'),
                    label=f'mult={int(mult)}x', linewidth=2, markersize=5)
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, alpha=0.85, zorder=5)
        ax.set_xlabel('Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('IForest: Spikes (Normal Only) — High vs Low Baseline', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'spikes_normal_high_vs_low.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: spikes_normal_high_vs_low.png")

    # Plot 5: Degradation %
    fig, ax = plt.subplots(figsize=(8, 5))
    for mult in sorted(df['multiplier'].unique()):
        sub = df[df['multiplier'] == mult]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        pct = ((means['AUC_ROC'] - baseline_auc) / baseline_auc) * 100
        ax.plot(means['fraction'], pct,
                marker=MULT_MARKERS.get(mult, 'o'), color=MULT_COLORS.get(mult, 'gray'),
                label=f'mult={int(mult)}x', linewidth=2, markersize=7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xlabel('Fraction', fontsize=12)
    ax.set_ylabel('AUC-ROC Change from Baseline (%)', fontsize=12)
    ax.set_title('IForest: Spikes (Normal Only) — Degradation', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'spikes_normal_degradation.png'), dpi=300)
    plt.close(fig)
    print("Saved: spikes_normal_degradation.png")

    # Plot 6: % of files with inverted detection (AUC < 0.5) per condition
    baseline_inverted_pct = (baseline['AUC_ROC'] < 0.5).mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    for mult in sorted(df['multiplier'].unique()):
        sub = df[df['multiplier'] == mult]
        inv = sub.groupby('fraction')['AUC_ROC'].apply(
            lambda g: (g < 0.5).mean() * 100
        ).reset_index(name='pct_inverted').sort_values('fraction')
        ax.plot(inv['fraction'], inv['pct_inverted'],
                marker=MULT_MARKERS.get(mult, 'o'), color=MULT_COLORS.get(mult, 'gray'),
                label=f'mult={int(mult)}x', linewidth=2, markersize=7)
    ax.axhline(y=baseline_inverted_pct, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5,
               label=f'Baseline ({baseline_inverted_pct:.0f}%)')
    ax.set_xlabel('Fraction', fontsize=12)
    ax.set_ylabel('Files with AUC-ROC < 0.5 (%)', fontsize=12)
    ax.set_title('IForest: Spikes (Normal Only) — Inverted Detection Rate', fontsize=13)
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'spikes_normal_inverted_rate.png'), dpi=300)
    plt.close(fig)
    print("Saved: spikes_normal_inverted_rate.png")

    # Plot 7: n_spikes vs AUC scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    for mult in sorted(df['multiplier'].unique()):
        sub = df[df['multiplier'] == mult]
        ax.scatter(sub['n_spikes'], sub['AUC_ROC'], alpha=0.3, s=15,
                   color=MULT_COLORS.get(mult, 'gray'), label=f'mult={int(mult)}x')
    ax.set_xlabel('Number of Spikes Injected', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: AUC-ROC vs Number of Spikes', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'spikes_normal_scatter.png'), dpi=300)
    plt.close(fig)
    print("Saved: spikes_normal_scatter.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
