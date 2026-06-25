"""Plot MNAR burst experiment results."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_mnar_burst")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

MECH_COLORS = {'mnar_extreme_burst': '#e41a1c', 'mnar_high_burst': '#ff7f00'}
MECH_MARKERS = {'mnar_extreme_burst': 's', 'mnar_high_burst': '^'}
MECH_LABELS = {'mnar_extreme_burst': 'MNAR extreme', 'mnar_high_burst': 'MNAR high'}

NB_COLORS = {1: '#e41a1c', 3: '#4daf4a', 5: '#984ea3', 10: '#ff7f00'}
NB_MARKERS = {1: 's', 3: '^', 5: 'D', 10: 'v'}


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]
    baseline = pd.read_csv(BASELINE_CSV)
    baseline_auc = baseline['AUC_ROC'].mean()

    print(f"Baseline: {baseline_auc:.4f}, {len(df)} rows")

    # Plot 1: AUC-ROC vs fraction per mechanism (nb=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech in ['mnar_extreme_burst', 'mnar_high_burst']:
        sub = df[(df['mechanism'] == mech) & (df['num_bursts'] == 1)]
        if sub.empty:
            continue
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                label=MECH_LABELS[mech], linewidth=2, markersize=7)
    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5,
               label=f'Baseline ({baseline_auc:.3f})')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: MNAR Burst (nb=1) — AUC-ROC', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'mnar_burst_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: mnar_burst_AUC_ROC.png")

    # Plot 2: AUC-ROC vs fraction per num_bursts (mnar_extreme only)
    fig, ax = plt.subplots(figsize=(8, 5))
    for nb in sorted(df['num_bursts'].unique()):
        nb = int(nb)
        sub = df[(df['mechanism'] == 'mnar_extreme_burst') & (df['num_bursts'] == nb)]
        if sub.empty:
            continue
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=NB_MARKERS.get(nb, 'o'), color=NB_COLORS.get(nb, 'gray'),
                label=f'nb={nb}', linewidth=2, markersize=7)
    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5,
               label=f'Baseline ({baseline_auc:.3f})')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xlabel('Fraction', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: MNAR Extreme Burst — AUC-ROC by num_bursts', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'mnar_burst_by_nb.png'), dpi=300)
    plt.close(fig)
    print("Saved: mnar_burst_by_nb.png")

    # Plot 3: % Anomalies Lost
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech in ['mnar_extreme_burst', 'mnar_high_burst']:
        sub = df[(df['mechanism'] == mech) & (df['num_bursts'] == 1)]
        if sub.empty:
            continue
        means = sub.groupby('fraction')['pct_anomalies_lost'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['pct_anomalies_lost'] * 100,
                marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                label=MECH_LABELS[mech], linewidth=2, markersize=7)
    ax.plot([0, 0.20], [0, 20], color='gray', linestyle=':', linewidth=1, alpha=0.5,
            label='Expected (uniform)')
    ax.set_xlabel('Fraction', fontsize=12)
    ax.set_ylabel('Anomalies Lost (%)', fontsize=12)
    ax.set_title('IForest: MNAR Burst — Anomalies Lost', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'mnar_burst_anomalies_lost.png'), dpi=300)
    plt.close(fig)
    print("Saved: mnar_burst_anomalies_lost.png")

    # Plot 4: Heatmap — num_bursts × fraction (mnar_extreme)
    mechs_for_heatmap = ['mnar_extreme_burst']
    for mech in mechs_for_heatmap:
        nbs = sorted(df[df['mechanism'] == mech]['num_bursts'].unique())
        fractions = sorted(df['fraction'].unique())
        matrix = np.full((len(nbs), len(fractions)), np.nan)

        for i, nb in enumerate(nbs):
            for j, frac in enumerate(fractions):
                sub = df[(df['mechanism'] == mech) & (df['num_bursts'] == nb) & (df['fraction'] == frac)]
                if not sub.empty:
                    matrix[i, j] = sub['AUC_ROC'].mean()

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                       vmin=0.3, vmax=baseline_auc + 0.02)
        ax.set_xticks(range(len(fractions)))
        ax.set_xticklabels([f'{f:.0%}' for f in fractions])
        ax.set_yticks(range(len(nbs)))
        ax.set_yticklabels([f'nb={int(nb)}' for nb in nbs])
        ax.set_xlabel('Fraction', fontsize=12)
        ax.set_title(f'IForest: {MECH_LABELS[mech]} Burst — Mean AUC-ROC', fontsize=13)
        for i in range(len(nbs)):
            for j in range(len(fractions)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if val < 0.45 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=10, color=color, fontweight='bold')
        fig.colorbar(im, ax=ax, label='AUC-ROC')
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f'mnar_burst_heatmap_{mech.replace("mnar_", "").replace("_burst", "")}.png'), dpi=300)
        plt.close(fig)
        print(f"Saved: mnar_burst_heatmap_{mech.replace('mnar_', '').replace('_burst', '')}.png")

    # Plot 5: Degradation %
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech in ['mnar_extreme_burst', 'mnar_high_burst']:
        sub = df[(df['mechanism'] == mech) & (df['num_bursts'] == 1)]
        if sub.empty:
            continue
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        pct = ((means['AUC_ROC'] - baseline_auc) / baseline_auc) * 100
        ax.plot(means['fraction'], pct,
                marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                label=MECH_LABELS[mech], linewidth=2, markersize=7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xlabel('Fraction', fontsize=12)
    ax.set_ylabel('AUC-ROC Change from Baseline (%)', fontsize=12)
    ax.set_title('IForest: MNAR Burst — Degradation', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'mnar_burst_degradation.png'), dpi=300)
    plt.close(fig)
    print("Saved: mnar_burst_degradation.png")

    # Plot 6: Inverted detection rate
    baseline_inv = (baseline['AUC_ROC'] < 0.5).mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech in ['mnar_extreme_burst', 'mnar_high_burst']:
        sub = df[(df['mechanism'] == mech) & (df['num_bursts'] == 1)]
        if sub.empty:
            continue
        inv = sub.groupby('fraction')['AUC_ROC'].apply(
            lambda g: (g < 0.5).mean() * 100
        ).reset_index(name='pct_inverted').sort_values('fraction')
        ax.plot(inv['fraction'], inv['pct_inverted'],
                marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                label=MECH_LABELS[mech], linewidth=2, markersize=7)
    ax.axhline(y=baseline_inv, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5,
               label=f'Baseline ({baseline_inv:.0f}%)')
    ax.set_xlabel('Fraction', fontsize=12)
    ax.set_ylabel('Files with AUC-ROC < 0.5 (%)', fontsize=12)
    ax.set_title('IForest: MNAR Burst — Inverted Detection Rate', fontsize=13)
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'mnar_burst_inverted_rate.png'), dpi=300)
    plt.close(fig)
    print("Saved: mnar_burst_inverted_rate.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
