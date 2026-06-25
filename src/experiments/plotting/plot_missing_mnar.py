"""Plot MCAR vs MNAR comparison results."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_mnar")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

MECH_COLORS = {'mcar': '#377eb8', 'mnar_extreme': '#e41a1c', 'mnar_high': '#ff7f00'}
MECH_MARKERS = {'mcar': 'o', 'mnar_extreme': 's', 'mnar_high': '^'}
MECH_LABELS = {'mcar': 'MCAR', 'mnar_extreme': 'MNAR (extreme)', 'mnar_high': 'MNAR (high)'}


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]
    baseline = pd.read_csv(BASELINE_CSV)
    baseline_auc = baseline['AUC_ROC'].mean()

    print(f"Baseline: {baseline_auc:.4f}, {len(df)} rows")

    # ================================================================
    # Plot 1: AUC-ROC vs fraction per mechanism
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech in ['mcar', 'mnar_extreme', 'mnar_high']:
        sub = df[df['mechanism'] == mech]
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
    ax.set_title('IForest: MCAR vs MNAR — AUC-ROC', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'mnar_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: mnar_AUC_ROC.png")

    # ================================================================
    # Plot 2: % Anomalies Lost per mechanism
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech in ['mcar', 'mnar_extreme', 'mnar_high']:
        sub = df[df['mechanism'] == mech]
        if sub.empty:
            continue
        means = sub.groupby('fraction')['pct_anomalies_lost'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['pct_anomalies_lost'] * 100,
                marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                label=MECH_LABELS[mech], linewidth=2, markersize=7)

    ax.plot([0, 0.20], [0, 20], color='gray', linestyle=':', linewidth=1, alpha=0.5,
            label='Expected (uniform)')
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('Anomalies Lost (%)', fontsize=12)
    ax.set_title('IForest: MCAR vs MNAR — Anomalies Lost', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'mnar_anomalies_lost.png'), dpi=300)
    plt.close(fig)
    print("Saved: mnar_anomalies_lost.png")

    # ================================================================
    # Plot 3: Heatmap — mechanism × fraction
    # ================================================================
    mechs = ['mcar', 'mnar_extreme', 'mnar_high']
    fractions = sorted(df['fraction'].unique())
    matrix = np.full((len(mechs), len(fractions)), np.nan)
    n_matrix = np.full((len(mechs), len(fractions)), 0, dtype=int)

    for i, mech in enumerate(mechs):
        for j, frac in enumerate(fractions):
            sub = df[(df['mechanism'] == mech) & (df['fraction'] == frac)]
            if not sub.empty:
                matrix[i, j] = sub['AUC_ROC'].mean()
                n_matrix[i, j] = len(sub)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                   vmin=0.4, vmax=baseline_auc + 0.02)
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_yticks(range(len(mechs)))
    ax.set_yticklabels([MECH_LABELS[m] for m in mechs])
    ax.set_xlabel('Fraction', fontsize=12)
    ax.set_title('IForest: MCAR vs MNAR — Mean AUC-ROC', fontsize=13)
    for i in range(len(mechs)):
        for j in range(len(fractions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}\n(n={n_matrix[i,j]})', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'mnar_heatmap.png'), dpi=300)
    plt.close(fig)
    print("Saved: mnar_heatmap.png")

    # ================================================================
    # Plot 4: AUC-ROC vs % anomalies lost (scatter)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech in ['mcar', 'mnar_extreme', 'mnar_high']:
        sub = df[df['mechanism'] == mech]
        if sub.empty:
            continue
        ax.scatter(sub['pct_anomalies_lost'] * 100, sub['AUC_ROC'],
                   alpha=0.3, s=20, color=MECH_COLORS[mech], label=MECH_LABELS[mech])

    ax.axhline(y=baseline_auc, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xlabel('Anomalies Lost (%)', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: AUC-ROC vs Anomalies Lost — MCAR vs MNAR', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'mnar_auc_vs_lost.png'), dpi=300)
    plt.close(fig)
    print("Saved: mnar_auc_vs_lost.png")

    # ================================================================
    # Plot 5: Degradation from baseline (%)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech in ['mcar', 'mnar_extreme', 'mnar_high']:
        sub = df[df['mechanism'] == mech]
        if sub.empty:
            continue
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        pct = ((means['AUC_ROC'] - baseline_auc) / baseline_auc) * 100
        ax.plot(means['fraction'], pct,
                marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                label=MECH_LABELS[mech], linewidth=2, markersize=7)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('AUC-ROC Change from Baseline (%)', fontsize=12)
    ax.set_title('IForest: MCAR vs MNAR — Degradation', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'mnar_degradation.png'), dpi=300)
    plt.close(fig)
    print("Saved: mnar_degradation.png")

    # ================================================================
    # Plot 6: Inverted detection rate
    # ================================================================
    baseline_inv = (baseline['AUC_ROC'] < 0.5).mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech in ['mcar', 'mnar_extreme', 'mnar_high']:
        sub = df[df['mechanism'] == mech]
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
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('Files with AUC-ROC < 0.5 (%)', fontsize=12)
    ax.set_title('IForest: MCAR vs MNAR — Inverted Detection Rate', fontsize=13)
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'mnar_inverted_rate.png'), dpi=300)
    plt.close(fig)
    print("Saved: mnar_inverted_rate.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
