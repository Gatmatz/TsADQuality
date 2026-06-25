"""Plot swap experiment results for IForest — mean score approach.

Generates two side-by-side plots:
  Left:  111 files with baseline AUC-ROC >= 0.5 (true degradation)
  Right: 30 files with baseline AUC-ROC < 0.5 (inverted ranking → converge to 0.5)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "swap")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}

COLORS = {1: '#e41a1c', 10: '#377eb8', 50: '#4daf4a', 100: '#984ea3', 200: '#ff7f00'}
MARKERS = {1: 's', 10: 'o', 50: '^', 100: 'D', 200: 'v'}


def load_data():
    """Load per-file results from checkpoint."""
    checkpoint = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    checkpoint = checkpoint[checkpoint['model'] == 'IForest']
    checkpoint = checkpoint[checkpoint['error'].isna()]
    return checkpoint


def get_file_splits():
    """Split files into high-baseline (>=0.5) and low-baseline (<0.5)."""
    baseline = pd.read_csv(BASELINE_CSV)
    high_files = set(baseline[baseline['AUC_ROC'] >= 0.5]['file'].tolist())
    low_files = set(baseline[baseline['AUC_ROC'] < 0.5]['file'].tolist())
    return high_files, low_files


def compute_summary(df):
    rows = []
    for (frac, sl), group in df.groupby(['fraction', 'swap_length']):
        row = {
            'fraction': frac,
            'swap_length': int(sl),
            'n_runs': len(group),
        }
        for metric in METRICS:
            row[f'mean_{metric}'] = group[metric].mean()
            row[f'std_{metric}'] = group[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_all_conditions(summary):
    """Plot 1: AUC-ROC vs fraction for each swap_length, with n_files annotations."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for sl in sorted(summary['swap_length'].unique()):
        sub = summary[summary['swap_length'] == sl].sort_values('fraction')
        label = 'Point swap (len=1)' if sl == 1 else f'Segment swap (len={sl})'
        ax.plot(sub['fraction'], sub['mean_AUC_ROC'],
                marker=MARKERS.get(sl, 'o'), color=COLORS.get(sl, 'gray'),
                label=label, linewidth=2, markersize=8)
        # Annotate n_files at each point
        for _, row in sub.iterrows():
            ax.annotate(f'n={int(row["n_runs"])}',
                        (row['fraction'], row['mean_AUC_ROC']),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=7, ha='center', color=COLORS.get(sl, 'gray'),
                        alpha=0.8)

    ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Swap Corruption — AUC-ROC vs Fraction', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'all_conditions_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: all_conditions_AUC_ROC.png")


def plot_per_metric(summary):
    """Plot 2-4: Per metric line plots."""
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        col = f'mean_{metric}'

        for sl in sorted(summary['swap_length'].unique()):
            sub = summary[summary['swap_length'] == sl].sort_values('fraction')
            label = 'Point swap (len=1)' if sl == 1 else f'Segment swap (len={sl})'
            ax.plot(sub['fraction'], sub[col],
                    marker=MARKERS.get(sl, 'o'), color=COLORS.get(sl, 'gray'),
                    label=label, linewidth=2, markersize=8)

        ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_title(f'IForest: Swap Corruption — {METRIC_LABELS[metric]}', fontsize=13)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f'swap_{metric}.png'), dpi=300)
        plt.close(fig)
        print(f"Saved: swap_{metric}.png")


def plot_heatmap(summary):
    """Plot 5: Heatmap of mean AUC-ROC."""
    fractions = sorted(summary['fraction'].unique())
    swap_lengths = sorted(summary['swap_length'].unique())

    matrix = np.full((len(swap_lengths), len(fractions)), np.nan)

    for i, sl in enumerate(swap_lengths):
        for j, frac in enumerate(fractions):
            sub = summary[(summary['swap_length'] == sl) & (summary['fraction'] == frac)]
            if not sub.empty:
                matrix[i, j] = sub['mean_AUC_ROC'].values[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_yticks(range(len(swap_lengths)))
    ax.set_yticklabels([f'Point (1)' if sl == 1 else f'Seg ({sl})' for sl in swap_lengths])
    ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax.set_ylabel('Swap Length', fontsize=12)
    ax.set_title('IForest: Mean AUC-ROC — Swap Corruption', fontsize=13)

    # Build n_runs matrix for annotations
    n_matrix = np.full((len(swap_lengths), len(fractions)), 0, dtype=int)
    for i, sl in enumerate(swap_lengths):
        for j, frac in enumerate(fractions):
            sub = summary[(summary['swap_length'] == sl) & (summary['fraction'] == frac)]
            if not sub.empty:
                n_matrix[i, j] = int(sub['n_runs'].values[0])

    for i in range(len(swap_lengths)):
        for j in range(len(fractions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}\n(n={n_matrix[i,j]})', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')

    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'heatmap_swap_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: heatmap_swap_AUC_ROC.png")


def plot_bar_chart(summary):
    """Plot 6: Grouped bar chart — all conditions."""
    fig, ax = plt.subplots(figsize=(12, 5))

    fractions = sorted(summary['fraction'].unique())
    swap_lengths = sorted(summary['swap_length'].unique())

    x = np.arange(len(fractions))
    width = 0.18
    n_groups = len(swap_lengths)

    for idx, sl in enumerate(swap_lengths):
        vals = []
        for frac in fractions:
            sub = summary[(summary['swap_length'] == sl) & (summary['fraction'] == frac)]
            if not sub.empty:
                vals.append(sub['mean_AUC_ROC'].values[0])
            else:
                vals.append(0)

        offset = (idx - n_groups / 2 + 0.5) * width
        label = 'Point (1)' if sl == 1 else f'Seg ({sl})'
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=COLORS.get(sl, 'gray'), alpha=0.85)
        # Annotate n_files on top of each bar
        for bar, frac in zip(bars, fractions):
            sub = summary[(summary['swap_length'] == sl) & (summary['fraction'] == frac)]
            if not sub.empty:
                n = int(sub['n_runs'].values[0])
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f'n={n}', ha='center', va='bottom', fontsize=6, alpha=0.7)

    ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Swap Corruption — All Conditions', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0.55)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'bar_chart_swap.png'), dpi=300)
    plt.close(fig)
    print("Saved: bar_chart_swap.png")


def plot_swap_length_effect(summary):
    """Plot 7: Faceted bar chart — effect of swap_length per fraction."""
    fractions = sorted(summary['fraction'].unique())
    n_frac = len(fractions)
    fig, axes = plt.subplots(1, n_frac, figsize=(4 * n_frac, 4), sharey=True)

    for ax, frac in zip(axes, fractions):
        sub = summary[summary['fraction'] == frac].sort_values('swap_length')
        colors_list = [COLORS.get(int(sl), 'gray') for sl in sub['swap_length']]
        ax.bar(range(len(sub)), sub['mean_AUC_ROC'], color=colors_list, alpha=0.85)

        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([str(int(sl)) for sl in sub['swap_length']], fontsize=9)
        ax.set_title(f'Fraction = {frac:.0%}', fontsize=11)
        ax.set_xlabel('Swap Length', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_ylim(bottom=0.60, top=0.80)

    axes[0].set_ylabel('AUC-ROC', fontsize=11)
    fig.suptitle('IForest: Swap Length Effect per Fraction', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'swap_length_effect.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: swap_length_effect.png")


def plot_high_vs_low_baseline(df):
    """Side-by-side: 111 files (baseline>=0.5) vs 30 files (baseline<0.5)."""
    high_files, low_files = get_file_splits()

    df_high = df[df['file'].isin(high_files)]
    df_low = df[df['file'].isin(low_files)]

    sum_high = compute_summary(df_high)
    sum_low = compute_summary(df_low)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: high-baseline files (111)
    for sl in sorted(sum_high['swap_length'].unique()):
        sub = sum_high[sum_high['swap_length'] == sl].sort_values('fraction')
        label = f'len={sl}' if sl > 1 else 'Point (len=1)'
        ax1.plot(sub['fraction'], sub['mean_AUC_ROC'],
                 marker=MARKERS.get(sl, 'o'), color=COLORS.get(sl, 'gray'),
                 label=label, linewidth=2, markersize=7)

    ax1.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax1.set_ylabel('AUC-ROC', fontsize=12)
    ax1.set_title('111 files (baseline AUC-ROC $\\geq$ 0.5)', fontsize=13)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: low-baseline files (30)
    for sl in sorted(sum_low['swap_length'].unique()):
        sub = sum_low[sum_low['swap_length'] == sl].sort_values('fraction')
        label = f'len={sl}' if sl > 1 else 'Point (len=1)'
        ax2.plot(sub['fraction'], sub['mean_AUC_ROC'],
                 marker=MARKERS.get(sl, 'o'), color=COLORS.get(sl, 'gray'),
                 label=label, linewidth=2, markersize=7)

    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5, label='Random (0.5)')
    ax2.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax2.set_ylabel('AUC-ROC', fontsize=12)
    ax2.set_title('30 files (baseline AUC-ROC < 0.5)', fontsize=13)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('IForest: Swap Corruption — Effect of Baseline AUC-ROC', fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, 'swap_high_vs_low_baseline.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: swap_high_vs_low_baseline.png")


def plot_corrected_auc(df):
    """Plot using max(AUC, 1-AUC) — discriminability regardless of ranking direction."""
    df_corr = df.copy()
    df_corr['AUC_ROC_corr'] = df_corr['AUC_ROC'].apply(lambda x: max(x, 1 - x))

    # Compute summary with corrected metric
    rows = []
    for (frac, sl), group in df_corr.groupby(['fraction', 'swap_length']):
        rows.append({
            'fraction': frac,
            'swap_length': int(sl),
            'n_runs': len(group),
            'mean_AUC_ROC_corr': group['AUC_ROC_corr'].mean(),
            'mean_AUC_ROC_raw': group['AUC_ROC'].mean(),
        })
    summary = pd.DataFrame(rows)

    # Also compute baseline corrected mean
    baseline = pd.read_csv(BASELINE_CSV)
    baseline_corr = baseline['AUC_ROC'].apply(lambda x: max(x, 1 - x)).mean()
    baseline_raw = baseline['AUC_ROC'].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: standard AUC-ROC (all 141)
    for sl in sorted(summary['swap_length'].unique()):
        sub = summary[summary['swap_length'] == sl].sort_values('fraction')
        label = f'len={sl}' if sl > 1 else 'Point (len=1)'
        ax1.plot(sub['fraction'], sub['mean_AUC_ROC_raw'],
                 marker=MARKERS.get(sl, 'o'), color=COLORS.get(sl, 'gray'),
                 label=label, linewidth=2, markersize=7)

    ax1.axhline(y=baseline_raw, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5,
                label=f'Baseline ({baseline_raw:.3f})')
    ax1.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax1.set_ylabel('AUC-ROC', fontsize=12)
    ax1.set_title('Standard AUC-ROC (all 141 files)', fontsize=13)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: max(AUC, 1-AUC) (all 141)
    for sl in sorted(summary['swap_length'].unique()):
        sub = summary[summary['swap_length'] == sl].sort_values('fraction')
        label = f'len={sl}' if sl > 1 else 'Point (len=1)'
        ax2.plot(sub['fraction'], sub['mean_AUC_ROC_corr'],
                 marker=MARKERS.get(sl, 'o'), color=COLORS.get(sl, 'gray'),
                 label=label, linewidth=2, markersize=7)

    ax2.axhline(y=baseline_corr, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5,
                label=f'Baseline ({baseline_corr:.3f})')
    ax2.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax2.set_ylabel('max(AUC, 1-AUC)', fontsize=12)
    ax2.set_title('Corrected AUC-ROC (all 141 files)', fontsize=13)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Standard vs Corrected AUC-ROC — Swap Corruption', fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, 'swap_corrected_auc.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: swap_corrected_auc.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading results...")
    df = load_data()
    summary = compute_summary(df)

    print(f"Results: {len(df)} rows")
    print(f"Fractions: {sorted(summary['fraction'].unique())}")
    print(f"Swap lengths: {sorted(summary['swap_length'].unique())}")
    print()

    print(f"{'fraction':>8} {'swap_len':>8} {'n':>5} {'AUC-ROC':>8} {'VUS-ROC':>8} {'AUC-PR':>8}")
    print("-" * 55)
    for _, row in summary.sort_values(['swap_length', 'fraction']).iterrows():
        print(f"{row['fraction']:>8.2f} {int(row['swap_length']):>8} {int(row['n_runs']):>5} "
              f"{row['mean_AUC_ROC']:>8.4f} {row['mean_VUS_ROC']:>8.4f} {row['mean_AUC_PR']:>8.4f}")
    print()

    plot_all_conditions(summary)
    plot_per_metric(summary)
    plot_heatmap(summary)
    plot_bar_chart(summary)
    plot_swap_length_effect(summary)
    plot_high_vs_low_baseline(df)
    plot_corrected_auc(df)

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
