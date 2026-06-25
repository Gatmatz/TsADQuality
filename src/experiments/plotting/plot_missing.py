"""Plot missing values experiment results — mean score approach."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "missing")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}

BURST_COLORS = {10: '#4daf4a', 50: '#ff7f00', 200: '#984ea3', 500: '#a65628', 1000: '#f781bf'}
BURST_MARKERS = {10: '^', 50: 's', 200: 'D', 500: 'v', 1000: 'p'}


def load_data():
    """Load per-file results from checkpoint."""
    checkpoint = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    checkpoint = checkpoint[checkpoint['model'] == 'IForest']
    checkpoint = checkpoint[checkpoint['error'].isna()]
    return checkpoint


def compute_summary(df):
    rows = []
    for keys, group in df.groupby(['missing_type', 'fraction', 'burst_length', 'imputation']):
        mtype, frac, blen, imp = keys
        row = {
            'missing_type': mtype, 'fraction': frac,
            'burst_length': int(blen), 'imputation': imp,
            'n_runs': len(group),
        }
        for metric in METRICS:
            row[f'mean_{metric}'] = group[metric].mean()
            row[f'std_{metric}'] = group[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_point_vs_burst(summary):
    """Plot 1: AUC-ROC vs fraction — point + all burst lengths (linear imputation)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    imp = 'linear'

    point = summary[(summary['missing_type'] == 'point') & (summary['imputation'] == imp)].sort_values('fraction')
    if not point.empty:
        ax.plot(point['fraction'], point['mean_AUC_ROC'],
                marker='o', color='#377eb8', label='Point missing', linewidth=2, markersize=8)

    for blen in sorted(summary[summary['missing_type'] == 'burst']['burst_length'].unique()):
        blen = int(blen)
        burst = summary[(summary['missing_type'] == 'burst') & (summary['burst_length'] == blen)
                        & (summary['imputation'] == imp)].sort_values('fraction')
        if not burst.empty:
            ax.plot(burst['fraction'], burst['mean_AUC_ROC'],
                    marker=BURST_MARKERS.get(blen, 'x'), color=BURST_COLORS.get(blen, 'gray'),
                    label=f'Burst (len={blen})', linewidth=2, markersize=8)

    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Point vs Burst Missing (linear imputation)', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'missing_point_vs_burst_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: missing_point_vs_burst_AUC_ROC.png")


def plot_imputation_comparison(summary):
    """Plot 2: Linear vs ffill — side by side for point and burst."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ax = axes[0]
    for imp, color, marker, ls in [('linear', '#377eb8', 'o', '-'), ('ffill', '#e41a1c', 's', '--')]:
        sub = summary[(summary['missing_type'] == 'point') & (summary['imputation'] == imp)].sort_values('fraction')
        if not sub.empty:
            ax.plot(sub['fraction'], sub['mean_AUC_ROC'],
                    marker=marker, color=color, label=imp, linewidth=2, markersize=8, linestyle=ls)
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Point Missing', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    burst_lens = sorted(summary[summary['missing_type'] == 'burst']['burst_length'].unique())
    target_blen = 200 if 200 in burst_lens else (burst_lens[-1] if burst_lens else None)
    if target_blen is not None:
        for imp, color, marker, ls in [('linear', '#377eb8', 'o', '-'), ('ffill', '#e41a1c', 's', '--')]:
            sub = summary[(summary['missing_type'] == 'burst') & (summary['burst_length'] == target_blen)
                          & (summary['imputation'] == imp)].sort_values('fraction')
            if not sub.empty:
                ax.plot(sub['fraction'], sub['mean_AUC_ROC'],
                        marker=marker, color=color, label=imp, linewidth=2, markersize=8, linestyle=ls)
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_title(f'Burst Missing (len={int(target_blen)})', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'missing_imputation_comparison.png'), dpi=300)
    plt.close(fig)
    print("Saved: missing_imputation_comparison.png")


def plot_burst_length_effect(summary):
    """Plot 3: Effect of burst length on AUC-ROC (linear imputation)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    frac_colors = {0.01: '#377eb8', 0.05: '#4daf4a', 0.10: '#ff7f00', 0.20: '#e41a1c'}
    frac_markers = {0.01: 'o', 0.05: '^', 0.10: 's', 0.20: 'D'}

    burst_linear = summary[(summary['missing_type'] == 'burst') & (summary['imputation'] == 'linear')]

    for frac in sorted(burst_linear['fraction'].unique()):
        sub = burst_linear[burst_linear['fraction'] == frac].sort_values('burst_length')
        if not sub.empty:
            ax.plot(sub['burst_length'], sub['mean_AUC_ROC'],
                    marker=frac_markers.get(frac, 'x'), color=frac_colors.get(frac, 'gray'),
                    label=f'Fraction={frac:.0%}', linewidth=2, markersize=8)

    ax.set_xlabel('Burst Length', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Burst Length Effect (linear imputation)', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'missing_burst_length_effect.png'), dpi=300)
    plt.close(fig)
    print("Saved: missing_burst_length_effect.png")


def plot_heatmap(summary):
    """Plot 4: Heatmap of mean AUC-ROC — burst missing, linear imputation."""
    burst_linear = summary[(summary['missing_type'] == 'burst') & (summary['imputation'] == 'linear')]
    point_linear = summary[(summary['missing_type'] == 'point') & (summary['imputation'] == 'linear')]

    fractions = sorted(summary['fraction'].unique())
    burst_lengths = sorted(burst_linear['burst_length'].unique())
    row_labels = ['Point'] + [f'Burst (len={int(b)})' for b in burst_lengths]

    matrix = np.full((len(row_labels), len(fractions)), np.nan)

    for j, frac in enumerate(fractions):
        sub = point_linear[point_linear['fraction'] == frac]
        if not sub.empty:
            matrix[0, j] = sub['mean_AUC_ROC'].values[0]

    for i, blen in enumerate(burst_lengths):
        for j, frac in enumerate(fractions):
            sub = burst_linear[(burst_linear['burst_length'] == blen) & (burst_linear['fraction'] == frac)]
            if not sub.empty:
                matrix[i + 1, j] = sub['mean_AUC_ROC'].values[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Fraction of Missing Points', fontsize=12)
    ax.set_ylabel('Missing Type', fontsize=12)
    ax.set_title('IForest: Mean AUC-ROC — Linear Imputation', fontsize=13)

    for i in range(len(row_labels)):
        for j in range(len(fractions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'heatmap_missing_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: heatmap_missing_AUC_ROC.png")


def plot_per_metric(summary):
    """Plot 5-7: Per metric — point + burst (linear imputation)."""
    imp = 'linear'
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        col = f'mean_{metric}'

        point = summary[(summary['missing_type'] == 'point') & (summary['imputation'] == imp)].sort_values('fraction')
        if not point.empty:
            ax.plot(point['fraction'], point[col],
                    marker='o', color='#377eb8', label='Point', linewidth=2, markersize=8)

        for blen in sorted(summary[summary['missing_type'] == 'burst']['burst_length'].unique()):
            blen = int(blen)
            burst = summary[(summary['missing_type'] == 'burst') & (summary['burst_length'] == blen)
                            & (summary['imputation'] == imp)].sort_values('fraction')
            if not burst.empty:
                ax.plot(burst['fraction'], burst[col],
                        marker=BURST_MARKERS.get(blen, 'x'), color=BURST_COLORS.get(blen, 'gray'),
                        label=f'Burst (len={blen})', linewidth=2, markersize=8)

        ax.set_xlabel('Fraction of Missing Points', fontsize=12)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_title(f'IForest: {METRIC_LABELS[metric]} vs Missing Fraction', fontsize=13)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f'missing_{metric}.png'), dpi=300)
        plt.close(fig)
        print(f"Saved: missing_{metric}.png")


def plot_bar_imputation(summary):
    """Plot 8: Bar chart — linear vs ffill at fraction=20%."""
    df_f20 = summary[summary['fraction'] == 0.20]

    conditions = []
    linear_vals = []
    ffill_vals = []

    for imp, vals in [('linear', linear_vals), ('ffill', ffill_vals)]:
        sub = df_f20[(df_f20['missing_type'] == 'point') & (df_f20['imputation'] == imp)]
        if not sub.empty:
            vals.append(sub['mean_AUC_ROC'].values[0])
        else:
            vals.append(0)
    conditions.append('Point')

    for blen in sorted(df_f20[df_f20['missing_type'] == 'burst']['burst_length'].unique()):
        blen = int(blen)
        for imp, vals in [('linear', linear_vals), ('ffill', ffill_vals)]:
            sub = df_f20[(df_f20['missing_type'] == 'burst') & (df_f20['burst_length'] == blen)
                         & (df_f20['imputation'] == imp)]
            if not sub.empty:
                vals.append(sub['mean_AUC_ROC'].values[0])
            else:
                vals.append(0)
        conditions.append(f'Burst\n(len={blen})')

    if not conditions:
        print("No data at fraction=0.20, skipping bar chart.")
        return

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, linear_vals, width, label='Linear', color='#377eb8')
    ax.bar(x + width / 2, ffill_vals, width, label='Forward-fill', color='#e41a1c')

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Imputation Comparison at Fraction=20%', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'missing_bar_f20_imputation.png'), dpi=300)
    plt.close(fig)
    print("Saved: missing_bar_f20_imputation.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading results...")
    df = load_data()
    summary = compute_summary(df)

    print(f"Results: {len(df)} rows")
    print(f"Missing types: {sorted(summary['missing_type'].unique())}")
    print(f"Fractions: {sorted(summary['fraction'].unique())}")
    print(f"Burst lengths: {sorted(summary[summary['missing_type'] == 'burst']['burst_length'].unique())}")
    print(f"Imputation: {sorted(summary['imputation'].unique())}")
    print()

    print(f"{'type':>8} {'frac':>6} {'blen':>6} {'imp':>8} {'n':>5} {'AUC-ROC':>8} {'VUS-ROC':>8} {'AUC-PR':>8}")
    print("-" * 70)
    for _, row in summary.sort_values(['missing_type', 'imputation', 'burst_length', 'fraction']).iterrows():
        blen = int(row['burst_length']) if row['missing_type'] == 'burst' else '-'
        print(f"{row['missing_type']:>8} {row['fraction']:>6.2f} {str(blen):>6} {row['imputation']:>8} "
              f"{int(row['n_runs']):>5} {row['mean_AUC_ROC']:>8.4f} {row['mean_VUS_ROC']:>8.4f} {row['mean_AUC_PR']:>8.4f}")
    print()

    plot_point_vs_burst(summary)
    plot_imputation_comparison(summary)
    plot_burst_length_effect(summary)
    plot_heatmap(summary)
    plot_per_metric(summary)
    plot_bar_imputation(summary)

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
