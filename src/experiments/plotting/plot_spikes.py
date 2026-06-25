"""Plot spike experiment results — mean score approach."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "spikes_no_zscore")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

METRICS = ['AUC_ROC', 'VUS_ROC', 'AUC_PR']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'VUS_ROC': 'VUS-ROC', 'AUC_PR': 'AUC-PR'}

CONDITIONS = [
    (False, 3.0,  'Point 3x',  '#377eb8', 'o'),
    (False, 10.0, 'Point 10x', '#e41a1c', 's'),
    (True,  3.0,  'Burst 3x',  '#4daf4a', '^'),
    (True,  10.0, 'Burst 10x', '#984ea3', 'D'),
]


def load_data():
    """Load per-file results from checkpoint."""
    checkpoint = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    checkpoint = checkpoint[checkpoint['model'] == 'IForest']
    checkpoint = checkpoint[checkpoint['error'].isna()]
    return checkpoint


def compute_summary(df):
    rows = []
    for (frac, seq, mult), group in df.groupby(['fraction', 'sequential', 'multiplier']):
        row = {
            'fraction': frac, 'sequential': seq, 'multiplier': mult,
            'n_runs': len(group),
        }
        for metric in METRICS:
            row[f'mean_{metric}'] = group[metric].mean()
            row[f'std_{metric}'] = group[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_all_conditions(summary):
    """Plot 1: AUC-ROC vs fraction for all 4 conditions."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for seq, mult, label, color, marker in CONDITIONS:
        sub = summary[(summary['sequential'] == seq) & (summary['multiplier'] == mult)].sort_values('fraction')
        if sub.empty:
            continue
        ax.plot(sub['fraction'], sub['mean_AUC_ROC'],
                marker=marker, color=color, label=label, linewidth=2, markersize=8)

    ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('IForest: Spike Impact on AUC-ROC', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'spikes_all_conditions_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: spikes_all_conditions_AUC_ROC.png")


def plot_per_metric(summary):
    """Plot 2-4: Each metric vs fraction for all conditions."""
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        col = f'mean_{metric}'

        for seq, mult, label, color, marker in CONDITIONS:
            sub = summary[(summary['sequential'] == seq) & (summary['multiplier'] == mult)].sort_values('fraction')
            if sub.empty:
                continue
            ax.plot(sub['fraction'], sub[col],
                    marker=marker, color=color, label=label, linewidth=2, markersize=8)

        ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_title(f'IForest: {METRIC_LABELS[metric]} vs Spike Fraction', fontsize=13)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f'spikes_{metric}.png'), dpi=300)
        plt.close(fig)
        print(f"Saved: spikes_{metric}.png")


def plot_heatmap(summary):
    """Plot 5: Heatmap of mean AUC-ROC."""
    fractions = sorted(summary['fraction'].unique())
    cond_keys = [(seq, mult, label) for seq, mult, label, _, _ in CONDITIONS]

    matrix = np.full((len(cond_keys), len(fractions)), np.nan)

    for i, (seq, mult, _) in enumerate(cond_keys):
        for j, frac in enumerate(fractions):
            sub = summary[(summary['sequential'] == seq) & (summary['multiplier'] == mult)
                          & (summary['fraction'] == frac)]
            if not sub.empty:
                matrix[i, j] = sub['mean_AUC_ROC'].values[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.set_yticks(range(len(cond_keys)))
    ax.set_yticklabels([label for _, _, label in cond_keys])
    ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax.set_ylabel('Spike Type', fontsize=12)
    ax.set_title('IForest: Mean AUC-ROC — Spike Corruption', fontsize=13)

    for i in range(len(cond_keys)):
        for j in range(len(fractions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

    fig.colorbar(im, ax=ax, label='AUC-ROC')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'heatmap_spikes_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("Saved: heatmap_spikes_AUC_ROC.png")


def plot_bar_chart(summary):
    """Plot 6: Grouped bar chart — 3 metrics at fraction=20%."""
    df_f20 = summary[summary['fraction'] == 0.20]

    cond_labels = []
    metric_vals = {m: [] for m in METRICS}

    for seq, mult, label, _, _ in CONDITIONS:
        sub = df_f20[(df_f20['sequential'] == seq) & (df_f20['multiplier'] == mult)]
        if sub.empty:
            continue
        cond_labels.append(label)
        for m in METRICS:
            metric_vals[m].append(sub[f'mean_{m}'].values[0])

    if not cond_labels:
        print("No data at fraction=0.20, skipping bar chart.")
        return

    x = np.arange(len(cond_labels))
    width = 0.25
    colors = ['#377eb8', '#4daf4a', '#ff7f00']

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(METRICS):
        ax.bar(x + i * width, metric_vals[m], width, label=METRIC_LABELS[m], color=colors[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(cond_labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('IForest: Metrics at Fraction=20% for Each Spike Type', fontsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'spikes_bar_f20_metrics.png'), dpi=300)
    plt.close(fig)
    print("Saved: spikes_bar_f20_metrics.png")


def plot_multiplier_comparison(summary):
    """Plot 7: Side-by-side 3x vs 10x (point and burst)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (seq_val, title) in zip(axes, [(False, 'Point Spikes'), (True, 'Burst Spikes')]):
        for mult, color, marker, ls in [(3.0, '#377eb8', 'o', '-'), (10.0, '#e41a1c', 's', '--')]:
            sub = summary[(summary['sequential'] == seq_val) & (summary['multiplier'] == mult)].sort_values('fraction')
            if sub.empty:
                continue
            ax.plot(sub['fraction'], sub['mean_AUC_ROC'],
                    marker=marker, color=color, label=f'{int(mult)}x multiplier',
                    linewidth=2, markersize=8, linestyle=ls)

        ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(f'IForest: {title}', fontsize=13)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'spikes_multiplier_comparison.png'), dpi=300)
    plt.close(fig)
    print("Saved: spikes_multiplier_comparison.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading results...")
    df = load_data()
    summary = compute_summary(df)

    print(f"Results: {len(df)} rows")
    print(f"Fractions: {sorted(summary['fraction'].unique())}")
    print()

    print(f"{'condition':>12} {'frac':>6} {'n':>5} {'AUC-ROC':>8} {'VUS-ROC':>8} {'AUC-PR':>8}")
    print("-" * 55)
    for _, row in summary.sort_values(['sequential', 'multiplier', 'fraction']).iterrows():
        ctype = 'Burst' if row['sequential'] else 'Point'
        label = f"{ctype} {int(row['multiplier'])}x"
        print(f"{label:>12} {row['fraction']:>6.2f} {int(row['n_runs']):>5} "
              f"{row['mean_AUC_ROC']:>8.4f} {row['mean_VUS_ROC']:>8.4f} {row['mean_AUC_PR']:>8.4f}")
    print()

    plot_all_conditions(summary)
    plot_per_metric(summary)
    plot_heatmap(summary)
    plot_bar_chart(summary)
    plot_multiplier_comparison(summary)

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
