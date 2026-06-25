"""
Publication-quality plots for the Swap robustness experiment.

Loads summary CSVs from all swap_pct_* subdirectories and generates
per-model heatmaps and line plots.
Output structure: results/plots/swap/{Model}/heatmap_auc_roc_swappct_X.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "swap")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "plots", "swap")

# Thesis-quality classic style
plt.style.use('classic')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

METRICS = [
    ('mean_AUC_ROC', 'AUC-ROC'),
    ('mean_AUC_PR', 'AUC-PR'),
    ('mean_VUS_ROC', 'VUS-ROC'),
    ('mean_VUS_PR', 'VUS-PR'),
]

MODEL_STYLES = {
    'IForest': {'color': '#D62728', 'marker': 's', 'label': 'Isolation Forest'},
    'PCA':     {'color': '#1F77B4', 'marker': 'o', 'label': 'PCA'},
    'LOF':     {'color': '#2CA02C', 'marker': '^', 'label': 'LOF'},
    'MP':      {'color': '#FF7F0E', 'marker': 'D', 'label': 'Matrix Profile'},
}

DISTANCE_ORDER = ['10', '50', '100', '500', '1000', 'Global']


def _style(model):
    return MODEL_STYLES.get(model, {'color': 'gray', 'marker': 'x', 'label': model})


def load_all_swap_data():
    """Load and combine summary CSVs from all swap_pct subdirectories."""
    all_dfs = []
    for subdir in sorted(os.listdir(EXPERIMENT_DIR)):
        if subdir.startswith('swap_pct_'):
            csv_path = os.path.join(EXPERIMENT_DIR, subdir, 'summary.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['swap_pct'] = float(subdir.replace('swap_pct_', ''))
                all_dfs.append(df)
    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


def plot_heatmaps(df, model, model_dir):
    """Heatmap: fraction vs max_distance for each swap_pct and metric."""
    df = df.copy()
    df['max_distance'] = df['max_distance'].astype(str)

    for metric_col, metric_label in METRICS:
        if metric_col not in df.columns:
            continue

        for pct in sorted(df['swap_pct'].unique()):
            pct_df = df[df['swap_pct'] == pct]
            pivot = pct_df.pivot_table(
                index='fraction', columns='max_distance',
                values=metric_col, aggfunc='mean'
            )
            ordered_cols = [c for c in DISTANCE_ORDER if c in pivot.columns]
            pivot = pivot[ordered_cols]

            fig, ax = plt.subplots(figsize=(9, 6))
            vmin = max(0, pivot.values[~np.isnan(pivot.values)].min() - 0.02)
            vmax = min(1, pivot.values[~np.isnan(pivot.values)].max() + 0.02)
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                           vmin=vmin, vmax=vmax)

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f'{v:.0%}' for v in pivot.index])

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val < (vmin + vmax) / 2 else 'black'
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                                fontsize=10, color=text_color, fontweight='bold')

            ax.set_xlabel('Max Swap Distance')
            ax.set_ylabel('Swap Fraction')
            ax.set_title(f'{_style(model)["label"]} — {metric_label}\n'
                         f'(Swap Length = {pct:.1f}% of series)')
            plt.colorbar(im, ax=ax, label=metric_label, shrink=0.8)

            plt.tight_layout()
            safe = metric_label.lower().replace('-', '_')
            out = os.path.join(model_dir, f"heatmap_{safe}_swappct_{pct:.0f}.png")
            plt.savefig(out)
            plt.close()
            print(f"  {out}")


def plot_lines_fraction(df, model, model_dir):
    """Line plot: metric vs fraction, one line per max_distance."""
    df = df.copy()
    df['max_distance'] = df['max_distance'].astype(str)
    colors = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD', '#8C564B']
    markers = ['s', 'o', '^', 'D', 'v', 'P']

    for metric_col, metric_label in METRICS:
        if metric_col not in df.columns:
            continue

        for pct in sorted(df['swap_pct'].unique()):
            pct_df = df[df['swap_pct'] == pct]
            distances = [d for d in DISTANCE_ORDER if d in pct_df['max_distance'].unique()]

            fig, ax = plt.subplots(figsize=(8, 6))

            for i, dist in enumerate(distances):
                dist_df = pct_df[pct_df['max_distance'] == dist].sort_values('fraction')
                ax.plot(dist_df['fraction'], dist_df[metric_col],
                        color=colors[i % len(colors)],
                        marker=markers[i % len(markers)],
                        linewidth=2, markersize=7, label=f'd={dist}')

            ax.set_xlabel('Swap Fraction')
            ax.set_ylabel(metric_label)
            ax.set_title(f'{_style(model)["label"]} — {metric_label} vs Fraction\n'
                         f'(Swap Length = {pct:.1f}% of series)')
            ax.legend(title='Max Distance', loc='best')
            ax.set_ylim(0, 1.05)

            plt.tight_layout()
            safe = metric_label.lower().replace('-', '_')
            out = os.path.join(model_dir, f"line_{safe}_vs_fraction_swappct_{pct:.0f}.png")
            plt.savefig(out)
            plt.close()
            print(f"  {out}")


def main():
    df = load_all_swap_data()
    if df is None:
        print(f"[ERROR] No summary CSVs found in {EXPERIMENT_DIR}")
        return

    print(f"Loaded {len(df)} rows from swap experiments")
    models = df['model'].unique()
    print(f"Models: {list(models)}")

    for model in models:
        model_dir = os.path.join(OUTPUT_DIR, model)
        os.makedirs(model_dir, exist_ok=True)
        mdf = df[df['model'] == model]

        print(f"\n--- {model} ---")
        plot_heatmaps(mdf, model, model_dir)
        plot_lines_fraction(mdf, model, model_dir)

    print(f"\nDone — plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
