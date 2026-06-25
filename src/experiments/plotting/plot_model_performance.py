"""
Publication-quality baseline performance plots.

Reads per-model baseline CSVs and generates comparison charts
across models and dataset families.
Output structure: results/plots/baselines/{Model}/metric_distribution.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
BASELINE_CSV = os.path.join(TABLES_DIR, "baseline_final_subset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "plots", "baselines")

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

MODEL_COLORS = {
    'IForest': '#1f77b4',
    'PCA':     '#1F77B4',
    'LOF':     '#d62728',
    'MP':      '#2ca02c',
    'Autoencoder': '#ff7f0e',
}

METRIC_LABELS = {
    'AUC_ROC': 'AUC-ROC',
    'AUC_PR':  'AUC-PR',
    'VUS_ROC': 'VUS-ROC',
    'VUS_PR':  'VUS-PR',
    'F1':      'F1 Score',
}


def load_baseline():
    """Load baseline CSV; try combined file first, then per-model files."""
    if os.path.exists(BASELINE_CSV):
        return pd.read_csv(BASELINE_CSV)

    # Fallback: merge per-model CSVs
    frames = []
    for f in os.listdir(TABLES_DIR):
        if f.startswith('baseline_') and f.endswith('.csv') and f != os.path.basename(BASELINE_CSV):
            frames.append(pd.read_csv(os.path.join(TABLES_DIR, f)))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return None


def plot_avg_per_model(df):
    """Bar chart: average metric per model."""
    out_dir = os.path.join(OUTPUT_DIR, "combined")
    os.makedirs(out_dir, exist_ok=True)

    for metric, label in METRIC_LABELS.items():
        if metric not in df.columns:
            continue

        summary = df.groupby('model')[metric].agg(['mean', 'std']).sort_values('mean', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [MODEL_COLORS.get(m, 'gray') for m in summary.index]

        bars = ax.bar(summary.index, summary['mean'], yerr=summary['std'],
                      color=colors, edgecolor='white', capsize=5, alpha=0.9)

        for bar, val in zip(bars, summary['mean']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

        ax.set_ylabel(label)
        ax.set_title(f'Baseline {label} per Model')
        ax.set_ylim(0, 1.15)

        plt.tight_layout()
        out = os.path.join(out_dir, f"baseline_avg_{metric.lower()}.png")
        plt.savefig(out)
        plt.close()
        print(f"  {out}")


def plot_boxplot_per_model(df):
    """Boxplot: metric distribution per model."""
    for model in df['model'].unique():
        model_dir = os.path.join(OUTPUT_DIR, model)
        os.makedirs(model_dir, exist_ok=True)

        mdf = df[df['model'] == model]
        metrics = [m for m in METRIC_LABELS if m in df.columns]

        fig, ax = plt.subplots(figsize=(8, 6))
        data = [mdf[m].dropna().values for m in metrics]
        labels = [METRIC_LABELS[m] for m in metrics]

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                        flierprops=dict(marker='o', markersize=3, alpha=0.3))

        color = MODEL_COLORS.get(model, 'gray')
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Score')
        ax.set_title(f'{model} — Baseline Metric Distributions')
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        out = os.path.join(model_dir, f"baseline_metric_distributions.png")
        plt.savefig(out)
        plt.close()
        print(f"  {out}")


def plot_heatmap_by_folder(df):
    """Heatmap: average metric per model × dataset family."""
    out_dir = os.path.join(OUTPUT_DIR, "combined")
    os.makedirs(out_dir, exist_ok=True)

    for metric, label in METRIC_LABELS.items():
        if metric not in df.columns:
            continue

        pivot = df.groupby(['folder', 'model'])[metric].mean().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
        im = ax.imshow(pivot.values, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=11)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                text_color = 'white' if val > 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=9, color=text_color)

        ax.set_xlabel('Model')
        ax.set_ylabel('Dataset Family')
        ax.set_title(f'Baseline {label} — Models × Dataset Families')
        plt.colorbar(im, ax=ax, label=label, shrink=0.8)

        plt.tight_layout()
        out = os.path.join(out_dir, f"heatmap_baseline_{metric.lower()}_by_family.png")
        plt.savefig(out)
        plt.close()
        print(f"  {out}")


def main():
    df = load_baseline()
    if df is None:
        print(f"[ERROR] No baseline CSVs found in {TABLES_DIR}")
        return

    print(f"Loaded {len(df)} baseline records ({df['model'].nunique()} models)")

    plot_avg_per_model(df)
    plot_boxplot_per_model(df)
    plot_heatmap_by_folder(df)

    print(f"\nDone — plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
