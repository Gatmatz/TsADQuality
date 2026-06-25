"""
Publication-quality time-series corruption visualizations.

Shows original vs corrupted signals side by side with anomaly regions
highlighted. Useful for thesis figures demonstrating corruption effects.
Output structure: results/plots/corruption_examples/comparison_*.png
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Path configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUBSET_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "robust_subset_TSB.csv")
CORRUPT_DIR = os.path.join(PROJECT_ROOT, "results", "corrupted_data", "adv_combinatorial_test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "plots", "corruption_examples")
NUM_EXAMPLES = 6

# Thesis-quality classic style
plt.style.use('classic')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})# How many datasets to plot


def load_experiment_log(corrupt_dir):
    log_path = os.path.join(corrupt_dir, "experiment_log.json")
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_single_comparison(original_path, corrupt_path, title, output_path, log_entry=None):
    """Plot original vs corrupted time series with anomaly regions highlighted."""
    df_orig = pd.read_csv(original_path, header=None, names=['value', 'label'])
    df_corr = pd.read_csv(corrupt_path, header=None, names=['value', 'label'])

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                              gridspec_kw={'height_ratios': [3, 3, 1]})
    fig.suptitle(title, fontsize=14, fontweight='bold')

    x = np.arange(len(df_orig))

    # -- Panel 1: Original --
    axes[0].plot(x, df_orig['value'], color='#2196F3', linewidth=0.5, alpha=0.9)
    axes[0].set_ylabel('Original', fontsize=11)
    axes[0].set_title('Clean Signal', fontsize=10, color='gray')

    # Highlight anomaly regions
    anomaly_mask = df_orig['label'] == 1
    if anomaly_mask.any():
        axes[0].fill_between(x, df_orig['value'].min(), df_orig['value'].max(),
                             where=anomaly_mask, alpha=0.15, color='red', label='Anomaly')

    # -- Panel 2: Corrupted --
    axes[1].plot(x, df_corr['value'], color='#FF5722', linewidth=0.5, alpha=0.9)
    axes[1].set_ylabel('Corrupted', fontsize=11)
    axes[1].set_title('After Data Issues', fontsize=10, color='gray')

    # Highlight NaN regions (burst missing)
    nan_mask = df_corr['value'].isna()
    if nan_mask.any():
        axes[1].fill_between(x, df_corr['value'].min(skipna=True), df_corr['value'].max(skipna=True),
                             where=nan_mask, alpha=0.3, color='orange', label='Missing Data')

    # Highlight anomaly regions on corrupted too
    if anomaly_mask.any():
        axes[1].fill_between(x, df_corr['value'].min(skipna=True), df_corr['value'].max(skipna=True),
                             where=anomaly_mask, alpha=0.15, color='red', label='Anomaly')

    # -- Panel 3: Difference --
    diff = df_corr['value'].fillna(0) - df_orig['value'].fillna(0)
    axes[2].fill_between(x, 0, diff, where=(diff > 0), color='#4CAF50', alpha=0.6)
    axes[2].fill_between(x, 0, diff, where=(diff < 0), color='#F44336', alpha=0.6)
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].set_ylabel('Difference', fontsize=11)
    axes[2].set_xlabel('Time Index', fontsize=11)
    axes[2].set_title('Signal Deviation (Corrupted - Original)', fontsize=10, color='gray')

    # Add legend
    if log_entry:
        summary = log_entry.get('summary', {})
        info_text = (f"Corrupted: {summary.get('corrupted_points', '?')} pts "
                     f"({summary.get('corruption_percentage', '?'):.1f}%) | "
                     f"Avg dist to anomaly: {summary.get('avg_distance_to_anomaly', '?'):.0f}")
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=9, style='italic', color='#666')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_corruption_summary(corrupt_dir):
    """Plot a bar chart summarizing corruption statistics across all datasets."""
    log = load_experiment_log(corrupt_dir)
    stats = log.get('dataset_stats', [])

    folders = [s['folder'] for s in stats]
    corrupted_pcts = [s['summary']['corruption_percentage'] for s in stats]

    df_stats = pd.DataFrame({'folder': folders, 'corruption_pct': corrupted_pcts})
    avg_by_folder = df_stats.groupby('folder')['corruption_pct'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(avg_by_folder)))
    bars = ax.bar(avg_by_folder.index, avg_by_folder.values, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Average Corruption %', fontsize=12)
    ax.set_xlabel('Dataset Family', fontsize=12)
    ax.set_title(f"Corruption Coverage by Dataset Family\n({log['config']['experiment_name']})",
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar, val in zip(bars, avg_by_folder.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "corruption_summary.png")
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"  Saved summary: {summary_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load metadata
    df_subset = pd.read_csv(SUBSET_CSV)
    log = load_experiment_log(CORRUPT_DIR)
    stats = log.get('dataset_stats', [])

    # Build lookup: baseline_name -> log entry
    stats_lookup = {s['baseline_name']: s for s in stats}

    # Pick diverse examples (one per folder where possible)
    seen_folders = set()
    examples = []
    for _, row in df_subset.iterrows():
        if row['folder'] not in seen_folders and row['baseline_name'] in stats_lookup:
            examples.append(row)
            seen_folders.add(row['folder'])
        if len(examples) >= NUM_EXAMPLES:
            break

    print(f"Plotting {len(examples)} example comparisons...\n")

    for i, row in enumerate(examples):
        original_path = row['filepath']
        corrupt_filename = f"{row['folder']}_{row['baseline_name']}"
        corrupt_path = os.path.join(CORRUPT_DIR, corrupt_filename)

        if not os.path.exists(corrupt_path):
            print(f"  Skipped (not found): {corrupt_filename}")
            continue

        title = f"{row['folder']} / {row['baseline_name']}"
        out_path = os.path.join(OUTPUT_DIR, f"comparison_{i+1}_{row['folder']}.png")

        log_entry = stats_lookup.get(row['baseline_name'])
        plot_single_comparison(original_path, corrupt_path, title, out_path, log_entry)

    # Summary bar chart
    print("\nGenerating summary chart...")
    plot_corruption_summary(CORRUPT_DIR)

    print(f"\nDone! All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
