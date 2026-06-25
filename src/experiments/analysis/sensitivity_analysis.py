"""
Sensitivity Analysis — Which series characteristics predict degradation?

For each corruption experiment, computes per-file degradation (baseline AUC - corrupted AUC)
and correlates it with series characteristics:
  - data_len:       Series length
  - ratio:          Anomaly ratio (% of points that are anomalies)
  - baseline_auc:   Baseline AUC-ROC (how well the model does without corruption)
  - anomaly_length: Mean anomaly segment length
  - n_anomalies:    Number of anomaly segments
  - variance:       Variance of the series values

Output:
  - Correlation table (Spearman) for each experiment × characteristic
  - Scatter plots: degradation vs each characteristic
  - Summary heatmap: all experiments × all characteristics

Usage:
    python sensitivity_analysis.py
    python sensitivity_analysis.py --experiments missing_true_impact freeze
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================
# PATH CONFIGURATION
# ==========================================
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path: sys.path.insert(0, project_root)
if src_path not in sys.path: sys.path.insert(0, src_path)

from data_loader import load_tsb_dataframe

RESULTS_BASE = os.path.join(project_root, "results", "experiments")
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
OUTPUT_DIR = os.path.join(project_root, "results", "analysis", "sensitivity")

# Experiments to analyze — name: (results_dir, condition_to_use)
# condition_to_use: the "moderate" condition for each experiment
EXPERIMENTS = {
    'missing':       {'dir': 'missing_true_impact', 'fraction': 0.10, 'extra_filter': {'num_bursts': 5}},
    'freeze':        {'dir': 'freeze',              'fraction': 0.10, 'extra_filter': {'num_stucks': 5}},
    'swap_point':    {'dir': 'swap_point',           'fraction': 0.10, 'extra_filter': {}},
    'swap_segment':  {'dir': 'swap_segment',         'fraction': 0.10, 'extra_filter': {'num_swaps': 5}},
    'noise_snr10':   {'dir': 'white_noise_snr',      'fraction': None, 'extra_filter': {'snr_db': 10}},
    'spikes':        {'dir': 'spikes',               'fraction': 0.10, 'extra_filter': {'multiplier': 5}},
}

CHARACTERISTICS = ['data_len', 'ratio', 'baseline_auc', 'anomaly_length', 'n_anomalies', 'variance']
CHAR_LABELS = {
    'data_len': 'Series Length',
    'ratio': 'Anomaly Ratio',
    'baseline_auc': 'Baseline AUC-ROC',
    'anomaly_length': 'Mean Anomaly Length',
    'n_anomalies': 'Number of Anomaly Segments',
    'variance': 'Series Variance',
}


def compute_series_characteristics(subset_df):
    """Compute characteristics for each series in the subset."""
    rows = []
    for _, meta in subset_df.iterrows():
        filepath = meta['filepath']
        filename = meta['filename_actual']
        try:
            df, _ = load_tsb_dataframe(filepath)
            values = df['value'].to_numpy(float)
            labels = df['is_anomaly'].to_numpy(int)

            # Count anomaly segments
            n_anomalies = 0
            anomaly_lengths = []
            in_anomaly = False
            current_len = 0
            for lab in labels:
                if lab == 1:
                    if not in_anomaly:
                        n_anomalies += 1
                        in_anomaly = True
                        current_len = 1
                    else:
                        current_len += 1
                else:
                    if in_anomaly:
                        anomaly_lengths.append(current_len)
                        in_anomaly = False
                        current_len = 0
            if in_anomaly:
                anomaly_lengths.append(current_len)

            rows.append({
                'file': filename,
                'data_len': len(df),
                'ratio': meta['ratio'],
                'baseline_auc': meta['mean_AUC_ROC'],
                'anomaly_length': np.mean(anomaly_lengths) if anomaly_lengths else 0,
                'n_anomalies': n_anomalies,
                'variance': np.var(values),
            })
        except Exception as e:
            print(f"  Warning: could not load {filename}: {e}")

    return pd.DataFrame(rows)


def load_experiment_results(exp_name, exp_config):
    """Load per-file results for a specific experiment condition."""
    checkpoint = os.path.join(RESULTS_BASE, exp_config['dir'], 'checkpoint.csv')
    if not os.path.exists(checkpoint):
        print(f"  [SKIP] {exp_name}: checkpoint not found at {checkpoint}")
        return None

    df = pd.read_csv(checkpoint)
    df = df[df['model'] == 'IForest']
    if 'error' in df.columns:
        df = df[df['error'].isna()]

    # Apply filters
    if exp_config['fraction'] is not None and 'fraction' in df.columns:
        df = df[df['fraction'] == exp_config['fraction']]

    for col, val in exp_config['extra_filter'].items():
        if col in df.columns:
            df = df[df[col] == val]

    if df.empty:
        print(f"  [SKIP] {exp_name}: no results after filtering")
        return None

    # Per-file mean AUC
    per_file = df.groupby('file')['AUC_ROC'].mean().reset_index()
    per_file.columns = ['file', 'corrupted_auc']
    return per_file


def compute_correlations(chars_df, experiments_results):
    """Compute Spearman correlation between degradation and each characteristic."""
    rows = []
    for exp_name, per_file in experiments_results.items():
        merged = chars_df.merge(per_file, on='file', how='inner')
        merged['degradation'] = merged['baseline_auc'] - merged['corrupted_auc']

        for char in CHARACTERISTICS:
            if char == 'baseline_auc':
                # Correlation with corrupted_auc directly (not degradation)
                rho, pval = stats.spearmanr(merged[char], merged['corrupted_auc'])
            else:
                rho, pval = stats.spearmanr(merged[char], merged['degradation'])

            rows.append({
                'experiment': exp_name,
                'characteristic': char,
                'spearman_rho': round(rho, 4),
                'p_value': round(pval, 6),
                'significant': pval < 0.05,
                'n_files': len(merged),
            })

    return pd.DataFrame(rows)


def plot_scatter_per_experiment(chars_df, experiments_results):
    """Scatter plots: degradation vs each characteristic, per experiment."""
    scatter_dir = os.path.join(OUTPUT_DIR, 'scatter_plots')
    os.makedirs(scatter_dir, exist_ok=True)

    for exp_name, per_file in experiments_results.items():
        merged = chars_df.merge(per_file, on='file', how='inner')
        merged['degradation'] = merged['baseline_auc'] - merged['corrupted_auc']

        chars_to_plot = [c for c in CHARACTERISTICS if c != 'baseline_auc']
        fig, axes = plt.subplots(1, len(chars_to_plot), figsize=(5 * len(chars_to_plot), 4))

        for ax, char in zip(axes, chars_to_plot):
            ax.scatter(merged[char], merged['degradation'], alpha=0.4, s=15)

            # Trend line
            rho, pval = stats.spearmanr(merged[char], merged['degradation'])
            z = np.polyfit(merged[char], merged['degradation'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(merged[char].min(), merged[char].max(), 100)
            ax.plot(x_range, p(x_range), 'r--', alpha=0.7)

            ax.set_xlabel(CHAR_LABELS[char], fontsize=10)
            ax.set_ylabel('Degradation (AUC)', fontsize=10)
            sig = '*' if pval < 0.05 else ''
            ax.set_title(f'ρ={rho:.3f}{sig}', fontsize=11)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{exp_name}: Degradation vs Series Characteristics', fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(scatter_dir, f'scatter_{exp_name}.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: scatter_{exp_name}.png")


def plot_heatmap(corr_df):
    """Heatmap: experiments × characteristics — Spearman rho values."""
    experiments = corr_df['experiment'].unique()
    chars = CHARACTERISTICS

    matrix = np.full((len(experiments), len(chars)), np.nan)
    pval_matrix = np.full((len(experiments), len(chars)), np.nan)

    for i, exp in enumerate(experiments):
        for j, char in enumerate(chars):
            sub = corr_df[(corr_df['experiment'] == exp) & (corr_df['characteristic'] == char)]
            if not sub.empty:
                matrix[i, j] = sub['spearman_rho'].values[0]
                pval_matrix[i, j] = sub['p_value'].values[0]

    fig, ax = plt.subplots(figsize=(10, max(4, len(experiments) * 0.8)))
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-0.6, vmax=0.6)

    ax.set_xticks(range(len(chars)))
    ax.set_xticklabels([CHAR_LABELS[c] for c in chars], rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels(experiments, fontsize=10)
    ax.set_title('Spearman Correlation: Degradation vs Series Characteristics', fontsize=13)

    for i in range(len(experiments)):
        for j in range(len(chars)):
            val = matrix[i, j]
            pval = pval_matrix[i, j]
            if not np.isnan(val):
                sig = '*' if pval < 0.05 else ''
                color = 'white' if abs(val) > 0.3 else 'black'
                ax.text(j, i, f'{val:.2f}{sig}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

    fig.colorbar(im, ax=ax, label='Spearman ρ', shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: correlation_heatmap.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', default=None,
                        help='Specific experiments to analyze (default: all)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load subset metadata
    print("Loading subset metadata...")
    subset_df = pd.read_csv(SUBSET_CSV)
    print(f"  {len(subset_df)} files in subset")

    # 2. Compute series characteristics
    print("Computing series characteristics...")
    chars_csv = os.path.join(OUTPUT_DIR, 'series_characteristics.csv')
    if os.path.exists(chars_csv):
        chars_df = pd.read_csv(chars_csv)
        print(f"  Loaded cached characteristics for {len(chars_df)} files")
    else:
        chars_df = compute_series_characteristics(subset_df)
        chars_df.to_csv(chars_csv, index=False)
        print(f"  Computed and saved characteristics for {len(chars_df)} files")

    # 3. Load experiment results
    print("\nLoading experiment results...")
    experiments_to_run = EXPERIMENTS
    if args.experiments:
        experiments_to_run = {k: v for k, v in EXPERIMENTS.items() if k in args.experiments}

    experiments_results = {}
    for exp_name, exp_config in experiments_to_run.items():
        per_file = load_experiment_results(exp_name, exp_config)
        if per_file is not None:
            experiments_results[exp_name] = per_file
            print(f"  {exp_name}: {len(per_file)} files loaded")

    if not experiments_results:
        print("\nNo experiment results found!")
        return

    # 4. Compute correlations
    print("\nComputing Spearman correlations...")
    corr_df = compute_correlations(chars_df, experiments_results)
    corr_df.to_csv(os.path.join(OUTPUT_DIR, 'correlations.csv'), index=False)

    # Print correlation table
    print(f"\n{'Experiment':<16} {'Characteristic':<20} {'ρ':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 65)
    for _, row in corr_df.sort_values(['experiment', 'characteristic']).iterrows():
        sig = '*' if row['significant'] else ''
        print(f"{row['experiment']:<16} {row['characteristic']:<20} {row['spearman_rho']:>8.4f} "
              f"{row['p_value']:>10.6f} {sig:>5}")

    # 5. Plot
    print("\nGenerating plots...")
    plot_heatmap(corr_df)
    plot_scatter_per_experiment(chars_df, experiments_results)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
