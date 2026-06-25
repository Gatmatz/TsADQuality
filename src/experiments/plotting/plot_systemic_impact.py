
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, spearmanr
from pathlib import Path
import os
import sys

# Paths
PROJECT_ROOT = Path(".").resolve()
src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path: sys.path.insert(0, str(src_path))

from data_loader import load_tsb_dataframe

SUBSET_CSV = PROJECT_ROOT / "results" / "tables" / "final_subset.csv"
BASELINE_CSV = PROJECT_ROOT / "results" / "tables" / "baseline_final_subset.csv"
NOISE_CKPT = PROJECT_ROOT / "results" / "experiments" / "white_noise_snr" / "checkpoint.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments" / "all_plots" / "systemic_impact"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Failure Thresholds
FN_THRESHOLD = 0.10 # Recall < 0.10 is a serious failure

def calculate_complexity(filepath):
    """Calculate DNA-like metrics for a time series."""
    try:
        df, _ = load_tsb_dataframe(filepath)
        v = df['value'].values
        # 1. Entropy (binned)
        hist, _ = np.histogram(v, bins=100, density=True)
        ent = entropy(hist + 1e-10)
        # 2. Volatility (Std Dev)
        std = np.std(v)
        # 3. Autocorrelation (lag 1)
        if len(v) > 1:
            ac = np.corrcoef(v[:-1], v[1:])[0, 1]
        else:
            ac = 0
        return {'entropy': ent, 'volatility': std, 'autocorr': ac}
    except:
        return {'entropy': np.nan, 'volatility': np.nan, 'autocorr': np.nan}

def main():
    print("Starting Systemic Impact Analysis...")
    
    # 1. Load Baseline and Noise Data
    df_base = pd.read_csv(BASELINE_CSV)
    df_base['model'] = df_base['model'].replace({'Autoencoder': 'AE'})
    
    df_noise = pd.read_csv(NOISE_CKPT)
    df_noise['model'] = df_noise['model'].replace({'Autoencoder': 'AE'})
    
    # 2. Identify Systemic Failures (Consensus)
    # Let's focus on SNR=0dB (Moderate Noise)
    snr_target = 0
    df_target = df_noise[df_noise['snr_db'] == snr_target].copy()
    
    # Pivot to see which models failed on which files
    # Failure = Recall < FN_THRESHOLD
    df_target['failed'] = (df_target['Recall'] < FN_THRESHOLD).astype(int)
    pivot_fail = df_target.pivot_table(index='file', columns='model', values='failed').fillna(0)
    
    # Count how many models failed for each file
    pivot_fail['n_models_failed'] = pivot_fail.sum(axis=1)
    consensus_counts = pivot_fail['n_models_failed'].value_counts().sort_index()
    
    # Plot Consensus Histogram
    plt.figure(figsize=(10, 6))
    sns.barplot(x=consensus_counts.index, y=consensus_counts.values, palette="Reds")
    plt.title(f"Systemic Failure Consensus (SNR={snr_target}dB)", fontsize=16)
    plt.xlabel("Number of Models Failing Simultaneously (Recall < 0.10)", fontsize=12)
    plt.ylabel("Number of Datasets", fontsize=12)
    for i, v in enumerate(consensus_counts.values):
        plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(OUTPUT_DIR / "systemic_failure_consensus.png", dpi=300)
    print(f"Saved: systemic_failure_consensus.png")

    # 3. Time Series "DNA" Correlation
    print("Calculating complexity for 141 datasets...")
    df_meta = pd.read_csv(SUBSET_CSV)
    complexity_results = []
    for _, row in df_meta.iterrows():
        c = calculate_complexity(row['filepath'])
        c['file'] = row['baseline_name']
        complexity_results.append(c)
    
    df_dna = pd.DataFrame(complexity_results)
    
    # Merge with AUC Drop
    df_agg_drop = df_noise[df_noise['snr_db'] == snr_target].copy()
    # Need to match with baseline to get individual drops
    df_merged = df_agg_drop.merge(df_base[['file', 'model', 'AUC_ROC']], on=['file', 'model'], suffixes=('_noise', '_base'))
    df_merged['auc_drop'] = df_merged['AUC_ROC_base'] - df_merged['AUC_ROC_noise']
    
    # Average drop per file across all models
    file_impact = df_merged.groupby('file')['auc_drop'].mean().reset_index()
    file_impact = file_impact.merge(df_dna, on='file')
    
    # Correlation Analysis
    corr_results = []
    for metric in ['entropy', 'volatility', 'autocorr']:
        r, p = spearmanr(file_impact[metric], file_impact['auc_drop'], nan_policy='omit')
        corr_results.append({'Metric': metric, 'Spearman_R': r, 'p_value': p})
    
    df_corr = pd.DataFrame(corr_results)
    df_corr.to_csv(OUTPUT_DIR / "dna_impact_correlation.csv", index=False)
    print("\nImpact Correlation (DNA vs. AUC Drop):")
    print(df_corr.to_string(index=False))

    # Scatter Plot: Entropy vs Impact
    plt.figure(figsize=(10, 6))
    sns.regplot(data=file_impact, x='entropy', y='auc_drop', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title("Correlation: Signal Entropy vs. Noise Impact (AUC Drop)", fontsize=16)
    plt.xlabel("Signal Entropy (Complexity)", fontsize=12)
    plt.ylabel("Mean AUC-ROC Drop (at 0dB)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(OUTPUT_DIR / "entropy_vs_impact_scatter.png", dpi=300)
    print(f"Saved: entropy_vs_impact_scatter.png")

    # 4. Systemic Resilience (Files where NO model failed)
    resilient_files = pivot_fail[pivot_fail['n_models_failed'] == 0].index.tolist()
    print(f"\nResilient Datasets (All models held up at 0dB): {len(resilient_files)}")
    with open(OUTPUT_DIR / "resilient_datasets.txt", "w") as f:
        f.write("\n".join(resilient_files))

if __name__ == "__main__":
    main()
