
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
PROJECT_ROOT = "."
JACCARD_CSV = os.path.join(PROJECT_ROOT, "results", "analysis", "failure_transferability", "pairwise_jaccard_all.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "all_plots", "systemic_impact")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_jaccard_heatmap(df, experiment, failure_type, title, filename):
    # Filter data
    sub = df[(df['experiment'] == experiment) & (df['failure_type'] == failure_type)].copy()
    if sub.empty:
        return

    # Average Jaccard across all severities for this experiment to get a "General Similarity"
    # Or we can pick the highest severity for a "Crisis Similarity"
    sub = sub.groupby(['model_a', 'model_b'])['jaccard'].mean().reset_index()

    models = sorted(list(set(sub['model_a']) | set(sub['model_b'])))
    matrix = pd.DataFrame(np.ones((len(models), len(models))), index=models, columns=models)

    for _, row in sub.iterrows():
        matrix.loc[row['model_a'], row['model_b']] = row['jaccard']
        matrix.loc[row['model_b'], row['model_a']] = row['jaccard']

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def main():
    if not os.path.exists(JACCARD_CSV):
        print("Jaccard CSV not found. Please run failure_transferability.py first.")
        return

    df = pd.read_csv(JACCARD_CSV)

    # 1. White Noise Failure Transferability (FN - Missing Anomalies)
    create_jaccard_heatmap(df, "White Noise", "fn", 
                           "Failure Transferability: White Noise (FN Jaccard)", 
                           "heatmap_transferability_noise_fn.png")

    # 2. Spikes Failure Transferability (FP - False Alarms)
    create_jaccard_heatmap(df, "Spikes", "fp", 
                           "Failure Transferability: Spikes (FP Jaccard)", 
                           "heatmap_transferability_spikes_fp.png")
    
    # 3. Missing Data Failure Transferability (FN)
    create_jaccard_heatmap(df, "Missing Data", "fn", 
                           "Failure Transferability: Missing Data (FN Jaccard)", 
                           "heatmap_transferability_missing_fn.png")

if __name__ == "__main__":
    main()
