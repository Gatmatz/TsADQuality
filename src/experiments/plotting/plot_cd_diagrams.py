
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, studentized_range
import os

def calculate_cd(n_datasets, n_models, alpha=0.05):
    """Calculate Critical Difference using Nemenyi test."""
    # q_alpha for alpha=0.05
    # Using scipy's studentized range distribution
    q_alpha = studentized_range.ppf(1 - alpha, n_models, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))
    return cd

def plot_cd_diagram(mean_ranks, models, cd, title, filename):
    """
    Simplified CD Diagram plot.
    mean_ranks: list/array of mean ranks
    models: list of model names
    cd: critical difference value
    """
    n_models = len(models)
    # Sort by rank
    sorted_indices = np.argsort(mean_ranks)
    sorted_ranks = mean_ranks[sorted_indices]
    sorted_models = [models[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Draw the main line
    ax.hlines(0, 1, n_models, colors='black', linewidth=1)
    
    # Draw ticks and labels
    for i in range(1, n_models + 1):
        ax.vlines(i, -0.1, 0.1, colors='black')
        ax.text(i, -0.3, str(i), ha='center', va='top')

    # Plot models
    for i, (rank, model) in enumerate(zip(sorted_ranks, sorted_models)):
        # Determine y position (zigzag to avoid overlap)
        y = 0.5 + (i % 2) * 0.5
        ax.vlines(rank, 0, y, colors='blue', alpha=0.5, linestyle='--')
        ax.text(rank, y + 0.1, model, ha='center', va='bottom', fontweight='bold')
        ax.plot(rank, 0, 'ro')

    # Draw CD bar
    cd_x = 1
    cd_y = 2.0
    ax.hlines(cd_y, cd_x, cd_x + cd, colors='red', linewidth=3)
    ax.text(cd_x + cd/2, cd_y + 0.1, f"CD={cd:.3f}", ha='center', va='bottom', color='red')

    # Draw connections (cliques of non-significant difference)
    # This is a simplified version: draw a line if diff < CD
    # In a real CD diagram, we find all maximal cliques
    for i in range(n_models):
        for j in range(i + 1, n_models):
            if abs(sorted_ranks[i] - sorted_ranks[j]) < cd:
                line_y = -1.0 - (i * 0.2)
                ax.hlines(line_y, sorted_ranks[i], sorted_ranks[j], colors='black', linewidth=4)

    ax.set_ylim(-2.5, 3.0)
    ax.set_xlim(0.5, n_models + 0.5)
    ax.set_axis_off()
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CD Diagram saved: {filename}")

def main():
    PROJECT_ROOT = "."
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "all_plots", "cd_diagrams")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. White Noise (at 0dB)
    noise_path = os.path.join(PROJECT_ROOT, "results", "experiments", "white_noise_snr", "checkpoint.csv")
    if os.path.exists(noise_path):
        df = pd.read_csv(noise_path)
        df = df[df['snr_db'] == 0]
        pivot = df.pivot_table(index='file', columns='model', values='AUC_ROC').dropna()
        if not pivot.empty:
            model_names = pivot.columns.tolist()
            n_datasets = len(pivot)
            n_models = len(model_names)
            
            ranks = np.zeros_like(pivot.values)
            for i in range(n_datasets):
                # rankdata returns 1 for lowest, so we negate AUC to get 1 for highest
                ranks[i] = rankdata(-pivot.values[i])
            
            mean_ranks = ranks.mean(axis=0)
            cd = calculate_cd(n_datasets, n_models)
            
            plot_cd_diagram(mean_ranks, model_names, cd, 
                            "CD Diagram: White Noise at SNR=0dB", 
                            os.path.join(OUTPUT_DIR, "cd_white_noise_0dB.png"))

    # 2. Burst Missing (Gilbert-Elliott, rate=0.1, len=10)
    ge_path = os.path.join(PROJECT_ROOT, "results", "experiments", "gilbert_elliott_true_impact", "checkpoint.csv")
    if os.path.exists(ge_path):
        df = pd.read_csv(ge_path)
        # Choose a representative condition
        df = df[(df['expected_rate'] > 0.09) & (df['expected_rate'] < 0.11) & (df['expected_burst_len'] == 10.0)]
        pivot = df.pivot_table(index='file', columns='model', values='AUC_ROC').dropna()
        if not pivot.empty:
            model_names = pivot.columns.tolist()
            n_datasets = len(pivot)
            n_models = len(model_names)
            
            ranks = np.zeros_like(pivot.values)
            for i in range(n_datasets):
                ranks[i] = rankdata(-pivot.values[i])
            
            mean_ranks = ranks.mean(axis=0)
            cd = calculate_cd(n_datasets, n_models)
            
            plot_cd_diagram(mean_ranks, model_names, cd, 
                            "CD Diagram: Burst Missing (Rate=0.1, Len=10)", 
                            os.path.join(OUTPUT_DIR, "cd_burst_missing.png"))

if __name__ == "__main__":
    main()
