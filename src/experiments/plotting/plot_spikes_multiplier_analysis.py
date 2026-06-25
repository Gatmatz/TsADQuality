
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
PROJECT_ROOT = "."
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
SPIKES_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "spikes_normal_only", "summary.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "all_plots", "spikes_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load baseline
df_base = pd.read_csv(BASELINE_CSV)
df_base['model'] = df_base['model'].replace({'Autoencoder': 'AE'})
baseline_auc = df_base.groupby('model')['AUC_ROC'].mean().to_dict()

# Load spikes summary
if os.path.exists(SPIKES_SUMMARY):
    df_spikes = pd.read_csv(SPIKES_SUMMARY)
    df_spikes['model'] = df_spikes['model'].replace({'Autoencoder': 'AE'})
    
    # Calculate drop
    def calc_drop(row):
        base = baseline_auc.get(row['model'], 0)
        return base - row['mean_AUC_ROC']
    df_spikes['auc_drop'] = df_spikes.apply(calc_drop, axis=1)

    multipliers = sorted(df_spikes['multiplier'].unique())
    models = sorted(df_spikes['model'].unique())

    # Create a figure with 1 row and N columns (one for each multiplier)
    fig, axes = plt.subplots(1, len(multipliers), figsize=(20, 6), sharey=True)
    
    for i, mult in enumerate(multipliers):
        ax = axes[i]
        df_mult = df_spikes[df_spikes['multiplier'] == mult]
        
        for model in models:
            m_data = df_mult[df_mult['model'] == model].sort_values('fraction')
            ax.plot(m_data['fraction'], m_data['auc_drop'], marker='o', label=model, linewidth=2)
        
        ax.set_title(f"Multiplier = {mult}x", fontsize=14)
        ax.set_xlabel("Fraction of Spikes", fontsize=12)
        if i == 0:
            ax.set_ylabel("AUC-ROC Drop", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xticks([0.01, 0.05, 0.10, 0.20])

    plt.suptitle("Impact of Spikes: AUC-ROC Drop by Multiplier and Fraction", fontsize=18, y=1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(models), bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "spikes_drop_by_multiplier.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Spikes multiplier plot saved to: {save_path}")

    # Also save individual plots per multiplier
    for mult in multipliers:
        plt.figure(figsize=(8, 5))
        df_mult = df_spikes[df_spikes['multiplier'] == mult]
        for model in models:
            m_data = df_mult[df_mult['model'] == model].sort_values('fraction')
            plt.plot(m_data['fraction'], m_data['auc_drop'], marker='o', label=model, linewidth=2)
        
        plt.title(f"Performance Drop: Spikes with {mult}x Multiplier", fontsize=14)
        plt.xlabel("Fraction of Spikes", fontsize=12)
        plt.ylabel("AUC-ROC Drop", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        filename = f"drop_spikes_mult_{int(mult)}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        plt.close()
        print(f"Saved individual spike plot: {filename}")
else:
    print("Spikes summary not found.")
