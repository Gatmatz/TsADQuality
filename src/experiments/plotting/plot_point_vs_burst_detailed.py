
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
PROJECT_ROOT = "."
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
GE_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "gilbert_elliott_true_impact", "summary.csv")
MCAR_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_mnar", "summary.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "all_plots", "missing_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load baseline
df_base = pd.read_csv(BASELINE_CSV)
df_base['model'] = df_base['model'].replace({'Autoencoder': 'AE'})
baseline_auc = df_base.groupby('model')['AUC_ROC'].mean().to_dict()

def get_drop(df, model_col='model', auc_col='mean_AUC_ROC'):
    df = df.copy()
    df[model_col] = df[model_col].replace({'Autoencoder': 'AE'})
    def calc_drop(row):
        base = baseline_auc.get(row[model_col], 0)
        return base - row[auc_col]
    df['auc_drop'] = df.apply(calc_drop, axis=1)
    return df

# Load Data
if os.path.exists(GE_SUMMARY) and os.path.exists(MCAR_SUMMARY):
    df_ge = get_drop(pd.read_csv(GE_SUMMARY))
    df_mcar = get_drop(pd.read_csv(MCAR_SUMMARY))
    df_mcar = df_mcar[df_mcar['mechanism'] == 'mcar']

    models = sorted(df_mcar['model'].unique())
    burst_lengths = sorted(df_ge['expected_burst_len'].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(22, 6), sharey=True)
    if len(models) == 1: axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]
        
        # 1. Plot Point Missing (MCAR)
        m_data = df_mcar[df_mcar['model'] == model].sort_values('fraction')
        ax.plot(m_data['fraction'], m_data['auc_drop'], color='black', marker='o', 
                label='Point (Random)', linewidth=3, zorder=5)
        
        # 2. Plot Burst Missing (GE) for different lengths
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
        for j, blen in enumerate(burst_lengths):
            b_data = df_ge[(df_ge['model'] == model) & (df_ge['expected_burst_len'] == blen)].sort_values('expected_rate')
            if not b_data.empty:
                ax.plot(b_data['expected_rate'], b_data['auc_drop'], marker='s', 
                        label=f'Burst (len={int(blen)})', alpha=0.7, linestyle='--')
        
        ax.set_title(f"Model: {model}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Missing Rate", fontsize=12)
        if i == 0:
            ax.set_ylabel("AUC-ROC Drop", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.8)

    plt.suptitle("Point vs. Burst Missing: The Severity of Temporal Contiguity", fontsize=18, y=1.05)
    
    # Unified Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.08), fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "detailed_point_vs_burst_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Detailed comparison plot saved to: {save_path}")
else:
    print("Required summaries not found.")
