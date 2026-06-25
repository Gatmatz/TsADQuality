
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
PROJECT_ROOT = "."
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "all_plots", "comparison")
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
mcar_path = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_mnar", "summary.csv")
burst_path = os.path.join(PROJECT_ROOT, "results", "experiments", "gilbert_elliott_true_impact", "summary.csv")

if os.path.exists(mcar_path) and os.path.exists(burst_path):
    df_mcar = get_drop(pd.read_csv(mcar_path))
    df_mcar = df_mcar[df_mcar['mechanism'] == 'mcar']
    
    df_burst = get_drop(pd.read_csv(burst_path))
    # Filter for burst length 10
    df_burst = df_burst[df_burst['expected_burst_len'] == 10.0]

    models = sorted(df_mcar['model'].unique())
    
    fig, axes = plt.subplots(1, len(models), figsize=(20, 6), sharey=True)
    if len(models) == 1: axes = [axes]
    
    for i, model in enumerate(models):
        ax = axes[i]
        
        # MCAR (Point)
        m_data = df_mcar[df_mcar['model'] == model].sort_values('fraction')
        ax.plot(m_data['fraction'], m_data['auc_drop'], marker='o', label='Point (MCAR)', linewidth=2)
        
        # Burst (Gilbert-Elliott)
        b_data = df_burst[df_burst['model'] == model].sort_values('expected_rate')
        ax.plot(b_data['expected_rate'], b_data['auc_drop'], marker='s', label='Burst (GE, len=10)', linewidth=2, linestyle='--')
        
        ax.set_title(f"{model}", fontsize=14)
        ax.set_xlabel("Missing Rate", fontsize=12)
        if i == 0:
            ax.set_ylabel("AUC-ROC Drop", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Point vs Burst Missing: The Impact of Context Erasure", fontsize=18, y=1.05)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "point_vs_burst_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
