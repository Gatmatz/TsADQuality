
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
PROJECT_ROOT = "."
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
GE_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "gilbert_elliott_true_impact", "summary.csv")
MNAR_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_mnar", "summary.csv")
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

# 1. Gilbert-Elliott Analysis (Burst Lengths)
if os.path.exists(GE_SUMMARY):
    df_ge = get_drop(pd.read_csv(GE_SUMMARY))
    burst_lengths = sorted(df_ge['expected_burst_len'].unique())
    models = sorted(df_ge['model'].unique())

    fig, axes = plt.subplots(1, len(burst_lengths), figsize=(20, 6), sharey=True)
    if len(burst_lengths) == 1: axes = [axes]

    for i, blen in enumerate(burst_lengths):
        ax = axes[i]
        df_sub = df_ge[df_ge['expected_burst_len'] == blen]
        for model in models:
            m_data = df_sub[df_sub['model'] == model].sort_values('expected_rate')
            ax.plot(m_data['expected_rate'], m_data['auc_drop'], marker='o', label=model, linewidth=2)
        
        ax.set_title(f"Burst Length = {int(blen)}", fontsize=14)
        ax.set_xlabel("Missing Rate", fontsize=12)
        if i == 0:
            ax.set_ylabel("AUC-ROC Drop", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Gilbert-Elliott Burst Missing: Impact of Burst Length and Rate", fontsize=18, y=1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(models), bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "missing_ge_burst_length_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"GE plot saved to: {save_path}")

# 2. MCAR vs MNAR (Point Missing)
if os.path.exists(MNAR_SUMMARY):
    df_mnar = get_drop(pd.read_csv(MNAR_SUMMARY))
    mechanisms = sorted(df_mnar['mechanism'].unique()) # mcar, mnar_extreme
    models = sorted(df_mnar['model'].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(20, 6), sharey=True)
    if len(models) == 1: axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]
        for mech in mechanisms:
            m_data = df_mnar[(df_mnar['model'] == model) & (df_mnar['mechanism'] == mech)].sort_values('fraction')
            label = "MCAR (Random)" if mech == 'mcar' else "MNAR (Extreme)"
            ax.plot(m_data['fraction'], m_data['auc_drop'], marker='s' if mech=='mcar' else 'x', label=label, linewidth=2)
        
        ax.set_title(f"{model}", fontsize=14)
        ax.set_xlabel("Fraction Missing", fontsize=12)
        if i == 0:
            ax.set_ylabel("AUC-ROC Drop", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Point Missing: MCAR (Random) vs MNAR (Extreme Values)", fontsize=18, y=1.05)
    handles, labels = ax.get_legend_handles_labels()
    # Unique labels only
    unique_labels = dict(zip(labels, handles))
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "missing_mcar_vs_mnar.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"MCAR vs MNAR plot saved to: {save_path}")
