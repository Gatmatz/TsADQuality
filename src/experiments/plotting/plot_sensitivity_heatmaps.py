
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
PROJECT_ROOT = "."
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "all_plots", "sensitivity")
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

def plot_heatmap(df, x_col, y_col, val_col, model, title, filename):
    plt.figure(figsize=(8, 6))
    pivot_df = df[df['model'] == model].pivot(index=y_col, columns=x_col, values=val_col)
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={'label': 'AUC Drop'})
    plt.title(f"{title} - {model}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

# 1. Gilbert-Elliott (Burst Length vs Rate)
ge_path = os.path.join(PROJECT_ROOT, "results", "experiments", "gilbert_elliott_true_impact", "summary.csv")
if os.path.exists(ge_path):
    df = get_drop(pd.read_csv(ge_path))
    for model in df['model'].unique():
        plot_heatmap(df, 'expected_burst_len', 'expected_rate', 'auc_drop', 
                     model, "Gilbert-Elliott Sensitivity", f"heatmap_ge_{model}.png")

# 2. Spikes (Multiplier vs Fraction)
spikes_path = os.path.join(PROJECT_ROOT, "results", "experiments", "spikes_normal_only", "summary.csv")
if os.path.exists(spikes_path):
    df = get_drop(pd.read_csv(spikes_path))
    for model in df['model'].unique():
        plot_heatmap(df, 'multiplier', 'fraction', 'auc_drop', 
                     model, "Spikes Sensitivity", f"heatmap_spikes_{model}.png")

# 3. Swap (Num Swaps vs Fraction)
swap_path = os.path.join(PROJECT_ROOT, "results", "experiments", "swap_segment", "summary.csv")
if os.path.exists(swap_path):
    df = get_drop(pd.read_csv(swap_path))
    for model in df['model'].unique():
        plot_heatmap(df, 'num_swaps', 'fraction', 'auc_drop', 
                     model, "Swap Sensitivity", f"heatmap_swap_{model}.png")

# 4. Freeze (Num Stucks vs Fraction)
freeze_path = os.path.join(PROJECT_ROOT, "results", "experiments", "freeze", "summary.csv")
if os.path.exists(freeze_path):
    df = get_drop(pd.read_csv(freeze_path))
    for model in df['model'].unique():
        plot_heatmap(df, 'num_stucks', 'fraction', 'auc_drop', 
                     model, "Freeze Sensitivity", f"heatmap_freeze_{model}.png")

print(f"Sensitivity heatmaps saved to: {OUTPUT_DIR}")
