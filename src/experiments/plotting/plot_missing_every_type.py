
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
PROJECT_ROOT = "."
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "all_plots", "missing_every_type")
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
        # Handle case where auc_col is already ROC (mean_AUC_ROC)
        return base - row[auc_col]
    df['auc_drop'] = df.apply(calc_drop, axis=1)
    return df

def save_mechanism_plot(df, x_col, title, filename, xlabel="Missing Rate"):
    plt.figure(figsize=(10, 6))
    models = sorted(df['model'].unique())
    for model in models:
        m_data = df[df['model'] == model].sort_values(x_col)
        if not m_data.empty:
            plt.plot(m_data[x_col], m_data['auc_drop'], marker='o', label=model, linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("AUC-ROC Drop", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"Saved: {filename}")

# 1. Point MCAR (from missing_mnar)
m_path = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_mnar", "summary.csv")
if os.path.exists(m_path):
    df = get_drop(pd.read_csv(m_path))
    save_mechanism_plot(df[df['mechanism'] == 'mcar'], 'fraction', "Point Missing: MCAR (Random)", "drop_point_mcar.png")

# 2. Point MNAR (from missing_mnar)
if os.path.exists(m_path):
    df = get_drop(pd.read_csv(m_path))
    save_mechanism_plot(df[df['mechanism'] == 'mnar_extreme'], 'fraction', "Point Missing: MNAR (Extreme)", "drop_point_mnar.png")

# 3. Gilbert-Elliott Burst (from gilbert_elliott_true_impact)
ge_path = os.path.join(PROJECT_ROOT, "results", "experiments", "gilbert_elliott_true_impact", "summary.csv")
if os.path.exists(ge_path):
    df_ge = get_drop(pd.read_csv(ge_path))
    for blen in sorted(df_ge['expected_burst_len'].unique()):
        save_mechanism_plot(df_ge[df_ge['expected_burst_len'] == blen], 'expected_rate', 
                            f"Burst Missing: Gilbert-Elliott (len={int(blen)})", f"drop_burst_ge_len{int(blen)}.png")

# 4. Long Burst Analysis (from missing_burst_length)
bl_path = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_burst_length", "summary.csv")
if os.path.exists(bl_path):
    df_bl = get_drop(pd.read_csv(bl_path))
    # This one often has columns like mean_burst_length or burst_ratio
    # Group by fraction to see the impact of burst ratio/length
    save_mechanism_plot(df_bl, 'fraction', "Burst Missing: Variable Burst Lengths", "drop_burst_long_variable.png", xlabel="Fraction missing")

# 5. Missing with Imputation (from missing/checkpoint.csv -> need to summarize)
missing_ckpt = os.path.join(PROJECT_ROOT, "results", "experiments", "missing", "checkpoint.csv")
if os.path.exists(missing_ckpt):
    df_ckpt = pd.read_csv(missing_ckpt)
    # Filter for successful runs and group
    df_sum = df_ckpt.groupby(['missing_type', 'imputation', 'fraction', 'model'])['AUC_ROC'].mean().reset_index()
    df_sum = get_drop(df_sum, auc_col='AUC_ROC')
    
    for mtype in df_sum['missing_type'].unique():
        for imp in df_sum['imputation'].unique():
            sub = df_sum[(df_sum['missing_type'] == mtype) & (df_sum['imputation'] == imp)]
            if not sub.empty:
                save_mechanism_plot(sub, 'fraction', f"Missing: {mtype.capitalize()} (Imputation: {imp})", 
                                    f"drop_{mtype}_{imp}.png")

print(f"Every type of missing data plot saved to: {OUTPUT_DIR}")
