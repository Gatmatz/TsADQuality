
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
PROJECT_ROOT = "."
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
GE_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "gilbert_elliott_true_impact", "summary.csv")
POINT_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_mnar", "summary.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "all_plots", "missing_detailed")
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

def save_mechanism_plot(df, x_col, title, filename, xlabel="Missing Rate"):
    plt.figure(figsize=(8, 5))
    models = sorted(df['model'].unique())
    for model in models:
        m_data = df[df['model'] == model].sort_values(x_col)
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

# 1. POINT MISSING - MCAR
if os.path.exists(POINT_SUMMARY):
    df_p = get_drop(pd.read_csv(POINT_SUMMARY))
    df_mcar = df_p[df_p['mechanism'] == 'mcar']
    save_mechanism_plot(df_mcar, 'fraction', "Point Missing: MCAR (Random)", "drop_point_mcar.png")

# 2. POINT MISSING - MNAR Extreme
if os.path.exists(POINT_SUMMARY):
    df_p = get_drop(pd.read_csv(POINT_SUMMARY))
    df_mnar = df_p[df_p['mechanism'] == 'mnar_extreme']
    save_mechanism_plot(df_mnar, 'fraction', "Point Missing: MNAR (Extreme Values)", "drop_point_mnar.png")

# 3. BURST MISSING - Gilbert-Elliott (Separated by length)
if os.path.exists(GE_SUMMARY):
    df_ge = get_drop(pd.read_csv(GE_SUMMARY))
    burst_lengths = sorted(df_ge['expected_burst_len'].unique())
    for blen in burst_lengths:
        df_blen = df_ge[df_ge['expected_burst_len'] == blen]
        title = f"Burst Missing: Gilbert-Elliott (Length={int(blen)})"
        filename = f"drop_burst_ge_len{int(blen)}.png"
        save_mechanism_plot(df_blen, 'expected_rate', title, filename)

print(f"Detailed missing data plots saved to: {OUTPUT_DIR}")
