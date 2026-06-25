
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
PROJECT_ROOT = "."
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "all_plots", "individual")
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

def save_plot(df, x_col, title, filename, invert_x=False, xlabel=None):
    plt.figure(figsize=(8, 5))
    models = sorted(df['model'].unique())
    for model in models:
        m_data = df[df['model'] == model].sort_values(x_col, ascending=not invert_x)
        plt.plot(m_data[x_col], m_data['auc_drop'], marker='o', label=model, linewidth=2)
    
    if invert_x:
        plt.gca().invert_xaxis()
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel if xlabel else x_col, fontsize=12)
    plt.ylabel("AUC-ROC Drop (Baseline - Corrupted)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"Saved: {filename}")

# 1. White Noise
snr_path = os.path.join(PROJECT_ROOT, "results", "experiments", "white_noise_snr", "summary.csv")
if os.path.exists(snr_path):
    df = get_drop(pd.read_csv(snr_path))
    save_plot(df, 'snr_db', "Performance Drop: White Noise (SNR)", "drop_white_noise.png", invert_x=True, xlabel="SNR (dB)")

# 2. Spikes
spikes_path = os.path.join(PROJECT_ROOT, "results", "experiments", "spikes_normal_only", "summary.csv")
if os.path.exists(spikes_path):
    df = get_drop(pd.read_csv(spikes_path))
    df_agg = df.groupby(['fraction', 'model'])['auc_drop'].mean().reset_index()
    save_plot(df_agg, 'fraction', "Performance Drop: Spikes", "drop_spikes.png", xlabel="Fraction of corrupted points")

# 3. Missing (MCAR)
missing_path = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_mnar", "summary.csv")
if os.path.exists(missing_path):
    df = get_drop(pd.read_csv(missing_path))
    df_mcar = df[df['mechanism'] == 'mcar']
    save_plot(df_mcar, 'fraction', "Performance Drop: Missing Data (MCAR)", "drop_missing_mcar.png", xlabel="Fraction missing")
    
    df_mnar = df[df['mechanism'] == 'mnar_extreme']
    save_plot(df_mnar, 'fraction', "Performance Drop: Missing Data (MNAR Extreme)", "drop_missing_mnar.png", xlabel="Fraction missing")

# 4. Swap
swap_path = os.path.join(PROJECT_ROOT, "results", "experiments", "swap_segment", "summary.csv")
if os.path.exists(swap_path):
    df = get_drop(pd.read_csv(swap_path))
    df_agg = df.groupby(['fraction', 'model'])['auc_drop'].mean().reset_index()
    save_plot(df_agg, 'fraction', "Performance Drop: Swap Segment", "drop_swap.png", xlabel="Fraction of signal swapped")

# 5. Freeze
freeze_path = os.path.join(PROJECT_ROOT, "results", "experiments", "freeze", "summary.csv")
if os.path.exists(freeze_path):
    df = get_drop(pd.read_csv(freeze_path))
    df_agg = df.groupby(['fraction', 'model'])['auc_drop'].mean().reset_index()
    save_plot(df_agg, 'fraction', "Performance Drop: Freeze / Sensor Stuck", "drop_freeze.png", xlabel="Fraction of signal stuck")

# 6. Gilbert-Elliott (Burst Missing)
ge_path = os.path.join(PROJECT_ROOT, "results", "experiments", "gilbert_elliott_true_impact", "summary.csv")
if os.path.exists(ge_path):
    df = get_drop(pd.read_csv(ge_path))
    df_burst = df[df['expected_burst_len'] == 10.0]
    save_plot(df_burst, 'expected_rate', "Performance Drop: Burst Missing (Gilbert-Elliott, len=10)", "drop_burst_missing.png", xlabel="Total missing rate")
