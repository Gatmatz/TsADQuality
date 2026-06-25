
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
PROJECT_ROOT = "."
SNR_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "white_noise_snr", "summary.csv")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
OUTPUT_PLOT = os.path.join(PROJECT_ROOT, "results", "experiments", "white_noise_snr", "plots", "snr_auc_drop_line.png")

os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)

# Load baseline
df_base = pd.read_csv(BASELINE_CSV)
df_base['model'] = df_base['model'].replace({'Autoencoder': 'AE'})
# Average baseline per model
baseline_auc = df_base.groupby('model')['AUC_ROC'].mean().to_dict()

# Load SNR summary
df_snr = pd.read_csv(SNR_SUMMARY)

# Calculate drop
# Drop = Baseline - Corrupted
def get_drop(row):
    model = row['model']
    base = baseline_auc.get(model, 0)
    return base - row['mean_AUC_ROC']

df_snr['auc_drop'] = df_snr.apply(get_drop, axis=1)

# Plot
plt.figure(figsize=(10, 6))

# Order models for consistent plotting
models = sorted(df_snr['model'].unique())
# Reverse SNR for x-axis (from clean to noisy)
snr_levels = sorted(df_snr['snr_db'].unique(), reverse=True)

for model in models:
    model_data = df_snr[df_snr['model'] == model].sort_values('snr_db', ascending=False)
    plt.plot(model_data['snr_db'], model_data['auc_drop'], marker='o', label=model, linewidth=2)

plt.gca().invert_xaxis() # SNR usually shown descending
plt.xlabel("SNR (dB) - Higher is cleaner", fontsize=12)
plt.ylabel("AUC-ROC Drop (Baseline - Corrupted)", fontsize=12)
plt.title("Performance Degradation under White Noise", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Add a horizontal line at 0
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
print(f"Plot saved to: {OUTPUT_PLOT}")
