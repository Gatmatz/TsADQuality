
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
PROJECT_ROOT = "."
DECOMP_CSV = os.path.join(PROJECT_ROOT, "results", "experiments", "fp_fn_decomposition", "fp_fn_decomposition.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "fp_fn_decomposition", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load decomposition data
df = pd.read_csv(DECOMP_CSV)

# Filter for White Noise
df_noise = df[df['corruption'] == 'noise'].copy()

# Extract SNR value from condition string "SNR=XdB"
df_noise['snr'] = df_noise['condition'].str.extract('SNR=(-?\d+)dB').astype(int)

# Create Plot
models = sorted(df_noise['model'].unique())

fig, axes = plt.subplots(1, len(models), figsize=(20, 5), sharey=True)
if len(models) == 1: axes = [axes]

for i, model in enumerate(models):
    m_data = df_noise[df_noise['model'] == model].sort_values('snr', ascending=False)
    ax = axes[i]
    
    # Plot Precision Drop (False Positives) and Recall Drop (False Negatives)
    ax.stackplot(m_data['snr'], m_data['precision_drop'], m_data['recall_drop'], 
                 labels=['Prec Drop (FPs)', 'Rec Drop (FNs)'], 
                 colors=['#ff9999', '#66b3ff'], alpha=0.8)
    
    ax.invert_xaxis()
    ax.set_title(f"{model}", fontsize=14)
    ax.set_xlabel("SNR (dB)")
    if i == 0:
        ax.set_ylabel("Metric Drop (Lower is worse)")
    ax.grid(True, linestyle='--', alpha=0.5)

plt.suptitle("FP vs FN Decomposition under White Noise (SNR)", fontsize=18, y=1.05)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=12)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "noise_fp_fn_stackplot.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {save_path}")
