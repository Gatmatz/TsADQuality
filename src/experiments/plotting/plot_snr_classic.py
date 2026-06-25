"""Plot white noise SNR experiment results — single AUC-ROC plot."""
import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "white_noise_snr")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]

    summary = df.groupby('snr_db')['AUC_ROC'].mean().reset_index()
    summary = summary.sort_values('snr_db')

    print(f"{'SNR (dB)':>10} {'AUC-ROC':>10}")
    print("-" * 22)
    for _, row in summary.iterrows():
        print(f"{int(row['snr_db']):>10} {row['AUC_ROC']:>10.3f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(summary['snr_db'], summary['AUC_ROC'],
            marker='s', color='#D62728', linewidth=2, markersize=7)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('AUC-ROC vs SNR (dB)', fontsize=13)
    ax.set_xlim(summary['snr_db'].max() + 3, summary['snr_db'].min() - 3)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, 'snr_AUC_ROC.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
