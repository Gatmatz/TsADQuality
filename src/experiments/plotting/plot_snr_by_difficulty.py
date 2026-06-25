"""
White-noise SNR degradation broken down by difficulty tercile (Easy/Medium/Hard).

Shows how baseline difficulty interacts with noise robustness.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "experiments", "white_noise_snr")
SUBSET_CSV   = os.path.join(PROJECT_ROOT, "results", "tables", "final_subset.csv")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")

METRICS = ['AUC_ROC', 'AUC_PR', 'VUS_ROC']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'AUC_PR': 'AUC-PR', 'VUS_ROC': 'VUS-ROC',
                 'corrected_AUC_ROC': 'max(AUC, 1-AUC)'}

DIFFICULTY_ORDER = ['Easy', 'Medium', 'Hard']
DIFFICULTY_COLORS = {'Easy': '#4daf4a', 'Medium': '#ff7f00', 'Hard': '#e41a1c'}


def load_data():
    # SNR results
    snr = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    snr = snr[(snr['model'] == 'IForest') & (snr['error'].isna())]

    # Subset metadata — recompute difficulty using max(AUC, 1-AUC)
    subset = pd.read_csv(SUBSET_CSV)
    subset['corrected_mean'] = subset['mean_AUC_ROC'].apply(lambda x: max(x, 1 - x))
    q33 = subset['corrected_mean'].quantile(0.33)
    q66 = subset['corrected_mean'].quantile(0.66)
    subset['difficulty'] = np.select(
        [subset['corrected_mean'] >= q66,
         subset['corrected_mean'] <= q33],
        ['Easy', 'Hard'], default='Medium')

    diff_map = dict(zip(subset['filename_actual'], subset['difficulty']))
    folder_map = dict(zip(subset['filename_actual'], subset['folder']))
    snr['difficulty'] = snr['file'].map(diff_map)
    snr['folder'] = snr['file'].map(folder_map)

    # Baseline results
    baseline = pd.read_csv(BASELINE_CSV)
    baseline = baseline[baseline['model'] == 'IForest']
    base_map = dict(zip(baseline['file'], baseline['AUC_ROC']))
    snr['baseline_AUC_ROC'] = snr['file'].map(base_map)

    # Corrected AUC: max(AUC, 1-AUC) — fixes sub-0.5 detectors
    snr['corrected_AUC_ROC'] = snr['AUC_ROC'].apply(lambda x: max(x, 1 - x))
    snr['baseline_corrected_AUC_ROC'] = snr['baseline_AUC_ROC'].apply(lambda x: max(x, 1 - x))

    return snr


def plot_absolute_by_difficulty(df):
    """Absolute metric values per SNR, split by difficulty."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_metrics = METRICS + ['corrected_AUC_ROC']
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(5 * len(plot_metrics), 5), sharey=False)

    for ax, metric in zip(axes, plot_metrics):
        for diff in DIFFICULTY_ORDER:
            sub = df[df['difficulty'] == diff]
            grouped = sub.groupby('snr_db')[metric].agg(['mean', 'std']).reset_index()
            grouped = grouped.sort_values('snr_db')

            n_per_group = sub.groupby('snr_db')[metric].count().values
            se = grouped['std'].values / np.sqrt(n_per_group)

            ax.fill_between(grouped['snr_db'],
                            grouped['mean'] - 1.96 * se,
                            grouped['mean'] + 1.96 * se,
                            alpha=0.15, color=DIFFICULTY_COLORS[diff])
            ax.plot(grouped['snr_db'], grouped['mean'],
                    'o-', color=DIFFICULTY_COLORS[diff], markersize=5,
                    linewidth=2, label=f'{diff} (n={sub["file"].nunique()})')

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(f'Mean {METRIC_LABELS[metric]}', fontsize=12)
        ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight='bold')
        ax.set_xlim(df['snr_db'].max() + 2, df['snr_db'].min() - 2)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('White Noise Degradation by Difficulty Tercile (IForest)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "snr_by_difficulty_absolute.png"), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, "snr_by_difficulty_absolute.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Saved: snr_by_difficulty_absolute")


def plot_relative_degradation(df):
    """Relative degradation (% drop from baseline) per SNR, split by difficulty.
    Two subplots: raw AUC-ROC and corrected max(AUC, 1-AUC)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = df.copy()
    df['rel_degradation_raw'] = (df['baseline_AUC_ROC'] - df['AUC_ROC']) / df['baseline_AUC_ROC'] * 100
    df['rel_degradation_corrected'] = (
        (df['baseline_corrected_AUC_ROC'] - df['corrected_AUC_ROC'])
        / df['baseline_corrected_AUC_ROC'] * 100
    )

    variants = [
        ('rel_degradation_raw', 'AUC-ROC', 'snr_by_difficulty_relative.png'),
        ('rel_degradation_corrected', 'max(AUC, 1\u2212AUC)', 'snr_by_difficulty_relative_corrected.png'),
    ]

    for col, label, fname in variants:
        fig, ax = plt.subplots(figsize=(8, 5.5))

        for diff in DIFFICULTY_ORDER:
            sub = df[df['difficulty'] == diff]
            grouped = sub.groupby('snr_db')[col].agg(['mean', 'std']).reset_index()
            grouped = grouped.sort_values('snr_db')

            n_per_group = sub.groupby('snr_db')[col].count().values
            se = grouped['std'].values / np.sqrt(n_per_group)

            ax.fill_between(grouped['snr_db'],
                            grouped['mean'] - 1.96 * se,
                            grouped['mean'] + 1.96 * se,
                            alpha=0.15, color=DIFFICULTY_COLORS[diff])
            ax.plot(grouped['snr_db'], grouped['mean'],
                    'o-', color=DIFFICULTY_COLORS[diff], markersize=6,
                    linewidth=2, label=f'{diff} (n={sub["file"].nunique()})')

        ax.axhline(0, color='gray', ls='--', lw=1)
        ax.axhline(10, color='gray', ls=':', lw=1, alpha=0.5)

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(f'Relative {label} Degradation (%)', fontsize=12)
        ax.set_title(f'Relative Performance Drop by Difficulty — {label} (IForest)',
                     fontsize=13, fontweight='bold')
        ax.set_xlim(df['snr_db'].max() + 2, df['snr_db'].min() - 2)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=200, bbox_inches='tight')
        fig.savefig(os.path.join(PLOTS_DIR, fname.replace('.png', '.pdf')), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fname}")


def print_summary_table(df):
    """Print summary statistics."""
    df = df.copy()
    df['rel_degradation'] = (df['baseline_AUC_ROC'] - df['AUC_ROC']) / df['baseline_AUC_ROC'] * 100

    print(f"\n{'='*70}")
    print(f"  SNR Degradation by Difficulty (IForest, AUC-ROC)")
    print(f"{'='*70}")
    print(f"{'SNR':>6} | {'Easy':>18} | {'Medium':>18} | {'Hard':>18}")
    print(f"{'(dB)':>6} | {'mean':>8} {'drop%':>8} | {'mean':>8} {'drop%':>8} | {'mean':>8} {'drop%':>8}")
    print("-" * 70)

    for snr in sorted(df['snr_db'].unique(), reverse=True):
        parts = []
        for diff in DIFFICULTY_ORDER:
            sub = df[(df['snr_db'] == snr) & (df['difficulty'] == diff)]
            m = sub['AUC_ROC'].mean()
            d = sub['rel_degradation'].mean()
            parts.append(f"{m:8.4f} {d:+7.1f}%")
        print(f"{snr:>6} | {parts[0]} | {parts[1]} | {parts[2]}")

    print(f"{'='*70}")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['file'].nunique()} datasets")
    print(f"Difficulty distribution: {df.groupby('difficulty')['file'].nunique().to_dict()}")

    print_summary_table(df)
    plot_absolute_by_difficulty(df)
    plot_relative_degradation(df)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
