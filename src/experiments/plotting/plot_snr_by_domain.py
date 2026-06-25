"""
White-noise SNR degradation broken down by application domain.

Groups dataset families into meaningful domains to compare noise
robustness across different real-world application areas.
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

DOMAIN_MAP = {
    'ECG': 'Medical', 'SVDB': 'Medical',
    'SMD': 'Industrial/IoT', 'GHL': 'Industrial/IoT',
    'SensorScope': 'Industrial/IoT', 'Occupancy': 'Industrial/IoT',
    'Genesis': 'Industrial/IoT', 'Dodgers': 'Industrial/IoT',
    'NASA-SMAP': 'Space', 'NASA-MSL': 'Space',
    'IOPS': 'Network/Cloud', 'NAB': 'Network/Cloud',
    'OPPORTUNITY': 'Wearable/Activity', 'Daphnet': 'Wearable/Activity',
    'YAHOO': 'Synthetic/Curated', 'MGAB': 'Synthetic/Curated',
    'KDD21': 'Synthetic/Curated',
}

DOMAIN_ORDER = ['Synthetic/Curated', 'Industrial/IoT', 'Wearable/Activity',
                'Medical', 'Space', 'Network/Cloud']

DOMAIN_COLORS = {
    'Synthetic/Curated': '#377eb8',
    'Industrial/IoT':    '#e41a1c',
    'Wearable/Activity': '#4daf4a',
    'Medical':           '#984ea3',
    'Space':             '#ff7f00',
    'Network/Cloud':     '#a65628',
}

DOMAIN_MARKERS = {
    'Synthetic/Curated': 'o',
    'Industrial/IoT':    's',
    'Wearable/Activity': '^',
    'Medical':           'D',
    'Space':             'v',
    'Network/Cloud':     'P',
}


def load_data():
    # SNR results
    snr = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    snr = snr[(snr['model'] == 'IForest') & (snr['error'].isna())]

    # Subset metadata
    subset = pd.read_csv(SUBSET_CSV)
    folder_map = dict(zip(subset['filename_actual'], subset['folder']))
    snr['folder'] = snr['file'].map(folder_map)
    snr['domain'] = snr['folder'].map(DOMAIN_MAP)

    # Baseline
    baseline = pd.read_csv(BASELINE_CSV)
    baseline = baseline[baseline['model'] == 'IForest']
    base_map = dict(zip(baseline['file'], baseline['AUC_ROC']))
    snr['baseline_AUC_ROC'] = snr['file'].map(base_map)

    # Corrected AUC
    snr['corrected_AUC_ROC'] = snr['AUC_ROC'].apply(lambda x: max(x, 1 - x))
    snr['baseline_corrected'] = snr['baseline_AUC_ROC'].apply(lambda x: max(x, 1 - x))

    return snr


def plot_absolute_by_domain(df):
    """Absolute corrected AUC-ROC per SNR, split by domain."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for domain in DOMAIN_ORDER:
        sub = df[df['domain'] == domain]
        n_datasets = sub['file'].nunique()
        grouped = sub.groupby('snr_db')['corrected_AUC_ROC'].agg(['mean', 'std']).reset_index()
        grouped = grouped.sort_values('snr_db')

        n_per = sub.groupby('snr_db')['corrected_AUC_ROC'].count().values
        se = grouped['std'].values / np.sqrt(n_per)

        ax.fill_between(grouped['snr_db'],
                        grouped['mean'] - 1.96 * se,
                        grouped['mean'] + 1.96 * se,
                        alpha=0.1, color=DOMAIN_COLORS[domain])
        ax.plot(grouped['snr_db'], grouped['mean'],
                marker=DOMAIN_MARKERS[domain], color=DOMAIN_COLORS[domain],
                markersize=6, linewidth=2,
                label=f'{domain} (n={n_datasets})')

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Mean max(AUC, 1\u2212AUC)', fontsize=12)
    ax.set_title('White Noise Degradation by Application Domain (IForest)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(df['snr_db'].max() + 2, df['snr_db'].min() - 2)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "snr_by_domain_absolute.png"), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, "snr_by_domain_absolute.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Saved: snr_by_domain_absolute")


def plot_relative_by_domain(df):
    """Relative degradation of corrected AUC-ROC per SNR, split by domain."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = df.copy()
    df['rel_deg'] = (df['baseline_corrected'] - df['corrected_AUC_ROC']) / df['baseline_corrected'] * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    for domain in DOMAIN_ORDER:
        sub = df[df['domain'] == domain]
        n_datasets = sub['file'].nunique()
        grouped = sub.groupby('snr_db')['rel_deg'].agg(['mean', 'std']).reset_index()
        grouped = grouped.sort_values('snr_db')

        n_per = sub.groupby('snr_db')['rel_deg'].count().values
        se = grouped['std'].values / np.sqrt(n_per)

        ax.fill_between(grouped['snr_db'],
                        grouped['mean'] - 1.96 * se,
                        grouped['mean'] + 1.96 * se,
                        alpha=0.1, color=DOMAIN_COLORS[domain])
        ax.plot(grouped['snr_db'], grouped['mean'],
                marker=DOMAIN_MARKERS[domain], color=DOMAIN_COLORS[domain],
                markersize=6, linewidth=2,
                label=f'{domain} (n={n_datasets})')

    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.axhline(10, color='gray', ls=':', lw=1, alpha=0.5)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Relative max(AUC, 1\u2212AUC) Degradation (%)', fontsize=12)
    ax.set_title('Relative Performance Drop by Domain (IForest)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(df['snr_db'].max() + 2, df['snr_db'].min() - 2)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "snr_by_domain_relative.png"), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, "snr_by_domain_relative.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Saved: snr_by_domain_relative")


def plot_heatmap(df):
    """Heatmap: domain × SNR → mean corrected AUC-ROC degradation."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = df.copy()
    df['rel_deg'] = (df['baseline_corrected'] - df['corrected_AUC_ROC']) / df['baseline_corrected'] * 100

    pivot = df.pivot_table(index='domain', columns='snr_db', values='rel_deg', aggfunc='mean')
    pivot = pivot.reindex(DOMAIN_ORDER)
    pivot = pivot[sorted(pivot.columns, reverse=True)]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{int(c)} dB' for c in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if abs(val) > 15 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Relative Degradation (%)', fontsize=11)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_title('Noise Robustness Heatmap by Domain (IForest)',
                 fontsize=14, fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "snr_domain_heatmap.png"), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, "snr_domain_heatmap.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Saved: snr_domain_heatmap")


def print_summary(df):
    df = df.copy()
    df['rel_deg'] = (df['baseline_corrected'] - df['corrected_AUC_ROC']) / df['baseline_corrected'] * 100

    print(f"\n{'='*80}")
    print(f"  SNR Degradation by Domain (IForest, max(AUC, 1-AUC))")
    print(f"{'='*80}")

    header = f"{'Domain':<22}"
    for snr in sorted(df['snr_db'].unique(), reverse=True):
        header += f"  {int(snr):>4}dB"
    print(header)
    print("-" * 80)

    for domain in DOMAIN_ORDER:
        sub = df[df['domain'] == domain]
        n = sub['file'].nunique()
        row = f"{domain:<18}({n:>2})"
        for snr in sorted(df['snr_db'].unique(), reverse=True):
            val = sub[sub['snr_db'] == snr]['rel_deg'].mean()
            row += f"  {val:>5.1f}%"
        print(row)

    print(f"{'='*80}")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['file'].nunique()} datasets")
    print(f"\nDatasets per domain:")
    for domain in DOMAIN_ORDER:
        n = df[df['domain'] == domain]['file'].nunique()
        print(f"  {domain:<22} {n}")

    print_summary(df)
    plot_absolute_by_domain(df)
    plot_relative_by_domain(df)
    plot_heatmap(df)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
