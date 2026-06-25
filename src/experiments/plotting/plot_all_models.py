"""
Universal multi-model plotting script.

Generates per-experiment plots with ALL available models overlaid,
plus per-model individual plots. Covers:
  1. White Noise (SNR)
  2. Missing True Impact
  3. Freeze
  4. Swap
  5. Spikes
  6. Gilbert-Elliott True Impact

Usage:
    python plot_all_models.py                 # all experiments
    python plot_all_models.py --exp snr       # specific experiment
    python plot_all_models.py --exp freeze snr
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")

# Consistent colors and markers across ALL plots
MODEL_COLORS = {
    'IForest':     '#1f77b4',
    'LOF':         '#d62728',
    'PCA':         '#2ca02c',
    'MP':          '#2ca02c',
    'Autoencoder': '#ff7f0e',
    'AE':          '#ff7f0e',
}
MODEL_MARKERS = {
    'IForest': 'o', 'LOF': 's', 'PCA': '^', 'MP': 'D',
    'Autoencoder': 'v', 'AE': 'v',
}
METRICS = ['AUC_ROC', 'AUC_PR', 'VUS_ROC']
METRIC_LABELS = {'AUC_ROC': 'AUC-ROC', 'AUC_PR': 'AUC-PR', 'VUS_ROC': 'VUS-ROC'}


def load_baselines():
    """Load per-model baselines from individual files or final_subset."""
    baselines = {}

    # Try individual files first
    for model in MODEL_COLORS:
        path = os.path.join(TABLES_DIR, f"baseline_{model}_144.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            baselines[model] = df

    # Always merge final_subset to fill gaps (e.g., AE/Autoencoder)
    path = os.path.join(TABLES_DIR, "baseline_final_subset.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Harmonize: final_subset uses 'Autoencoder', experiments use 'AE'
        df_ae = df[df['model'] == 'Autoencoder']
        if not df_ae.empty and 'AE' not in baselines and 'Autoencoder' not in baselines:
            baselines['AE'] = df_ae
            baselines['Autoencoder'] = df_ae
        for m in df['model'].unique():
            if m not in baselines:
                baselines[m] = df[df['model'] == m]

    return baselines


def get_baseline_value(baselines, model, metric='AUC_ROC'):
    if model in baselines:
        return baselines[model][metric].mean()
    return None


# ================================================================
# 1. WHITE NOISE SNR
# ================================================================
def plot_snr(baselines):
    exp_dir = os.path.join(RESULTS_DIR, "white_noise_snr")
    checkpoint = os.path.join(exp_dir, "checkpoint.csv")
    if not os.path.exists(checkpoint):
        print("[SKIP] white_noise_snr: no checkpoint.csv")
        return

    df = pd.read_csv(checkpoint)
    df = df[df['error'].isna()]
    models = sorted(df['model'].unique())
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n=== White Noise SNR === models: {models}")

    # --- Combined: all models, AUC-ROC ---
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline_plotted = False
    for model in models:
        sub = df[df['model'] == model]
        means = sub.groupby('snr_db')['AUC_ROC'].mean().reset_index().sort_values('snr_db')
        ax.plot(means['snr_db'], means['AUC_ROC'],
                marker=MODEL_MARKERS.get(model, 'o'),
                color=MODEL_COLORS.get(model, 'gray'),
                label=model, linewidth=2, markersize=7)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            lbl = 'Clean baseline' if not baseline_plotted else None
            ax.axhline(y=bl, color=MODEL_COLORS.get(model, 'gray'),
                       linestyle='--', alpha=0.7, linewidth=1.5, label=lbl)
            baseline_plotted = True

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('All Models: AUC-ROC vs White Noise SNR', fontsize=13)
    ax.set_xlim(df['snr_db'].max() + 3, df['snr_db'].min() - 3)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'snr_all_models_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("  Saved: snr_all_models_AUC_ROC.png")

    # --- Combined: multi-metric ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, METRICS):
        if metric not in df.columns:
            continue
        for model in models:
            sub = df[df['model'] == model]
            means = sub.groupby('snr_db')[metric].mean().reset_index().sort_values('snr_db')
            ax.plot(means['snr_db'], means[metric],
                    marker=MODEL_MARKERS.get(model, 'o'),
                    color=MODEL_COLORS.get(model, 'gray'),
                    label=model, linewidth=2, markersize=6)
        ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=12)
        ax.set_xlim(df['snr_db'].max() + 3, df['snr_db'].min() - 3)
        ax.grid(True, alpha=0.3)
        if ax == axes[-1]:
            ax.legend(fontsize=9)
    fig.suptitle('All Models: White Noise SNR — Multiple Metrics', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'snr_all_models_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: snr_all_models_multi_metric.png")

    # --- Per-model individual plots ---
    for model in models:
        sub = df[df['model'] == model]
        means = sub.groupby('snr_db')['AUC_ROC'].mean().reset_index().sort_values('snr_db')
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(means['snr_db'], means['AUC_ROC'],
                marker=MODEL_MARKERS.get(model, 'o'),
                color=MODEL_COLORS.get(model, 'gray'),
                linewidth=2, markersize=7)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0,
                       alpha=0.85, zorder=5, label=f'Baseline ({bl:.3f})')
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(f'{model}: AUC-ROC vs White Noise SNR', fontsize=13)
        ax.set_xlim(means['snr_db'].max() + 3, means['snr_db'].min() - 3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'snr_{model}_AUC_ROC.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved: snr_{model}_AUC_ROC.png")


# ================================================================
# 2. MISSING TRUE IMPACT
# ================================================================
def plot_missing_true_impact(baselines):
    exp_dir = os.path.join(RESULTS_DIR, "missing_true_impact")
    checkpoint = os.path.join(exp_dir, "checkpoint.csv")
    if not os.path.exists(checkpoint):
        print("[SKIP] missing_true_impact: no checkpoint.csv")
        return

    df = pd.read_csv(checkpoint)
    df = df[df['error'].isna()]
    models = sorted(df['model'].unique())
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n=== Missing True Impact === models: {models}")

    # Get missing types
    missing_types = df['missing_type'].unique() if 'missing_type' in df.columns else ['point']

    for mtype in missing_types:
        sub_type = df[df['missing_type'] == mtype] if 'missing_type' in df.columns else df

        # --- Combined: all models ---
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in models:
            sub = sub_type[sub_type['model'] == model]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=MODEL_MARKERS.get(model, 'o'),
                    color=MODEL_COLORS.get(model, 'gray'),
                    label=model, linewidth=2, markersize=7)
            bl = get_baseline_value(baselines, model)
            if bl is not None:
                ax.axhline(y=bl, color=MODEL_COLORS.get(model, 'gray'),
                           linestyle='--', alpha=0.7, linewidth=1.5)

        ax.set_xlabel('Missing Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(f'All Models: Missing True Impact ({mtype})', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fname = f'missing_true_impact_{mtype}_all_models.png'
        fig.savefig(os.path.join(plots_dir, fname), dpi=300)
        plt.close(fig)
        print(f"  Saved: {fname}")

        # --- Per-model ---
        for model in models:
            sub = sub_type[sub_type['model'] == model]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=MODEL_MARKERS.get(model, 'o'),
                    color=MODEL_COLORS.get(model, 'gray'),
                    linewidth=2, markersize=7)
            bl = get_baseline_value(baselines, model)
            if bl is not None:
                ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0,
                           alpha=0.85, zorder=5, label=f'Baseline ({bl:.3f})')
            ax.set_xlabel('Missing Fraction', fontsize=12)
            ax.set_ylabel('AUC-ROC', fontsize=12)
            ax.set_title(f'{model}: Missing True Impact ({mtype})', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            fig.tight_layout()
            fname = f'missing_true_impact_{mtype}_{model}.png'
            fig.savefig(os.path.join(plots_dir, fname), dpi=300)
            plt.close(fig)
            print(f"  Saved: {fname}")


# ================================================================
# 3. FREEZE
# ================================================================
def plot_freeze(baselines):
    exp_dir = os.path.join(RESULTS_DIR, "freeze")
    checkpoint = os.path.join(exp_dir, "checkpoint.csv")
    if not os.path.exists(checkpoint):
        print("[SKIP] freeze: no checkpoint.csv")
        return

    df = pd.read_csv(checkpoint)
    df = df[df['error'].isna()]
    models = sorted(df['model'].unique())
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    NS_COLORS = {1: '#e41a1c', 3: '#4daf4a', 5: '#984ea3', 10: '#ff7f00'}
    NS_MARKERS = {1: 's', 3: '^', 5: 'D', 10: 'v'}

    print(f"\n=== Freeze === models: {models}")

    # --- Combined: all models (ns=3 representative) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    representative_ns = 3
    for model in models:
        sub = df[(df['model'] == model) & (df['num_stucks'] == representative_ns)]
        if sub.empty:
            sub = df[df['model'] == model]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=MODEL_MARKERS.get(model, 'o'),
                color=MODEL_COLORS.get(model, 'gray'),
                label=model, linewidth=2, markersize=7)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color=MODEL_COLORS.get(model, 'gray'),
                       linestyle='--', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title(f'All Models: Freeze (num_stucks={representative_ns})', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'freeze_all_models_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("  Saved: freeze_all_models_AUC_ROC.png")

    # --- Per-model: lines per num_stucks ---
    for model in models:
        model_df = df[df['model'] == model]
        fig, ax = plt.subplots(figsize=(8, 5))
        for ns in sorted(model_df['num_stucks'].unique()):
            ns_int = int(ns)
            sub = model_df[model_df['num_stucks'] == ns]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=NS_MARKERS.get(ns_int, 'o'),
                    color=NS_COLORS.get(ns_int, 'gray'),
                    label=f'ns={ns_int}', linewidth=2, markersize=7)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0,
                       alpha=0.85, zorder=5, label=f'Baseline ({bl:.3f})')
        ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(f'{model}: Freeze — AUC-ROC vs Fraction', fontsize=13)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'freeze_{model}_AUC_ROC.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved: freeze_{model}_AUC_ROC.png")

    # --- Heatmap per model ---
    for model in models:
        model_df = df[df['model'] == model]
        fractions = sorted(model_df['fraction'].unique())
        ns_list = sorted(model_df['num_stucks'].unique())
        row_labels = [f'ns={int(ns)}' for ns in ns_list]

        matrix = np.full((len(ns_list), len(fractions)), np.nan)
        for i, ns in enumerate(ns_list):
            for j, frac in enumerate(fractions):
                sub = model_df[(model_df['num_stucks'] == ns) & (model_df['fraction'] == frac)]
                if not sub.empty:
                    matrix[i, j] = sub['AUC_ROC'].mean()

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(fractions)))
        ax.set_xticklabels([f'{f:.0%}' for f in fractions])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel('Fraction of Corrupted Points', fontsize=12)
        ax.set_title(f'{model}: Mean AUC-ROC — Freeze', fontsize=13)
        for i in range(len(ns_list)):
            for j in range(len(fractions)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if val < 0.6 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=10, color=color, fontweight='bold')
        fig.colorbar(im, ax=ax, label='AUC-ROC')
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'freeze_{model}_heatmap.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved: freeze_{model}_heatmap.png")


# ================================================================
# 4. SWAP
# ================================================================
def plot_swap(baselines):
    exp_dir = os.path.join(RESULTS_DIR, "swap")
    checkpoint = os.path.join(exp_dir, "checkpoint.csv")
    if not os.path.exists(checkpoint):
        print("[SKIP] swap: no checkpoint.csv")
        return

    df = pd.read_csv(checkpoint)
    df = df[df['error'].isna()]
    models = sorted(df['model'].unique())
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n=== Swap === models: {models}")

    # Use point swap (swap_length=1) if available
    if 'swap_length' in df.columns:
        df_point = df[df['swap_length'] == 1]
        if df_point.empty:
            df_point = df
    else:
        df_point = df

    # --- Combined: all models ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        sub = df_point[df_point['model'] == model]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=MODEL_MARKERS.get(model, 'o'),
                color=MODEL_COLORS.get(model, 'gray'),
                label=model, linewidth=2, markersize=7)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color=MODEL_COLORS.get(model, 'gray'),
                       linestyle='--', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Swap Fraction', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('All Models: Swap (Point) — AUC-ROC', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'swap_all_models_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("  Saved: swap_all_models_AUC_ROC.png")

    # --- Per-model with swap_length variants ---
    for model in models:
        model_df = df[df['model'] == model]
        if 'swap_length' in model_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            SL_COLORS = {1: '#e41a1c', 5: '#377eb8', 10: '#4daf4a', 20: '#984ea3'}
            SL_MARKERS = {1: 'o', 5: 's', 10: '^', 20: 'D'}
            for sl in sorted(model_df['swap_length'].unique()):
                sl_int = int(sl)
                sub = model_df[model_df['swap_length'] == sl]
                means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
                ax.plot(means['fraction'], means['AUC_ROC'],
                        marker=SL_MARKERS.get(sl_int, 'o'),
                        color=SL_COLORS.get(sl_int, 'gray'),
                        label=f'len={sl_int}', linewidth=2, markersize=6)
            bl = get_baseline_value(baselines, model)
            if bl is not None:
                ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0,
                           alpha=0.85, zorder=5, label=f'Baseline ({bl:.3f})')
            ax.set_xlabel('Swap Fraction', fontsize=12)
            ax.set_ylabel('AUC-ROC', fontsize=12)
            ax.set_title(f'{model}: Swap — AUC-ROC vs Fraction', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f'swap_{model}_AUC_ROC.png'), dpi=300)
            plt.close(fig)
            print(f"  Saved: swap_{model}_AUC_ROC.png")


# ================================================================
# 5. SPIKES
# ================================================================
def plot_spikes(baselines):
    exp_dir = os.path.join(RESULTS_DIR, "spikes")
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n=== Spikes ===")

    # Spikes has per-model subdirectories
    all_data = []
    for model_dir in os.listdir(exp_dir):
        model_path = os.path.join(exp_dir, model_dir)
        if not os.path.isdir(model_path) or model_dir == 'plots':
            continue
        for spike_type in ['point', 'burst']:
            raw_path = os.path.join(model_path, spike_type, 'raw_results.csv')
            if os.path.exists(raw_path):
                df = pd.read_csv(raw_path)
                df['model'] = model_dir
                df['spike_type'] = spike_type
                all_data.append(df)

    if not all_data:
        print("  [SKIP] No spikes data found")
        return

    df = pd.concat(all_data, ignore_index=True)
    if 'error' in df.columns:
        df = df[df['error'].isna()]
    models = sorted(df['model'].unique())
    print(f"  models: {models}")

    # Use multiplier=10, point spikes
    for spike_type in df['spike_type'].unique():
        sub_type = df[df['spike_type'] == spike_type]
        # Use highest multiplier as representative
        if 'multiplier' in sub_type.columns:
            max_mult = sub_type['multiplier'].max()
            sub_repr = sub_type[sub_type['multiplier'] == max_mult]
        else:
            sub_repr = sub_type
            max_mult = '?'

        # --- Combined: all models ---
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in models:
            sub = sub_repr[sub_repr['model'] == model]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=MODEL_MARKERS.get(model, 'o'),
                    color=MODEL_COLORS.get(model, 'gray'),
                    label=model, linewidth=2, markersize=7)
            bl = get_baseline_value(baselines, model)
            if bl is not None:
                ax.axhline(y=bl, color=MODEL_COLORS.get(model, 'gray'),
                           linestyle='--', alpha=0.7, linewidth=1.5)

        ax.set_xlabel('Spike Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(f'All Models: Spikes ({spike_type}, mult={max_mult})', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fname = f'spikes_{spike_type}_all_models_AUC_ROC.png'
        fig.savefig(os.path.join(plots_dir, fname), dpi=300)
        plt.close(fig)
        print(f"  Saved: {fname}")

        # --- Per-model: lines per multiplier ---
        for model in models:
            model_df = sub_type[sub_type['model'] == model]
            if 'multiplier' not in model_df.columns:
                continue
            MULT_COLORS = {3: '#377eb8', 5: '#4daf4a', 10: '#e41a1c', 20: '#984ea3'}
            MULT_MARKERS = {3: 'o', 5: 's', 10: '^', 20: 'D'}

            fig, ax = plt.subplots(figsize=(8, 5))
            for mult in sorted(model_df['multiplier'].unique()):
                m_int = int(mult)
                sub = model_df[model_df['multiplier'] == mult]
                means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
                ax.plot(means['fraction'], means['AUC_ROC'],
                        marker=MULT_MARKERS.get(m_int, 'o'),
                        color=MULT_COLORS.get(m_int, 'gray'),
                        label=f'mult={m_int}', linewidth=2, markersize=6)
            bl = get_baseline_value(baselines, model)
            if bl is not None:
                ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0,
                           alpha=0.85, zorder=5, label=f'Baseline ({bl:.3f})')
            ax.set_xlabel('Spike Fraction', fontsize=12)
            ax.set_ylabel('AUC-ROC', fontsize=12)
            ax.set_title(f'{model}: Spikes ({spike_type})', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            fig.tight_layout()
            fname = f'spikes_{spike_type}_{model}_AUC_ROC.png'
            fig.savefig(os.path.join(plots_dir, fname), dpi=300)
            plt.close(fig)
            print(f"  Saved: {fname}")


# ================================================================
# 6. GILBERT-ELLIOTT TRUE IMPACT
# ================================================================
def plot_ge_true_impact(baselines):
    exp_dir = os.path.join(RESULTS_DIR, "gilbert_elliott_true_impact")
    checkpoint = os.path.join(exp_dir, "checkpoint.csv")
    if not os.path.exists(checkpoint):
        print("[SKIP] gilbert_elliott_true_impact: no checkpoint.csv")
        return

    df = pd.read_csv(checkpoint)
    df = df[df['error'].isna()]
    models = sorted(df['model'].unique())
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n=== Gilbert-Elliott True Impact === models: {models}")

    # Use beta=0.5 as representative
    if 'beta' in df.columns:
        betas = df['beta'].unique()
        repr_beta = 0.5 if 0.5 in betas else betas[0]
        df_repr = df[df['beta'] == repr_beta]
    else:
        df_repr = df
        repr_beta = '?'

    # --- Combined: all models ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        sub = df_repr[df_repr['model'] == model]
        means = sub.groupby('alpha')['AUC_ROC'].mean().reset_index().sort_values('alpha')
        ax.plot(means['alpha'], means['AUC_ROC'],
                marker=MODEL_MARKERS.get(model, 'o'),
                color=MODEL_COLORS.get(model, 'gray'),
                label=model, linewidth=2, markersize=7)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color=MODEL_COLORS.get(model, 'gray'),
                       linestyle='--', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Alpha (error rate)', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title(f'All Models: Gilbert-Elliott (beta={repr_beta})', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'ge_all_models_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("  Saved: ge_all_models_AUC_ROC.png")

    # --- Per-model: lines per beta ---
    for model in models:
        model_df = df[df['model'] == model]
        if 'beta' not in model_df.columns:
            continue

        BETA_COLORS = {0.1: '#e41a1c', 0.3: '#377eb8', 0.5: '#4daf4a',
                       0.7: '#984ea3', 0.9: '#ff7f00'}
        BETA_MARKERS = {0.1: 'o', 0.3: 's', 0.5: '^', 0.7: 'D', 0.9: 'v'}

        fig, ax = plt.subplots(figsize=(8, 5))
        for beta in sorted(model_df['beta'].unique()):
            sub = model_df[model_df['beta'] == beta]
            means = sub.groupby('alpha')['AUC_ROC'].mean().reset_index().sort_values('alpha')
            ax.plot(means['alpha'], means['AUC_ROC'],
                    marker=BETA_MARKERS.get(beta, 'o'),
                    color=BETA_COLORS.get(beta, 'gray'),
                    label=f'β={beta}', linewidth=2, markersize=6)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0,
                       alpha=0.85, zorder=5, label=f'Baseline ({bl:.3f})')
        ax.set_xlabel('Alpha (error rate)', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(f'{model}: Gilbert-Elliott', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'ge_{model}_AUC_ROC.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved: ge_{model}_AUC_ROC.png")


# ================================================================
# 7. SWAP SEGMENT
# ================================================================
def plot_swap_segment(baselines):
    exp_dir = os.path.join(RESULTS_DIR, "swap_segment")
    checkpoint = os.path.join(exp_dir, "checkpoint.csv")
    if not os.path.exists(checkpoint):
        print("[SKIP] swap_segment: no checkpoint.csv")
        return

    df = pd.read_csv(checkpoint)
    df = df[df['error'].isna()]
    models = sorted(df['model'].unique())
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n=== Swap Segment === models: {models}")

    # --- Combined: all models, group by fraction (mean over swap_length/num_swaps) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        sub = df[df['model'] == model]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=MODEL_MARKERS.get(model, 'o'),
                color=MODEL_COLORS.get(model, 'gray'),
                label=model, linewidth=2, markersize=7)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color=MODEL_COLORS.get(model, 'gray'),
                       linestyle='--', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Swap Fraction', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('All Models: Swap Segment — AUC-ROC', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'swap_segment_all_models_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("  Saved: swap_segment_all_models_AUC_ROC.png")

    # --- Per-model: lines per num_swaps ---
    for model in models:
        model_df = df[df['model'] == model]
        if 'num_swaps' not in model_df.columns:
            continue

        NS_COLORS = {1: '#e41a1c', 3: '#4daf4a', 5: '#984ea3', 10: '#ff7f00'}
        NS_MARKERS = {1: 's', 3: '^', 5: 'D', 10: 'v'}

        fig, ax = plt.subplots(figsize=(8, 5))
        for ns in sorted(model_df['num_swaps'].unique()):
            ns_int = int(ns)
            sub = model_df[model_df['num_swaps'] == ns]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=NS_MARKERS.get(ns_int, 'o'),
                    color=NS_COLORS.get(ns_int, 'gray'),
                    label=f'n_swaps={ns_int}', linewidth=2, markersize=6)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0,
                       alpha=0.85, zorder=5, label=f'Baseline ({bl:.3f})')
        ax.set_xlabel('Swap Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(f'{model}: Swap Segment', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'swap_segment_{model}_AUC_ROC.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved: swap_segment_{model}_AUC_ROC.png")


# ================================================================
# 8. SWAP PERMUTATION
# ================================================================
def plot_swap_permutation(baselines):
    exp_dir = os.path.join(RESULTS_DIR, "swap_permutation")
    checkpoint = os.path.join(exp_dir, "checkpoint.csv")
    if not os.path.exists(checkpoint):
        print("[SKIP] swap_permutation: no checkpoint.csv")
        return

    df = pd.read_csv(checkpoint)
    df = df[df['error'].isna()]
    models = sorted(df['model'].unique())
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n=== Swap Permutation === models: {models}")

    # --- Combined: all models, AUC vs n_segments ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        sub = df[df['model'] == model]
        means = sub.groupby('n_segments')['AUC_ROC'].mean().reset_index().sort_values('n_segments')
        ax.plot(means['n_segments'], means['AUC_ROC'],
                marker=MODEL_MARKERS.get(model, 'o'),
                color=MODEL_COLORS.get(model, 'gray'),
                label=model, linewidth=2, markersize=7)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color=MODEL_COLORS.get(model, 'gray'),
                       linestyle='--', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Number of Segments', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('All Models: Swap Permutation — AUC-ROC', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'swap_permutation_all_models_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("  Saved: swap_permutation_all_models_AUC_ROC.png")

    # --- Per-model ---
    for model in models:
        sub = df[df['model'] == model]
        means = sub.groupby('n_segments')['AUC_ROC'].mean().reset_index().sort_values('n_segments')
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(means['n_segments'], means['AUC_ROC'],
                marker=MODEL_MARKERS.get(model, 'o'),
                color=MODEL_COLORS.get(model, 'gray'),
                linewidth=2, markersize=7)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0,
                       alpha=0.85, zorder=5, label=f'Baseline ({bl:.3f})')
        ax.set_xlabel('Number of Segments', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(f'{model}: Swap Permutation', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'swap_permutation_{model}_AUC_ROC.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved: swap_permutation_{model}_AUC_ROC.png")


# ================================================================
# 9. SPIKES NORMAL ONLY
# ================================================================
def plot_spikes_normal_only(baselines):
    exp_dir = os.path.join(RESULTS_DIR, "spikes_normal_only")
    checkpoint = os.path.join(exp_dir, "checkpoint.csv")
    if not os.path.exists(checkpoint):
        print("[SKIP] spikes_normal_only: no checkpoint.csv")
        return

    df = pd.read_csv(checkpoint)
    df = df[df['error'].isna()]
    models = sorted(df['model'].unique())
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n=== Spikes Normal Only === models: {models}")

    # Use multiplier=10 as representative
    max_mult = df['multiplier'].max()
    df_repr = df[df['multiplier'] == max_mult]

    # --- Combined: all models ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        sub = df_repr[df_repr['model'] == model]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=MODEL_MARKERS.get(model, 'o'),
                color=MODEL_COLORS.get(model, 'gray'),
                label=model, linewidth=2, markersize=7)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color=MODEL_COLORS.get(model, 'gray'),
                       linestyle='--', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Spike Fraction', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title(f'All Models: Spikes Normal Only (mult={int(max_mult)})', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'spikes_normal_all_models_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("  Saved: spikes_normal_all_models_AUC_ROC.png")

    # --- Per-model: lines per multiplier ---
    MULT_COLORS = {3: '#377eb8', 5: '#4daf4a', 10: '#e41a1c'}
    MULT_MARKERS = {3: 'o', 5: 's', 10: '^'}

    for model in models:
        model_df = df[df['model'] == model]
        fig, ax = plt.subplots(figsize=(8, 5))
        for mult in sorted(model_df['multiplier'].unique()):
            m_int = int(mult)
            sub = model_df[model_df['multiplier'] == mult]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=MULT_MARKERS.get(m_int, 'o'),
                    color=MULT_COLORS.get(m_int, 'gray'),
                    label=f'mult={m_int}', linewidth=2, markersize=6)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0,
                       alpha=0.85, zorder=5, label=f'Baseline ({bl:.3f})')
        ax.set_xlabel('Spike Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(f'{model}: Spikes Normal Only', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'spikes_normal_{model}_AUC_ROC.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved: spikes_normal_{model}_AUC_ROC.png")


# ================================================================
# 10. MISSING MNAR
# ================================================================
def plot_mnar(baselines):
    exp_dir = os.path.join(RESULTS_DIR, "missing_mnar")
    checkpoint = os.path.join(exp_dir, "checkpoint.csv")
    if not os.path.exists(checkpoint):
        print("[SKIP] missing_mnar: no checkpoint.csv")
        return

    df = pd.read_csv(checkpoint)
    df = df[df['error'].isna()]
    models = sorted(df['model'].unique())
    mechanisms = sorted(df['mechanism'].unique())
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n=== Missing MNAR === models: {models}, mechanisms: {mechanisms}")

    MECH_COLORS = {'mcar': '#377eb8', 'mnar_high': '#4daf4a', 'mnar_extreme': '#e41a1c'}
    MECH_MARKERS = {'mcar': 'o', 'mnar_high': 's', 'mnar_extreme': '^'}

    # --- Combined: all models, per mechanism ---
    for mechanism in mechanisms:
        sub_mech = df[df['mechanism'] == mechanism]
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in models:
            sub = sub_mech[sub_mech['model'] == model]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=MODEL_MARKERS.get(model, 'o'),
                    color=MODEL_COLORS.get(model, 'gray'),
                    label=model, linewidth=2, markersize=7)
            bl = get_baseline_value(baselines, model)
            if bl is not None:
                ax.axhline(y=bl, color=MODEL_COLORS.get(model, 'gray'),
                           linestyle='--', alpha=0.7, linewidth=1.5)

        ax.set_xlabel('Missing Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(f'All Models: MNAR ({mechanism})', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fname = f'mnar_{mechanism}_all_models_AUC_ROC.png'
        fig.savefig(os.path.join(plots_dir, fname), dpi=300)
        plt.close(fig)
        print(f"  Saved: {fname}")

    # --- Per-model: lines per mechanism ---
    for model in models:
        model_df = df[df['model'] == model]
        fig, ax = plt.subplots(figsize=(8, 5))
        for mechanism in mechanisms:
            sub = model_df[model_df['mechanism'] == mechanism]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=MECH_MARKERS.get(mechanism, 'o'),
                    color=MECH_COLORS.get(mechanism, 'gray'),
                    label=mechanism, linewidth=2, markersize=6)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0,
                       alpha=0.85, zorder=5, label=f'Baseline ({bl:.3f})')
        ax.set_xlabel('Missing Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(f'{model}: MNAR — Mechanisms Compared', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'mnar_{model}_mechanisms.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved: mnar_{model}_mechanisms.png")


# ================================================================
# 11. MISSING MNAR BURST
# ================================================================
def plot_mnar_burst(baselines):
    exp_dir = os.path.join(RESULTS_DIR, "missing_mnar_burst")
    checkpoint = os.path.join(exp_dir, "checkpoint.csv")
    if not os.path.exists(checkpoint):
        print("[SKIP] missing_mnar_burst: no checkpoint.csv")
        return

    df = pd.read_csv(checkpoint)
    df = df[df['error'].isna()]
    models = sorted(df['model'].unique())
    mechanisms = sorted(df['mechanism'].unique())
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n=== Missing MNAR Burst === models: {models}, mechanisms: {mechanisms}")

    MECH_COLORS = {'mnar_high_burst': '#4daf4a', 'mnar_extreme_burst': '#e41a1c'}
    MECH_MARKERS = {'mnar_high_burst': 's', 'mnar_extreme_burst': '^'}
    NB_COLORS = {1: '#e41a1c', 3: '#4daf4a', 5: '#984ea3', 10: '#ff7f00'}
    NB_MARKERS = {1: 's', 3: '^', 5: 'D', 10: 'v'}

    # --- Combined: all models (use num_bursts=3, mnar_extreme_burst as representative) ---
    repr_mech = 'mnar_extreme_burst' if 'mnar_extreme_burst' in mechanisms else mechanisms[0]
    repr_nb = 3

    fig, ax = plt.subplots(figsize=(8, 5))
    sub_repr = df[(df['mechanism'] == repr_mech) & (df['num_bursts'] == repr_nb)]
    for model in models:
        sub = sub_repr[sub_repr['model'] == model]
        means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        ax.plot(means['fraction'], means['AUC_ROC'],
                marker=MODEL_MARKERS.get(model, 'o'),
                color=MODEL_COLORS.get(model, 'gray'),
                label=model, linewidth=2, markersize=7)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color=MODEL_COLORS.get(model, 'gray'),
                       linestyle='--', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Missing Fraction', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title(f'All Models: MNAR Burst ({repr_mech}, nb={repr_nb})', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'mnar_burst_all_models_AUC_ROC.png'), dpi=300)
    plt.close(fig)
    print("  Saved: mnar_burst_all_models_AUC_ROC.png")

    # --- Per-model: lines per num_bursts, per mechanism ---
    for model in models:
        for mechanism in mechanisms:
            model_df = df[(df['model'] == model) & (df['mechanism'] == mechanism)]
            if model_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            for nb in sorted(model_df['num_bursts'].unique()):
                nb_int = int(nb)
                sub = model_df[model_df['num_bursts'] == nb]
                means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
                ax.plot(means['fraction'], means['AUC_ROC'],
                        marker=NB_MARKERS.get(nb_int, 'o'),
                        color=NB_COLORS.get(nb_int, 'gray'),
                        label=f'nb={nb_int}', linewidth=2, markersize=6)
            bl = get_baseline_value(baselines, model)
            if bl is not None:
                ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0,
                           alpha=0.85, zorder=5, label=f'Baseline ({bl:.3f})')
            ax.set_xlabel('Missing Fraction', fontsize=12)
            ax.set_ylabel('AUC-ROC', fontsize=12)
            ax.set_title(f'{model}: MNAR Burst ({mechanism})', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            fig.tight_layout()
            fname = f'mnar_burst_{model}_{mechanism}.png'
            fig.savefig(os.path.join(plots_dir, fname), dpi=300)
            plt.close(fig)
            print(f"  Saved: {fname}")

    # --- Per-model: mechanism comparison (fixed nb=3) ---
    for model in models:
        model_df = df[(df['model'] == model) & (df['num_bursts'] == repr_nb)]
        if model_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for mechanism in mechanisms:
            sub = model_df[model_df['mechanism'] == mechanism]
            means = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            ax.plot(means['fraction'], means['AUC_ROC'],
                    marker=MECH_MARKERS.get(mechanism, 'o'),
                    color=MECH_COLORS.get(mechanism, 'gray'),
                    label=mechanism, linewidth=2, markersize=6)
        bl = get_baseline_value(baselines, model)
        if bl is not None:
            ax.axhline(y=bl, color='black', linestyle='--', linewidth=2.0,
                       alpha=0.85, zorder=5, label=f'Baseline ({bl:.3f})')
        ax.set_xlabel('Missing Fraction', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title(f'{model}: MNAR Burst — Mechanisms (nb={repr_nb})', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'mnar_burst_{model}_mechanisms.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved: mnar_burst_{model}_mechanisms.png")


# ================================================================
# MAIN
# ================================================================
EXPERIMENT_MAP = {
    'snr': plot_snr,
    'missing': plot_missing_true_impact,
    'freeze': plot_freeze,
    'swap': plot_swap,
    'spikes': plot_spikes,
    'ge': plot_ge_true_impact,
    'swap_segment': plot_swap_segment,
    'swap_permutation': plot_swap_permutation,
    'spikes_normal': plot_spikes_normal_only,
    'mnar': plot_mnar,
    'mnar_burst': plot_mnar_burst,
}


def main():
    parser = argparse.ArgumentParser(description="Generate multi-model plots for all experiments")
    parser.add_argument('--exp', nargs='*', default=None,
                        choices=list(EXPERIMENT_MAP.keys()),
                        help="Experiments to plot (default: all)")
    args = parser.parse_args()

    baselines = load_baselines()
    print(f"Baselines loaded: {list(baselines.keys())}")

    exps = args.exp if args.exp else list(EXPERIMENT_MAP.keys())

    for exp in exps:
        EXPERIMENT_MAP[exp](baselines)

    print("\n[Done]")


if __name__ == '__main__':
    main()
