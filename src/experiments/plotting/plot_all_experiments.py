"""
Unified plotting script for all corruption experiments.

Generates metric vs fraction plots for: swap, spikes, freeze, missing.
Each experiment gets combined (all models) and per-secondary-parameter plots.

Output: results/plots/{experiment}/...
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

# ==========================================
# PATH CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")

# ==========================================
# SHARED STYLE (same as plot_snr_classic.py)
# ==========================================
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '-',
})

METRICS = [
    ('mean_AUC_ROC', 'AUC-ROC', 'std_AUC_ROC'),
    ('mean_AUC_PR', 'AUC-PR', 'std_AUC_PR'),
    ('mean_VUS_ROC', 'VUS-ROC', 'std_VUS_ROC'),
    ('mean_VUS_PR', 'VUS-PR', 'std_VUS_PR'),
    ('mean_R_AUC_ROC', 'Range AUC-ROC', 'std_R_AUC_ROC'),
    ('mean_R_AUC_PR', 'Range AUC-PR', 'std_R_AUC_PR'),
]

MODEL_STYLES = {
    'IForest': {'color': '#1f77b4', 'marker': 's', 'label': 'Isolation Forest'},
    'PCA':     {'color': '#1F77B4', 'marker': 'o', 'label': 'PCA'},
    'LOF':     {'color': '#d62728', 'marker': '^', 'label': 'LOF'},
    'MP':      {'color': '#2ca02c', 'marker': 'D', 'label': 'Matrix Profile'},
}


def _style(model):
    return MODEL_STYLES.get(model, {'color': 'gray', 'marker': 'x', 'label': model})


def _fname(ylabel):
    return ylabel.lower().replace('-', '_').replace(' ', '_')


def plot_metric_vs_fraction(df, models, x_col, xlabel, title_suffix, output_dir):
    """Generic: plot each metric vs x_col, one line per model."""
    os.makedirs(output_dir, exist_ok=True)

    for metric_col, ylabel, std_col in METRICS:
        if metric_col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for model in models:
            mdf = df[df['model'] == model].sort_values(x_col)
            if mdf.empty:
                continue
            s = _style(model)

            ax.plot(mdf[x_col], mdf[metric_col],
                    color=s['color'], marker=s['marker'],
                    linewidth=2, markersize=7, label=s['label'])

            if std_col in mdf.columns:
                ax.fill_between(mdf[x_col],
                                mdf[metric_col] - mdf[std_col],
                                mdf[metric_col] + mdf[std_col],
                                alpha=0.15, color=s['color'])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs {xlabel} — {title_suffix}')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.legend(loc='best')

        plt.tight_layout()
        out = os.path.join(output_dir, f"{_fname(ylabel)}_vs_{_fname(xlabel)}.png")
        plt.savefig(out)
        plt.close()
        print(f"  {out}")


# ==========================================
# SWAP PLOTS
# ==========================================
def plot_swap():
    summary_csv = os.path.join(EXPERIMENTS_DIR, "swap", "summary.csv")
    if not os.path.exists(summary_csv):
        print("[SKIP] swap — no summary.csv")
        return

    df = pd.read_csv(summary_csv)
    models = df['model'].unique()
    out_base = os.path.join(PLOTS_DIR, "swap")
    print("\n=== SWAP ===")

    # Combined: all models, point swap only
    point_df = df[df['swap_ratio'] == 0.0]
    if not point_df.empty:
        plot_metric_vs_fraction(point_df, models, 'fraction', 'Fraction',
                                'Point Swap', os.path.join(out_base, "point"))

    # Per swap_ratio (segment swap)
    for ratio in sorted(df[df['swap_ratio'] > 0]['swap_ratio'].unique()):
        ratio_df = df[df['swap_ratio'] == ratio]
        pct = f"{ratio * 100:.0f}%"

        # Per max_distance
        for dist in ratio_df['max_distance'].unique():
            dist_df = ratio_df[ratio_df['max_distance'] == dist]
            dist_str = str(dist)
            plot_metric_vs_fraction(dist_df, models, 'fraction', 'Fraction',
                                    f'Segment {pct}, dist={dist_str}',
                                    os.path.join(out_base, f"segment_pct{ratio*100:.0f}", f"dist_{dist_str}"))


# ==========================================
# SPIKES PLOTS
# ==========================================
def plot_spikes():
    summary_csv = os.path.join(EXPERIMENTS_DIR, "spikes", "summary.csv")
    if not os.path.exists(summary_csv):
        print("[SKIP] spikes — no summary.csv")
        return

    df = pd.read_csv(summary_csv)
    models = df['model'].unique()
    out_base = os.path.join(PLOTS_DIR, "spikes")
    print("\n=== SPIKES ===")

    # Per mode (point vs burst)
    for seq, seq_label in [(False, 'point'), (True, 'burst')]:
        mode_df = df[df['sequential'] == seq]

        # Per multiplier
        for mult in sorted(mode_df['multiplier'].unique()):
            mult_df = mode_df[mode_df['multiplier'] == mult]
            plot_metric_vs_fraction(mult_df, models, 'fraction', 'Fraction',
                                    f'{seq_label.title()} Spike ({mult:.0f}x std)',
                                    os.path.join(out_base, seq_label, f"mult_{mult:.0f}"))


# ==========================================
# FREEZE PLOTS
# ==========================================
def plot_freeze():
    summary_csv = os.path.join(EXPERIMENTS_DIR, "freeze", "summary.csv")
    if not os.path.exists(summary_csv):
        print("[SKIP] freeze — no summary.csv")
        return

    df = pd.read_csv(summary_csv)
    models = df['model'].unique()
    out_base = os.path.join(PLOTS_DIR, "freeze")
    print("\n=== FREEZE ===")

    # Per stuck_length
    for slen in sorted(df['stuck_length'].unique()):
        slen_df = df[df['stuck_length'] == slen]
        plot_metric_vs_fraction(slen_df, models, 'fraction', 'Fraction',
                                f'Freeze (block={int(slen)})',
                                os.path.join(out_base, f"len_{int(slen)}"))

    # Combined: all stuck_lengths for one model at a time
    length_styles = {10: '--', 50: '-', 200: '-.'}
    for model in models:
        model_df = df[df['model'] == model]
        s = _style(model)
        model_dir = os.path.join(out_base, "by_length", model)
        os.makedirs(model_dir, exist_ok=True)

        for metric_col, ylabel, std_col in METRICS:
            if metric_col not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            for slen in sorted(model_df['stuck_length'].unique()):
                sdf = model_df[model_df['stuck_length'] == slen].sort_values('fraction')
                ls = length_styles.get(int(slen), '-')
                ax.plot(sdf['fraction'], sdf[metric_col],
                        color=s['color'], marker=s['marker'], linestyle=ls,
                        linewidth=2, markersize=7, label=f'block={int(slen)}')

            ax.set_xlabel('Fraction')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{s["label"]} — {ylabel} vs Fraction')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.legend(loc='best')
            plt.tight_layout()
            out = os.path.join(model_dir, f"{_fname(ylabel)}_by_length.png")
            plt.savefig(out)
            plt.close()
            print(f"  {out}")


# ==========================================
# MISSING PLOTS
# ==========================================
def plot_missing():
    summary_csv = os.path.join(EXPERIMENTS_DIR, "missing", "summary.csv")
    if not os.path.exists(summary_csv):
        print("[SKIP] missing — no summary.csv")
        return

    df = pd.read_csv(summary_csv)
    models = df['model'].unique()
    out_base = os.path.join(PLOTS_DIR, "missing")
    print("\n=== MISSING ===")

    # Point missing — per imputation
    point_df = df[df['missing_type'] == 'point']
    for imp in point_df['imputation'].unique():
        imp_df = point_df[point_df['imputation'] == imp]
        plot_metric_vs_fraction(imp_df, models, 'fraction', 'Fraction',
                                f'Point Missing ({imp})',
                                os.path.join(out_base, "point", imp))

    # Burst missing — per burst_length, per imputation
    burst_df = df[df['missing_type'] == 'burst']
    for blen in sorted(burst_df['burst_length'].unique()):
        for imp in burst_df['imputation'].unique():
            sub = burst_df[(burst_df['burst_length'] == blen) & (burst_df['imputation'] == imp)]
            plot_metric_vs_fraction(sub, models, 'fraction', 'Fraction',
                                    f'Burst Missing (len={int(blen)}, {imp})',
                                    os.path.join(out_base, "burst", f"len_{int(blen)}", imp))

    # Imputation comparison: linear vs ffill for one model at a time
    for model in models:
        model_df = df[df['model'] == model]
        s = _style(model)

        # Point missing only
        pm = model_df[model_df['missing_type'] == 'point']
        if pm.empty:
            continue

        imp_dir = os.path.join(out_base, "imputation_comparison", model)
        os.makedirs(imp_dir, exist_ok=True)

        imp_styles = {'linear': '-', 'ffill': '--'}

        for metric_col, ylabel, std_col in METRICS:
            if metric_col not in pm.columns:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            for imp in pm['imputation'].unique():
                idf = pm[pm['imputation'] == imp].sort_values('fraction')
                ls = imp_styles.get(imp, '-')
                ax.plot(idf['fraction'], idf[metric_col],
                        color=s['color'], marker=s['marker'], linestyle=ls,
                        linewidth=2, markersize=7, label=imp)

            ax.set_xlabel('Fraction')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{s["label"]} — {ylabel} (linear vs ffill)')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.legend(loc='best')
            plt.tight_layout()
            out = os.path.join(imp_dir, f"{_fname(ylabel)}_imputation_cmp.png")
            plt.savefig(out)
            plt.close()
            print(f"  {out}")


# ==========================================
# HEATMAP — Cross-corruption comparison
# ==========================================
def plot_heatmap():
    """
    Heatmap: corruption condition × model → % AUC-ROC drop from baseline.
    This is THE central result of the thesis.
    """
    baseline_csv = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_results_5_corrected_z-score.csv")
    if not os.path.exists(baseline_csv):
        print("[SKIP] heatmap — no baseline CSV")
        return

    out_dir = os.path.join(PLOTS_DIR, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    print("\n=== HEATMAP ===")

    # Load baseline
    df_base = pd.read_csv(baseline_csv)
    if 'model' not in df_base.columns or 'AUC_ROC' not in df_base.columns:
        print("[SKIP] heatmap — baseline CSV missing required columns")
        return

    baseline_auc = df_base.groupby('model')['AUC_ROC'].mean().to_dict()

    # Collect representative conditions from each experiment
    conditions = {}

    # Swap
    swap_csv = os.path.join(EXPERIMENTS_DIR, "swap", "summary.csv")
    if os.path.exists(swap_csv):
        df = pd.read_csv(swap_csv)
        # Point swap, fraction=0.10
        sub = df[(df['swap_ratio'] == 0.0) & (df['fraction'] == 0.10)]
        if not sub.empty:
            conditions['Swap\n(point, 10%)'] = sub.set_index('model')['mean_AUC_ROC'].to_dict()
        # Segment swap, fraction=0.10
        sub = df[(df['swap_ratio'] == 0.05) & (df['fraction'] == 0.10)]
        if not sub.empty:
            conditions['Swap\n(segment, 10%)'] = sub.set_index('model')['mean_AUC_ROC'].to_dict()

    # Spikes
    spikes_csv = os.path.join(EXPERIMENTS_DIR, "spikes", "summary.csv")
    if os.path.exists(spikes_csv):
        df = pd.read_csv(spikes_csv)
        sub = df[(df['sequential'] == False) & (df['multiplier'] == 10.0) & (df['fraction'] == 0.10)]
        if not sub.empty:
            conditions['Spikes\n(10x, 10%)'] = sub.set_index('model')['mean_AUC_ROC'].to_dict()

    # Freeze
    freeze_csv = os.path.join(EXPERIMENTS_DIR, "freeze", "summary.csv")
    if os.path.exists(freeze_csv):
        df = pd.read_csv(freeze_csv)
        sub = df[(df['stuck_length'] == 50) & (df['fraction'] == 0.10)]
        if not sub.empty:
            conditions['Freeze\n(len=50, 10%)'] = sub.set_index('model')['mean_AUC_ROC'].to_dict()

    # Missing
    missing_csv = os.path.join(EXPERIMENTS_DIR, "missing", "summary.csv")
    if os.path.exists(missing_csv):
        df = pd.read_csv(missing_csv)
        sub = df[(df['missing_type'] == 'point') & (df['imputation'] == 'linear') & (df['fraction'] == 0.10)]
        if not sub.empty:
            conditions['Missing\n(point, 10%)'] = sub.set_index('model')['mean_AUC_ROC'].to_dict()
        sub = df[(df['missing_type'] == 'burst') & (df['burst_length'] == 50) & (df['imputation'] == 'linear') & (df['fraction'] == 0.10)]
        if not sub.empty:
            conditions['Missing\n(burst, 10%)'] = sub.set_index('model')['mean_AUC_ROC'].to_dict()

    # SNR
    snr_csv = os.path.join(EXPERIMENTS_DIR, "white_noise_snr", "summary.csv")
    if os.path.exists(snr_csv):
        df = pd.read_csv(snr_csv)
        sub = df[df['snr_db'] == 5]
        if not sub.empty:
            conditions['Noise\n(SNR=5dB)'] = sub.set_index('model')['mean_AUC_ROC'].to_dict()

    if not conditions:
        print("[SKIP] heatmap — no experiment data found")
        return

    models = sorted(baseline_auc.keys())
    cond_names = list(conditions.keys())

    # Build drop matrix (%)
    drop_matrix = np.full((len(cond_names), len(models)), np.nan)
    for i, cond in enumerate(cond_names):
        for j, model in enumerate(models):
            if model in conditions[cond] and model in baseline_auc:
                base = baseline_auc[model]
                corrupted = conditions[cond][model]
                if base > 0:
                    drop_matrix[i, j] = ((base - corrupted) / base) * 100

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(cond_names) * 0.8 + 2)))

    cmap = plt.cm.RdYlGn_r  # red = bad (big drop), green = good (small drop)
    im = ax.imshow(drop_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=np.nanmax(drop_matrix) * 1.1)

    model_labels = [MODEL_STYLES.get(m, {}).get('label', m) for m in models]
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(cond_names)))
    ax.set_yticklabels(cond_names)

    # Annotate cells
    for i in range(len(cond_names)):
        for j in range(len(models)):
            val = drop_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > np.nanmax(drop_matrix) * 0.6 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        fontsize=12, fontweight='bold', color=color)

    ax.set_title('AUC-ROC Drop (%) per Corruption Type', fontsize=14, pad=15)
    plt.colorbar(im, ax=ax, label='Drop (%)', shrink=0.8)

    plt.tight_layout()
    out = os.path.join(out_dir, "heatmap_auc_roc_drop.png")
    plt.savefig(out)
    plt.close()
    print(f"  {out}")


# ==========================================
# BOX PLOTS — Distribution per corruption
# ==========================================
def plot_boxplots():
    """
    Box plots showing distribution of AUC-ROC across files for each
    corruption type at fraction=0.10.
    """
    out_dir = os.path.join(PLOTS_DIR, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    print("\n=== BOX PLOTS ===")

    # Collect raw results (not summaries) at fraction=0.10
    data_by_model = {}

    experiments = {
        'swap': {'file': 'checkpoint.csv', 'filter': lambda df: df[(df['swap_ratio'] == 0.0) & (df['fraction'] == 0.10)]},
        'spikes': {'file': 'checkpoint.csv', 'filter': lambda df: df[(df['sequential'] == False) & (df['multiplier'] == 10.0) & (df['fraction'] == 0.10)]},
        'freeze': {'file': 'checkpoint.csv', 'filter': lambda df: df[(df['stuck_length'] == 50) & (df['fraction'] == 0.10)]},
        'missing': {'file': 'checkpoint.csv', 'filter': lambda df: df[(df['missing_type'] == 'point') & (df['imputation'] == 'linear') & (df['fraction'] == 0.10)]},
    }

    for exp_name, config in experiments.items():
        csv_path = os.path.join(EXPERIMENTS_DIR, exp_name, config['file'])
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if 'error' in df.columns:
            df = df[df['error'].isna()]
        if 'AUC_ROC' not in df.columns:
            continue

        filtered = config['filter'](df)
        if filtered.empty:
            continue

        for model in filtered['model'].unique():
            if model not in data_by_model:
                data_by_model[model] = {}
            model_data = filtered[filtered['model'] == model]['AUC_ROC'].dropna().values
            if len(model_data) > 0:
                data_by_model[model][exp_name] = model_data

    if not data_by_model:
        print("[SKIP] box plots — no data found")
        return

    # One box plot per model
    for model in sorted(data_by_model.keys()):
        exp_data = data_by_model[model]
        if not exp_data:
            continue

        s = _style(model)
        labels = list(exp_data.keys())
        values = [exp_data[l] for l in labels]

        fig, ax = plt.subplots(figsize=(8, 6))

        bp = ax.boxplot(values, labels=[l.title() for l in labels], patch_artist=True,
                        boxprops=dict(facecolor=s['color'], alpha=0.3),
                        medianprops=dict(color=s['color'], linewidth=2),
                        whiskerprops=dict(color=s['color']),
                        capprops=dict(color=s['color']),
                        flierprops=dict(markeredgecolor=s['color'], marker='o', markersize=4))

        ax.set_ylabel('AUC-ROC')
        ax.set_title(f'{s["label"]} — AUC-ROC Distribution (fraction=0.10)')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.tight_layout()
        out = os.path.join(out_dir, f"boxplot_{model}_auc_roc.png")
        plt.savefig(out)
        plt.close()
        print(f"  {out}")

    # Combined: all models side by side for each experiment
    all_experiments = set()
    for model_data in data_by_model.values():
        all_experiments.update(model_data.keys())
    all_experiments = sorted(all_experiments)

    if not all_experiments:
        return

    models = sorted(data_by_model.keys())
    n_models = len(models)
    n_exp = len(all_experiments)

    fig, ax = plt.subplots(figsize=(max(10, n_exp * 3), 6))

    width = 0.8 / n_models
    positions_base = np.arange(n_exp)

    for i, model in enumerate(models):
        s = _style(model)
        model_values = []
        model_positions = []

        for j, exp in enumerate(all_experiments):
            if exp in data_by_model.get(model, {}):
                model_values.append(data_by_model[model][exp])
                model_positions.append(positions_base[j] + i * width - (n_models - 1) * width / 2)

        if model_values:
            bp = ax.boxplot(model_values, positions=model_positions, widths=width * 0.9,
                            patch_artist=True,
                            boxprops=dict(facecolor=s['color'], alpha=0.3),
                            medianprops=dict(color=s['color'], linewidth=2),
                            whiskerprops=dict(color=s['color']),
                            capprops=dict(color=s['color']),
                            flierprops=dict(markeredgecolor=s['color'], marker='o', markersize=3))

    ax.set_xticks(positions_base)
    ax.set_xticklabels([e.title() for e in all_experiments])
    ax.set_ylabel('AUC-ROC')
    ax.set_title('AUC-ROC Distribution per Corruption Type (fraction=0.10)')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Custom legend
    legend_patches = []
    for model in models:
        s = _style(model)
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=s['color'], alpha=0.3, edgecolor=s['color'], label=s['label']))
    ax.legend(handles=legend_patches, loc='best')

    plt.tight_layout()
    out = os.path.join(out_dir, "boxplot_all_models_comparison.png")
    plt.savefig(out)
    plt.close()
    print(f"  {out}")


# ==========================================
# MAIN
# ==========================================
def main():
    print("Generating plots for all experiments...")
    plot_swap()
    plot_spikes()
    plot_freeze()
    plot_missing()
    plot_heatmap()
    plot_boxplots()
    print(f"\nDone — all plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
