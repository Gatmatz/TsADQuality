"""
Unified Causal Chain Analysis — Connects all experiments for deep interpretation.

Loads results from ALL experiments and produces:
  1. cross_corruption_ranking.csv — which corruption worst per model (Q2)
  2. cri_table.csv + radar_chart.png — Corruption Robustness Index (Q1 visual)
  3. ranking_inversion.csv — does best model change under corruption?
  4. causal_chain.csv — internal change + segment loss + FP/FN -> AUC drop
  5. cross_correlations.csv — internal metrics <-> AUC drops
  6. dataset_characteristics.csv — impact by dataset properties

Usage:
    python src/analysis/unified_causal_analysis.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "analysis", "unified")

MODEL_ORDER = ['IForest', 'LOF', 'MP', 'AE']
MODEL_NAME_MAP = {'Autoencoder': 'AE', 'MatrixProfile': 'MP'}

# ================================================================
# DATA LOADING
# ================================================================

def load_csv(path, name=""):
    if not os.path.exists(path):
        print(f"  [SKIP] {name or path} not found")
        return None
    df = pd.read_csv(path)
    print(f"  [OK]   {name or os.path.basename(path)}: {len(df)} rows")
    return df


def load_all_data():
    """Load all experiment results. Returns dict of DataFrames."""
    print("Loading data...")
    data = {}

    # Baselines (clean)
    data['baseline'] = load_csv(
        os.path.join(TABLES_DIR, "baseline_final_subset.csv"), "baseline")
    if data['baseline'] is not None:
        data['baseline']['model'] = data['baseline']['model'].replace(MODEL_NAME_MAP)

    # Main experiments
    data['noise'] = load_csv(
        os.path.join(RESULTS_DIR, "white_noise_snr", "summary.csv"), "noise (SNR)")
    data['spikes'] = load_csv(
        os.path.join(RESULTS_DIR, "spikes_normal_only", "summary.csv"), "spikes")
    data['missing'] = load_csv(
        os.path.join(RESULTS_DIR, "missing_true_impact", "summary.csv"), "missing")
    data['swap'] = load_csv(
        os.path.join(RESULTS_DIR, "swap_segment", "summary.csv"), "swap")
    data['freeze'] = load_csv(
        os.path.join(RESULTS_DIR, "freeze", "summary.csv"), "freeze")

    # Analysis experiments
    data['internal'] = load_csv(
        os.path.join(RESULTS_DIR, "internal_analysis", "aggregate_summary.csv"), "internal")
    data['segment'] = load_csv(
        os.path.join(RESULTS_DIR, "segment_vulnerability", "bin_detection_drop.csv"), "segment vuln")
    data['fpfn'] = load_csv(
        os.path.join(RESULTS_DIR, "fp_fn_decomposition", "fp_fn_decomposition.csv"), "FP/FN")
    data['propagation'] = load_csv(
        os.path.join(RESULTS_DIR, "propagation", "propagation_compact.csv"), "propagation")

    # Per-file baselines (for dataset characteristics)
    data['baseline_perfile'] = load_csv(
        os.path.join(TABLES_DIR, "baseline_final_subset.csv"), "baseline per-file")
    if data['baseline_perfile'] is not None:
        data['baseline_perfile']['model'] = data['baseline_perfile']['model'].replace(MODEL_NAME_MAP)

    # Per-file experiment results (checkpoints)
    for exp_name in ['white_noise_snr', 'spikes_normal_only', 'swap_segment', 'freeze']:
        ckpt = os.path.join(RESULTS_DIR, exp_name, "checkpoint.csv")
        data[f'{exp_name}_perfile'] = load_csv(ckpt, f"{exp_name} per-file")

    print()
    return data


# ================================================================
# 1. CROSS-CORRUPTION RANKING
# ================================================================

def compute_clean_auc(data):
    """Get mean clean AUC per model from baseline."""
    if data['baseline'] is None:
        return {}
    bl = data['baseline'].groupby('model')['AUC_ROC'].mean()
    return bl.to_dict()


def build_cross_corruption_ranking(data):
    """Rank corruption types by AUC drop per model."""
    clean_auc = compute_clean_auc(data)
    if not clean_auc:
        print("[SKIP] No baseline data for ranking")
        return None

    rows = []

    # Noise — use representative severities
    if data['noise'] is not None:
        for snr in [20, 10, 5, 0, -5]:
            sub = data['noise'][data['noise']['snr_db'] == snr]
            for _, r in sub.iterrows():
                model = r['model']
                if model in clean_auc:
                    rows.append({
                        'corruption': 'noise', 'condition': f'SNR={snr}dB',
                        'severity_param': -snr,  # higher = worse
                        'model': model,
                        'mean_AUC_ROC': r['mean_AUC_ROC'],
                        'clean_AUC_ROC': clean_auc[model],
                        'AUC_drop': r['mean_AUC_ROC'] - clean_auc[model],
                        'AUC_drop_pct': (r['mean_AUC_ROC'] - clean_auc[model]) / clean_auc[model] * 100,
                    })

    # Spikes — representative: multiplier=5
    if data['spikes'] is not None:
        for frac in [0.01, 0.05, 0.10, 0.20]:
            sub = data['spikes'][(data['spikes']['fraction'] == frac) &
                                  (data['spikes']['multiplier'] == 5.0)]
            for _, r in sub.iterrows():
                model = r['model']
                if model in clean_auc:
                    rows.append({
                        'corruption': 'spikes', 'condition': f'frac={frac},mult=5',
                        'severity_param': frac,
                        'model': model,
                        'mean_AUC_ROC': r['mean_AUC_ROC'],
                        'clean_AUC_ROC': clean_auc[model],
                        'AUC_drop': r['mean_AUC_ROC'] - clean_auc[model],
                        'AUC_drop_pct': (r['mean_AUC_ROC'] - clean_auc[model]) / clean_auc[model] * 100,
                    })

    # Missing — MCAR
    if data['missing'] is not None:
        for frac in [0.01, 0.05, 0.10, 0.20]:
            sub = data['missing'][(data['missing']['fraction'] == frac) &
                                   (data['missing']['missing_type'] == 'mcar')]
            if sub.empty:
                sub = data['missing'][data['missing']['fraction'] == frac]
            for _, r in sub.iterrows():
                model = r['model']
                if model in clean_auc:
                    rows.append({
                        'corruption': 'missing', 'condition': f'frac={frac}',
                        'severity_param': frac,
                        'model': model,
                        'mean_AUC_ROC': r['mean_AUC_ROC'],
                        'clean_AUC_ROC': clean_auc[model],
                        'AUC_drop': r['mean_AUC_ROC'] - clean_auc[model],
                        'AUC_drop_pct': (r['mean_AUC_ROC'] - clean_auc[model]) / clean_auc[model] * 100,
                    })

    # Swap — representative: num_swaps=5
    if data['swap'] is not None:
        for frac in [0.01, 0.05, 0.10, 0.20]:
            sub = data['swap'][(data['swap']['fraction'] == frac) &
                                (data['swap']['num_swaps'] == 5)]
            if sub.empty:
                sub = data['swap'][data['swap']['fraction'] == frac].head(len(MODEL_ORDER))
            for _, r in sub.iterrows():
                model = r['model']
                if model in clean_auc:
                    rows.append({
                        'corruption': 'swap', 'condition': f'frac={frac},ns=5',
                        'severity_param': frac,
                        'model': model,
                        'mean_AUC_ROC': r['mean_AUC_ROC'],
                        'clean_AUC_ROC': clean_auc[model],
                        'AUC_drop': r['mean_AUC_ROC'] - clean_auc[model],
                        'AUC_drop_pct': (r['mean_AUC_ROC'] - clean_auc[model]) / clean_auc[model] * 100,
                    })

    # Freeze
    if data['freeze'] is not None:
        for frac in [0.01, 0.05, 0.10, 0.20]:
            sub = data['freeze'][data['freeze']['fraction'] == frac]
            if 'num_stucks' in sub.columns:
                sub = sub[sub['num_stucks'] == 1] if not sub[sub['num_stucks'] == 1].empty else sub
            for _, r in sub.iterrows():
                model = r['model']
                if model in clean_auc:
                    rows.append({
                        'corruption': 'freeze', 'condition': f'frac={frac}',
                        'severity_param': frac,
                        'model': model,
                        'mean_AUC_ROC': r['mean_AUC_ROC'],
                        'clean_AUC_ROC': clean_auc[model],
                        'AUC_drop': r['mean_AUC_ROC'] - clean_auc[model],
                        'AUC_drop_pct': (r['mean_AUC_ROC'] - clean_auc[model]) / clean_auc[model] * 100,
                    })

    if not rows:
        return None

    df = pd.DataFrame(rows).round(4)
    df = df.sort_values(['model', 'AUC_drop'])
    return df


# ================================================================
# 2. CORRUPTION ROBUSTNESS INDEX (CRI)
# ================================================================

def compute_cri(data):
    """Compute CRI per (model, corruption_type).

    CRI = normalized area under the severity->AUC curve.
    Higher = more robust. Range [0, 1].
    """
    clean_auc = compute_clean_auc(data)
    if not clean_auc:
        return None

    cri_rows = []

    for model in MODEL_ORDER:
        if model not in clean_auc:
            continue
        baseline = clean_auc[model]

        # Noise: severity = inverse SNR (use SNR values as x-axis, AUC as y-axis)
        if data['noise'] is not None:
            sub = data['noise'][data['noise']['model'] == model].copy()
            if not sub.empty:
                sub = sub.sort_values('snr_db', ascending=False)  # worst first
                aucs = [baseline] + sub['mean_AUC_ROC'].tolist()
                # Normalize x to 0-1
                n_points = len(aucs)
                x = np.linspace(0, 1, n_points)
                cri = float(np.trapezoid(aucs, x))
                cri_rows.append({'model': model, 'corruption': 'noise', 'CRI': round(cri, 4)})

        # Spikes: use fraction as severity (fix multiplier=5)
        if data['spikes'] is not None:
            sub = data['spikes'][(data['spikes']['model'] == model) &
                                  (data['spikes']['multiplier'] == 5.0)].copy()
            if not sub.empty:
                sub = sub.sort_values('fraction')
                aucs = [baseline] + sub['mean_AUC_ROC'].tolist()
                x = np.linspace(0, 1, len(aucs))
                cri = float(np.trapezoid(aucs, x))
                cri_rows.append({'model': model, 'corruption': 'spikes', 'CRI': round(cri, 4)})

        # Missing: fraction as severity
        if data['missing'] is not None:
            sub = data['missing'][data['missing']['model'] == model].copy()
            if 'missing_type' in sub.columns:
                mcar = sub[sub['missing_type'] == 'mcar']
                if not mcar.empty:
                    sub = mcar
            if not sub.empty:
                sub = sub.sort_values('fraction')
                aucs = [baseline] + sub['mean_AUC_ROC'].tolist()
                x = np.linspace(0, 1, len(aucs))
                cri = float(np.trapezoid(aucs, x))
                cri_rows.append({'model': model, 'corruption': 'missing', 'CRI': round(cri, 4)})

        # Swap: fraction as severity (fix num_swaps=5)
        if data['swap'] is not None:
            sub = data['swap'][(data['swap']['model'] == model) &
                                (data['swap']['num_swaps'] == 5)].copy()
            if sub.empty:
                sub = data['swap'][data['swap']['model'] == model].copy()
            if not sub.empty:
                sub = sub.sort_values('fraction')
                aucs = [baseline] + sub['mean_AUC_ROC'].tolist()
                x = np.linspace(0, 1, len(aucs))
                cri = float(np.trapezoid(aucs, x))
                cri_rows.append({'model': model, 'corruption': 'swap', 'CRI': round(cri, 4)})

        # Freeze: fraction as severity
        if data['freeze'] is not None:
            sub = data['freeze'][data['freeze']['model'] == model].copy()
            if 'num_stucks' in sub.columns:
                s1 = sub[sub['num_stucks'] == 1]
                if not s1.empty:
                    sub = s1
            if not sub.empty:
                sub = sub.sort_values('fraction')
                aucs = [baseline] + sub['mean_AUC_ROC'].tolist()
                x = np.linspace(0, 1, len(aucs))
                cri = float(np.trapezoid(aucs, x))
                cri_rows.append({'model': model, 'corruption': 'freeze', 'CRI': round(cri, 4)})

    if not cri_rows:
        return None
    return pd.DataFrame(cri_rows)


def plot_radar_chart(cri_df, output_path):
    """Radar chart: one polygon per model, axes = corruption types."""
    if cri_df is None or cri_df.empty:
        return

    corruptions = cri_df['corruption'].unique().tolist()
    n_axes = len(corruptions)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    colors = {'IForest': '#1f77b4', 'LOF': '#ff7f0e', 'MP': '#d62728', 'AE': '#7f7f7f'}

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in MODEL_ORDER:
        model_data = cri_df[cri_df['model'] == model]
        if model_data.empty:
            continue
        values = []
        for c in corruptions:
            row = model_data[model_data['corruption'] == c]
            values.append(float(row['CRI'].iloc[0]) if not row.empty else 0)
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model,
                color=colors.get(model, '#333333'))
        ax.fill(angles, values, alpha=0.1, color=colors.get(model, '#333333'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in corruptions], fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Corruption Robustness Index (CRI)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.2, 0), fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Radar chart saved: {output_path}")


# ================================================================
# 3. MODEL RANKING INVERSION
# ================================================================

def compute_ranking_inversion(ranking_df):
    """Check if the best model changes under corruption."""
    if ranking_df is None:
        return None

    clean_auc = ranking_df.groupby('model')['clean_AUC_ROC'].first()
    clean_rank = clean_auc.sort_values(ascending=False)
    clean_best = clean_rank.index[0]

    rows = []
    for (corruption, condition), group in ranking_df.groupby(['corruption', 'condition']):
        corrupted_rank = group.set_index('model')['mean_AUC_ROC'].sort_values(ascending=False)
        corrupted_best = corrupted_rank.index[0]

        # Full ranking
        rank_order = corrupted_rank.index.tolist()

        rows.append({
            'corruption': corruption,
            'condition': condition,
            'clean_rank1': clean_best,
            'corrupted_rank1': corrupted_best,
            'inverted': clean_best != corrupted_best,
            'clean_ranking': ' > '.join(clean_rank.index.tolist()),
            'corrupted_ranking': ' > '.join(rank_order),
        })

    return pd.DataFrame(rows)


# ================================================================
# 4. CAUSAL CHAIN TABLE
# ================================================================

def build_causal_chain(data, ranking_df):
    """Build unified table connecting internal change -> segment loss -> AUC drop."""
    if ranking_df is None:
        return None

    # Get worst AUC drop per (corruption_type, model)
    worst = ranking_df.loc[
        ranking_df.groupby(['corruption', 'model'])['AUC_drop'].idxmin()
    ][['corruption', 'model', 'condition', 'AUC_drop', 'AUC_drop_pct']].copy()

    # Internal analysis — map corruption names
    if data['internal'] is not None:
        internal = data['internal'].copy()
        # Extract corruption type from name
        internal['corruption_type'] = internal['corruption'].apply(lambda x:
            'noise' if x.startswith('noise') else
            'spikes' if x.startswith('spikes') else
            'missing' if x.startswith('mcar') or x.startswith('mnar') else
            'swap' if x.startswith('swap') else 'other'
        )
        # Get worst internal change per (corruption_type, model)
        if 'separation_gap_change_pct_mean' in internal.columns:
            int_summary = internal.groupby(['corruption_type', 'model']).agg(
                max_gap_change=('separation_gap_change_pct_mean', 'min'),  # most negative = worst
            ).reset_index()
            worst = worst.merge(int_summary, left_on=['corruption', 'model'],
                                right_on=['corruption_type', 'model'], how='left')
            worst.drop(columns=['corruption_type'], errors='ignore', inplace=True)

    # Segment vulnerability — worst bin per (corruption_type, model)
    if data['segment'] is not None:
        seg = data['segment'].copy()
        worst_seg = seg.loc[seg.groupby(['corruption_type', 'model'])['detection_drop'].idxmax()]
        worst_seg = worst_seg[['corruption_type', 'model', 'combined_bin',
                                'detection_drop', 'clean_rate', 'detection_rate']].copy()
        worst_seg.columns = ['corruption_type', 'model', 'worst_bin',
                              'worst_bin_drop', 'worst_bin_clean_rate', 'worst_bin_corrupted_rate']
        worst = worst.merge(worst_seg, left_on=['corruption', 'model'],
                            right_on=['corruption_type', 'model'], how='left')
        worst.drop(columns=['corruption_type'], errors='ignore', inplace=True)

    # FP/FN — dominant error per (corruption, model)
    if data['fpfn'] is not None:
        fpfn = data['fpfn'].copy()
        # Get the dominant error at worst severity per corruption type
        fpfn_worst = fpfn.loc[fpfn.groupby(['corruption', 'model'])['auc_drop'].idxmax()]
        fpfn_summary = fpfn_worst[['corruption', 'model', 'dominant_error',
                                    'precision_drop_pct', 'recall_drop_pct']].copy()
        fpfn_summary.columns = ['corruption', 'model', 'dominant_error',
                                 'precision_drop_pct', 'recall_drop_pct']
        worst = worst.merge(fpfn_summary, on=['corruption', 'model'], how='left')

    return worst.round(4)


# ================================================================
# 5. CROSS-EXPERIMENT CORRELATIONS
# ================================================================

def compute_cross_correlations(data):
    """Correlate internal metric changes with AUC drops."""
    if data['internal'] is None:
        print("  [SKIP] No internal analysis data for correlations")
        return None

    internal = data['internal'].copy()
    if 'separation_gap_change_pct_mean' not in internal.columns:
        return None

    # We need per-corruption AUC values to correlate
    # Use FP/FN decomposition which has per-condition AUC
    if data['fpfn'] is None:
        return None

    fpfn = data['fpfn'].copy()

    # Map internal corruption names to fpfn conditions
    corr_results = []

    # For IForest: correlate separation_gap_change with auc_drop
    iforest_internal = internal[internal['model'] == 'IForest'].copy()
    iforest_fpfn = fpfn[fpfn['model'] == 'IForest'].copy()

    if not iforest_internal.empty and not iforest_fpfn.empty:
        # Match by corruption name patterns
        # Internal: noise_snr5, noise_snr10, etc.
        # FP/FN: noise with condition SNR=5dB, etc.
        # Manual mapping for noise
        pairs = []
        for _, row in iforest_internal.iterrows():
            corr_name = row['corruption']
            gap_change = row['separation_gap_change_pct_mean']

            # Try to find matching FP/FN row
            if corr_name.startswith('noise_snr'):
                snr = corr_name.replace('noise_snr', '')
                match = iforest_fpfn[iforest_fpfn['condition'].str.contains(f'SNR={snr}dB')]
                if not match.empty:
                    pairs.append((gap_change, match.iloc[0]['auc_drop']))
            elif corr_name.startswith('spikes_f'):
                parts = corr_name.replace('spikes_f', '').split('_m')
                if len(parts) == 2:
                    frac, mult = parts
                    match = iforest_fpfn[iforest_fpfn['condition'].str.contains(f'f={frac}')]
                    if not match.empty:
                        pairs.append((gap_change, match.iloc[0]['auc_drop']))

        if len(pairs) >= 5:
            x_vals, y_vals = zip(*pairs)
            r, p = stats.spearmanr(x_vals, y_vals)
            corr_results.append({
                'x_metric': 'IForest_separation_gap_change_pct',
                'y_metric': 'AUC_drop',
                'spearman_r': round(r, 4),
                'p_value': round(p, 4),
                'n_points': len(pairs),
                'significant': 'yes' if p < 0.05 else 'no',
            })

    if not corr_results:
        return None
    return pd.DataFrame(corr_results)


# ================================================================
# 6. DATASET CHARACTERISTICS
# ================================================================

def compute_dataset_characteristics(data):
    """Split AUC drops by dataset properties."""
    bl = data.get('baseline_perfile')
    if bl is None:
        return None

    # Compute per-file properties
    file_props = bl.groupby('file').agg(
        mean_clean_auc=('AUC_ROC', 'mean'),
        anomaly_ratio=('anomaly_ratio', 'first'),
    ).reset_index()

    # Bin by anomaly ratio
    median_ratio = file_props['anomaly_ratio'].median()
    file_props['ratio_group'] = np.where(
        file_props['anomaly_ratio'] <= median_ratio, 'low_ratio', 'high_ratio')

    # Bin by clean AUC
    median_auc = file_props['mean_clean_auc'].median()
    file_props['difficulty_group'] = np.where(
        file_props['mean_clean_auc'] <= median_auc, 'hard', 'easy')

    # Try to merge with per-file experiment results
    results = []
    for exp_key in ['white_noise_snr_perfile', 'spikes_normal_only_perfile',
                     'swap_segment_perfile', 'freeze_perfile']:
        exp_df = data.get(exp_key)
        if exp_df is None:
            continue

        exp_name = exp_key.replace('_perfile', '')

        # Merge with file properties
        if 'file' not in exp_df.columns:
            continue

        merged = exp_df.merge(file_props[['file', 'ratio_group', 'difficulty_group']],
                               on='file', how='left')
        merged = merged.dropna(subset=['ratio_group'])

        if 'AUC_ROC' not in merged.columns:
            continue

        # Get clean AUC per file×model from baseline
        clean = bl[['file', 'model', 'AUC_ROC']].rename(columns={'AUC_ROC': 'clean_AUC'})
        if 'model' in merged.columns:
            merged['model'] = merged['model'].replace(MODEL_NAME_MAP)
        merged = merged.merge(clean, on=['file', 'model'], how='left')
        merged['auc_drop'] = merged['AUC_ROC'] - merged['clean_AUC']

        # Aggregate by group
        for group_col in ['ratio_group', 'difficulty_group']:
            group_summary = merged.groupby([group_col, 'model']).agg(
                mean_auc_drop=('auc_drop', 'mean'),
                std_auc_drop=('auc_drop', 'std'),
                n_files=('file', 'nunique'),
            ).round(4).reset_index()
            group_summary['experiment'] = exp_name
            group_summary['group_type'] = group_col
            results.append(group_summary)

    if not results:
        return None
    return pd.concat(results, ignore_index=True)


# ================================================================
# MAIN
# ================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_all_data()

    # 1. Cross-corruption ranking
    print("=" * 60)
    print("  1. Cross-Corruption Ranking")
    print("=" * 60)
    ranking_df = build_cross_corruption_ranking(data)
    if ranking_df is not None:
        ranking_df.to_csv(os.path.join(OUTPUT_DIR, "cross_corruption_ranking.csv"), index=False)
        print(f"  Saved: {len(ranking_df)} rows")

        # Print summary: worst corruption per model
        print("\n  Worst corruption per model (largest AUC drop):")
        for model in MODEL_ORDER:
            m_df = ranking_df[ranking_df['model'] == model]
            if not m_df.empty:
                worst = m_df.loc[m_df['AUC_drop'].idxmin()]
                print(f"    {model:8s}: {worst['corruption']:8s} {worst['condition']:20s} "
                      f"AUC: {worst['clean_AUC_ROC']:.3f} -> {worst['mean_AUC_ROC']:.3f} "
                      f"({worst['AUC_drop_pct']:+.1f}%)")

    # 2. CRI + Radar
    print(f"\n{'=' * 60}")
    print("  2. Corruption Robustness Index (CRI)")
    print("=" * 60)
    cri_df = compute_cri(data)
    if cri_df is not None:
        cri_df.to_csv(os.path.join(OUTPUT_DIR, "cri_table.csv"), index=False)

        # Pivot for display
        pivot = cri_df.pivot(index='model', columns='corruption', values='CRI')
        pivot = pivot.reindex(MODEL_ORDER)
        print("\n  CRI Table (higher = more robust):")
        print(f"  {pivot.to_string()}")

        plot_radar_chart(cri_df, os.path.join(OUTPUT_DIR, "cri_radar_chart.png"))

    # 3. Ranking inversion
    print(f"\n{'=' * 60}")
    print("  3. Model Ranking Inversion")
    print("=" * 60)
    inversion_df = compute_ranking_inversion(ranking_df)
    if inversion_df is not None:
        inversion_df.to_csv(os.path.join(OUTPUT_DIR, "ranking_inversion.csv"), index=False)
        n_inverted = inversion_df['inverted'].sum()
        n_total = len(inversion_df)
        print(f"  Inversions: {n_inverted}/{n_total} conditions ({n_inverted/n_total*100:.0f}%)")

        inverted = inversion_df[inversion_df['inverted']]
        if not inverted.empty:
            print("\n  Conditions where best model changes:")
            for _, r in inverted.iterrows():
                print(f"    {r['corruption']:8s} {r['condition']:20s}: "
                      f"{r['clean_rank1']} -> {r['corrupted_rank1']}")

    # 4. Causal chain
    print(f"\n{'=' * 60}")
    print("  4. Causal Chain Table")
    print("=" * 60)
    causal_df = build_causal_chain(data, ranking_df)
    if causal_df is not None:
        causal_df.to_csv(os.path.join(OUTPUT_DIR, "causal_chain.csv"), index=False)
        print(f"  Saved: {len(causal_df)} rows")
        print("\n  Preview:")
        cols = ['corruption', 'model', 'AUC_drop_pct']
        extra = [c for c in ['max_gap_change', 'worst_bin', 'worst_bin_drop',
                              'dominant_error'] if c in causal_df.columns]
        print(causal_df[cols + extra].to_string(index=False))

    # 5. Cross-experiment correlations
    print(f"\n{'=' * 60}")
    print("  5. Cross-Experiment Correlations")
    print("=" * 60)
    corr_df = compute_cross_correlations(data)
    if corr_df is not None:
        corr_df.to_csv(os.path.join(OUTPUT_DIR, "cross_correlations.csv"), index=False)
        for _, r in corr_df.iterrows():
            sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
            print(f"  {r['x_metric']} <-> {r['y_metric']}: "
                  f"r={r['spearman_r']:+.3f} p={r['p_value']:.4f} {sig} (n={r['n_points']})")
    else:
        print("  [SKIP] Insufficient data for correlations")

    # 6. Dataset characteristics
    print(f"\n{'=' * 60}")
    print("  6. Dataset Characteristics")
    print("=" * 60)
    char_df = compute_dataset_characteristics(data)
    if char_df is not None:
        char_df.to_csv(os.path.join(OUTPUT_DIR, "dataset_characteristics.csv"), index=False)
        print(f"  Saved: {len(char_df)} rows")

        # Print key finding
        for gt in ['ratio_group', 'difficulty_group']:
            sub = char_df[char_df['group_type'] == gt]
            if sub.empty:
                continue
            print(f"\n  By {gt}:")
            pivot = sub.groupby([gt.replace('group_type', '').strip(),
                                  'model' if 'model' in sub.columns else gt])
            for col_name in [gt.replace('_group', '') + '_group']:
                if col_name in sub.columns:
                    summary = sub.groupby([col_name, 'model'])['mean_auc_drop'].mean().round(4)
                    print(f"  {summary.to_string()}")
    else:
        print("  [SKIP] Insufficient per-file data")

    print(f"\n{'=' * 60}")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
