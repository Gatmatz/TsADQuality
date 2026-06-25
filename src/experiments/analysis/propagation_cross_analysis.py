"""
Cross-Experiment Analysis: Propagation vs AUC Drop

Investigates whether a model's propagation behavior (how localized corruption
spreads through anomaly scores) predicts its actual AUC degradation under
global corruption from the main experiments.

Analyses:
  1. Per-file Spearman correlation: propagation_ratio vs AUC_drop
     per (model, corruption_type, severity)
  2. Aggregate model-level comparison: mean propagation vs mean AUC drop
  3. Zone decay profiles: how score changes decay with distance from corruption
  4. Model sensitivity classification: global-sensitive vs local-sensitive

Output (results/experiments/propagation/cross_analysis/):
  - per_file_correlations.csv
  - aggregate_comparison.csv
  - zone_decay_profiles.csv
  - model_classification.csv
  - Console summary

Usage:
    python propagation_cross_analysis.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Data paths
PROP_RESULTS = os.path.join(PROJECT_ROOT, "results", "experiments", "propagation", "propagation_results.csv")
PROP_ZONES = os.path.join(PROJECT_ROOT, "results", "experiments", "propagation", "propagation_zones.csv")
NOISE_CKPT = os.path.join(PROJECT_ROOT, "results", "experiments", "white_noise_snr", "checkpoint.csv")
SPIKES_CKPT = os.path.join(PROJECT_ROOT, "results", "experiments", "spikes_normal_only", "checkpoint.csv")
MISSING_CKPT = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_true_impact", "checkpoint.csv")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "propagation", "cross_analysis")

# Severity mapping: propagation severity -> main experiment condition
# Propagation uses localized corruption (10% of series)
# Main experiments use global corruption (entire series)
SEVERITY_MAP = {
    'noise': {
        'low':  {'condition': 'snr_20dB'},
        'med':  {'condition': 'snr_10dB'},
        'high': {'condition': 'snr_5dB'},
    },
    'spikes': {
        'low':  {'fraction': 0.05, 'multiplier': 3.0},
        'med':  {'fraction': 0.05, 'multiplier': 5.0},
        'high': {'fraction': 0.05, 'multiplier': 10.0},
    },
    'missing': {
        'low':  {'fraction': 0.05, 'missing_type': 'point'},
        'med':  {'fraction': 0.10, 'missing_type': 'point'},
        'high': {'fraction': 0.20, 'missing_type': 'point'},
    },
}


def load_data():
    """Load all experiment data."""
    prop = pd.read_csv(PROP_RESULTS)
    bl = pd.read_csv(BASELINE_CSV)
    bl['model'] = bl['model'].replace('Autoencoder', 'AE')

    noise = pd.read_csv(NOISE_CKPT)
    noise = noise[noise['error'].isna()] if 'error' in noise.columns else noise

    spikes = pd.read_csv(SPIKES_CKPT)
    spikes = spikes[spikes['error'].isna()] if 'error' in spikes.columns else spikes

    missing = pd.read_csv(MISSING_CKPT)
    missing = missing[missing['error'].isna()] if 'error' in missing.columns else missing

    return prop, bl, noise, spikes, missing


def get_main_experiment_auc(corruption_type, severity, model, noise_df, spikes_df, missing_df):
    """Get per-file AUC from the main experiment matching a propagation condition."""
    mapping = SEVERITY_MAP[corruption_type][severity]

    if corruption_type == 'noise':
        sub = noise_df[(noise_df['model'] == model) & (noise_df['condition'] == mapping['condition'])]
        return sub[['file', 'AUC_ROC']].copy()

    elif corruption_type == 'spikes':
        sub = spikes_df[
            (spikes_df['model'] == model) &
            (spikes_df['fraction'] == mapping['fraction']) &
            (spikes_df['multiplier'] == mapping['multiplier'])
        ]
        return sub[['file', 'AUC_ROC']].copy()

    elif corruption_type == 'missing':
        sub = missing_df[
            (missing_df['model'] == model) &
            (missing_df['fraction'] == mapping['fraction']) &
            (missing_df['missing_type'] == mapping['missing_type'])
        ]
        return sub[['file', 'AUC_ROC']].copy()

    return pd.DataFrame()


# =============================================
# Analysis 1: Per-file correlations
# =============================================
def per_file_correlations(prop, bl, noise, spikes, missing):
    """Spearman correlation between propagation_ratio and AUC_drop per file."""
    print("=" * 65)
    print("  Analysis 1: Per-File Correlation (Propagation Ratio vs AUC Drop)")
    print("=" * 65)

    results = []

    for model in prop['model'].unique():
        for ctype in prop['corruption_type'].unique():
            for sev in ['low', 'med', 'high']:
                # Propagation data
                prop_sub = prop[
                    (prop['model'] == model) &
                    (prop['corruption_type'] == ctype) &
                    (prop['severity'] == sev)
                ][['file', 'propagation_ratio', 'mean_diff_corruption_zone',
                   'mean_diff_far']].copy()

                if prop_sub.empty:
                    continue

                # Main experiment AUC
                main_auc = get_main_experiment_auc(ctype, sev, model, noise, spikes, missing)
                if main_auc.empty:
                    continue

                # Baseline AUC
                bl_sub = bl[bl['model'] == model][['file', 'AUC_ROC']].rename(
                    columns={'AUC_ROC': 'bl_auc'})

                # Merge
                merged = prop_sub.merge(
                    main_auc.rename(columns={'AUC_ROC': 'corrupt_auc'}),
                    on='file', how='inner'
                ).merge(bl_sub, on='file', how='inner')

                merged['auc_drop'] = merged['bl_auc'] - merged['corrupt_auc']

                if len(merged) < 10:
                    continue

                # Spearman correlation
                r, p = stats.spearmanr(merged['propagation_ratio'], merged['auc_drop'])

                # Also correlate zone metrics
                r_zone, p_zone = stats.spearmanr(
                    merged['mean_diff_corruption_zone'], merged['auc_drop'])
                r_far, p_far = stats.spearmanr(
                    merged['mean_diff_far'], merged['auc_drop'])

                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

                results.append({
                    'model': model,
                    'corruption_type': ctype,
                    'severity': sev,
                    'n_files': len(merged),
                    'r_propagation_ratio': round(r, 3),
                    'p_propagation_ratio': round(p, 4),
                    'significant': sig,
                    'r_corruption_zone': round(r_zone, 3),
                    'p_corruption_zone': round(p_zone, 4),
                    'r_far_zone': round(r_far, 3),
                    'p_far_zone': round(p_far, 4),
                    'mean_prop_ratio': round(merged['propagation_ratio'].mean(), 2),
                    'mean_auc_drop': round(merged['auc_drop'].mean(), 4),
                })

    df = pd.DataFrame(results)

    # Print summary
    print(f"\n{'Model':<10} {'Corruption':<10} {'Severity':<8} {'n':>4} "
          f"{'r':>7} {'p':>8} {'Sig':>4}  {'mean_prop':>9} {'mean_drop':>9}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['model']:<10} {row['corruption_type']:<10} {row['severity']:<8} "
              f"{row['n_files']:>4} {row['r_propagation_ratio']:>+7.3f} "
              f"{row['p_propagation_ratio']:>8.4f} {row['significant']:>4}  "
              f"{row['mean_prop_ratio']:>9.2f} {row['mean_auc_drop']:>+9.4f}")

    # Summary stats
    sig_count = len(df[df['p_propagation_ratio'] < 0.05])
    total = len(df)
    print(f"\nSignificant correlations: {sig_count}/{total}")

    pos_sig = df[(df['p_propagation_ratio'] < 0.05) & (df['r_propagation_ratio'] > 0)]
    neg_sig = df[(df['p_propagation_ratio'] < 0.05) & (df['r_propagation_ratio'] < 0)]
    print(f"  Positive (more propagation -> more drop): {len(pos_sig)}")
    print(f"  Negative (more propagation -> less drop): {len(neg_sig)}")

    return df


# =============================================
# Analysis 2: Aggregate model-level comparison
# =============================================
def aggregate_comparison(prop, bl, noise, spikes, missing):
    """Compare mean propagation ratio vs mean AUC drop per (model, corruption_type)."""
    print("\n" + "=" * 65)
    print("  Analysis 2: Aggregate Model-Level Comparison")
    print("=" * 65)

    results = []

    for model in prop['model'].unique():
        bl_auc = bl[bl['model'] == model]['AUC_ROC'].mean()

        for ctype in prop['corruption_type'].unique():
            # Mean propagation ratio (across severities)
            prop_sub = prop[(prop['model'] == model) & (prop['corruption_type'] == ctype)]
            mean_prop = prop_sub['propagation_ratio'].mean()
            mean_zone = prop_sub['mean_diff_corruption_zone'].mean()
            mean_far = prop_sub['mean_diff_far'].mean()

            # Mean AUC from main experiment (high severity)
            main_auc = get_main_experiment_auc(ctype, 'high', model, noise, spikes, missing)
            if main_auc.empty:
                continue

            mean_corrupt_auc = main_auc['AUC_ROC'].mean()
            mean_drop = bl_auc - mean_corrupt_auc

            # Decay ratio: far / corruption_zone (1.0 = no decay = global effect)
            decay_ratio = mean_far / mean_zone if mean_zone > 0.001 else 0

            results.append({
                'model': model,
                'corruption_type': ctype,
                'baseline_auc': round(bl_auc, 4),
                'corrupted_auc': round(mean_corrupt_auc, 4),
                'auc_drop': round(mean_drop, 4),
                'mean_propagation_ratio': round(mean_prop, 2),
                'mean_diff_corruption_zone': round(mean_zone, 3),
                'mean_diff_far': round(mean_far, 3),
                'decay_ratio': round(decay_ratio, 3),
            })

    df = pd.DataFrame(results)

    print(f"\n{'Model':<10} {'Corruption':<10} {'AUC drop':>9} {'Prop ratio':>11} "
          f"{'Zone diff':>10} {'Far diff':>9} {'Decay':>6}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['model']:<10} {row['corruption_type']:<10} "
              f"{row['auc_drop']:>+9.4f} {row['mean_propagation_ratio']:>11.2f} "
              f"{row['mean_diff_corruption_zone']:>10.3f} "
              f"{row['mean_diff_far']:>9.3f} {row['decay_ratio']:>6.3f}")

    # Cross-model correlation
    if len(df) >= 5:
        r, p = stats.spearmanr(df['mean_propagation_ratio'], df['auc_drop'])
        print(f"\nCross-model Spearman (propagation vs AUC drop): r={r:+.3f}, p={p:.4f}")
        r2, p2 = stats.spearmanr(df['decay_ratio'], df['auc_drop'])
        print(f"Cross-model Spearman (decay_ratio vs AUC drop):  r={r2:+.3f}, p={p2:.4f}")

    return df


# =============================================
# Analysis 3: Zone decay profiles
# =============================================
def zone_decay_profiles(prop):
    """Characterize zone decay patterns per model."""
    print("\n" + "=" * 65)
    print("  Analysis 3: Zone Decay Profiles")
    print("=" * 65)

    zone_cols = ['mean_diff_corruption_zone', 'mean_diff_near', 'mean_diff_mid', 'mean_diff_far']
    zone_names = ['corruption', 'near', 'mid', 'far']

    results = []

    for model in sorted(prop['model'].unique()):
        for ctype in sorted(prop['corruption_type'].unique()):
            sub = prop[(prop['model'] == model) & (prop['corruption_type'] == ctype)]
            if sub.empty:
                continue

            means = [sub[col].mean() for col in zone_cols]
            zone_val = means[0]
            far_val = means[3]

            # Decay type classification
            decay_ratio = far_val / zone_val if zone_val > 0.001 else 0

            if decay_ratio > 0.25:
                decay_type = 'global (flat)'
            elif decay_ratio > 0.12:
                decay_type = 'moderate'
            else:
                decay_type = 'local (sharp)'

            row = {
                'model': model,
                'corruption_type': ctype,
                'decay_type': decay_type,
                'decay_ratio': round(decay_ratio, 3),
            }
            for name, val in zip(zone_names, means):
                row[f'mean_{name}'] = round(val, 4)

            results.append(row)

    df = pd.DataFrame(results)

    print(f"\n{'Model':<10} {'Corruption':<10} {'Corrupt':>8} {'Near':>8} "
          f"{'Mid':>8} {'Far':>8} {'Decay':>6} {'Type':<15}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['model']:<10} {row['corruption_type']:<10} "
              f"{row['mean_corruption']:>8.4f} {row['mean_near']:>8.4f} "
              f"{row['mean_mid']:>8.4f} {row['mean_far']:>8.4f} "
              f"{row['decay_ratio']:>6.3f} {row['decay_type']:<15}")

    return df


# =============================================
# Analysis 4: Model sensitivity classification
# =============================================
def model_classification(prop, bl, noise, spikes, missing):
    """Classify models as global-sensitive vs local-sensitive."""
    print("\n" + "=" * 65)
    print("  Analysis 4: Model Sensitivity Classification")
    print("=" * 65)

    results = []

    for model in sorted(prop['model'].unique()):
        sub = prop[prop['model'] == model]

        mean_prop_ratio = sub['propagation_ratio'].mean()
        mean_zone = sub['mean_diff_corruption_zone'].mean()
        mean_far = sub['mean_diff_far'].mean()
        decay_ratio = mean_far / mean_zone if mean_zone > 0.001 else 0

        # Classification based on mean decay ratio across corruption types
        # IForest ~0.40 (scores change far from corruption)
        # LOF ~0.14, AE ~0.15, MP ~0.07 (changes stay local)
        if decay_ratio > 0.25:
            sensitivity_type = 'global-sensitive'
        else:
            sensitivity_type = 'local-sensitive'

        # Mean AUC drop across all corruption types (high severity)
        drops = []
        for ctype in sub['corruption_type'].unique():
            main_auc = get_main_experiment_auc(ctype, 'high', model, noise, spikes, missing)
            if not main_auc.empty:
                bl_auc = bl[bl['model'] == model]['AUC_ROC'].mean()
                drops.append(bl_auc - main_auc['AUC_ROC'].mean())

        mean_auc_drop = np.mean(drops) if drops else np.nan

        results.append({
            'model': model,
            'sensitivity_type': sensitivity_type,
            'mean_propagation_ratio': round(mean_prop_ratio, 2),
            'mean_decay_ratio': round(decay_ratio, 3),
            'mean_diff_corruption_zone': round(mean_zone, 3),
            'mean_diff_far': round(mean_far, 3),
            'mean_auc_drop_global': round(mean_auc_drop, 4) if not np.isnan(mean_auc_drop) else np.nan,
        })

    df = pd.DataFrame(results)

    print(f"\n{'Model':<10} {'Type':<20} {'Prop ratio':>11} {'Decay':>6} "
          f"{'Zone diff':>10} {'Far diff':>9} {'AUC drop':>9}")
    print("-" * 80)
    for _, row in df.iterrows():
        drop_str = f"{row['mean_auc_drop_global']:>+9.4f}" if pd.notna(row['mean_auc_drop_global']) else "      N/A"
        print(f"{row['model']:<10} {row['sensitivity_type']:<20} "
              f"{row['mean_propagation_ratio']:>11.2f} {row['mean_decay_ratio']:>6.3f} "
              f"{row['mean_diff_corruption_zone']:>10.3f} "
              f"{row['mean_diff_far']:>9.3f} {drop_str}")

    print("\n  Key insight: sensitivity_type does NOT predict AUC drop magnitude.")
    print("  Global-sensitive models (IForest) shift scores uniformly -> rankings preserved.")
    print("  Local-sensitive models (MP) confine changes locally, but under global corruption")
    print("  every local comparison is affected -> larger AUC degradation.")

    return df


def main():
    print("Loading data...")
    prop, bl, noise, spikes, missing = load_data()
    print(f"  Propagation: {len(prop)} rows")
    print(f"  Baseline: {bl['model'].nunique()} models, {bl['file'].nunique()} files")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run analyses
    df_corr = per_file_correlations(prop, bl, noise, spikes, missing)
    df_agg = aggregate_comparison(prop, bl, noise, spikes, missing)
    df_zones = zone_decay_profiles(prop)
    df_class = model_classification(prop, bl, noise, spikes, missing)

    # Save
    df_corr.to_csv(os.path.join(OUTPUT_DIR, "per_file_correlations.csv"), index=False)
    df_agg.to_csv(os.path.join(OUTPUT_DIR, "aggregate_comparison.csv"), index=False)
    df_zones.to_csv(os.path.join(OUTPUT_DIR, "zone_decay_profiles.csv"), index=False)
    df_class.to_csv(os.path.join(OUTPUT_DIR, "model_classification.csv"), index=False)

    print(f"\n{'=' * 65}")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
