"""
Advanced Internal Analysis (items 2 + 5, all available models)

Builds:
1) Distribution diagnostics for internal separation metrics (p10/p50/p90/IQR).
2) Correlations between internal degradation and downstream performance drop (AUC/F1).

Uses existing outputs:
  - results/experiments/internal_analysis/raw_internals.csv
  - main corruption checkpoints (noise/spikes/missing/swap/freeze)
  - results/tables/baseline_final_subset.csv for clean baseline metrics

Usage:
    python run_internal_advanced_analysis.py
"""
import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INTERNAL_RAW = os.path.join(PROJECT_ROOT, "results", "experiments", "internal_analysis", "raw_internals.csv")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")

NOISE_CKPT = os.path.join(PROJECT_ROOT, "results", "experiments", "white_noise_snr", "checkpoint.csv")
SPIKES_CKPT = os.path.join(PROJECT_ROOT, "results", "experiments", "spikes_normal_only", "checkpoint.csv")
MISSING_CKPT = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_mnar", "checkpoint.csv")
SWAP_CKPT = os.path.join(PROJECT_ROOT, "results", "experiments", "swap_segment", "checkpoint.csv")
FREEZE_CKPT = os.path.join(PROJECT_ROOT, "results", "experiments", "freeze", "checkpoint.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "internal_analysis", "advanced_analysis")


MODEL_METRIC_CONFIG = {
    'IForest': {
        'clean': 'separation_gap_clean',
        'corrupted': 'separation_gap_corrupted',
        'change_pct': 'separation_gap_change_pct',
        'non_separation_threshold': 0.0,
    },
    'LOF': {
        'clean': 'kdist_gap_clean',
        'corrupted': 'kdist_gap_corrupted',
        'change_pct': 'kdist_gap_change_pct',
        'non_separation_threshold': 0.0,
    },
    'MP': {
        'clean': 'nn_dist_gap_clean',
        'corrupted': 'nn_dist_gap_corrupted',
        'change_pct': 'nn_dist_gap_change_pct',
        'non_separation_threshold': 0.0,
    },
    'AE': {
        'clean': 'error_ratio_clean',
        'corrupted': 'error_ratio_corrupted',
        'change_pct': 'error_ratio_change_pct',
        'non_separation_threshold': 1.0,
    },
    'PCA': {
        'clean': None,      # derived from score_anomaly - score_normal
        'corrupted': None,  # derived from score_anomaly - score_normal
        'change_pct': 'score_gap_change_pct',
        'non_separation_threshold': 0.0,
    },
}


def _clean_ok(df):
    if 'error' not in df.columns:
        return df
    err = df['error'].astype(str).str.strip()
    return df[(df['error'].isna()) | (err == '') | (err == 'nan')]


def _fmt_fraction(x):
    s = f"{float(x):.6f}".rstrip('0').rstrip('.')
    return s if s != '-0' else '0'


def _q_stats(series):
    vals = pd.to_numeric(series, errors='coerce').dropna().to_numpy()
    if len(vals) == 0:
        return {'p10': np.nan, 'p50': np.nan, 'p90': np.nan, 'iqr': np.nan}
    q10, q50, q90 = np.percentile(vals, [10, 50, 90])
    q25, q75 = np.percentile(vals, [25, 75])
    return {
        'p10': float(q10),
        'p50': float(q50),
        'p90': float(q90),
        'iqr': float(q75 - q25),
    }


def _safe_spearman(x, y, min_n=8):
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    mask = (~x.isna()) & (~y.isna())
    n = int(mask.sum())
    if n < min_n:
        return np.nan, np.nan, n
    xv = x[mask]
    yv = y[mask]
    if xv.nunique() < 2 or yv.nunique() < 2:
        return np.nan, np.nan, n
    r, p = stats.spearmanr(xv, yv)
    return float(r), float(p), n


def prepare_internal_df(path):
    raw = pd.read_csv(path)
    raw = _clean_ok(raw).copy()

    # numeric coercion for potential metric columns
    for c in raw.columns:
        if c in ('file', 'corruption', 'model', 'error'):
            continue
        raw[c] = pd.to_numeric(raw[c], errors='coerce')

    raw['internal_metric_clean'] = np.nan
    raw['internal_metric_corrupted'] = np.nan
    raw['internal_metric_change_pct'] = np.nan
    raw['non_separation_threshold'] = np.nan

    for model, cfg in MODEL_METRIC_CONFIG.items():
        m = raw['model'] == model
        if not m.any():
            continue
        if model == 'PCA':
            clean = raw.loc[m, 'score_anomaly_clean'] - raw.loc[m, 'score_normal_clean']
            corr = raw.loc[m, 'score_anomaly_corrupted'] - raw.loc[m, 'score_normal_corrupted']
            raw.loc[m, 'internal_metric_clean'] = clean
            raw.loc[m, 'internal_metric_corrupted'] = corr
            if cfg['change_pct'] in raw.columns:
                raw.loc[m, 'internal_metric_change_pct'] = raw.loc[m, cfg['change_pct']]
            # fallback if no stored change%
            missing_change = raw.loc[m, 'internal_metric_change_pct'].isna()
            denom = clean.abs().replace(0, np.nan)
            calc_change = ((corr - clean) / denom) * 100.0
            raw.loc[m & missing_change, 'internal_metric_change_pct'] = calc_change[missing_change]
        else:
            raw.loc[m, 'internal_metric_clean'] = raw.loc[m, cfg['clean']]
            raw.loc[m, 'internal_metric_corrupted'] = raw.loc[m, cfg['corrupted']]
            raw.loc[m, 'internal_metric_change_pct'] = raw.loc[m, cfg['change_pct']]

        raw.loc[m, 'non_separation_threshold'] = cfg['non_separation_threshold']

    raw['metric_degraded'] = raw['internal_metric_corrupted'] < raw['internal_metric_clean']
    raw['non_separated_clean'] = raw['internal_metric_clean'] <= raw['non_separation_threshold']
    raw['non_separated_corrupted'] = raw['internal_metric_corrupted'] <= raw['non_separation_threshold']
    return raw


def build_distribution_report(df_internal):
    rows = []
    grouped = df_internal.groupby(['model', 'corruption'], dropna=False)
    for (model, corruption), g in grouped:
        clean_stats = _q_stats(g['internal_metric_clean'])
        corr_stats = _q_stats(g['internal_metric_corrupted'])
        chg_stats = _q_stats(g['internal_metric_change_pct'])

        rows.append({
            'model': model,
            'corruption': corruption,
            'n_files': int(len(g)),
            'internal_clean_mean': round(float(g['internal_metric_clean'].mean()), 6),
            'internal_corrupted_mean': round(float(g['internal_metric_corrupted'].mean()), 6),
            'internal_change_pct_mean': round(float(g['internal_metric_change_pct'].mean()), 4),
            'internal_clean_p10': round(clean_stats['p10'], 6),
            'internal_clean_p50': round(clean_stats['p50'], 6),
            'internal_clean_p90': round(clean_stats['p90'], 6),
            'internal_clean_iqr': round(clean_stats['iqr'], 6),
            'internal_corrupted_p10': round(corr_stats['p10'], 6),
            'internal_corrupted_p50': round(corr_stats['p50'], 6),
            'internal_corrupted_p90': round(corr_stats['p90'], 6),
            'internal_corrupted_iqr': round(corr_stats['iqr'], 6),
            'internal_change_pct_p10': round(chg_stats['p10'], 4),
            'internal_change_pct_p50': round(chg_stats['p50'], 4),
            'internal_change_pct_p90': round(chg_stats['p90'], 4),
            'internal_change_pct_iqr': round(chg_stats['iqr'], 4),
            'degraded_rate_pct': round(float(g['metric_degraded'].mean() * 100.0), 2),
            'non_separated_clean_pct': round(float(g['non_separated_clean'].mean() * 100.0), 2),
            'non_separated_corrupted_pct': round(float(g['non_separated_corrupted'].mean() * 100.0), 2),
        })
    return pd.DataFrame(rows)


def _load_baseline():
    bl = pd.read_csv(BASELINE_CSV)
    bl['model'] = bl['model'].replace({'Autoencoder': 'AE'})
    return bl[['file', 'model', 'AUC_ROC', 'F1']].rename(
        columns={'AUC_ROC': 'auc_clean', 'F1': 'f1_clean'}
    )


def _noise_perf_rows():
    df = _clean_ok(pd.read_csv(NOISE_CKPT)).copy()
    df['model'] = df['model'].replace({'Autoencoder': 'AE'})
    df['corruption'] = df['snr_db'].apply(lambda x: f"noise_snr{int(float(x))}")
    return df[['file', 'model', 'corruption', 'AUC_ROC', 'F']].rename(
        columns={'AUC_ROC': 'auc_corrupted', 'F': 'f1_corrupted'}
    )


def _spikes_perf_rows():
    df = _clean_ok(pd.read_csv(SPIKES_CKPT)).copy()
    df['model'] = df['model'].replace({'Autoencoder': 'AE'})
    df['corruption'] = df.apply(
        lambda r: f"spikes_f{_fmt_fraction(r['fraction'])}_m{float(r['multiplier']):.1f}", axis=1
    )
    return df[['file', 'model', 'corruption', 'AUC_ROC', 'F']].rename(
        columns={'AUC_ROC': 'auc_corrupted', 'F': 'f1_corrupted'}
    )


def _missing_perf_rows():
    df = _clean_ok(pd.read_csv(MISSING_CKPT)).copy()
    df = df[df['mechanism'].isin(['mcar', 'mnar_extreme'])].copy()
    df['model'] = df['model'].replace({'Autoencoder': 'AE'})
    df['corruption'] = df.apply(
        lambda r: f"{r['mechanism']}_{_fmt_fraction(r['fraction'])}", axis=1
    )
    return df[['file', 'model', 'corruption', 'AUC_ROC', 'F']].rename(
        columns={'AUC_ROC': 'auc_corrupted', 'F': 'f1_corrupted'}
    )


def _swap_perf_rows():
    df = _clean_ok(pd.read_csv(SWAP_CKPT)).copy()
    df['model'] = df['model'].replace({'Autoencoder': 'AE'})
    df['corruption'] = df.apply(
        lambda r: f"swap_f{_fmt_fraction(r['fraction'])}_ns{int(float(r['num_swaps']))}", axis=1
    )
    return df[['file', 'model', 'corruption', 'AUC_ROC', 'F']].rename(
        columns={'AUC_ROC': 'auc_corrupted', 'F': 'f1_corrupted'}
    )


def _freeze_perf_rows():
    df = _clean_ok(pd.read_csv(FREEZE_CKPT)).copy()
    df['model'] = df['model'].replace({'Autoencoder': 'AE'})
    df['corruption'] = df.apply(
        lambda r: f"freeze_f{_fmt_fraction(r['fraction'])}_ns{int(float(r['num_stucks']))}", axis=1
    )
    return df[['file', 'model', 'corruption', 'AUC_ROC', 'F']].rename(
        columns={'AUC_ROC': 'auc_corrupted', 'F': 'f1_corrupted'}
    )


def build_internal_performance_merge(df_internal):
    baseline = _load_baseline()
    perf = pd.concat([
        _noise_perf_rows(),
        _spikes_perf_rows(),
        _missing_perf_rows(),
        _swap_perf_rows(),
        _freeze_perf_rows(),
    ], ignore_index=True).drop_duplicates(subset=['file', 'model', 'corruption'])

    merged = df_internal.merge(perf, on=['file', 'model', 'corruption'], how='inner')
    merged = merged.merge(baseline, on=['file', 'model'], how='inner')
    merged['auc_drop'] = merged['auc_clean'] - merged['auc_corrupted']
    merged['f1_drop'] = merged['f1_clean'] - merged['f1_corrupted']
    return merged


def build_correlation_report(df_merged):
    rows = []

    # per (model, corruption)
    for (model, corruption), g in df_merged.groupby(['model', 'corruption']):
        r_auc, p_auc, n_auc = _safe_spearman(g['internal_metric_change_pct'], g['auc_drop'])
        r_f1, p_f1, n_f1 = _safe_spearman(g['internal_metric_change_pct'], g['f1_drop'])
        rows.append({
            'scope': 'per_corruption',
            'model': model,
            'corruption': corruption,
            'n_auc': n_auc,
            'spearman_internal_vs_auc_drop': round(r_auc, 4) if not np.isnan(r_auc) else np.nan,
            'p_auc': round(p_auc, 6) if not np.isnan(p_auc) else np.nan,
            'n_f1': n_f1,
            'spearman_internal_vs_f1_drop': round(r_f1, 4) if not np.isnan(r_f1) else np.nan,
            'p_f1': round(p_f1, 6) if not np.isnan(p_f1) else np.nan,
        })

    # per model (all corruptions pooled)
    for model, g in df_merged.groupby('model'):
        r_auc, p_auc, n_auc = _safe_spearman(g['internal_metric_change_pct'], g['auc_drop'])
        r_f1, p_f1, n_f1 = _safe_spearman(g['internal_metric_change_pct'], g['f1_drop'])
        rows.append({
            'scope': 'per_model_all_corruptions',
            'model': model,
            'corruption': 'ALL',
            'n_auc': n_auc,
            'spearman_internal_vs_auc_drop': round(r_auc, 4) if not np.isnan(r_auc) else np.nan,
            'p_auc': round(p_auc, 6) if not np.isnan(p_auc) else np.nan,
            'n_f1': n_f1,
            'spearman_internal_vs_f1_drop': round(r_f1, 4) if not np.isnan(r_f1) else np.nan,
            'p_f1': round(p_f1, 6) if not np.isnan(p_f1) else np.nan,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Advanced internal analysis (distribution + correlation)')
    parser.add_argument('--internal', default=INTERNAL_RAW, help='Path to raw internal metrics CSV')
    parser.add_argument('--outdir', default=OUTPUT_DIR, help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_internal = prepare_internal_df(args.internal)
    dist = build_distribution_report(df_internal)
    dist_path = os.path.join(args.outdir, "internal_gap_distributions.csv")
    dist.to_csv(dist_path, index=False)

    merged = build_internal_performance_merge(df_internal)
    merged_path = os.path.join(args.outdir, "merged_internal_performance_rows.csv")
    merged.to_csv(merged_path, index=False)

    corr = build_correlation_report(merged)
    corr_path = os.path.join(args.outdir, "internal_vs_performance_correlations.csv")
    corr.to_csv(corr_path, index=False)

    print(f"Internal rows used: {len(df_internal)}")
    print(f"Merged rows (with AUC/F1): {len(merged)}")
    print(f"Distribution report: {dist_path} ({len(dist)} rows)")
    print(f"Correlation report:  {corr_path} ({len(corr)} rows)")


if __name__ == "__main__":
    main()
