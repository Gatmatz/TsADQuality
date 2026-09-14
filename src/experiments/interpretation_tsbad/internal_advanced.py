"""
Advanced internal analysis on TSB-AD — distributions + internal-vs-performance correlations.

The TSB-AD counterpart of src/experiments/analysis/run_internal_advanced_analysis.py (thesis).
Same outputs:
  1) distribution diagnostics of each model's primary internal separation metric
     (p10/p50/p90/IQR, degraded rate, non-separation rates)
  2) Spearman correlation between internal degradation and performance drop,
     per (model, experiment, condition) and pooled per model

Three deliberate differences:
  * performance comes from the TSB-AD corruption checkpoints, joined on
    experiment|file|condition|seed|model — no condition-name reformatting (in the thesis that
    join silently lost freeze and PCA); the clean reference is the clean anchor of the same
    checkpoints, not a separate baseline file
  * internal rows without a matching performance row stop the analysis with a report
    (unless --allow-unmatched) instead of being dropped
  * drops are computed for AUC_ROC (as in the thesis), Standard_F1 (TSB-AD has no plain 'F')
    and VUS_PR; every per-condition p-value also gets a Benjamini-Hochberg q-value

Usage:
    python src/experiments/interpretation_tsbad/internal_advanced.py
    python src/experiments/interpretation_tsbad/internal_advanced.py --internal <raw_internals.csv> --results-root <dir>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # src/
import tsbad.env  # noqa: E402,F401

import argparse  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from tsbad import paths  # noqa: E402
from tsbad.checkpoint import load_all_for_summary  # noqa: E402

INTERNAL_RAW = paths.results_dir('internal_analysis_tsbad') / 'raw_internals.csv'
OUTPUT_DIR = paths.results_dir('internal_analysis_tsbad') / 'advanced_analysis'

KEYS = ['experiment', 'file', 'condition', 'seed', 'model']
PERF_METRICS = {'AUC_ROC': 'auc', 'Standard_F1': 'f1', 'VUS_PR': 'vus_pr'}

# The thesis configuration, keyed by the TSB-AD model each thesis model maps to
MODEL_METRIC_CONFIG = {
    'IForest': {
        'clean': 'separation_gap_clean',
        'corrupted': 'separation_gap_corrupted',
        'change_pct': 'separation_gap_change_pct',
        'non_separation_threshold': 0.0,
    },
    'Sub_LOF': {
        'clean': 'kdist_gap_clean',
        'corrupted': 'kdist_gap_corrupted',
        'change_pct': 'kdist_gap_change_pct',
        'non_separation_threshold': 0.0,
    },
    'MatrixProfile': {
        'clean': 'nn_dist_gap_clean',
        'corrupted': 'nn_dist_gap_corrupted',
        'change_pct': 'nn_dist_gap_change_pct',
        'non_separation_threshold': 0.0,
    },
    'AutoEncoder_2': {
        'clean': 'error_ratio_clean',
        'corrupted': 'error_ratio_corrupted',
        'change_pct': 'error_ratio_change_pct',
        'non_separation_threshold': 1.0,
    },
    'Sub_PCA': {
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


def _bh(pvals):
    """Benjamini-Hochberg q-values; NaN p-values stay NaN and do not count as tests."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    out = np.empty(m)
    out[order] = ranked
    q[ok] = out
    return q


# ==========================================
# INTERNALS
# ==========================================
def prepare_internal_df(path):
    raw = _clean_ok(pd.read_csv(path, low_memory=False)).copy()

    for c in raw.columns:
        if c in KEYS + ['error']:
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
        if model == 'Sub_PCA':
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
    for (model, experiment, condition), g in df_internal.groupby(['model', 'experiment', 'condition'], dropna=False):
        clean_stats = _q_stats(g['internal_metric_clean'])
        corr_stats = _q_stats(g['internal_metric_corrupted'])
        chg_stats = _q_stats(g['internal_metric_change_pct'])
        rows.append({
            'model': model,
            'experiment': experiment,
            'condition': condition,
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


# ==========================================
# PERFORMANCE JOIN
# ==========================================
def load_performance(results_root, experiments):
    frames = []
    for experiment in experiments:
        d = load_all_for_summary(os.path.join(results_root, experiment))
        if d.empty:
            continue
        d = _clean_ok(d).copy()
        d['experiment'] = experiment
        frames.append(d[KEYS + list(PERF_METRICS)])
    if not frames:
        return pd.DataFrame(columns=KEYS + list(PERF_METRICS))
    return pd.concat(frames, ignore_index=True)


def build_internal_performance_merge(df_internal, results_root, allow_unmatched=False):
    perf = load_performance(results_root, sorted(df_internal['experiment'].unique()))
    corrupted = perf[perf['condition'] != 'clean'].rename(
        columns={m: f'{short}_corrupted' for m, short in PERF_METRICS.items()})
    clean = (perf[perf['condition'] == 'clean']
             .drop(columns=['condition', 'seed'])
             .drop_duplicates(['experiment', 'file', 'model'])
             .rename(columns={m: f'{short}_clean' for m, short in PERF_METRICS.items()}))

    merged = df_internal.merge(corrupted, on=KEYS, how='left', indicator='_perf')
    merged = merged.merge(clean, on=['experiment', 'file', 'model'], how='left', indicator='_clean')
    unmatched = merged[(merged['_perf'] != 'both') | (merged['_clean'] != 'both')]
    if len(unmatched):
        report = unmatched.groupby(['experiment', 'model']).size().rename('unmatched_rows')
        print(f"[WARN] {len(unmatched)} internal rows have no performance (or no clean anchor) to relate to:")
        print(report.to_string())
        print("  e.g.", unmatched[KEYS].head(3).to_dict('records'))
        if not allow_unmatched:
            raise SystemExit("Run the corruption experiments for these models/files first, "
                             "or pass --allow-unmatched to analyse only the matched rows.")
    merged = merged[(merged['_perf'] == 'both') & (merged['_clean'] == 'both')].drop(columns=['_perf', '_clean'])
    for short in PERF_METRICS.values():
        merged[f'{short}_drop'] = merged[f'{short}_clean'] - merged[f'{short}_corrupted']
    return merged


def build_correlation_report(df_merged):
    rows = []
    groupings = [('per_condition', ['model', 'experiment', 'condition']),
                 ('per_model_all_conditions', ['model'])]
    for scope, keys in groupings:
        for name, g in df_merged.groupby(keys):
            name = name if isinstance(name, tuple) else (name,)
            row = {'scope': scope, 'model': name[0],
                   'experiment': name[1] if len(name) > 1 else 'ALL',
                   'condition': name[2] if len(name) > 2 else 'ALL'}
            for short in PERF_METRICS.values():
                r, p, n = _safe_spearman(g['internal_metric_change_pct'], g[f'{short}_drop'])
                row[f'n_{short}'] = n
                row[f'spearman_internal_vs_{short}_drop'] = round(r, 4) if not np.isnan(r) else np.nan
                row[f'p_{short}'] = round(p, 6) if not np.isnan(p) else np.nan
            rows.append(row)
    corr = pd.DataFrame(rows)
    per_cond = corr['scope'] == 'per_condition'
    for short in PERF_METRICS.values():
        corr[f'q_{short}'] = np.nan
        corr.loc[per_cond, f'q_{short}'] = np.round(_bh(corr.loc[per_cond, f'p_{short}']), 6)
    return corr


def main():
    ap = argparse.ArgumentParser(description='Advanced internal analysis — TSB-AD (distribution + correlation)')
    ap.add_argument('--internal', default=str(INTERNAL_RAW), help='raw_internals.csv from internals.py')
    ap.add_argument('--results-root', default=str(paths.RESULTS_ROOT),
                    help='Directory holding the <experiment>_tsbad checkpoint folders')
    ap.add_argument('--outdir', default=str(OUTPUT_DIR))
    ap.add_argument('--allow-unmatched', action='store_true',
                    help='Analyse the matched rows even if some internal rows have no performance row')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df_internal = prepare_internal_df(args.internal)

    dist = build_distribution_report(df_internal)
    dist_path = os.path.join(args.outdir, "internal_gap_distributions.csv")
    dist.to_csv(dist_path, index=False)

    merged = build_internal_performance_merge(df_internal, args.results_root, args.allow_unmatched)
    merged_path = os.path.join(args.outdir, "merged_internal_performance_rows.csv")
    merged.to_csv(merged_path, index=False)

    corr = build_correlation_report(merged)
    corr_path = os.path.join(args.outdir, "internal_vs_performance_correlations.csv")
    corr.to_csv(corr_path, index=False)

    print(f"Internal rows used: {len(df_internal)}")
    print(f"Merged rows (with performance): {len(merged)}")
    print(f"Distribution report: {dist_path} ({len(dist)} rows)")
    print(f"Correlation report:  {corr_path} ({len(corr)} rows)")


if __name__ == "__main__":
    main()
