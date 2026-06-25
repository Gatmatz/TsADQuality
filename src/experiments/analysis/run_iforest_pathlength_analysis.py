"""
IForest Internal Analysis (True Path Lengths)

Computes true Isolation Forest path lengths (not only score proxy) for
anomaly vs normal windows, before and after corruption.

Outputs:
  - raw_iforest_pathlengths.csv
  - aggregate_iforest_pathlengths.csv
  - case_studies/*_iforest_pathlengths.csv

Usage:
    python run_iforest_pathlength_analysis.py --test
    python run_iforest_pathlength_analysis.py --workers 4
"""
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Prevent thread oversubscription when using ProcessPoolExecutor
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import traceback

# ==========================================
# PATH CONFIGURATION
# ==========================================
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
tsb_uad_path = os.path.join(project_root, 'TSB-UAD')
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if tsb_uad_path not in sys.path:
    sys.path.insert(0, tsb_uad_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from TSB_UAD.models.iforest import IForest
    from TSB_UAD.models.feature import Window
    from TSB_UAD.utils.slidingWindows import find_length
except ImportError as e:
    warnings.warn(f"TSB_UAD components could not be imported: {e}")

from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors
from data_loader import load_tsb_dataframe

# ==========================================
# CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
OUTPUT_DIR = os.path.join(project_root, "results", "experiments", "internal_analysis_iforest_pathlengths")
SEED = 0

# Build corruption conditions — matching main experiments exactly
CORRUPTIONS = []

# Noise: 9 SNR levels (run_whitenoise_snr.py)
for snr in [40, 30, 20, 10, 5, 0, -5, -10, -20]:
    CORRUPTIONS.append({'name': f'noise_snr{snr}', 'type': 'noise', 'snr_db': snr})

# Spikes: 4 fractions × 3 multipliers (run_spikes_normal_only.py)
for frac in [0.01, 0.05, 0.10, 0.20]:
    for mult in [3.0, 5.0, 10.0]:
        CORRUPTIONS.append({'name': f'spikes_f{frac}_m{mult}', 'type': 'spikes',
                            'fraction': frac, 'multiplier': mult})

# MCAR missing: 4 fractions (run_missing_mnar.py)
for frac in [0.01, 0.05, 0.10, 0.20]:
    CORRUPTIONS.append({'name': f'mcar_{frac}', 'type': 'mcar', 'fraction': frac})

# MNAR extreme missing: 4 fractions (run_missing_mnar.py)
for frac in [0.01, 0.05, 0.10, 0.20]:
    CORRUPTIONS.append({'name': f'mnar_extreme_{frac}', 'type': 'mnar_extreme', 'fraction': frac})

# Swap segment: 5 fractions × 5 num_swaps (run_swap_segment.py)
for frac in [0.01, 0.05, 0.10, 0.20, 0.30]:
    for ns in [1, 3, 5, 10, 20]:
        CORRUPTIONS.append({'name': f'swap_f{frac}_ns{ns}', 'type': 'swap',
                            'fraction': frac, 'num_swaps': ns})

# Freeze (sensor stuck): 4 fractions × 5 num_stucks (run_freeze.py)
for frac in [0.01, 0.05, 0.10, 0.20]:
    for ns in [1, 3, 5, 10, 20]:
        CORRUPTIONS.append({'name': f'freeze_f{frac}_ns{ns}', 'type': 'freeze',
                            'fraction': frac, 'num_stucks': ns})


# ==========================================
# CORRUPTION FUNCTIONS
# ==========================================
def select_missing_indices(values, labels, fraction, mechanism, rng):
    """Select missing indices — copied from run_missing_mnar.py for consistency."""
    n = len(values)
    n_missing = max(1, int(fraction * n))

    if mechanism == 'mcar':
        indices = rng.choice(n, size=n_missing, replace=False)
    elif mechanism == 'mnar_extreme':
        z = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-10)
        weights = np.abs(z)
        weights = weights / weights.sum()
        indices = rng.choice(n, size=n_missing, replace=False, p=weights)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    return indices


def apply_corruption(df, corruption, seed):
    """Apply corruption and return corrupted values + metadata.

    Returns:
        For non-missing: (corrupted_values, labels, None)
        For missing: (kept_values, kept_labels, nan_mask)
    """
    ctype = corruption['type']
    values = df['value'].to_numpy('float')
    labels = df['is_anomaly'].to_numpy('int')
    n = len(values)

    if ctype == 'noise':
        corruptor = TSCorruptor(df.copy(), value_col='value', label_col='is_anomaly', seed=seed)
        ts_corruptor.injectors.inject_white_noise_snr(corruptor, snr_db=corruption['snr_db'])
        corrupted = corruptor.get_corrupted_df()['value'].to_numpy('float')
        return corrupted, labels, None

    if ctype == 'spikes':
        corruptor = TSCorruptor(df.copy(), value_col='value', label_col='is_anomaly',
                                seed=seed, corruption_target='only_normal')
        ts_corruptor.injectors.inject_spikes(
            corruptor, fraction=corruption['fraction'],
            multiplier=corruption['multiplier'],
            sequential=False, sequence_length=1)
        corrupted = corruptor.get_corrupted_df()['value'].to_numpy('float')
        return corrupted, labels, None

    if ctype in ('mcar', 'mnar_extreme'):
        rng = np.random.default_rng(seed)
        missing_idx = select_missing_indices(values, labels, corruption['fraction'], ctype, rng)
        nan_mask = np.zeros(n, dtype=bool)
        nan_mask[missing_idx] = True
        kept_values = values[~nan_mask]
        kept_labels = labels[~nan_mask]
        return kept_values, kept_labels, nan_mask

    if ctype == 'swap':
        frac = corruption['fraction']
        ns = corruption['num_swaps']
        swap_length = max(1, int(frac * n / (2 * ns)))
        corruptor = TSCorruptor(df.copy(), value_col='value', label_col='is_anomaly', seed=seed)
        ts_corruptor.injectors.inject_swap(
            corruptor, fraction=frac, swap_length=swap_length, max_distance=None)
        corrupted = corruptor.get_corrupted_df()['value'].to_numpy('float')
        return corrupted, labels, None

    if ctype == 'freeze':
        frac = corruption['fraction']
        ns = corruption['num_stucks']
        stuck_length = max(1, int(frac * n / ns))
        corruptor = TSCorruptor(df.copy(), value_col='value', label_col='is_anomaly', seed=seed)
        ts_corruptor.injectors.inject_sensor_stuck(
            corruptor, num_stucks=ns, stuck_length=stuck_length)
        corrupted = corruptor.get_corrupted_df()['value'].to_numpy('float')
        return corrupted, labels, None

    raise ValueError(f"Unknown corruption type: {ctype}")


# ==========================================
# IFOREST PATH-LENGTH EXTRACTION
# ==========================================
def map_window_labels(labels, window):
    """Map point-level labels to window-level. Window is anomaly if ANY point in it is anomaly."""
    n_windows = len(labels) - window + 1
    cum = np.cumsum(np.concatenate(([0], labels)))
    window_sums = cum[window:] - cum[:n_windows]
    return window_sums > 0


def _average_path_length(n_samples_leaf):
    """Average path-length correction c(n) for leaves with n samples."""
    n = np.asarray(n_samples_leaf, dtype=float)
    out = np.zeros_like(n, dtype=float)

    mask_two = n == 2
    mask_gt_two = n > 2

    out[mask_two] = 1.0
    out[mask_gt_two] = (
        2.0 * (np.log(n[mask_gt_two] - 1.0) + np.euler_gamma)
        - 2.0 * (n[mask_gt_two] - 1.0) / n[mask_gt_two]
    )
    return out


def _compute_node_depths(tree):
    """Compute depth of each node once for a fitted sklearn tree."""
    tree_ = tree.tree_
    node_depth = np.zeros(tree_.node_count, dtype=np.int32)
    stack = [(0, 0)]  # (node_id, depth)
    while stack:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        if left != right:  # split node
            stack.append((left, depth + 1))
            stack.append((right, depth + 1))
    return node_depth


def _quantile_stats(values):
    """Return p10/p50/p90/iqr for a 1D array."""
    if values is None or len(values) == 0:
        return {'p10': np.nan, 'p50': np.nan, 'p90': np.nan, 'iqr': np.nan}
    q10, q50, q90 = np.percentile(values, [10, 50, 90])
    q25, q75 = np.percentile(values, [25, 75])
    return {
        'p10': float(q10),
        'p50': float(q50),
        'p90': float(q90),
        'iqr': float(q75 - q25),
    }


def _iforest_internals(detector, X):
    """Return sample-level path internals and tree-structure summaries."""
    n_samples = X.shape[0]
    n_estimators = len(detector.estimators_)
    if n_estimators == 0:
        zeros = np.zeros(n_samples, dtype=float)
        return {
            'path_lengths': zeros,
            'raw_depths': zeros,
            'leaf_sizes': zeros,
            'tree_max_depth_mean': np.nan,
            'tree_max_depth_std': np.nan,
            'tree_n_leaves_mean': np.nan,
            'tree_n_leaves_std': np.nan,
            'split_feature_entropy': np.nan,
            'top_split_feature_idx': np.nan,
            'top_split_feature_frac': np.nan,
        }

    path_sum = np.zeros(n_samples, dtype=float)
    raw_depth_sum = np.zeros(n_samples, dtype=float)
    leaf_size_sum = np.zeros(n_samples, dtype=float)

    tree_max_depths = []
    tree_n_leaves = []
    split_counts = {}
    total_splits = 0

    for tree, feat_idx in zip(detector.estimators_, detector.estimators_features_):
        X_tree = X[:, feat_idx] if feat_idx is not None else X

        leaves = tree.apply(X_tree)
        node_depth = _compute_node_depths(tree)
        depths = node_depth[leaves]
        leaf_sizes = tree.tree_.n_node_samples[leaves]

        path_sum += depths + _average_path_length(leaf_sizes)
        raw_depth_sum += depths
        leaf_size_sum += leaf_sizes

        tree_max_depths.append(float(np.max(node_depth)))
        t = tree.tree_
        is_leaf = t.children_left == -1
        tree_n_leaves.append(float(np.sum(is_leaf)))

        split_nodes = t.children_left != t.children_right
        split_features_local = t.feature[split_nodes]
        split_features_local = split_features_local[split_features_local >= 0]
        if len(split_features_local) > 0:
            if feat_idx is not None:
                split_features_global = np.asarray(feat_idx)[split_features_local]
            else:
                split_features_global = split_features_local
            u, c = np.unique(split_features_global.astype(int), return_counts=True)
            for uu, cc in zip(u, c):
                split_counts[int(uu)] = split_counts.get(int(uu), 0) + int(cc)
            total_splits += int(np.sum(c))

    split_feature_entropy = np.nan
    top_split_feature_idx = np.nan
    top_split_feature_frac = np.nan
    if total_splits > 0:
        counts = np.array(list(split_counts.values()), dtype=float)
        probs = counts / counts.sum()
        split_feature_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        top_idx, top_count = max(split_counts.items(), key=lambda x: x[1])
        top_split_feature_idx = int(top_idx)
        top_split_feature_frac = float(top_count / total_splits)

    return {
        'path_lengths': path_sum / n_estimators,
        'raw_depths': raw_depth_sum / n_estimators,
        'leaf_sizes': leaf_size_sum / n_estimators,
        'tree_max_depth_mean': float(np.mean(tree_max_depths)),
        'tree_max_depth_std': float(np.std(tree_max_depths)),
        'tree_n_leaves_mean': float(np.mean(tree_n_leaves)),
        'tree_n_leaves_std': float(np.std(tree_n_leaves)),
        'split_feature_entropy': split_feature_entropy,
        'top_split_feature_idx': top_split_feature_idx,
        'top_split_feature_frac': top_split_feature_frac,
    }


def extract_iforest_paths(X, window_labels):
    """Extract path, score, and tree-structure internals for anomaly vs normal windows."""
    clf = IForest(n_estimators=100, random_state=42)
    clf.fit(X)

    # score proxy (higher = more anomalous)
    raw_scores = -clf.detector_.score_samples(X)
    # true path/structure internals from fitted trees
    internals = _iforest_internals(clf.detector_, X)
    mean_path_lengths = internals['path_lengths']
    mean_depths = internals['raw_depths']
    mean_leaf_sizes = internals['leaf_sizes']

    anom_mask = window_labels[:len(mean_path_lengths)]
    norm_mask = ~anom_mask

    path_anom = mean_path_lengths[anom_mask]
    path_norm = mean_path_lengths[norm_mask]
    score_anom = raw_scores[anom_mask]
    score_norm = raw_scores[norm_mask]
    depth_anom = mean_depths[anom_mask]
    depth_norm = mean_depths[norm_mask]
    leaf_anom = mean_leaf_sizes[anom_mask]
    leaf_norm = mean_leaf_sizes[norm_mask]

    path_anom_q = _quantile_stats(path_anom)
    path_norm_q = _quantile_stats(path_norm)
    score_anom_q = _quantile_stats(score_anom)
    score_norm_q = _quantile_stats(score_norm)

    return {
        'path_length_anomaly': float(np.mean(path_anom)) if anom_mask.any() else np.nan,
        'path_length_normal': float(np.mean(path_norm)) if norm_mask.any() else np.nan,
        'path_length_anomaly_p10': path_anom_q['p10'],
        'path_length_anomaly_p50': path_anom_q['p50'],
        'path_length_anomaly_p90': path_anom_q['p90'],
        'path_length_anomaly_iqr': path_anom_q['iqr'],
        'path_length_normal_p10': path_norm_q['p10'],
        'path_length_normal_p50': path_norm_q['p50'],
        'path_length_normal_p90': path_norm_q['p90'],
        'path_length_normal_iqr': path_norm_q['iqr'],
        'score_proxy_anomaly': float(np.mean(score_anom)) if anom_mask.any() else np.nan,
        'score_proxy_normal': float(np.mean(score_norm)) if norm_mask.any() else np.nan,
        'score_proxy_anomaly_p10': score_anom_q['p10'],
        'score_proxy_anomaly_p50': score_anom_q['p50'],
        'score_proxy_anomaly_p90': score_anom_q['p90'],
        'score_proxy_normal_p10': score_norm_q['p10'],
        'score_proxy_normal_p50': score_norm_q['p50'],
        'score_proxy_normal_p90': score_norm_q['p90'],
        'mean_raw_depth_anomaly': float(np.mean(depth_anom)) if anom_mask.any() else np.nan,
        'mean_raw_depth_normal': float(np.mean(depth_norm)) if norm_mask.any() else np.nan,
        'mean_leaf_size_anomaly': float(np.mean(leaf_anom)) if anom_mask.any() else np.nan,
        'mean_leaf_size_normal': float(np.mean(leaf_norm)) if norm_mask.any() else np.nan,
        'tree_max_depth_mean': internals['tree_max_depth_mean'],
        'tree_max_depth_std': internals['tree_max_depth_std'],
        'tree_n_leaves_mean': internals['tree_n_leaves_mean'],
        'tree_n_leaves_std': internals['tree_n_leaves_std'],
        'split_feature_entropy': internals['split_feature_entropy'],
        'top_split_feature_idx': internals['top_split_feature_idx'],
        'top_split_feature_frac': internals['top_split_feature_frac'],
    }


def compute_metrics(clean, corrupted):
    # For path length, good separation means normal > anomaly (anomalies isolate faster)
    gap_clean = (clean['path_length_normal'] or 0) - (clean['path_length_anomaly'] or 0)
    gap_corr = (corrupted['path_length_normal'] or 0) - (corrupted['path_length_anomaly'] or 0)
    change = ((gap_corr - gap_clean) / abs(gap_clean) * 100) if abs(gap_clean) > 1e-10 else np.nan

    score_gap_clean = (clean['score_proxy_anomaly'] or 0) - (clean['score_proxy_normal'] or 0)
    score_gap_corr = (corrupted['score_proxy_anomaly'] or 0) - (corrupted['score_proxy_normal'] or 0)

    path_p50_gap_clean = (clean['path_length_normal_p50'] or 0) - (clean['path_length_anomaly_p50'] or 0)
    path_p50_gap_corr = (corrupted['path_length_normal_p50'] or 0) - (corrupted['path_length_anomaly_p50'] or 0)
    score_p50_gap_clean = (clean['score_proxy_anomaly_p50'] or 0) - (clean['score_proxy_normal_p50'] or 0)
    score_p50_gap_corr = (corrupted['score_proxy_anomaly_p50'] or 0) - (corrupted['score_proxy_normal_p50'] or 0)

    return {
        'path_length_anomaly_clean': clean['path_length_anomaly'],
        'path_length_anomaly_corrupted': corrupted['path_length_anomaly'],
        'path_length_normal_clean': clean['path_length_normal'],
        'path_length_normal_corrupted': corrupted['path_length_normal'],
        'path_length_anomaly_p10_clean': clean['path_length_anomaly_p10'],
        'path_length_anomaly_p10_corrupted': corrupted['path_length_anomaly_p10'],
        'path_length_anomaly_p50_clean': clean['path_length_anomaly_p50'],
        'path_length_anomaly_p50_corrupted': corrupted['path_length_anomaly_p50'],
        'path_length_anomaly_p90_clean': clean['path_length_anomaly_p90'],
        'path_length_anomaly_p90_corrupted': corrupted['path_length_anomaly_p90'],
        'path_length_normal_p10_clean': clean['path_length_normal_p10'],
        'path_length_normal_p10_corrupted': corrupted['path_length_normal_p10'],
        'path_length_normal_p50_clean': clean['path_length_normal_p50'],
        'path_length_normal_p50_corrupted': corrupted['path_length_normal_p50'],
        'path_length_normal_p90_clean': clean['path_length_normal_p90'],
        'path_length_normal_p90_corrupted': corrupted['path_length_normal_p90'],
        'path_length_gap_clean': round(gap_clean, 6),
        'path_length_gap_corrupted': round(gap_corr, 6),
        'path_length_gap_change_pct': round(change, 2) if not np.isnan(change) else np.nan,
        'path_length_p50_gap_clean': round(path_p50_gap_clean, 6),
        'path_length_p50_gap_corrupted': round(path_p50_gap_corr, 6),
        'score_proxy_anomaly_clean': clean['score_proxy_anomaly'],
        'score_proxy_anomaly_corrupted': corrupted['score_proxy_anomaly'],
        'score_proxy_normal_clean': clean['score_proxy_normal'],
        'score_proxy_normal_corrupted': corrupted['score_proxy_normal'],
        'score_proxy_gap_clean': round(score_gap_clean, 6),
        'score_proxy_gap_corrupted': round(score_gap_corr, 6),
        'score_proxy_anomaly_p50_clean': clean['score_proxy_anomaly_p50'],
        'score_proxy_anomaly_p50_corrupted': corrupted['score_proxy_anomaly_p50'],
        'score_proxy_normal_p50_clean': clean['score_proxy_normal_p50'],
        'score_proxy_normal_p50_corrupted': corrupted['score_proxy_normal_p50'],
        'score_proxy_p50_gap_clean': round(score_p50_gap_clean, 6),
        'score_proxy_p50_gap_corrupted': round(score_p50_gap_corr, 6),
        'mean_raw_depth_anomaly_clean': clean['mean_raw_depth_anomaly'],
        'mean_raw_depth_anomaly_corrupted': corrupted['mean_raw_depth_anomaly'],
        'mean_raw_depth_normal_clean': clean['mean_raw_depth_normal'],
        'mean_raw_depth_normal_corrupted': corrupted['mean_raw_depth_normal'],
        'mean_leaf_size_anomaly_clean': clean['mean_leaf_size_anomaly'],
        'mean_leaf_size_anomaly_corrupted': corrupted['mean_leaf_size_anomaly'],
        'mean_leaf_size_normal_clean': clean['mean_leaf_size_normal'],
        'mean_leaf_size_normal_corrupted': corrupted['mean_leaf_size_normal'],
        'tree_max_depth_mean_clean': clean['tree_max_depth_mean'],
        'tree_max_depth_mean_corrupted': corrupted['tree_max_depth_mean'],
        'tree_n_leaves_mean_clean': clean['tree_n_leaves_mean'],
        'tree_n_leaves_mean_corrupted': corrupted['tree_n_leaves_mean'],
        'split_feature_entropy_clean': clean['split_feature_entropy'],
        'split_feature_entropy_corrupted': corrupted['split_feature_entropy'],
        'top_split_feature_idx_clean': clean['top_split_feature_idx'],
        'top_split_feature_idx_corrupted': corrupted['top_split_feature_idx'],
        'top_split_feature_frac_clean': clean['top_split_feature_frac'],
        'top_split_feature_frac_corrupted': corrupted['top_split_feature_frac'],
    }


# ==========================================
# CORE WORKER
# ==========================================
def process_file(job_args):
    """Process one file: clean + all corruptions."""
    file_path, corruptions_list = job_args
    file_name = os.path.basename(file_path)

    try:
        df, file_name = load_tsb_dataframe(file_path)
        values = df['value'].to_numpy('float')
        labels = df['is_anomaly'].to_numpy('int')

        clean_scaled = StandardScaler().fit_transform(values.reshape(-1, 1)).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)
        window_labels = map_window_labels(labels, sliding_window)

        X_clean = Window(window=sliding_window).convert(clean_scaled).to_numpy()
        clean_internals = extract_iforest_paths(X_clean, window_labels)

        rows = []
        for corruption in corruptions_list:
            try:
                corr_values, corr_labels, _ = apply_corruption(df, corruption, SEED)
                is_missing = corruption['type'] in ('mcar', 'mnar_extreme')

                if is_missing:
                    n_kept = len(corr_values)
                    if n_kept < 30:
                        rows.append({'file': file_name, 'corruption': corruption['name'],
                                     'model': 'IForestPaths', 'error': 'Too few points after masking'})
                        continue
                    sw = max(min(sliding_window, n_kept // 4), 10)
                    corr_scaled = StandardScaler().fit_transform(corr_values.reshape(-1, 1)).flatten()
                    corr_window_labels = map_window_labels(corr_labels, sw)
                else:
                    if np.isnan(corr_values).any():
                        corr_values = np.nan_to_num(corr_values, nan=np.nanmean(corr_values))
                    sw = sliding_window
                    corr_scaled = StandardScaler().fit_transform(corr_values.reshape(-1, 1)).flatten()
                    corr_window_labels = window_labels

                X_corr = Window(window=sw).convert(corr_scaled).to_numpy()
                corr_internals = extract_iforest_paths(X_corr, corr_window_labels)

                row = {'file': file_name, 'corruption': corruption['name'],
                       'model': 'IForestPaths', 'error': None}
                row.update(compute_metrics(clean_internals, corr_internals))
                rows.append(row)

            except Exception as e:
                rows.append({'file': file_name, 'corruption': corruption['name'],
                             'model': 'IForestPaths', 'error': str(e)[:200]})

        return {'status': 'success', 'rows': rows, 'job_id': file_name}

    except Exception:
        return {'status': 'error', 'file': file_name, 'error': traceback.format_exc()[:300]}


# ==========================================
# SUMMARY & CASE STUDIES
# ==========================================
def compute_aggregate_summary(df_all, output_path):
    """Compute mean ± std per corruption for each metric."""
    df_ok = df_all[df_all['error'].isna()] if 'error' in df_all.columns else df_all
    if df_ok.empty:
        print("No successful results to summarize.")
        return None

    metric_cols = [c for c in df_ok.columns if c not in ('file', 'corruption', 'model', 'error')]

    summary_rows = []
    for corruption, group in df_ok.groupby('corruption'):
        row = {'corruption': corruption, 'model': 'IForestPaths', 'n_files': len(group)}
        for col in metric_cols:
            vals = group[col].dropna()
            if len(vals) > 0:
                row[f'{col}_mean'] = round(vals.mean(), 6)
                row[f'{col}_std'] = round(vals.std(), 6)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_path, index=False)
    print(f"Aggregate summary saved: {output_path} ({len(df_summary)} rows)")
    return df_summary


def save_case_studies(df_all, output_dir):
    """Save detailed results for 5 representative files."""
    df_ok = df_all[df_all['error'].isna()] if 'error' in df_all.columns else df_all
    if df_ok.empty:
        return

    files = df_ok['file'].unique()
    indices = np.linspace(0, len(files) - 1, min(5, len(files)), dtype=int)
    case_files = [files[i] for i in indices]

    case_dir = os.path.join(output_dir, "case_studies")
    os.makedirs(case_dir, exist_ok=True)

    for cf in case_files:
        sub = df_ok[df_ok['file'] == cf]
        safe_name = cf.replace('.', '_').replace('/', '_').replace('\\', '_')
        sub.to_csv(os.path.join(case_dir, f"{safe_name}_iforest_pathlengths.csv"), index=False)

    print(f"Case studies saved for {len(case_files)} files in {case_dir}")


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='IForest path-length internal analysis')
    parser.add_argument('--test', action='store_true', help='3 files, reduced corruptions')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(SUBSET_CSV):
        print(f"[ERROR] Subset CSV not found: {SUBSET_CSV}")
        return

    df_meta = pd.read_csv(SUBSET_CSV)
    file_paths = df_meta['filepath'].tolist()
    corruptions_list = CORRUPTIONS

    if args.test:
        file_paths = file_paths[:3]
        # 2 per type = 12 conditions
        corruptions_list = [c for c in CORRUPTIONS if c['name'] in (
            'noise_snr20', 'noise_snr5',
            'spikes_f0.1_m5.0', 'spikes_f0.1_m10.0',
            'mcar_0.05', 'mcar_0.2',
            'mnar_extreme_0.05', 'mnar_extreme_0.2',
            'swap_f0.1_ns5', 'swap_f0.2_ns10',
            'freeze_f0.1_ns5', 'freeze_f0.2_ns10',
        )]
        print("!!! TEST MODE !!!")

    n_workers = args.workers or os.cpu_count() or 1
    jobs = [(fp, corruptions_list) for fp in file_paths]
    total_individual = len(jobs) * len(corruptions_list)

    print(f"\n{'=' * 60}")
    print("  IForest Path-Length Internal Analysis")
    print(f"{'=' * 60}")
    print(f"  Files:        {len(file_paths)}")
    print(f"  Corruptions:  {len(corruptions_list)}")
    print(f"  Total rows:   {total_individual}")
    print(f"  Workers:      {n_workers}")
    print(f"{'=' * 60}\n")

    all_results = []
    if len(jobs) > 0:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_file, job): job for job in jobs}
            with tqdm(total=total_individual, desc="IForest Paths", unit="corruption") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        n_rows = len(res['rows'])
                        all_results.extend(res['rows'])
                        pbar.update(n_rows)
                    else:
                        all_results.append({
                            'file': res.get('file', 'unknown'),
                            'corruption': 'ALL',
                            'model': 'IForestPaths',
                            'error': res.get('error', 'unknown'),
                        })
                        pbar.update(len(corruptions_list))

    if all_results:
        df_all = pd.DataFrame(all_results)
        raw_path = os.path.join(OUTPUT_DIR, "raw_iforest_pathlengths.csv")
        df_all.to_csv(raw_path, index=False)
        print(f"Raw internals: {raw_path} ({len(df_all)} rows)")

        summary_path = os.path.join(OUTPUT_DIR, "aggregate_iforest_pathlengths.csv")
        compute_aggregate_summary(df_all, summary_path)

        save_case_studies(df_all, OUTPUT_DIR)

        df_ok = df_all[df_all['error'].isna()] if 'error' in df_all.columns else df_all
        if not df_ok.empty and 'path_length_gap_change_pct' in df_ok.columns:
            vals = df_ok.groupby('corruption')['path_length_gap_change_pct'].mean()
            worst = vals.idxmin() if len(vals) > 0 else 'N/A'
            print(f"Worst path-length gap loss: {worst} ({vals.min():.1f}%)")

    print("\n[Done] IForest path-length analysis complete.")


if __name__ == "__main__":
    main()
