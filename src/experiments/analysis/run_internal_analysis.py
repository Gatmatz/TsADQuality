"""
Internal Algorithmic Analysis — WHY each model is robust or vulnerable

Extracts internal model metrics (path lengths, k-distances, NN indices,
reconstruction errors, eigenvectors) before and after corruption, using
the SAME parameters as the main experiments.

Answers: "WHY does IForest drop 15% under noise while MP only drops 3%?"
by showing what changes INSIDE each algorithm.

Corruptions (74 conditions, matching main experiments + freeze):
  - Noise:  9 SNR levels (40 to -20 dB)     — from run_whitenoise_snr.py
  - Spikes: 4 fractions × 3 multipliers = 12 — from run_spikes_normal_only.py
  - MCAR:   4 fractions                       — from run_missing_mnar.py
  - MNAR:   4 fractions (extreme)             — from run_missing_mnar.py
  - Swap:   5 fractions × 5 num_swaps = 25   — from run_swap_segment.py
  - Freeze: 4 fractions × 5 num_stucks = 20  — from run_freeze.py

Models: IForest, LOF, MP, AE, PCA

Usage:
    python run_internal_analysis.py --test --models IForest
    python run_internal_analysis.py --models IForest LOF MP AE PCA --workers 4
"""
import os
import sys
import math
import argparse

# Prevent thread oversubscription when using ProcessPoolExecutor
# Each worker gets 1 thread — parallelism comes from multiprocessing instead
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import stumpy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
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

if project_root not in sys.path: sys.path.insert(0, project_root)
if tsb_uad_path not in sys.path: sys.path.insert(0, tsb_uad_path)
if src_path not in sys.path: sys.path.insert(0, src_path)

try:
    from TSB_UAD.models.iforest import IForest
    from TSB_UAD.models.pca import PCA
    from TSB_UAD.models.matrix_profile import MatrixProfile
    from TSB_UAD.models.lof import LOF
    from TSB_UAD.models.feature import Window
    from TSB_UAD.utils.slidingWindows import find_length
except ImportError as e:
    warnings.warn(f"TSB_UAD components could not be imported: {e}")

from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors
from data_loader import load_tsb_dataframe, load_pretrained_ae

# ==========================================
# CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
OUTPUT_DIR = os.path.join(project_root, "results", "experiments", "internal_analysis")
SEED = 0
ALL_MODELS = ['IForest', 'LOF', 'MP', 'AE', 'PCA']
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

    elif ctype == 'spikes':
        corruptor = TSCorruptor(df.copy(), value_col='value', label_col='is_anomaly',
                                seed=seed, corruption_target='only_normal')
        ts_corruptor.injectors.inject_spikes(
            corruptor, fraction=corruption['fraction'],
            multiplier=corruption['multiplier'],
            sequential=False, sequence_length=1)
        corrupted = corruptor.get_corrupted_df()['value'].to_numpy('float')
        return corrupted, labels, None

    elif ctype in ('mcar', 'mnar_extreme'):
        rng = np.random.default_rng(seed)
        missing_idx = select_missing_indices(values, labels, corruption['fraction'], ctype, rng)
        nan_mask = np.zeros(n, dtype=bool)
        nan_mask[missing_idx] = True
        kept_values = values[~nan_mask]
        kept_labels = labels[~nan_mask]
        return kept_values, kept_labels, nan_mask

    elif ctype == 'swap':
        frac = corruption['fraction']
        ns = corruption['num_swaps']
        swap_length = max(1, int(frac * n / (2 * ns)))
        corruptor = TSCorruptor(df.copy(), value_col='value', label_col='is_anomaly', seed=seed)
        ts_corruptor.injectors.inject_swap(
            corruptor, fraction=frac, swap_length=swap_length, max_distance=None)
        corrupted = corruptor.get_corrupted_df()['value'].to_numpy('float')
        return corrupted, labels, None

    elif ctype == 'freeze':
        frac = corruption['fraction']
        ns = corruption['num_stucks']
        stuck_length = max(1, int(frac * n / ns))
        corruptor = TSCorruptor(df.copy(), value_col='value', label_col='is_anomaly', seed=seed)
        ts_corruptor.injectors.inject_sensor_stuck(
            corruptor, num_stucks=ns, stuck_length=stuck_length)
        corrupted = corruptor.get_corrupted_df()['value'].to_numpy('float')
        return corrupted, labels, None

    else:
        raise ValueError(f"Unknown corruption type: {ctype}")


# ==========================================
# WINDOW LABEL MAPPING
# ==========================================

def map_window_labels(labels, window):
    """Map point-level labels to window-level. Window is anomaly if ANY point in it is anomaly."""
    n_windows = len(labels) - window + 1
    # Use cumsum for efficiency
    cum = np.cumsum(np.concatenate(([0], labels)))
    window_sums = cum[window:] - cum[:n_windows]
    return window_sums > 0


# ==========================================
# INTERNAL EXTRACTION FUNCTIONS
# ==========================================

def extract_iforest(X, window_labels):
    """Extract IForest internals: path lengths for anomaly vs normal windows."""
    clf = IForest(n_estimators=100, random_state=42)
    clf.fit(X)
    # -score_samples = average path length proxy (higher = more anomalous)
    raw_scores = -clf.detector_.score_samples(X)

    anom_mask = window_labels[:len(raw_scores)]
    norm_mask = ~anom_mask

    return {
        'path_anomaly': float(np.mean(raw_scores[anom_mask])) if anom_mask.any() else np.nan,
        'path_normal': float(np.mean(raw_scores[norm_mask])) if norm_mask.any() else np.nan,
    }


def extract_lof(X, window_labels, n_neighbors=20):
    """Extract LOF internals: k-distances for anomaly vs normal windows."""
    k = min(n_neighbors, len(X) - 1)
    clf = LOF(n_neighbors=k)
    clf.fit(X)
    # k-distance: distance to k-th nearest neighbor
    kdist = clf.detector_._distances_fit_X_[:, -1]

    anom_mask = window_labels[:len(kdist)]
    norm_mask = ~anom_mask

    return {
        'kdist_anomaly': float(np.mean(kdist[anom_mask])) if anom_mask.any() else np.nan,
        'kdist_normal': float(np.mean(kdist[norm_mask])) if norm_mask.any() else np.nan,
    }


def extract_mp(scaled_data, window, window_labels):
    """Extract MP internals: NN distances and indices via stumpy directly."""
    try:
        profile = stumpy.gpu_stump(scaled_data, m=window)
    except Exception:
        profile = stumpy.stump(scaled_data, m=window)
        # Note: running on CPU (gpu_stump not available)
    nn_distances = profile[:, 0].astype(float)
    nn_indices = profile[:, 1].astype(int)

    anom_mask = window_labels[:len(nn_distances)]
    norm_mask = ~anom_mask

    return {
        'nn_dist_anomaly': float(np.mean(nn_distances[anom_mask])) if anom_mask.any() else np.nan,
        'nn_dist_normal': float(np.mean(nn_distances[norm_mask])) if norm_mask.any() else np.nan,
        'nn_indices': nn_indices,
    }


def extract_ae(scaled_data, clf):
    """Extract AE internals: reconstruction errors for anomaly vs normal points.
    clf must be a pre-loaded AE model (load once, predict many times)."""
    clf.predict(scaled_data)
    return {
        'recon_errors': clf.decision_scores_,  # full-length, MinMax scaled
    }


def extract_pca(X, window_labels, n_components=10):
    """Extract PCA internals: eigenvectors, explained variance, scores."""
    n_comp = min(n_components, X.shape[1] - 1) if X.shape[1] > 1 else 1
    clf = PCA(n_components=n_comp)
    clf.fit(X)

    anom_mask = window_labels[:len(clf.decision_scores_)]
    norm_mask = ~anom_mask

    return {
        'components': clf.detector_.components_.copy(),
        'explained_variance_ratio': clf.detector_.explained_variance_ratio_.copy(),
        'score_anomaly': float(np.mean(clf.decision_scores_[anom_mask])) if anom_mask.any() else np.nan,
        'score_normal': float(np.mean(clf.decision_scores_[norm_mask])) if norm_mask.any() else np.nan,
    }


# ==========================================
# METRIC COMPUTATION
# ==========================================

def compute_iforest_metrics(clean, corrupted):
    # gap = anomaly - normal: positive means anomalies score higher (good separation)
    gap_clean = (clean['path_anomaly'] or 0) - (clean['path_normal'] or 0)
    gap_corr = (corrupted['path_anomaly'] or 0) - (corrupted['path_normal'] or 0)
    change = ((gap_corr - gap_clean) / abs(gap_clean) * 100) if abs(gap_clean) > 1e-10 else np.nan
    return {
        'path_anomaly_clean': clean['path_anomaly'],
        'path_anomaly_corrupted': corrupted['path_anomaly'],
        'path_normal_clean': clean['path_normal'],
        'path_normal_corrupted': corrupted['path_normal'],
        'separation_gap_clean': round(gap_clean, 6),
        'separation_gap_corrupted': round(gap_corr, 6),
        'separation_gap_change_pct': round(change, 2) if not np.isnan(change) else np.nan,
    }


def compute_lof_metrics(clean, corrupted):
    gap_clean = (clean['kdist_anomaly'] or 0) - (clean['kdist_normal'] or 0)
    gap_corr = (corrupted['kdist_anomaly'] or 0) - (corrupted['kdist_normal'] or 0)
    change = ((gap_corr - gap_clean) / abs(gap_clean) * 100) if abs(gap_clean) > 1e-10 else np.nan
    return {
        'kdist_anomaly_clean': clean['kdist_anomaly'],
        'kdist_anomaly_corrupted': corrupted['kdist_anomaly'],
        'kdist_normal_clean': clean['kdist_normal'],
        'kdist_normal_corrupted': corrupted['kdist_normal'],
        'kdist_gap_clean': round(gap_clean, 6),
        'kdist_gap_corrupted': round(gap_corr, 6),
        'kdist_gap_change_pct': round(change, 2) if not np.isnan(change) else np.nan,
    }


def compute_mp_metrics(clean, corrupted, can_compare_nn=True):
    gap_clean = (clean['nn_dist_anomaly'] or 0) - (clean['nn_dist_normal'] or 0)
    gap_corr = (corrupted['nn_dist_anomaly'] or 0) - (corrupted['nn_dist_normal'] or 0)
    change = ((gap_corr - gap_clean) / abs(gap_clean) * 100) if abs(gap_clean) > 1e-10 else np.nan

    pct_changed = np.nan
    pct_changed_anom = np.nan
    if can_compare_nn and 'nn_indices' in clean and 'nn_indices' in corrupted:
        idx_c = clean['nn_indices']
        idx_r = corrupted['nn_indices']
        min_len = min(len(idx_c), len(idx_r))
        if min_len > 0:
            pct_changed = float(np.mean(idx_c[:min_len] != idx_r[:min_len]) * 100)

    return {
        'nn_dist_anomaly_clean': clean['nn_dist_anomaly'],
        'nn_dist_anomaly_corrupted': corrupted['nn_dist_anomaly'],
        'nn_dist_normal_clean': clean['nn_dist_normal'],
        'nn_dist_normal_corrupted': corrupted['nn_dist_normal'],
        'nn_dist_gap_clean': round(gap_clean, 6),
        'nn_dist_gap_corrupted': round(gap_corr, 6),
        'nn_dist_gap_change_pct': round(change, 2) if not np.isnan(change) else np.nan,
        'pct_nn_changed': round(pct_changed, 2) if not np.isnan(pct_changed) else np.nan,
    }


def compute_ae_metrics(clean_errors, corrupted_errors, clean_labels, corrupted_labels):
    anom_c = clean_labels == 1
    norm_c = clean_labels == 0
    anom_r = corrupted_labels == 1
    norm_r = corrupted_labels == 0

    re_anom_clean = float(np.mean(clean_errors[anom_c])) if anom_c.any() else np.nan
    re_norm_clean = float(np.mean(clean_errors[norm_c])) if norm_c.any() else np.nan
    re_anom_corr = float(np.mean(corrupted_errors[anom_r])) if anom_r.any() else np.nan
    re_norm_corr = float(np.mean(corrupted_errors[norm_r])) if norm_r.any() else np.nan

    ratio_clean = re_anom_clean / (re_norm_clean + 1e-10) if not np.isnan(re_anom_clean) else np.nan
    ratio_corr = re_anom_corr / (re_norm_corr + 1e-10) if not np.isnan(re_anom_corr) else np.nan
    change = ((ratio_corr - ratio_clean) / abs(ratio_clean) * 100) if (
        not np.isnan(ratio_clean) and abs(ratio_clean) > 1e-10) else np.nan

    return {
        'recon_error_anomaly_clean': round(re_anom_clean, 6) if not np.isnan(re_anom_clean) else np.nan,
        'recon_error_anomaly_corrupted': round(re_anom_corr, 6) if not np.isnan(re_anom_corr) else np.nan,
        'recon_error_normal_clean': round(re_norm_clean, 6) if not np.isnan(re_norm_clean) else np.nan,
        'recon_error_normal_corrupted': round(re_norm_corr, 6) if not np.isnan(re_norm_corr) else np.nan,
        'error_ratio_clean': round(ratio_clean, 4) if not np.isnan(ratio_clean) else np.nan,
        'error_ratio_corrupted': round(ratio_corr, 4) if not np.isnan(ratio_corr) else np.nan,
        'error_ratio_change_pct': round(change, 2) if not np.isnan(change) else np.nan,
    }


def compute_pca_metrics(clean, corrupted):
    # Cosine similarity for top principal components (use abs to handle sign flips)
    n_pc = min(3, len(clean['components']), len(corrupted['components']))
    cosines = {}
    for i in range(n_pc):
        v1 = clean['components'][i]
        v2 = corrupted['components'][i]
        min_d = min(len(v1), len(v2))
        v1, v2 = v1[:min_d], v2[:min_d]
        cos_sim = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
        cosines[f'pc{i+1}_cosine'] = round(float(cos_sim), 6)
    for i in range(n_pc, 3):
        cosines[f'pc{i+1}_cosine'] = np.nan

    ev_clean = float(np.sum(clean['explained_variance_ratio'][:3]))
    ev_corr = float(np.sum(corrupted['explained_variance_ratio'][:3]))

    gap_clean = (clean['score_anomaly'] or 0) - (clean['score_normal'] or 0)
    gap_corr = (corrupted['score_anomaly'] or 0) - (corrupted['score_normal'] or 0)
    change = ((gap_corr - gap_clean) / abs(gap_clean) * 100) if abs(gap_clean) > 1e-10 else np.nan

    return {
        **cosines,
        'explained_var_top3_clean': round(ev_clean, 6),
        'explained_var_top3_corrupted': round(ev_corr, 6),
        'score_anomaly_clean': clean['score_anomaly'],
        'score_anomaly_corrupted': corrupted['score_anomaly'],
        'score_normal_clean': clean['score_normal'],
        'score_normal_corrupted': corrupted['score_normal'],
        'score_gap_change_pct': round(change, 2) if not np.isnan(change) else np.nan,
    }


# ==========================================
# CORE WORKER
# ==========================================

def process_file_model(job_args):
    """Process one (file, model) pair: clean + all 54 corruptions."""
    file_path, model_name, corruptions_list = job_args
    file_name = os.path.basename(file_path)

    try:
        # 1. Load data
        df, file_name = load_tsb_dataframe(file_path)
        values = df['value'].to_numpy('float')
        labels = df['is_anomaly'].to_numpy('int')
        n = len(df)

        # 2. Clean preprocessing
        clean_scaled = StandardScaler().fit_transform(values.reshape(-1, 1)).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)
        window_labels = map_window_labels(labels, sliding_window)

        # 3. Extract clean internals (once)
        if model_name == 'IForest':
            X_clean = Window(window=sliding_window).convert(clean_scaled).to_numpy()
            clean_internals = extract_iforest(X_clean, window_labels)
        elif model_name == 'LOF':
            X_clean = Window(window=sliding_window).convert(clean_scaled).to_numpy()
            clean_internals = extract_lof(X_clean, window_labels)
        elif model_name == 'MP':
            clean_internals = extract_mp(clean_scaled, sliding_window, window_labels)
        elif model_name == 'AE':
            ae_clf, _ = load_pretrained_ae(file_name, project_root)  # load once
            ae_result = extract_ae(clean_scaled, ae_clf)
            clean_internals = {'recon_errors': ae_result['recon_errors'], 'labels': labels}
        elif model_name == 'PCA':
            X_clean = Window(window=sliding_window).convert(clean_scaled).to_numpy()
            clean_internals = extract_pca(X_clean, window_labels)

        # 4. Process each corruption
        rows = []
        for corruption in corruptions_list:
            try:
                corr_values, corr_labels, nan_mask = apply_corruption(df, corruption, SEED)
                is_missing = corruption['type'] in ('mcar', 'mnar_extreme')

                if is_missing:
                    n_kept = len(corr_values)
                    if n_kept < 30:
                        rows.append({'file': file_name, 'corruption': corruption['name'],
                                     'model': model_name, 'error': 'Too few points after masking'})
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

                # Extract corrupted internals
                row = {'file': file_name, 'corruption': corruption['name'],
                       'model': model_name, 'error': None}

                if model_name == 'IForest':
                    X_corr = Window(window=sw).convert(corr_scaled).to_numpy()
                    corr_int = extract_iforest(X_corr, corr_window_labels)
                    row.update(compute_iforest_metrics(clean_internals, corr_int))

                elif model_name == 'LOF':
                    X_corr = Window(window=sw).convert(corr_scaled).to_numpy()
                    corr_int = extract_lof(X_corr, corr_window_labels)
                    row.update(compute_lof_metrics(clean_internals, corr_int))

                elif model_name == 'MP':
                    corr_int = extract_mp(corr_scaled, sw, corr_window_labels)
                    can_compare = not is_missing  # NN indices incomparable if series length changed
                    row.update(compute_mp_metrics(clean_internals, corr_int, can_compare_nn=can_compare))

                elif model_name == 'AE':
                    ae_result = extract_ae(corr_scaled, ae_clf)
                    row.update(compute_ae_metrics(
                        clean_internals['recon_errors'], ae_result['recon_errors'],
                        clean_internals['labels'], corr_labels))

                elif model_name == 'PCA':
                    X_corr = Window(window=sw).convert(corr_scaled).to_numpy()
                    corr_int = extract_pca(X_corr, corr_window_labels)
                    row.update(compute_pca_metrics(clean_internals, corr_int))

                rows.append(row)

            except Exception as e:
                rows.append({'file': file_name, 'corruption': corruption['name'],
                             'model': model_name, 'error': str(e)[:200]})

        return {'status': 'success', 'rows': rows, 'job_id': f"{file_name}_{model_name}"}

    except Exception as e:
        return {'status': 'error', 'file': file_name, 'model': model_name,
                'error': traceback.format_exc()[:300]}


# ==========================================
# SUMMARY & CASE STUDIES
# ==========================================

def compute_aggregate_summary(df_all, output_path):
    """Compute mean ± std per (corruption, model) for each metric."""
    df_ok = df_all[df_all['error'].isna()] if 'error' in df_all.columns else df_all
    if df_ok.empty:
        print("No successful results to summarize.")
        return

    metric_cols = [c for c in df_ok.columns if c not in ('file', 'corruption', 'model', 'error')]

    summary_rows = []
    for (corruption, model), group in df_ok.groupby(['corruption', 'model']):
        row = {'corruption': corruption, 'model': model, 'n_files': len(group)}
        for col in metric_cols:
            vals = group[col].dropna()
            if len(vals) > 0:
                row[f'{col}_mean'] = round(vals.mean(), 4)
                row[f'{col}_std'] = round(vals.std(), 4)
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
        sub.to_csv(os.path.join(case_dir, f"{safe_name}_internals.csv"), index=False)

    print(f"Case studies saved for {len(case_files)} files in {case_dir}")


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Internal Algorithmic Analysis')
    parser.add_argument('--test', action='store_true', help='3 files, reduced corruptions')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
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

    print(f"\n{'=' * 60}")
    print(f"  Internal Algorithmic Analysis")
    print(f"{'=' * 60}")
    print(f"  Files:        {len(file_paths)}")
    print(f"  Models:       {args.models}")
    print(f"  Corruptions:  {len(corruptions_list)}")
    print(f"  Total rows:   {len(file_paths) * len(args.models) * len(corruptions_list)}")
    print(f"  Workers:      {n_workers}")
    print(f"{'=' * 60}\n")

    # Checkpoint
    checkpoint_file = os.path.join(OUTPUT_DIR, "checkpoint.csv")
    requested_corruptions = {c['name'] for c in corruptions_list}
    completed_jobs = set()
    pair_to_corruptions = {}
    all_results = []

    if os.path.exists(checkpoint_file):
        try:
            df_ckpt = pd.read_csv(checkpoint_file)
            if 'error' in df_ckpt.columns:
                df_ok = df_ckpt[df_ckpt['error'].isna()]
            else:
                df_ok = df_ckpt
            all_results = df_ok.to_dict('records')

            # Mark a (file, model) job as complete only if ALL requested corruptions exist.
            for r in all_results:
                corr = r.get('corruption')
                if corr not in requested_corruptions:
                    continue
                pair = f"{r['file']}_{r['model']}"
                if pair not in pair_to_corruptions:
                    pair_to_corruptions[pair] = set()
                pair_to_corruptions[pair].add(corr)

            completed_jobs = {
                pair for pair, corr_set in pair_to_corruptions.items()
                if requested_corruptions.issubset(corr_set)
            }
            print(f"Loaded checkpoint: {len(all_results)} rows, {len(completed_jobs)} fully-complete (file, model) pairs")
        except Exception as e:
            print(f"Warning: Could not read checkpoint: {e}")

    # Build jobs — pass only MISSING corruptions per (file, model) pair
    jobs = []
    for fp in file_paths:
        fname = os.path.basename(fp)
        for model in args.models:
            job_id = f"{fname}_{model}"
            if job_id in completed_jobs:
                continue
            done_set = pair_to_corruptions.get(job_id, set())
            missing_corruptions = [c for c in corruptions_list if c['name'] not in done_set]
            if not missing_corruptions:
                continue
            jobs.append((fp, model, missing_corruptions))

    total_individual = sum(len(j[2]) for j in jobs)
    print(f"Jobs to run: {len(jobs)} (file, model) pairs, {total_individual} (file, model, corruption) combos to compute")

    if len(jobs) > 0:
        new_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_file_model, job): job for job in jobs}

            with tqdm(total=total_individual, desc="Internal Analysis",
                      unit="corruption") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        n_rows = len(res['rows'])
                        all_results.extend(res['rows'])
                        new_count += 1
                        pbar.update(n_rows)
                    elif res['status'] == 'error':
                        all_results.append({
                            'file': res.get('file', 'unknown'),
                            'model': res.get('model', 'unknown'),
                            'corruption': 'ALL',
                            'error': res.get('error', 'unknown'),
                        })
                        pbar.update(len(corruptions_list))

                    if new_count % 50 == 0 and new_count > 0:
                        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)

        df_ckpt_save = pd.DataFrame(all_results)
        if not df_ckpt_save.empty and all(c in df_ckpt_save.columns for c in ('file', 'model', 'corruption')):
            df_ckpt_save = df_ckpt_save.drop_duplicates(
                subset=['file', 'model', 'corruption'], keep='last'
            )
            all_results = df_ckpt_save.to_dict('records')
        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
        print(f"\nCheckpoint saved: {len(all_results)} rows")

    # Outputs
    if all_results:
        df_all = pd.DataFrame(all_results)
        if all(c in df_all.columns for c in ('file', 'model', 'corruption')):
            df_all = df_all.drop_duplicates(subset=['file', 'model', 'corruption'], keep='last')

        # Raw internals (clean version)
        df_ok = df_all[df_all['error'].isna()] if 'error' in df_all.columns else df_all
        raw_path = os.path.join(OUTPUT_DIR, "raw_internals.csv")
        df_ok.to_csv(raw_path, index=False)
        print(f"Raw internals: {raw_path} ({len(df_ok)} rows)")

        # Aggregate summary
        summary_path = os.path.join(OUTPUT_DIR, "aggregate_summary.csv")
        compute_aggregate_summary(df_all, summary_path)

        # Case studies
        save_case_studies(df_all, OUTPUT_DIR)

        # Print compact summary
        if not df_ok.empty:
            print(f"\n{'=' * 60}")
            print(f"  KEY FINDINGS")
            print(f"{'=' * 60}")
            for model in args.models:
                model_df = df_ok[df_ok['model'] == model]
                if model_df.empty:
                    continue
                print(f"\n  {model}:")
                if model == 'IForest' and 'separation_gap_change_pct' in model_df.columns:
                    vals = model_df.groupby('corruption')['separation_gap_change_pct'].mean()
                    worst = vals.idxmin() if len(vals) > 0 else 'N/A'
                    print(f"    Worst separation gap loss: {worst} ({vals.min():.1f}%)")
                elif model == 'LOF' and 'kdist_gap_change_pct' in model_df.columns:
                    vals = model_df.groupby('corruption')['kdist_gap_change_pct'].mean()
                    worst = vals.idxmin() if len(vals) > 0 else 'N/A'
                    print(f"    Worst k-dist gap loss: {worst} ({vals.min():.1f}%)")
                elif model == 'MP' and 'pct_nn_changed' in model_df.columns:
                    vals = model_df.groupby('corruption')['pct_nn_changed'].mean()
                    worst = vals.idxmax() if len(vals) > 0 else 'N/A'
                    print(f"    Most NN disruption: {worst} ({vals.max():.1f}%)")
                elif model == 'AE' and 'error_ratio_change_pct' in model_df.columns:
                    vals = model_df.groupby('corruption')['error_ratio_change_pct'].mean()
                    worst = vals.idxmin() if len(vals) > 0 else 'N/A'
                    print(f"    Worst error ratio loss: {worst} ({vals.min():.1f}%)")
                elif model == 'PCA' and 'pc1_cosine' in model_df.columns:
                    vals = model_df.groupby('corruption')['pc1_cosine'].mean()
                    worst = vals.idxmin() if len(vals) > 0 else 'N/A'
                    print(f"    Most eigenvector rotation: {worst} (cos={vals.min():.3f})")

    print(f"\n[Done] Internal analysis complete.")


if __name__ == "__main__":
    main()
