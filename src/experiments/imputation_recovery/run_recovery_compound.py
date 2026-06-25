"""
Recovery for: Compound Corruptions

Applies the same compound corruptions as run_compound_corruptions.py,
then applies cleaning/imputation and re-runs the model.

Recovery strategy per corruption type:
  - missing / ge_missing: linear interpolation (fill NaN)
  - freeze: detect constant segments → interpolate
  - noise: Savitzky-Golay smoothing (window=5, order=2)
  - spikes: z-score clipping (|z|>4) → interpolate

All combinations are tested. Noise+Spikes (no missing/freeze) tests
whether smoothing+clipping alone can recover performance.

Compare recovered_AUC with corrupted AUC from:
  results/experiments/compound_corruptions/checkpoint.csv

Usage:
    python run_recovery_compound.py --models IForest
    python run_recovery_compound.py --models IForest PCA LOF MP --workers 4
    python run_recovery_compound.py --test
"""
import os
import sys
import math
import argparse
import itertools
import pandas as pd
import numpy as np
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

from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    from TSB_UAD.models.iforest import IForest
    from TSB_UAD.models.pca import PCA
    from TSB_UAD.models.matrix_profile import MatrixProfile
    from TSB_UAD.models.lof import LOF
    from TSB_UAD.models.feature import Window
    from TSB_UAD.utils.slidingWindows import find_length
    from TSB_UAD.vus.metrics import get_metrics
except ImportError as e:
    warnings.warn(f"TSB_UAD components could not be imported: {e}")

from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors
from data_loader import load_tsb_dataframe

# ==========================================
# MUST MATCH run_compound_corruptions.py
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "recovery_compound")
BASELINE_CSV = os.path.join(project_root, "results", "tables", "baseline_final_subset.csv")


N_SEEDS = 1

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
]

# Severity levels — EXACT COPY from compound script
SEVERITY = {
    'noise':   {'low': {'snr_db': 20},
                'med': {'snr_db': 10},
                'high': {'snr_db': 5}},
    'missing': {'low': {'fraction': 0.05},
                'med': {'fraction': 0.10},
                'high': {'fraction': 0.20}},
    'spikes':  {'low': {'fraction': 0.05, 'multiplier': 3, 'sequential': False, 'sequence_length': 1},
                'med': {'fraction': 0.10, 'multiplier': 3, 'sequential': False, 'sequence_length': 1},
                'high': {'fraction': 0.10, 'multiplier': 10, 'sequential': False, 'sequence_length': 1}},
    'freeze':  {'low': {'num_stucks': 3, 'freeze_fraction': 0.05},
                'med': {'num_stucks': 5, 'freeze_fraction': 0.10},
                'high': {'num_stucks': 5, 'freeze_fraction': 0.20}},
    'ge_missing': {'low': {'alpha': 0.01, 'beta': 0.25},
                   'med': {'alpha': 0.01, 'beta': 0.10},
                   'high': {'alpha': 0.05, 'beta': 0.10}},
}

# Only combinations that include recoverable corruptions
COMBINATIONS = [
    ('noise_missing',        ['noise', 'missing']),
    ('noise_spikes',         ['noise', 'spikes']),
    ('spikes_missing',       ['spikes', 'missing']),
    ('missing_freeze',       ['missing', 'freeze']),
    ('noise_spikes_missing', ['noise', 'spikes', 'missing']),
    ('noise_ge_missing',     ['noise', 'ge_missing']),
]

APPLICATION_ORDER = {'freeze': 0, 'noise': 1, 'spikes': 2, 'missing': 3, 'ge_missing': 3}

MIN_CONSTANT_LENGTH = 3
SPIKE_Z_THRESHOLD = 4.0       # conservative: only clip extreme outliers
SAVGOL_WINDOW = 5             # Savitzky-Golay smoothing window
SAVGOL_ORDER = 2              # polynomial order


# ==========================================
# RECOVERY FUNCTIONS
# ==========================================

def detect_and_fix_freeze(data, min_length=3):
    """Detect constant segments and replace with linear interpolation."""
    cleaned = data.copy()
    n = len(cleaned)
    stuck_mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        j = i + 1
        while j < n and cleaned[j] == cleaned[i]:
            j += 1
        if j - i >= min_length:
            stuck_mask[i + 1:j] = True
        i = j
    if not stuck_mask.any():
        return cleaned
    cleaned[stuck_mask] = np.nan
    s = pd.Series(cleaned)
    return s.interpolate(method='linear').bfill().ffill().to_numpy('float')


def recover_noise(data, window=SAVGOL_WINDOW, order=SAVGOL_ORDER):
    """Smooth noisy data using Savitzky-Golay filter.

    Preserves peaks better than moving average while reducing noise.
    """
    from scipy.signal import savgol_filter
    if len(data) < window:
        return data
    return savgol_filter(data, window_length=window, polyorder=order)


def recover_spikes(data, z_threshold=SPIKE_Z_THRESHOLD):
    """Clip extreme outliers (|z| > threshold) and interpolate.

    Uses z_threshold=4 (conservative) to avoid removing real anomalies.
    """
    cleaned = data.copy()
    mean = np.nanmean(cleaned)
    std = np.nanstd(cleaned)
    if std < 1e-10:
        return cleaned
    z = np.abs((cleaned - mean) / std)
    spike_mask = z > z_threshold
    if not spike_mask.any():
        return cleaned
    cleaned[spike_mask] = np.nan
    s = pd.Series(cleaned)
    return s.interpolate(method='linear').bfill().ffill().to_numpy('float')


def apply_recovery(data, condition):
    """Apply recovery to compound-corrupted data.

    Order: impute missing → fix freeze → clip spikes → smooth noise
    (reverse of corruption application order)
    """
    recovered = data.copy()
    corruption_types = [c['type'] for c in condition['corruptions']]

    has_missing = any(t in ('missing', 'ge_missing') for t in corruption_types)
    has_freeze = 'freeze' in corruption_types
    has_spikes = 'spikes' in corruption_types
    has_noise = 'noise' in corruption_types

    # Step 1: impute missing values
    if has_missing and np.isnan(recovered).any():
        s = pd.Series(recovered)
        recovered = s.interpolate(method='linear').bfill().ffill().to_numpy('float')

    # Step 2: fix freeze segments
    if has_freeze:
        recovered = detect_and_fix_freeze(recovered, min_length=MIN_CONSTANT_LENGTH)

    # Step 3: clip spikes (before smoothing — spikes would bias the smoother)
    if has_spikes:
        recovered = recover_spikes(recovered, z_threshold=SPIKE_Z_THRESHOLD)

    # Step 4: smooth noise
    if has_noise:
        recovered = recover_noise(recovered, window=SAVGOL_WINDOW, order=SAVGOL_ORDER)

    return recovered


# ==========================================
# CONDITION BUILDER (same as compound script)
# ==========================================

def build_conditions():
    """Build compound conditions that have recoverable corruptions."""
    conditions = []
    for combo_name, corruption_types in COMBINATIONS:
        for sev_combo in itertools.product(['low', 'med', 'high'], repeat=len(corruption_types)):
            corr_list = []
            name_parts = []
            has_missing = False
            for ctype, sev in zip(corruption_types, sev_combo):
                corr_list.append({'type': ctype, 'severity': sev,
                                  'params': SEVERITY[ctype][sev].copy()})
                name_parts.append(f"{ctype}_{sev}")
                if ctype in ('missing', 'ge_missing'):
                    has_missing = True
            conditions.append({
                'name': '+'.join(name_parts),
                'combination_name': combo_name,
                'condition_type': 'compound',
                'corruptions': corr_list,
                'has_missing': has_missing,
            })
    return conditions


# ==========================================
# CORRUPTION APPLICATION (same as compound script)
# ==========================================

def apply_corruptions(corruptor, condition, series_length):
    """Apply corruptions in canonical physical order."""
    corruptions = sorted(condition['corruptions'],
                         key=lambda c: APPLICATION_ORDER[c['type']])
    for corr in corruptions:
        ctype = corr['type']
        params = corr['params'].copy()
        if ctype == 'noise':
            ts_corruptor.injectors.inject_white_noise_snr(corruptor, **params)
        elif ctype == 'missing':
            ts_corruptor.injectors.inject_point_missing(corruptor, **params)
        elif ctype == 'spikes':
            ts_corruptor.injectors.inject_spikes(corruptor, **params)
        elif ctype == 'freeze':
            freeze_frac = params.pop('freeze_fraction')
            num_stucks = params['num_stucks']
            stuck_length = max(1, int(freeze_frac * series_length / num_stucks))
            ts_corruptor.injectors.inject_sensor_stuck(
                corruptor, num_stucks=num_stucks, stuck_length=stuck_length)
        elif ctype == 'ge_missing':
            ts_corruptor.injectors.inject_gilbert_elliott(
                corruptor, p_good_to_bad=params['alpha'],
                p_bad_to_good=params['beta'], noise_type='missing')


# ==========================================
# MODEL RUNNER
# ==========================================

def run_model(model_name, X, scaled_data, sw):
    if model_name == 'IForest':
        clf = IForest(n_estimators=100, random_state=42)
        clf.fit(X)
        return clf.decision_scores_
    elif model_name == 'PCA':
        n_comp = min(10, X.shape[1] - 1) if X.shape[1] > 1 else 1
        clf = PCA(n_components=n_comp)
        clf.fit(X)
        return clf.decision_scores_
    elif model_name == 'MP':
        clf = MatrixProfile(window=sw)
        clf.fit(scaled_data)
        return clf.decision_scores_
    elif model_name == 'LOF':
        n_neigh = min(20, len(X) - 1)
        clf = LOF(n_neighbors=n_neigh)
        clf.fit(X)
        return clf.decision_scores_
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==========================================
# CORE WORKER
# ==========================================

def process_single_job(job_args):
    (file_path, condition, seed, model_names, baseline_dict) = job_args

    file_name = os.path.basename(file_path)
    condition_name = condition['name']
    results = []

    try:
        df, file_name = load_tsb_dataframe(file_path)
        labels = df['is_anomaly'].to_numpy('int')
        n = len(df)

        clean_data = df['value'].to_numpy('float')
        clean_scaled = StandardScaler().fit_transform(clean_data.reshape(-1, 1)).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # Apply same corruptions as compound experiment
        corruptor = TSCorruptor(df.copy(), value_col='value', label_col='is_anomaly', seed=seed)
        apply_corruptions(corruptor, condition, n)
        df_corrupted = corruptor.get_corrupted_df()
        corrupted_data = df_corrupted['value'].to_numpy('float')

        nan_mask = np.isnan(corrupted_data)
        nan_count = int(nan_mask.sum())
        actual_missing = nan_count / n
        has_missing = nan_count > 0

        # ── Corrupted evaluation (mirrors compound experiment) ──
        masked_anomaly = nan_mask & (labels == 1)
        masked_normal = nan_mask & (labels == 0)

        if has_missing:
            corr_model_data = corrupted_data[~nan_mask]
            corr_n_kept = len(corr_model_data)
            corr_sw = max(min(sliding_window, corr_n_kept // 4), 10)
        else:
            corr_model_data = corrupted_data
            corr_n_kept = n
            corr_sw = sliding_window

        # ── Recovered data ──
        recovered_data = apply_recovery(corrupted_data, condition)

        if np.isnan(recovered_data).any():
            s = pd.Series(recovered_data)
            recovered_data = s.interpolate(method='linear').bfill().ffill().to_numpy('float')

        for model_name in model_names:
            try:
                # --- Corrupted AUC (same method as compound experiment) ---
                corrupted_auc = None
                if corr_n_kept >= corr_sw + 10:
                    corr_scaled = StandardScaler().fit_transform(
                        corr_model_data.reshape(-1, 1)).flatten()
                    corr_X = Window(window=corr_sw).convert(corr_scaled).to_numpy()
                    corr_score = run_model(model_name, corr_X, corr_scaled, corr_sw)
                    corr_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                        corr_score.reshape(-1, 1)).ravel()
                    corr_padded = np.array(
                        [corr_score[0]] * math.ceil((corr_sw - 1) / 2) +
                        list(corr_score) +
                        [corr_score[-1]] * ((corr_sw - 1) // 2)
                    )
                    if not (np.isnan(corr_padded).any() or len(np.unique(corr_padded)) <= 1):
                        if has_missing:
                            full_corr = np.full(n, np.nan)
                            full_corr[~nan_mask] = corr_padded
                            full_corr[masked_anomaly] = 0.0
                            eval_mask = ~masked_normal
                            corr_metrics = get_metrics(
                                full_corr[eval_mask], labels[eval_mask],
                                metric="all", slidingWindow=corr_sw)
                        else:
                            corr_metrics = get_metrics(
                                corr_padded, labels, metric="all", slidingWindow=corr_sw)
                        corrupted_auc = round(corr_metrics.get('AUC_ROC', 0.0), 4)

                # --- Recovered AUC (full series evaluation) ---
                rec_scaled = StandardScaler().fit_transform(
                    recovered_data.reshape(-1, 1)).flatten()
                rec_X = Window(window=sliding_window).convert(rec_scaled).to_numpy()

                rec_score = run_model(model_name, rec_X, rec_scaled, sliding_window)
                rec_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                    rec_score.reshape(-1, 1)).ravel()

                rec_padded = np.array(
                    [rec_score[0]] * math.ceil((sliding_window - 1) / 2) +
                    list(rec_score) +
                    [rec_score[-1]] * ((sliding_window - 1) // 2)
                )

                if np.isnan(rec_padded).any() or len(np.unique(rec_padded)) <= 1:
                    results.append({
                        'file': file_name, 'condition': condition_name,
                        'combination_name': condition['combination_name'],
                        'model': model_name, 'seed': seed,
                        'recovery': 'combined',
                        'actual_missing_rate': round(actual_missing, 4),
                        'error': 'Invalid scores',
                    })
                    continue

                rec_metrics = get_metrics(rec_padded, labels, metric="all",
                                          slidingWindow=sliding_window)
                baseline_auc = baseline_dict.get(file_name, {}).get(model_name, None)

                row = {
                    'file': file_name,
                    'condition': condition_name,
                    'combination_name': condition['combination_name'],
                    'model': model_name,
                    'seed': seed,
                    'recovery': 'combined',
                    'actual_missing_rate': round(actual_missing, 4),
                    'baseline_AUC_ROC': round(baseline_auc, 4) if baseline_auc else None,
                    'corrupted_AUC_ROC': corrupted_auc,
                    'error': None,
                }
                for key in SELECTED_METRICS:
                    row[f'recovered_{key}'] = round(rec_metrics.get(key, 0.0), 4)

                results.append(row)

            except Exception as e:
                results.append({
                    'file': file_name, 'condition': condition_name,
                    'combination_name': condition['combination_name'],
                    'model': model_name, 'seed': seed,
                    'recovery': 'combined',
                    'actual_missing_rate': round(actual_missing, 4),
                    'error': str(e),
                })

        return {'status': 'success', 'results': results}

    except Exception as e:
        return {'status': 'error', 'file': file_name, 'error': traceback.format_exc()}


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Recovery for Compound Corruptions")
    parser.add_argument('--models', nargs='+', default=['IForest'],
                        choices=['IForest', 'PCA', 'LOF', 'MP'])
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(RESULTS_DIR, "checkpoint.csv")

    # Load baseline
    baseline_df = pd.read_csv(BASELINE_CSV)
    baseline_dict = {}
    for _, r in baseline_df.iterrows():
        baseline_dict.setdefault(r['file'], {})[r['model']] = r['AUC_ROC']

    # Load file list
    subset_df = pd.read_csv(SUBSET_CSV)
    file_paths = subset_df['filepath'].tolist()
    if args.test:
        file_paths = file_paths[:3]
        print(f"[TEST MODE] Using {len(file_paths)} files")

    # Build conditions
    conditions = build_conditions()

    # Resume
    done_results = set()
    if os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        for _, r in existing.iterrows():
            key = (r.get('file'), r.get('condition'), int(r.get('seed', 0)),
                   r.get('model'), r.get('recovery'))
            done_results.add(key)
        print(f"[Resume] Loaded {len(existing)} rows from checkpoint")

    # Build jobs
    jobs = []
    skipped = 0
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        for seed in range(N_SEEDS):
            for cond in conditions:
                if done_results and all(
                    (file_name, cond['name'], seed, m, 'combined') in done_results
                    for m in args.models
                ):
                    skipped += 1
                    continue
                jobs.append((file_path, cond, seed, args.models, baseline_dict))

    if skipped:
        print(f"[Resume] Skipped {skipped} completed jobs")

    print(f"\n{'='*60}")
    print(f"Recovery — Compound Corruptions")
    print(f"{'='*60}")
    print(f"Files: {len(file_paths)}")
    print(f"Models: {args.models}")
    print(f"Conditions: {len(conditions)}")
    print(f"Total jobs: {len(jobs)}")
    print(f"{'='*60}\n")

    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_job, j): j for j in jobs}
        with tqdm(total=len(futures), desc="recovery (compound)") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result['status'] == 'success' and result.get('results'):
                        all_results.extend(result['results'])
                        if len(all_results) % 100 == 0:
                            df_save = pd.DataFrame(all_results)
                            if os.path.exists(checkpoint_path):
                                df_existing = pd.read_csv(checkpoint_path)
                                df_save = pd.concat([df_existing, df_save], ignore_index=True)
                            df_save.to_csv(checkpoint_path, index=False)
                            all_results = []
                except Exception as e:
                    print(f"  [Error] {e}")
                pbar.update(1)

    if all_results:
        df_save = pd.DataFrame(all_results)
        if os.path.exists(checkpoint_path):
            df_existing = pd.read_csv(checkpoint_path)
            df_save = pd.concat([df_existing, df_save], ignore_index=True)
        df_save.to_csv(checkpoint_path, index=False)

    print(f"\n[Done] Results saved to {checkpoint_path}")

    # Summary
    if os.path.exists(checkpoint_path):
        df_all = pd.read_csv(checkpoint_path)
        df_ok = df_all[df_all['error'].isna()]
        if not df_ok.empty:
            summary = df_ok.groupby(['combination_name', 'condition', 'model']).agg(
                n=('file', 'count'),
                mean_corrupted_AUC_ROC=('corrupted_AUC_ROC', 'mean'),
                mean_recovered_AUC_ROC=('recovered_AUC_ROC', 'mean'),
                mean_baseline_AUC_ROC=('baseline_AUC_ROC', 'mean'),
            ).reset_index()
            summary['recovery_gap'] = summary['mean_baseline_AUC_ROC'] - summary['mean_recovered_AUC_ROC']
            gap = (summary['mean_baseline_AUC_ROC'] - summary['mean_corrupted_AUC_ROC']).replace(0, np.nan)
            summary['recovery_pct'] = (
                (summary['mean_recovered_AUC_ROC'] - summary['mean_corrupted_AUC_ROC']) / gap * 100
            ).round(1)
            summary_path = os.path.join(RESULTS_DIR, "summary.csv")
            summary.to_csv(summary_path, index=False)
            print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
