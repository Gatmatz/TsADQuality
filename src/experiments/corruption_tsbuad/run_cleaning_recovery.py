"""
Cleaning Recovery Experiment (RQ3) — Optimized.

Reuses baseline and corrupted results from existing experiments.
Only runs corruption -> cleaning -> detection (the new part).

For missing data, cleaned results are also loaded from the existing
missing experiment (which already tested linear and ffill imputation).

Pipeline:
  - Baseline: loaded from baseline_final_subset.csv
  - Corrupted (no cleaning): loaded from existing experiment checkpoints
  - Cleaned: corrupt (seed=0) -> clean -> detect  (NEW computation)

Usage:
    python run_cleaning_recovery.py                # full run
    python run_cleaning_recovery.py --test         # quick test (3 files)
    python run_cleaning_recovery.py --workers 6    # parallel workers
"""
import os
import sys
import math
import argparse
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

from sklearn.preprocessing import MinMaxScaler

try:
    from TSB_UAD.models.iforest import IForest
    from TSB_UAD.models.feature import Window
    from TSB_UAD.utils.slidingWindows import find_length
    from TSB_UAD.vus.metrics import get_metrics
except ImportError as e:
    warnings.warn(f"TSB_UAD import error: {e}")

from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors
from data_loader import load_tsb_dataframe
from data_prep.cleaning import CLEANING_METHODS

# ==========================================
# CONFIGURATION
# ==========================================
EXPERIMENTS_DIR = os.path.join(project_root, "results", "experiments")
BASELINE_CSV = os.path.join(project_root, "results", "tables", "baseline_final_subset.csv")
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "cleaning_recovery")

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'VUS_ROC', 'VUS_PR',
    'R_AUC_ROC', 'R_AUC_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]

# Corruption conditions — matched to existing experiment parameters
CORRUPTION_CONDITIONS = [
    # White noise (from white_noise_snr experiment)
    {'type': 'white_noise', 'label': 'noise_snr_10dB', 'params': {'snr_db': 10.0}},
    {'type': 'white_noise', 'label': 'noise_snr_0dB', 'params': {'snr_db': 0.0}},
    {'type': 'white_noise', 'label': 'noise_snr_-10dB', 'params': {'snr_db': -10.0}},

    # Spikes — point (matching existing: multipliers 3.0 and 10.0)
    {'type': 'spikes', 'label': 'spikes_5pct_10x', 'params': {'fraction': 0.05, 'multiplier': 10.0}},
    {'type': 'spikes', 'label': 'spikes_20pct_10x', 'params': {'fraction': 0.20, 'multiplier': 10.0}},

    # Missing — point (results loaded entirely from existing experiment)
    {'type': 'missing', 'label': 'missing_5pct_point', 'params': {'fraction': 0.05}},
    {'type': 'missing', 'label': 'missing_20pct_point', 'params': {'fraction': 0.20}},

    # Freeze / stuck sensor
    {'type': 'freeze', 'label': 'freeze_5pct_len50', 'params': {'fraction': 0.05, 'stuck_length': 50}},
    {'type': 'freeze', 'label': 'freeze_20pct_len50', 'params': {'fraction': 0.20, 'stuck_length': 50}},

    # Swap — point
    {'type': 'swap', 'label': 'swap_5pct_len1', 'params': {'fraction': 0.05, 'swap_length': 1}},
    {'type': 'swap', 'label': 'swap_20pct_len1', 'params': {'fraction': 0.20, 'swap_length': 1}},
]

# Corruption types that need NEW cleaning runs (missing is fully loaded)
TYPES_NEEDING_CLEANING = {'white_noise', 'spikes', 'freeze', 'swap'}


# ==========================================
# LOAD EXISTING RESULTS
# ==========================================

def _extract_metrics(row):
    """Extract selected metrics from a DataFrame row."""
    return {m: round(float(row[m]), 4)
            for m in SELECTED_METRICS
            if m in row.index and pd.notna(row[m])}


def _filter_errors(df):
    """Remove rows with errors."""
    if 'error' in df.columns:
        return df[df['error'].isna() | (df['error'] == '')]
    return df


def load_existing_results():
    """Load baseline + corrupted + missing-cleaned from existing experiments."""
    results = []
    _cache = {}  # file path -> DataFrame

    def _load(rel_path):
        if rel_path not in _cache:
            _cache[rel_path] = pd.read_csv(os.path.join(EXPERIMENTS_DIR, rel_path))
        return _cache[rel_path]

    # --- 1. Baseline ---
    baseline = pd.read_csv(BASELINE_CSV)
    baseline = baseline[baseline['model'] == 'IForest']
    for _, row in baseline.iterrows():
        r = {'file': row['file'], 'corruption': 'none', 'cleaning': 'none',
             'phase': 'baseline', **_extract_metrics(row)}
        results.append(r)
    print(f"  Baseline: {len(baseline)} rows loaded")

    # --- 2. Corrupted (no cleaning) from existing experiments ---
    corrupted_count = 0

    # White noise SNR
    snr_df = _filter_errors(_load("white_noise_snr/checkpoint.csv"))
    snr_df = snr_df[snr_df['model'] == 'IForest']
    for snr_val, label in [(10, 'noise_snr_10dB'), (0, 'noise_snr_0dB'), (-10, 'noise_snr_-10dB')]:
        sub = snr_df[snr_df['snr_db'] == snr_val]
        for _, row in sub.iterrows():
            results.append({'file': row['file'], 'corruption': label,
                           'cleaning': 'none', 'phase': 'corrupted',
                           **_extract_metrics(row)})
            corrupted_count += 1

    # Spikes
    spikes_df = _filter_errors(_load("spikes/IForest/point/raw_results.csv"))
    for frac, mult, label in [(0.05, 10.0, 'spikes_5pct_10x'),
                               (0.20, 10.0, 'spikes_20pct_10x')]:
        sub = spikes_df[(spikes_df['fraction'] == frac) & (spikes_df['multiplier'] == mult)]
        for _, row in sub.iterrows():
            results.append({'file': row['file'], 'corruption': label,
                           'cleaning': 'none', 'phase': 'corrupted',
                           **_extract_metrics(row)})
            corrupted_count += 1

    # Missing — corrupted baseline (ffill imputation = minimal cleaning)
    missing_df = _filter_errors(_load("missing/checkpoint.csv"))
    missing_df = missing_df[(missing_df['model'] == 'IForest') &
                            (missing_df['missing_type'] == 'point')]
    for frac, label in [(0.05, 'missing_5pct_point'), (0.20, 'missing_20pct_point')]:
        sub = missing_df[(missing_df['fraction'] == frac) & (missing_df['imputation'] == 'ffill')]
        for _, row in sub.iterrows():
            results.append({'file': row['file'], 'corruption': label,
                           'cleaning': 'none', 'phase': 'corrupted',
                           **_extract_metrics(row)})
            corrupted_count += 1

    # Freeze
    freeze_df = _filter_errors(_load("freeze/checkpoint.csv"))
    freeze_df = freeze_df[freeze_df['model'] == 'IForest']
    for cond_str, label in [('frac_0.05_len_50', 'freeze_5pct_len50'),
                             ('frac_0.2_len_50', 'freeze_20pct_len50')]:
        sub = freeze_df[freeze_df['condition'] == cond_str]
        for _, row in sub.iterrows():
            results.append({'file': row['file'], 'corruption': label,
                           'cleaning': 'none', 'phase': 'corrupted',
                           **_extract_metrics(row)})
            corrupted_count += 1

    # Swap
    swap_df = _filter_errors(_load("swap/checkpoint.csv"))
    swap_df = swap_df[swap_df['model'] == 'IForest']
    for cond_str, label in [('frac_0.05_dist_Global_len_1', 'swap_5pct_len1'),
                             ('frac_0.2_dist_Global_len_1', 'swap_20pct_len1')]:
        sub = swap_df[swap_df['condition'] == cond_str]
        for _, row in sub.iterrows():
            results.append({'file': row['file'], 'corruption': label,
                           'cleaning': 'none', 'phase': 'corrupted',
                           **_extract_metrics(row)})
            corrupted_count += 1

    print(f"  Corrupted: {corrupted_count} rows loaded")

    # --- 3. Missing cleaned results (already in existing experiment) ---
    cleaned_count = 0
    for frac, label in [(0.05, 'missing_5pct_point'), (0.20, 'missing_20pct_point')]:
        # Linear interpolation
        sub = missing_df[(missing_df['fraction'] == frac) & (missing_df['imputation'] == 'linear')]
        for _, row in sub.iterrows():
            results.append({'file': row['file'], 'corruption': label,
                           'cleaning': 'Linear interpolation', 'phase': 'cleaned',
                           **_extract_metrics(row)})
            cleaned_count += 1
        # Forward fill (same as corrupted — shows 0% recovery, which is informative)
        sub = missing_df[(missing_df['fraction'] == frac) & (missing_df['imputation'] == 'ffill')]
        for _, row in sub.iterrows():
            results.append({'file': row['file'], 'corruption': label,
                           'cleaning': 'Forward fill', 'phase': 'cleaned',
                           **_extract_metrics(row)})
            cleaned_count += 1

    print(f"  Cleaned (missing, from existing): {cleaned_count} rows loaded")

    return results


# ==========================================
# CORRUPTION + CLEANING + DETECTION
# ==========================================

def apply_corruption(corruptor, corr_type, params):
    """Apply corruption based on type and params dict."""
    if corr_type == 'white_noise':
        ts_corruptor.injectors.inject_white_noise_snr(corruptor, snr_db=params['snr_db'])
    elif corr_type == 'spikes':
        ts_corruptor.injectors.inject_spikes(
            corruptor, fraction=params['fraction'], multiplier=params['multiplier'],
            sequential=False, sequence_length=1)
    elif corr_type == 'freeze':
        n = len(corruptor.df)
        num_stucks = max(1, int(params['fraction'] * n / params['stuck_length']))
        ts_corruptor.injectors.inject_sensor_stuck(
            corruptor, num_stucks=num_stucks, stuck_length=params['stuck_length'])
    elif corr_type == 'swap':
        ts_corruptor.injectors.inject_swap(
            corruptor, fraction=params['fraction'],
            swap_length=params.get('swap_length', 1), max_distance=None)


def run_detector(data, labels, sliding_window):
    """Run IForest and return metrics dict or None on failure."""
    try:
        X = Window(window=sliding_window).convert(data).to_numpy()
        clf = IForest(n_estimators=100, random_state=42)
        clf.fit(X)
        score = clf.decision_scores_

        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        full_score = np.array(
            [score[0]] * math.ceil((sliding_window - 1) / 2) +
            list(score) +
            [score[-1]] * ((sliding_window - 1) // 2)
        )

        if np.isnan(full_score).any() or len(np.unique(full_score)) <= 1:
            return None

        metrics = get_metrics(full_score, labels, metric="all", slidingWindow=sliding_window)
        return {k: round(metrics.get(k, 0.0), 4) for k in SELECTED_METRICS}
    except Exception:
        return None


def process_single_file_cleaning(args):
    """Process one file: corrupt -> clean -> detect (only cleaning variants)."""
    file_path, condition_indices = args
    results = []

    try:
        df, file_name = load_tsb_dataframe(file_path)
        labels = df['is_anomaly'].to_numpy('int')
        clean_data = df['value'].to_numpy('float')
        sliding_window = max(int(find_length(clean_data)), 10)

        for idx in condition_indices:
            cond = CORRUPTION_CONDITIONS[idx]
            corr_type = cond['type']
            if corr_type not in TYPES_NEEDING_CLEANING:
                continue

            corr_label = cond['label']

            # Apply corruption (seed=0, matching existing experiments)
            corruptor = TSCorruptor(df, value_col='value', label_col='is_anomaly', seed=0)
            apply_corruption(corruptor, corr_type, cond['params'])
            corrupted_data = corruptor.get_corrupted_df()['value'].to_numpy('float')

            # Apply each cleaning method
            for clean_name, clean_fn in CLEANING_METHODS.get(corr_type, []):
                try:
                    cleaned_data = clean_fn(corrupted_data)

                    # Safety: fill any remaining NaNs
                    if np.isnan(cleaned_data).any():
                        cleaned_data = np.nan_to_num(cleaned_data,
                                                      nan=np.nanmean(cleaned_data))

                    cleaned_metrics = run_detector(cleaned_data, labels, sliding_window)
                    if cleaned_metrics:
                        results.append({
                            'file': file_name, 'corruption': corr_label,
                            'cleaning': clean_name, 'phase': 'cleaned',
                            **cleaned_metrics
                        })
                except Exception as e:
                    results.append({
                        'file': file_name, 'corruption': corr_label,
                        'cleaning': clean_name, 'phase': 'cleaned',
                        'error': str(e)
                    })

        return {'status': 'success', 'results': results}

    except Exception as e:
        return {'status': 'error', 'file': os.path.basename(file_path),
                'error': traceback.format_exc()}


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(SUBSET_CSV):
        print(f"[ERROR] {SUBSET_CSV} not found")
        return

    # --- Load existing results (baseline + corrupted + missing cleaned) ---
    print("Loading existing results...")
    existing_results = load_existing_results()

    # --- Prepare cleaning jobs ---
    df_files = pd.read_csv(SUBSET_CSV)['filepath'].tolist()
    all_cond_indices = list(range(len(CORRUPTION_CONDITIONS)))

    # Only conditions that need new cleaning runs
    cleaning_indices = [i for i, c in enumerate(CORRUPTION_CONDITIONS)
                        if c['type'] in TYPES_NEEDING_CLEANING]

    if args.test:
        df_files = df_files[:3]
        cleaning_indices = cleaning_indices[:4]  # subset for testing
        print("!!! TEST MODE !!!")

    # Count new cleaning evaluations
    n_cleaning_per_file = sum(
        len(CLEANING_METHODS.get(CORRUPTION_CONDITIONS[i]['type'], []))
        for i in cleaning_indices
    )

    print(f"\n{'=' * 60}")
    print(f"  Cleaning Recovery Experiment (RQ3) — Optimized")
    print(f"{'=' * 60}")
    print(f"  Files: {len(df_files)}")
    print(f"  Corruption conditions (needing cleaning): {len(cleaning_indices)}")
    print(f"  Cleaning methods per file: {n_cleaning_per_file}")
    print(f"  Total NEW evaluations: {len(df_files) * n_cleaning_per_file}")
    print(f"  (Baseline + corrupted + missing cleaned: loaded from existing)")
    print(f"{'=' * 60}\n")

    # --- Checkpoint for cleaning results ---
    checkpoint_file = os.path.join(RESULTS_DIR, "cleaning_checkpoint.csv")
    completed_files = set()
    cleaning_results = []

    if os.path.exists(checkpoint_file):
        try:
            df_cp = pd.read_csv(checkpoint_file)
            cleaning_results = df_cp.to_dict('records')
            completed_files = set(df_cp['file'].unique())
            print(f"Loaded checkpoint: {len(completed_files)} files done.")
        except Exception as e:
            print(f"Warning: checkpoint error: {e}")

    jobs = [(fp, cleaning_indices) for fp in df_files
            if os.path.basename(fp) not in completed_files]

    print(f"Jobs remaining: {len(jobs)}")

    if jobs:
        new_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_file_cleaning, job): job
                       for job in jobs}

            with tqdm(total=len(jobs), desc="cleaning recovery") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        cleaning_results.extend(res['results'])
                        new_count += 1
                    pbar.update(1)

                    if new_count % 20 == 0 and new_count > 0:
                        pd.DataFrame(cleaning_results).to_csv(
                            checkpoint_file, index=False)

        pd.DataFrame(cleaning_results).to_csv(checkpoint_file, index=False)

    # --- Combine all results ---
    all_results = existing_results + cleaning_results
    df_all = pd.DataFrame(all_results)

    # Save combined
    combined_path = os.path.join(RESULTS_DIR, "combined_results.csv")
    df_all.to_csv(combined_path, index=False)
    print(f"\nSaved combined results: {combined_path}")

    # --- Recovery summary ---
    if len(df_all) > 0:
        # Filter out errors
        if 'error' in df_all.columns:
            df_valid = df_all[df_all['error'].isna() | (df_all['error'] == '')]
        else:
            df_valid = df_all

        # Per-file recovery (paired comparison)
        print(f"\n{'=' * 70}")
        print(f"  Recovery Summary (AUC-ROC)")
        print(f"{'=' * 70}")

        baseline_mean = df_valid[df_valid['phase'] == 'baseline']['AUC_ROC'].mean()
        print(f"  Baseline mean AUC-ROC: {baseline_mean:.4f}\n")

        for corr_label in sorted(df_valid['corruption'].unique()):
            if corr_label == 'none':
                continue
            corrupted = df_valid[(df_valid['corruption'] == corr_label) &
                                 (df_valid['cleaning'] == 'none')]
            corr_mean = corrupted['AUC_ROC'].mean()
            drop = baseline_mean - corr_mean

            print(f"  {corr_label}:")
            print(f"    Corrupted:  {corr_mean:.4f} (drop: {drop:+.4f})")

            cleaned_methods = df_valid[(df_valid['corruption'] == corr_label) &
                                       (df_valid['cleaning'] != 'none')]
            for method in cleaned_methods['cleaning'].unique():
                m_df = cleaned_methods[cleaned_methods['cleaning'] == method]
                clean_mean = m_df['AUC_ROC'].mean()
                recovery = ((clean_mean - corr_mean) / drop * 100
                           if abs(drop) > 1e-6 else 0)
                print(f"    {method:30s}  {clean_mean:.4f}  "
                      f"recovery: {recovery:+.1f}%")
            print()

        # Save summary CSV
        summary_path = os.path.join(RESULTS_DIR, "summary.csv")
        summary_rows = []
        for corr_label in df_valid['corruption'].unique():
            if corr_label == 'none':
                continue
            for cleaning in df_valid[df_valid['corruption'] == corr_label]['cleaning'].unique():
                sub = df_valid[(df_valid['corruption'] == corr_label) &
                               (df_valid['cleaning'] == cleaning)]
                row = {'corruption': corr_label, 'cleaning': cleaning,
                       'n': len(sub)}
                for m in SELECTED_METRICS:
                    if m in sub.columns:
                        row[f'mean_{m}'] = round(sub[m].mean(), 4)
                        row[f'std_{m}'] = round(sub[m].std(), 4)
                summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\n[Done] Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
