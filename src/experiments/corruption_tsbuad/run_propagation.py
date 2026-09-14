"""
Corruption Propagation Experiment — Full run on 141 files.

Measures how localized corruption (10% of series, centered) affects
anomaly scores at distant points. Reveals whether models propagate
corruption effects locally or globally.

Conditions:
  - 3 corruption types: noise, spikes, missing
  - 3 severity levels each
  - 5 models: IForest, PCA, LOF, MP, AE

Usage:
    python run_propagation.py                              # full run
    python run_propagation.py --test                       # 5 files only
    python run_propagation.py --models IForest PCA         # select models
    python run_propagation.py --workers 4                  # parallel
"""
import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
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

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.pca import PCA
from TSB_UAD.models.matrix_profile import MatrixProfile
from TSB_UAD.models.lof import LOF
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from data_loader import load_tsb_file, remap_filepath, load_pretrained_ae

# ==========================================
# CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
OUTPUT_DIR = os.path.join(project_root, "results", "experiments", "propagation")
CORRUPTION_RATIO = 0.10  # corrupt 10% of the time series
SEED = 42
SCORE_DIFF_THRESHOLD = 0.05  # 5% of max diff to define "impacted"

# Corruption severity grid
CORRUPTIONS = {
    'noise': {
        'low':  {'snr_db': 20},
        'med':  {'snr_db': 10},
        'high': {'snr_db': 5},
    },
    'spikes': {
        'low':  {'magnitude': 3},
        'med':  {'magnitude': 6},
        'high': {'magnitude': 10},
    },
    'missing': {
        'low':  {'fraction': 0.3},   # 30% of the corruption zone becomes NaN
        'med':  {'fraction': 0.5},
        'high': {'fraction': 0.8},
    },
}

ALL_MODELS = ['IForest',  'LOF', 'MP', 'AE']


# ==========================================
# CORRUPTION INJECTION (localized)
# ==========================================

def inject_localized_noise(data, start, end, snr_db, rng):
    """Inject white noise into data[start:end] at given SNR."""
    corrupted = data.copy()
    segment = data[start:end]
    signal_power = np.mean(segment ** 2)
    if signal_power < 1e-10:
        signal_power = 1e-10
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), size=end - start)
    corrupted[start:end] = segment + noise
    return corrupted


def inject_localized_spikes(data, start, end, magnitude, rng):
    """Inject random spikes into data[start:end]."""
    corrupted = data.copy()
    n_spikes = max(1, (end - start) // 10)  # ~10% of segment are spikes
    spike_indices = rng.choice(range(start, end), size=n_spikes, replace=False)
    std = np.std(data)
    if std < 1e-10:
        std = 1.0
    for idx in spike_indices:
        sign = rng.choice([-1, 1])
        corrupted[idx] = data[idx] + sign * magnitude * std
    return corrupted


def inject_localized_missing(data, start, end, fraction, rng):
    """Set a fraction of data[start:end] to NaN."""
    corrupted = data.copy()
    segment_len = end - start
    n_missing = max(1, int(fraction * segment_len))
    missing_indices = rng.choice(range(start, end), size=n_missing, replace=False)
    corrupted[missing_indices] = np.nan
    return corrupted


# ==========================================
# MODEL RUNNER
# ==========================================

def run_model(model_name, scaled_data, sliding_window, file_name=None):
    """Run a single model and return full-length anomaly scores."""
    if model_name == 'AE':
        clf, meta = load_pretrained_ae(file_name, project_root)
        clf.predict(scaled_data)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            score.reshape(-1, 1)).ravel()
        return score  # already full length

    if model_name == 'MP':
        clf = MatrixProfile(window=sliding_window)
        clf.fit(scaled_data)
        score = clf.decision_scores_
    else:
        X = Window(window=sliding_window).convert(scaled_data).to_numpy()
        if model_name == 'IForest':
            clf = IForest(n_estimators=100, random_state=42)
        elif model_name == 'PCA':
            clf = PCA(n_components=min(10, X.shape[1] - 1))
        elif model_name == 'LOF':
            clf = LOF(n_neighbors=20)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        clf.fit(X)
        score = clf.decision_scores_

    # MinMaxScale
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        score.reshape(-1, 1)).ravel()

    # Pad to full length (TSB-UAD convention)
    full_score = np.array(
        [score[0]] * math.ceil((sliding_window - 1) / 2)
        + list(score)
        + [score[-1]] * ((sliding_window - 1) // 2)
    )

    if len(full_score) > len(scaled_data):
        full_score = full_score[:len(scaled_data)]
    elif len(full_score) < len(scaled_data):
        full_score = np.pad(full_score, (0, len(scaled_data) - len(full_score)),
                            mode='edge')

    return full_score


# ==========================================
# PROPAGATION ANALYSIS
# ==========================================

def compute_propagation(scores_clean, scores_corrupted, corrupt_start, corrupt_end, n_total):
    """Compute propagation metrics from score differences."""
    score_diff = np.abs(scores_corrupted - scores_clean)

    max_diff = np.max(score_diff)
    if max_diff < 1e-10:
        return {
            'mean_diff_corruption_zone': 0,
            'mean_diff_near': 0,
            'mean_diff_mid': 0,
            'mean_diff_far': 0,
            'propagation_ratio': 1.0,
            'max_diff': 0,
        }

    score_diff_norm = score_diff / max_diff
    corrupt_size = corrupt_end - corrupt_start
    near_dist = int(0.10 * n_total)
    mid_dist = int(0.30 * n_total)

    # Define zones
    corruption_zone = score_diff_norm[corrupt_start:corrupt_end]

    near_left_start = max(0, corrupt_start - near_dist)
    near_right_end = min(n_total, corrupt_end + near_dist)
    near_zone = np.concatenate([
        score_diff_norm[near_left_start:corrupt_start],
        score_diff_norm[corrupt_end:near_right_end]
    ])

    mid_left_start = max(0, corrupt_start - mid_dist)
    mid_right_end = min(n_total, corrupt_end + mid_dist)
    mid_zone = np.concatenate([
        score_diff_norm[mid_left_start:near_left_start],
        score_diff_norm[near_right_end:mid_right_end]
    ])

    far_zone = np.concatenate([
        score_diff_norm[:mid_left_start],
        score_diff_norm[mid_right_end:]
    ])

    impacted = np.sum(score_diff_norm > SCORE_DIFF_THRESHOLD)
    propagation_ratio = impacted / corrupt_size if corrupt_size > 0 else 0

    return {
        'mean_diff_corruption_zone': float(np.mean(corruption_zone)) if len(corruption_zone) > 0 else 0,
        'mean_diff_near': float(np.mean(near_zone)) if len(near_zone) > 0 else 0,
        'mean_diff_mid': float(np.mean(mid_zone)) if len(mid_zone) > 0 else 0,
        'mean_diff_far': float(np.mean(far_zone)) if len(far_zone) > 0 else 0,
        'propagation_ratio': float(propagation_ratio),
        'max_diff': float(max_diff),
    }


# ==========================================
# CORE WORKER
# ==========================================

def process_single_job(job_args):
    """Process one (file, corruption_type, severity, model) job."""
    file_path, folder, corruption_type, severity, severity_params, model_name = job_args

    file_name = os.path.basename(file_path)
    rng = np.random.RandomState(SEED)

    try:
        data, label, canonical_name = load_tsb_file(file_path)
        n = len(data)

        scaled_clean = StandardScaler().fit_transform(data.reshape(-1, 1)).flatten()
        sliding_window = max(int(find_length(scaled_clean)), 10)

        # Corruption segment: centered at midpoint
        corrupt_size = int(CORRUPTION_RATIO * n)
        corrupt_start = (n - corrupt_size) // 2
        corrupt_end = corrupt_start + corrupt_size

        # Clean scores
        scores_clean = run_model(model_name, scaled_clean, sliding_window, canonical_name)

        # Inject localized corruption
        if corruption_type == 'noise':
            corrupted_raw = inject_localized_noise(
                data, corrupt_start, corrupt_end, severity_params['snr_db'], rng)
            scaled_corrupted = StandardScaler().fit_transform(
                corrupted_raw.reshape(-1, 1)).flatten()
        elif corruption_type == 'spikes':
            corrupted_raw = inject_localized_spikes(
                data, corrupt_start, corrupt_end, severity_params['magnitude'], rng)
            scaled_corrupted = StandardScaler().fit_transform(
                corrupted_raw.reshape(-1, 1)).flatten()
        elif corruption_type == 'missing':
            corrupted_raw = inject_localized_missing(
                data, corrupt_start, corrupt_end, severity_params['fraction'], rng)
            # Interpolate NaNs for model input
            valid = ~np.isnan(corrupted_raw)
            if valid.sum() < sliding_window + 10:
                return {'status': 'skipped', 'file': canonical_name,
                        'reason': 'Too few valid points after missing injection'}
            corrupted_filled = corrupted_raw.copy()
            corrupted_filled[~valid] = np.interp(
                np.where(~valid)[0], np.where(valid)[0], corrupted_raw[valid])
            scaled_corrupted = StandardScaler().fit_transform(
                corrupted_filled.reshape(-1, 1)).flatten()
        else:
            return {'status': 'error', 'file': canonical_name,
                    'error': f'Unknown corruption: {corruption_type}'}

        # Corrupted scores
        scores_corrupted = run_model(model_name, scaled_corrupted, sliding_window, canonical_name)

        # Propagation analysis
        props = compute_propagation(
            scores_clean, scores_corrupted,
            corrupt_start, corrupt_end, n
        )

        row = {
            'file': canonical_name,
            'folder': folder,
            'length': n,
            'corruption_type': corruption_type,
            'severity': severity,
            'model': model_name,
            'sliding_window': sliding_window,
            'corrupt_start': corrupt_start,
            'corrupt_end': corrupt_end,
            'error': None,
            **props,
        }

        # Add severity params
        for k, v in severity_params.items():
            row[f'param_{k}'] = v

        return {'status': 'success', 'result': row}

    except Exception as e:
        return {'status': 'error', 'file': file_name,
                'error': str(e)[:300]}


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Corruption Propagation Experiment')
    parser.add_argument('--test', action='store_true', help='5 files only')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help='Models to evaluate')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_meta = pd.read_csv(SUBSET_CSV)
    file_paths = df_meta['filepath'].tolist()
    folders = df_meta['folder'].tolist()
    file_paths = [remap_filepath(fp, project_root) for fp in file_paths]

    if args.test:
        # Pick 5 spread across folders
        indices = np.linspace(0, len(file_paths) - 1, 5, dtype=int)
        file_paths = [file_paths[i] for i in indices]
        folders = [folders[i] for i in indices]
        print("!!! TEST MODE !!!")

    # Build jobs
    checkpoint_file = os.path.join(OUTPUT_DIR, "checkpoint.csv")
    completed_jobs = set()

    if os.path.exists(checkpoint_file):
        try:
            df_ckpt = pd.read_csv(checkpoint_file)
            df_ok = df_ckpt[df_ckpt['error'].isna()] if 'error' in df_ckpt.columns else df_ckpt
            for _, r in df_ok.iterrows():
                job_id = f"{r['file']}_{r['corruption_type']}_{r['severity']}_{r['model']}"
                completed_jobs.add(job_id)
            print(f"[Resume] {len(completed_jobs)} jobs already done")
        except Exception as e:
            print(f"Warning: checkpoint error: {e}")

    jobs = []
    for fp, folder in zip(file_paths, folders):
        fname = os.path.basename(fp)
        for ctype, severities in CORRUPTIONS.items():
            for sev, params in severities.items():
                for model_name in args.models:
                    job_id = f"{fname}_{ctype}_{sev}_{model_name}"
                    if job_id not in completed_jobs:
                        jobs.append((fp, folder, ctype, sev, params, model_name))

    n_total_jobs = len(file_paths) * len(CORRUPTIONS) * 3 * len(args.models)

    print(f"\n{'=' * 60}")
    print(f"  Corruption Propagation Experiment")
    print(f"{'=' * 60}")
    print(f"  Files:       {len(file_paths)}")
    print(f"  Models:      {args.models}")
    print(f"  Corruptions: {list(CORRUPTIONS.keys())}")
    print(f"  Severities:  3 per corruption (low/med/high)")
    print(f"  Total jobs:  {n_total_jobs}")
    print(f"  Already done:{len(completed_jobs)}")
    print(f"  To run:      {len(jobs)}")
    print(f"  Workers:     {args.workers}")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    if not jobs:
        print("All jobs done!")
    else:
        # Load existing results from checkpoint
        all_results = []
        if os.path.exists(checkpoint_file):
            try:
                all_results = pd.read_csv(checkpoint_file).to_dict('records')
            except:
                pass

        new_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_job, job): job for job in jobs}

            with tqdm(total=len(jobs), desc="Propagation") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        all_results.append(res['result'])
                        new_count += 1
                    elif res['status'] == 'error':
                        all_results.append({
                            'file': res.get('file', 'unknown'),
                            'error': res.get('error', 'unknown'),
                        })

                    pbar.update(1)

                    # Checkpoint every 200 results
                    if new_count % 200 == 0 and new_count > 0:
                        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)

        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
        print(f"\nCheckpoint saved: {len(all_results)} results")

    # Load all results for summary
    if os.path.exists(checkpoint_file):
        df = pd.read_csv(checkpoint_file)
        df_ok = df[df['error'].isna()] if 'error' in df.columns else df

        if not df_ok.empty:
            # Save clean results
            results_path = os.path.join(OUTPUT_DIR, "propagation_results.csv")
            df_ok.to_csv(results_path, index=False)

            # Summary: corruption_type x severity x model -> propagation metrics
            print(f"\n{'=' * 60}")
            print(f"  PROPAGATION RATIO (mean)")
            print(f"{'=' * 60}")
            summary = df_ok.groupby(['corruption_type', 'severity', 'model']).agg(
                mean_ratio=('propagation_ratio', 'mean'),
                std_ratio=('propagation_ratio', 'std'),
                mean_diff_far=('mean_diff_far', 'mean'),
                n=('file', 'count'),
            ).round(3)
            print(summary.to_string())

            summary_path = os.path.join(OUTPUT_DIR, "propagation_summary.csv")
            summary.to_csv(summary_path)

            # Compact table: model x corruption_type (averaged over severities)
            print(f"\n{'=' * 60}")
            print(f"  COMPACT: mean propagation ratio")
            print(f"{'=' * 60}")
            compact = df_ok.groupby(['corruption_type', 'model'])['propagation_ratio'].mean().round(2)
            compact_pivot = compact.unstack('model')
            print(compact_pivot.to_string())

            compact_path = os.path.join(OUTPUT_DIR, "propagation_compact.csv")
            compact_pivot.to_csv(compact_path)

            # Zone decay table
            print(f"\n{'=' * 60}")
            print(f"  ZONE DECAY (normalized score diff)")
            print(f"{'=' * 60}")
            zone_cols = ['mean_diff_corruption_zone', 'mean_diff_near', 'mean_diff_mid', 'mean_diff_far']
            zone_summary = df_ok.groupby(['corruption_type', 'model'])[zone_cols].mean().round(3)
            print(zone_summary.to_string())

            zone_path = os.path.join(OUTPUT_DIR, "propagation_zones.csv")
            zone_summary.to_csv(zone_path)

            print(f"\nResults: {results_path}")
            print(f"Summary: {summary_path}")
            print(f"Compact: {compact_path}")
            print(f"Zones:   {zone_path}")

    print(f"\n[Done]")


if __name__ == "__main__":
    main()
