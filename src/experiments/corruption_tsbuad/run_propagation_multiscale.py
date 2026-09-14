"""
Multi-Scale Corruption Propagation Experiment

Measures how localized corruption at different scales (5%, 10%, 20%, 30%)
affects anomaly scores at distant points. Reveals whether there is a
threshold at which algorithms lose their robustness.

Conditions:
  - 3 corruption ratios: 5%, 10%, 20%
  - 3 corruption types: noise, spikes, freeze
  - 3 severity levels each
  - 4 models: IForest, LOF, MP, AE

Optimization: jobs grouped by (file, model) so clean scores are computed once.

Usage:
    python run_propagation_multiscale.py                   # full run
    python run_propagation_multiscale.py --test            # 5 files only
    python run_propagation_multiscale.py --models IForest  # select models
    python run_propagation_multiscale.py --workers 4       # parallel
"""
import os
import sys
import math
import argparse

# Thread contention fix — MUST be before numpy/sklearn imports
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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
from TSB_UAD.models.matrix_profile import MatrixProfile
from TSB_UAD.models.lof import LOF
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics
from data_loader import load_tsb_file, remap_filepath, load_pretrained_ae

# ==========================================
# CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
OUTPUT_DIR = os.path.join(project_root, "results", "experiments", "propagation_multiscale")
CORRUPTION_RATIOS = [0.05, 0.10, 0.20]  # 0.30 excluded (far zone too small)
SEED = 42

# Metrics to compute
SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]
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
    'freeze': {
        'low':  {'num_blocks': 1},
        'med':  {'num_blocks': 3},
        'high': {'num_blocks': 5},
    },
}

ALL_MODELS = ['IForest', 'LOF', 'MP', 'AE']

# Build flat list of conditions to iterate inside each job
CONDITIONS = []
for cr in CORRUPTION_RATIOS:
    for ctype, severities in CORRUPTIONS.items():
        for sev, params in severities.items():
            CONDITIONS.append((cr, ctype, sev, params))


# ==========================================
# CORRUPTION INJECTION (localized)
# ==========================================

def inject_localized_noise(data, start, end, snr_db, rng):
    corrupted = data.copy()
    segment = data[start:end]
    signal_power = max(np.mean(segment ** 2), 1e-10)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    corrupted[start:end] = segment + rng.normal(0, np.sqrt(noise_power), size=end - start)
    return corrupted


def inject_localized_spikes(data, start, end, magnitude, rng):
    corrupted = data.copy()
    n_spikes = max(1, (end - start) // 10)
    spike_indices = rng.choice(range(start, end), size=n_spikes, replace=False)
    std = max(np.std(data), 1.0)
    for idx in spike_indices:
        corrupted[idx] = data[idx] + rng.choice([-1, 1]) * magnitude * std
    return corrupted


def inject_localized_freeze(data, start, end, num_blocks, rng):
    """Freeze signal at random positions within [start, end]."""
    corrupted = data.copy()
    segment_len = end - start
    block_len = max(1, segment_len // (num_blocks * 2))  # each block ~half the available space per block

    # Pick random start positions for freeze blocks
    possible_starts = list(range(start, end - block_len))
    if len(possible_starts) < num_blocks:
        possible_starts = list(range(start, end))
    block_starts = rng.choice(possible_starts, size=min(num_blocks, len(possible_starts)), replace=False)

    for bs in block_starts:
        stuck_value = data[bs]
        for i in range(block_len):
            if bs + i < end:
                corrupted[bs + i] = stuck_value
    return corrupted


# ==========================================
# MODEL RUNNER
# ==========================================

def run_model_score(model_name, scaled_data, sliding_window, clf_ae=None):
    """Run a single model and return full-length anomaly scores.

    For AE, pass a pre-loaded clf_ae to avoid reloading the model.
    """
    if model_name == 'AE':
        clf_ae.predict(scaled_data)
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            clf_ae.decision_scores_.reshape(-1, 1)).ravel()
        return score  # already full length

    if model_name == 'MP':
        clf = MatrixProfile(window=sliding_window)
        clf.fit(scaled_data)
        score = clf.decision_scores_
    else:
        X = Window(window=sliding_window).convert(scaled_data).to_numpy()
        if model_name == 'IForest':
            clf = IForest(n_estimators=100, random_state=42)
        elif model_name == 'LOF':
            clf = LOF(n_neighbors=20)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        clf.fit(X)
        score = clf.decision_scores_

    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        score.reshape(-1, 1)).ravel()

    # Pad to full length
    full_score = np.array(
        [score[0]] * math.ceil((sliding_window - 1) / 2)
        + list(score)
        + [score[-1]] * ((sliding_window - 1) // 2)
    )
    if len(full_score) > len(scaled_data):
        full_score = full_score[:len(scaled_data)]
    elif len(full_score) < len(scaled_data):
        full_score = np.pad(full_score, (0, len(scaled_data) - len(full_score)), mode='edge')
    return full_score


# ==========================================
# PROPAGATION ANALYSIS
# ==========================================

def compute_propagation(scores_clean, scores_corrupted, corrupt_start, corrupt_end, n_total):
    score_diff = np.abs(scores_corrupted - scores_clean)
    max_diff = np.max(score_diff)
    if max_diff < 1e-10:
        return {
            'mean_diff_corruption_zone': 0, 'mean_diff_near': 0,
            'mean_diff_mid': 0, 'mean_diff_far': 0,
            'propagation_ratio': 1.0, 'max_diff': 0,
        }

    score_diff_norm = score_diff / max_diff
    corrupt_size = corrupt_end - corrupt_start
    near_dist = int(0.10 * n_total)
    mid_dist = int(0.30 * n_total)

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
# CORE WORKER — one (file, model) job
# ==========================================

def process_file_model(job_args):
    """Process all conditions for one (file, model) pair.

    Clean scores computed once, then reused for all 27 corrupted conditions.
    Returns list of result rows.
    """
    file_path, folder, model_name, skip_keys = job_args

    file_name = os.path.basename(file_path)
    rows = []

    try:
        data, label, canonical_name = load_tsb_file(file_path)
        n = len(data)

        scaled_clean = StandardScaler().fit_transform(data.reshape(-1, 1)).flatten()
        sliding_window = max(int(find_length(scaled_clean)), 10)

        # Load AE once if needed
        clf_ae = None
        if model_name == 'AE':
            clf_ae, _ = load_pretrained_ae(canonical_name, project_root)

        # Clean scores — computed ONCE
        scores_clean = run_model_score(model_name, scaled_clean, sliding_window, clf_ae)
        
        # Clean metrics — computed ONCE
        metrics_clean = get_metrics(scores_clean, label, metric="all", slidingWindow=sliding_window)

        # Iterate over all (ratio, type, severity) conditions
        for cr, ctype, sev, params in CONDITIONS:
            cond_key = f"{canonical_name}_{cr}_{ctype}_{sev}_{model_name}"
            if cond_key in skip_keys:
                continue

            rng = np.random.RandomState(SEED)

            corrupt_size = int(cr * n)
            corrupt_start = (n - corrupt_size) // 2
            corrupt_end = corrupt_start + corrupt_size

            # Inject corruption + run model
            if ctype == 'noise':
                corrupted_raw = inject_localized_noise(
                    data, corrupt_start, corrupt_end, params['snr_db'], rng)
                scaled_corrupted = StandardScaler().fit_transform(
                    corrupted_raw.reshape(-1, 1)).flatten()
                scores_corrupted = run_model_score(model_name, scaled_corrupted, sliding_window, clf_ae)
            elif ctype == 'spikes':
                corrupted_raw = inject_localized_spikes(
                    data, corrupt_start, corrupt_end, params['magnitude'], rng)
                scaled_corrupted = StandardScaler().fit_transform(
                    corrupted_raw.reshape(-1, 1)).flatten()
                scores_corrupted = run_model_score(model_name, scaled_corrupted, sliding_window, clf_ae)
            elif ctype == 'freeze':
                corrupted_raw = inject_localized_freeze(
                    data, corrupt_start, corrupt_end, params['num_blocks'], rng)
                scaled_corrupted = StandardScaler().fit_transform(
                    corrupted_raw.reshape(-1, 1)).flatten()
                scores_corrupted = run_model_score(model_name, scaled_corrupted, sliding_window, clf_ae)
            else:
                continue

            # Propagation
            props = compute_propagation(scores_clean, scores_corrupted,
                                        corrupt_start, corrupt_end, n)

            # AUC metrics
            metrics_corrupted = get_metrics(scores_corrupted, label, metric="all", slidingWindow=sliding_window)

            row = {
                'file': canonical_name, 'folder': folder, 'length': n,
                'corruption_ratio': cr, 'corruption_type': ctype,
                'severity': sev, 'model': model_name,
                'sliding_window': sliding_window,
                'corrupt_start': corrupt_start, 'corrupt_end': corrupt_end,
                'error': None, **props,
            }
            
            # Add clean metrics, corrupted metrics, and drops
            for metric in SELECTED_METRICS:
                clean_val = metrics_clean.get(metric, 0.0)
                corr_val = metrics_corrupted.get(metric, 0.0)
                row[f'{metric}_clean'] = round(clean_val, 4)
                row[f'{metric}_corrupted'] = round(corr_val, 4)
                # Drop = corrupted - clean (negative means performance decreased)
                row[f'{metric}_drop'] = round(corr_val - clean_val, 4)
            
            for k, v in params.items():
                row[f'param_{k}'] = v
            rows.append(row)

        return {'status': 'success', 'rows': rows,
                'job_id': f"{canonical_name}_{model_name}",
                'n_conditions': len(rows)}

    except Exception as e:
        return {'status': 'error', 'rows': [],
                'job_id': f"{file_name}_{model_name}",
                'error': str(e)[:300]}


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Scale Corruption Propagation Experiment')
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
        indices = np.linspace(0, len(file_paths) - 1, 5, dtype=int)
        file_paths = [file_paths[i] for i in indices]
        folders = [folders[i] for i in indices]
        print("!!! TEST MODE !!!")

    # Load checkpoint
    checkpoint_file = os.path.join(OUTPUT_DIR, "checkpoint.csv")
    completed_keys = set()

    if os.path.exists(checkpoint_file):
        try:
            df_ckpt = pd.read_csv(checkpoint_file)
            df_ok = df_ckpt[df_ckpt['error'].isna()] if 'error' in df_ckpt.columns else df_ckpt
            for _, r in df_ok.iterrows():
                key = f"{r['file']}_{r['corruption_ratio']}_{r['corruption_type']}_{r['severity']}_{r['model']}"
                completed_keys.add(key)
            print(f"[Resume] {len(completed_keys)} conditions already done")
        except Exception as e:
            print(f"Warning: checkpoint error: {e}")

    # Build (file, model) jobs — each processes all conditions internally
    n_conditions_per_job = len(CONDITIONS)  # 3 ratios x 3 types x 3 severities = 27
    jobs = []
    n_conditions_to_run = 0
    for fp, folder in zip(file_paths, folders):
        for model_name in args.models:
            fname = os.path.basename(fp)
            remaining = 0
            for cr, ctype, sev, _ in CONDITIONS:
                key = f"{fname}_{cr}_{ctype}_{sev}_{model_name}"
                if key not in completed_keys:
                    remaining += 1
            if remaining > 0:
                jobs.append((fp, folder, model_name, completed_keys))
                n_conditions_to_run += remaining

    n_total_conditions = len(file_paths) * len(args.models) * n_conditions_per_job

    print(f"\n{'=' * 60}")
    print(f"  Multi-Scale Corruption Propagation Experiment")
    print(f"{'=' * 60}")
    print(f"  Files:          {len(file_paths)}")
    print(f"  Models:         {args.models}")
    print(f"  Corruptions:    {list(CORRUPTIONS.keys())}")
    print(f"  Severities:     3 per corruption (low/med/high)")
    print(f"  Scales:         {CORRUPTION_RATIOS}")
    print(f"  Conditions/job: {n_conditions_per_job}")
    print(f"  Jobs (file×model): {len(jobs)}")
    print(f"  Total conditions:  {n_total_conditions}")
    print(f"  Already done:      {len(completed_keys)}")
    print(f"  Workers:           {args.workers}")
    print(f"  Output:            {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    if not jobs:
        print("All jobs done!")
    else:
        # Load existing checkpoint rows
        all_results = []
        if os.path.exists(checkpoint_file):
            try:
                all_results = pd.read_csv(checkpoint_file).to_dict('records')
            except:
                pass

        n_conditions_already_done = len(completed_keys)
        n_new = 0
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_file_model, job): job for job in jobs}

            with tqdm(total=n_total_conditions, initial=n_conditions_already_done,
                      desc="Propagation (conditions)") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        all_results.extend(res['rows'])
                        n_new += res['n_conditions']
                        pbar.update(res['n_conditions'])
                    elif res['status'] == 'error':
                        tqdm.write(f"  Error: {res['job_id']}: {res.get('error', '?')[:80]}")
                        pbar.update(n_conditions_per_job)  # Still count as processed

                    # Checkpoint every ~20 jobs
                    if n_new > 0 and n_new % (20 * n_conditions_per_job) < n_conditions_per_job:
                        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)

        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
        print(f"\nCheckpoint saved: {len(all_results)} results ({n_new} new)")

    # ==========================================
    # SUMMARY
    # ==========================================
    if os.path.exists(checkpoint_file):
        df_ok = pd.read_csv(checkpoint_file)
        df_ok = df_ok[df_ok['error'].isna()] if 'error' in df_ok.columns else df_ok

    if os.path.exists(checkpoint_file) and not df_ok.empty:
        results_path = os.path.join(OUTPUT_DIR, "propagation_results.csv")
        df_ok.to_csv(results_path, index=False)

        # Summary
        print(f"\n{'=' * 60}")
        print(f"  PROPAGATION RATIO (mean)")
        print(f"{'=' * 60}")
        summary = df_ok.groupby(['corruption_ratio', 'corruption_type', 'severity', 'model']).agg(
            mean_ratio=('propagation_ratio', 'mean'),
            std_ratio=('propagation_ratio', 'std'),
            mean_diff_far=('mean_diff_far', 'mean'),
            n=('file', 'count'),
        ).round(3)
        print(summary.to_string())

        summary_path = os.path.join(OUTPUT_DIR, "propagation_summary.csv")
        summary.to_csv(summary_path)

        # Compact
        print(f"\n{'=' * 60}")
        print(f"  COMPACT: mean propagation ratio")
        print(f"{'=' * 60}")
        compact = df_ok.groupby(['corruption_ratio', 'corruption_type', 'model'])['propagation_ratio'].mean().round(2)
        compact_pivot = compact.unstack('model')
        print(compact_pivot.to_string())

        compact_path = os.path.join(OUTPUT_DIR, "propagation_compact.csv")
        compact_pivot.to_csv(compact_path)

        # Zone decay
        print(f"\n{'=' * 60}")
        print(f"  ZONE DECAY (normalized score diff)")
        print(f"{'=' * 60}")
        zone_cols = ['mean_diff_corruption_zone', 'mean_diff_near', 'mean_diff_mid', 'mean_diff_far']
        zone_summary = df_ok.groupby(['corruption_ratio', 'corruption_type', 'model'])[zone_cols].mean().round(3)
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
