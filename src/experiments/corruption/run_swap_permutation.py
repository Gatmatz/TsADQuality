"""
Swap Corruption — Segment Permutation (Literature Approach)

Splits the time series into N equal segments and randomly shuffles
their order, following the permutation augmentation literature.

Based on:
  - Um et al. (2017) — Permutation augmentation for wearable sensor data
  - Grover et al. (NeurIPS 2024) — Segment, Shuffle, and Stitch (S3)

Parameters:
  - n_segments: Number of equal segments to split the series into
                [2, 4, 8, 16, 24]

The entire series is permuted — values are rearranged but preserved
(no new values created, no values removed). Labels stay in their
original positions.

Key question: "How much temporal reordering can anomaly detectors
              tolerate before performance breaks down?"

Usage:
    python run_swap_permutation.py --models IForest
    python run_swap_permutation.py --test
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

from data_loader import load_tsb_dataframe, load_pretrained_ae

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "swap_permutation")

N_SEGMENTS_LIST = [2, 4, 8, 16, 24]
N_SEEDS = 3  # Multiple seeds since permutation is random

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]


def permute_segments(values, n_segments, rng):
    """Split values into n_segments equal parts and shuffle their order."""
    n = len(values)
    seg_len = n // n_segments
    remainder = n % n_segments

    # Build segment boundaries
    segments = []
    start = 0
    for i in range(n_segments):
        # Distribute remainder across first segments
        end = start + seg_len + (1 if i < remainder else 0)
        segments.append(values[start:end].copy())
        start = end

    # Shuffle until the order is actually different
    original_order = list(range(n_segments))
    perm = original_order.copy()
    for _ in range(100):
        rng.shuffle(perm)
        if perm != original_order:
            break

    # Reassemble in permuted order
    permuted = np.concatenate([segments[i] for i in perm])
    return permuted, perm


def run_model(model_name, X, scaled_data, sw, file_name=None):
    if model_name == 'IForest':
        clf = IForest(n_estimators=100, random_state=42); clf.fit(X); return clf.decision_scores_
    elif model_name == 'PCA':
        clf = PCA(n_components=10); clf.fit(X); return clf.decision_scores_
    elif model_name == 'MP':
        clf = MatrixProfile(window=sw); clf.fit(scaled_data); return clf.decision_scores_
    elif model_name == 'LOF':
        clf = LOF(n_neighbors=20); clf.fit(X); return clf.decision_scores_
    elif model_name == 'AE':
        clf, _ = load_pretrained_ae(file_name, project_root)
        clf.predict(scaled_data)
        return clf.decision_scores_
    else:
        raise ValueError(f"Unknown model: {model_name}")


def process_single_job(job_args):
    (file_path, n_segments, seed, model_names) = job_args

    file_name = os.path.basename(file_path)
    condition_name = f"nseg_{n_segments}"

    try:
        # 1. Load Data
        df, file_name = load_tsb_dataframe(file_path)
        labels = df['is_anomaly'].to_numpy('int')
        n = len(df)

        if n < n_segments * 2:
            return {'status': 'skipped', 'file': file_name,
                    'reason': f'Series too short ({n}) for {n_segments} segments'}

        # 2. Sliding window from clean data
        clean_scaled = StandardScaler().fit_transform(
            df['value'].to_numpy('float').reshape(-1, 1)
        ).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # 3. Permute segments
        rng = np.random.RandomState(seed)
        original_values = df['value'].to_numpy('float')
        permuted_values, perm_order = permute_segments(original_values, n_segments, rng)

        seg_len = n // n_segments

        # 4. Preprocessing on permuted data
        scaled_data = StandardScaler().fit_transform(
            permuted_values.reshape(-1, 1)
        ).flatten()
        X = Window(window=sliding_window).convert(scaled_data).to_numpy()

        # 5. Modeling & Evaluation
        results = []
        for model_name in model_names:
            try:
                score = run_model(model_name, X, scaled_data, sliding_window, file_name=file_name)
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

                if model_name == 'AE':
                    full_score = score
                else:
                    full_score = np.array(
                        [score[0]] * math.ceil((sliding_window - 1) / 2) +
                        list(score) +
                        [score[-1]] * ((sliding_window - 1) // 2)
                    )

                if np.isnan(full_score).any() or len(np.unique(full_score)) <= 1:
                    results.append({
                        'file': file_name, 'n_segments': n_segments,
                        'segment_length': seg_len,
                        'permutation': str(perm_order),
                        'seed': seed, 'model': model_name,
                        'condition': condition_name,
                        'error': "Invalid scores generated"
                    })
                    continue

                # Labels stay in original positions
                metrics = get_metrics(full_score, labels, metric="all", slidingWindow=sliding_window)

                row = {
                    'file': file_name, 'n_segments': n_segments,
                    'segment_length': seg_len,
                    'permutation': str(perm_order),
                    'seed': seed, 'model': model_name,
                    'condition': condition_name, 'error': None
                }
                for key in SELECTED_METRICS:
                    row[key] = round(metrics.get(key, 0.0), 4)
                results.append(row)

            except Exception as e:
                results.append({
                    'file': file_name, 'n_segments': n_segments,
                    'segment_length': seg_len,
                    'permutation': str(perm_order),
                    'seed': seed, 'model': model_name,
                    'condition': condition_name, 'error': str(e)
                })

        return {'status': 'success', 'results': results}

    except Exception as e:
        return {'status': 'error', 'file': file_name, 'error': traceback.format_exc()}


def compute_summary(df_results, output_path):
    df_success = df_results[df_results['error'].isnull()]

    if df_success.empty:
        print("No successful runs to summarize.")
        return

    grouped = df_success.groupby(['n_segments', 'model'])

    summary_data = []
    for name, group in grouped:
        nseg, model = name
        row = {
            'n_segments': nseg,
            'mean_segment_length': round(group['segment_length'].mean(), 1),
            'model': model, 'n_runs': len(group),
        }
        for col in SELECTED_METRICS:
            if col in group.columns:
                row[f'mean_{col}'] = round(group[col].mean(), 4)
                row[f'std_{col}'] = round(group[col].std(), 4)

        summary_data.append(row)

    pd.DataFrame(summary_data).to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--models', nargs='+', default=['AE'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(SUBSET_CSV):
        print(f"[ERROR] Subset CSV not found: {SUBSET_CSV}")
        return

    df_files_metadata = pd.read_csv(SUBSET_CSV)
    df_files = df_files_metadata['filepath'].tolist()

    n_segments_list = N_SEGMENTS_LIST
    n_seeds = N_SEEDS

    if args.test:
        df_files = df_files[:3]
        n_segments_list = [2, 8]
        n_seeds = 1
        print("!!! RUNNING IN TEST MODE !!!")

    n_workers = args.workers or os.cpu_count() or 1
    print(f"\n{'=' * 60}")
    print(f"  Swap — Segment Permutation (Literature Approach)")
    print(f"{'=' * 60}")
    print(f"  Um et al. (2017), S3 (Grover et al., NeurIPS 2024)")
    print(f"  Split into N segments, shuffle order")
    print(f"{'=' * 60}")
    print(f"N segments: {n_segments_list}")
    print(f"Seeds:      {n_seeds}")
    print(f"Datasets:   {len(df_files)}")
    print(f"Total conditions: {len(n_segments_list)} x {n_seeds} seeds")
    print(f"{'=' * 60}\n")

    # Checkpoint
    checkpoint_file = os.path.join(RESULTS_DIR, "checkpoint.csv")
    completed_jobs = set()
    all_results = []

    if os.path.exists(checkpoint_file):
        try:
            df_checkpoint = pd.read_csv(checkpoint_file)
            if 'error' in df_checkpoint.columns:
                df_success = df_checkpoint[df_checkpoint['error'].isna()]
            else:
                df_success = df_checkpoint
            all_results = df_success.to_dict('records')
            for r in all_results:
                job_id = f"{r['file']}_{r['condition']}_{r['seed']}_{r['model']}"
                completed_jobs.add(job_id)
            print(f"Loaded {len(all_results)} successful jobs from checkpoint.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint: {e}")

    # Build jobs
    jobs = []
    for file_path in df_files:
        file_name = os.path.basename(file_path)
        for nseg in n_segments_list:
            for seed in range(n_seeds):
                condition = f"nseg_{nseg}"
                needed_models = []
                for m in args.models:
                    job_id = f"{file_name}_{condition}_{seed}_{m}"
                    if job_id not in completed_jobs:
                        needed_models.append(m)
                if needed_models:
                    jobs.append((file_path, nseg, seed, needed_models))

    print(f"Jobs Scheduled: {len(jobs)}")

    if len(jobs) > 0:
        new_results_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_job, job): job for job in jobs}

            with tqdm(total=len(jobs), desc="permutation") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        all_results.extend(res['results'])
                        new_results_count += len(res['results'])

                    pbar.update(1)

                    if new_results_count >= 500:
                        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
                        new_results_count = 0

        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)

    if all_results:
        df_all = pd.DataFrame(all_results)
        compute_summary(df_all, os.path.join(RESULTS_DIR, "summary.csv"))

        for model_name in args.models:
            model_df = df_all[df_all['model'] == model_name]
            if model_df.empty:
                continue

            for nseg in model_df['n_segments'].unique():
                out_dir = os.path.join(RESULTS_DIR, model_name, f"nseg_{int(nseg)}")
                os.makedirs(out_dir, exist_ok=True)
                sub_df = model_df[model_df['n_segments'] == nseg]
                sub_df.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)

    print(f"\n[Done] All permutation experiments complete.")


if __name__ == "__main__":
    main()
