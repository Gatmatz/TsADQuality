"""
Missing Values — Burst Length Effect Experiment

Tests how burst LENGTH affects anomaly detection, keeping total missing
fraction constant. Uses burst_ratio (burst length as % of series length)
so that burst_length scales correctly with each series.

Parameters:
  - fraction:    Total % of data missing (0.01–0.20)
  - burst_ratio: Each burst as % of series length (0.001–0.05)
  - num_bursts = round(fraction / burst_ratio)
  - burst_length = round(burst_ratio * n)

Evaluation: True impact approach
  - Masking (remove NaN points)
  - Score=0 for lost anomalies (false negatives)
  - Exclude masked normal points from evaluation

Key question: "For the same amount of missing data,
              is it worse to have many short gaps or few long gaps?"

Usage:
    python run_missing_burst_length.py --models IForest
    python run_missing_burst_length.py --test
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

from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors
from data_loader import load_tsb_dataframe

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "missing_burst_length")

FRACTIONS = [0.01, 0.05, 0.10, 0.20]
BURST_RATIOS = [0.001, 0.005, 0.01, 0.05]   # each burst as % of series length
N_SEEDS = 1

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]


def run_model(model_name, X, scaled_data, sw):
    if model_name == 'IForest':
        clf = IForest(n_estimators=100, random_state=42); clf.fit(X); return clf.decision_scores_
    elif model_name == 'PCA':
        n_comp = min(10, X.shape[1] - 1) if X.shape[1] > 1 else 1
        clf = PCA(n_components=n_comp); clf.fit(X); return clf.decision_scores_
    elif model_name == 'MP':
        clf = MatrixProfile(window=sw); clf.fit(scaled_data); return clf.decision_scores_
    elif model_name == 'LOF':
        n_neigh = min(20, len(X) - 1)
        clf = LOF(n_neighbors=n_neigh); clf.fit(X); return clf.decision_scores_
    else:
        raise ValueError(f"Unknown model: {model_name}")


def process_single_job(job_args):
    (file_path, fraction, burst_ratio, seed, model_names) = job_args

    file_name = os.path.basename(file_path)
    condition_name = f"frac_{fraction}_br_{burst_ratio}"

    try:
        # 1. Load Data
        df, file_name = load_tsb_dataframe(file_path)
        labels = df['is_anomaly'].to_numpy('int')
        n = len(df)

        # 2. Compute burst parameters from ratio and series length
        burst_length = max(1, round(burst_ratio * n))
        num_bursts = max(1, round(fraction / burst_ratio))
        expected_missing = num_bursts * burst_length

        # 3. Sliding window from clean data
        clean_scaled = StandardScaler().fit_transform(
            df['value'].to_numpy('float').reshape(-1, 1)
        ).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # 4. Inject Burst Missing — global target
        corruptor = TSCorruptor(df, value_col='value', label_col='is_anomaly', seed=seed)
        ts_corruptor.injectors.inject_burst_missing(
            corruptor, num_bursts=num_bursts, burst_length=burst_length
        )

        df_corrupted = corruptor.get_corrupted_df()
        corrupted_data = df_corrupted['value'].to_numpy('float')

        nan_mask = np.isnan(corrupted_data)
        nan_count = int(nan_mask.sum())
        actual_missing_rate = nan_count / n

        # 5. Identify what was masked
        masked_normal = nan_mask & (labels == 0)
        masked_anomaly = nan_mask & (labels == 1)
        n_lost_anomalies = int(masked_anomaly.sum())

        # 6. Run model on non-NaN data
        model_data = corrupted_data[~nan_mask]
        n_kept = len(model_data)

        if n_kept < sliding_window + 10:
            return {'status': 'skipped', 'file': file_name,
                    'reason': f'Too few points after masking: {n_kept}'}

        sw = min(sliding_window, n_kept // 4)
        sw = max(sw, 10)

        scaled_data = StandardScaler().fit_transform(
            model_data.reshape(-1, 1)
        ).flatten()
        X = Window(window=sw).convert(scaled_data).to_numpy()

        # 7. Modeling & Evaluation
        results = []
        for model_name in model_names:
            try:
                score = run_model(model_name, X, scaled_data, sw)
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

                padded_score = np.array(
                    [score[0]] * math.ceil((sw - 1) / 2) +
                    list(score) +
                    [score[-1]] * ((sw - 1) // 2)
                )

                if np.isnan(padded_score).any() or len(np.unique(padded_score)) <= 1:
                    results.append({
                        'file': file_name, 'fraction': fraction,
                        'burst_ratio': burst_ratio, 'burst_length': burst_length,
                        'num_bursts': num_bursts,
                        'actual_missing_rate': actual_missing_rate,
                        'n_lost_anomalies': n_lost_anomalies,
                        'n_original': n, 'n_kept': n_kept,
                        'seed': seed, 'model': model_name,
                        'condition': condition_name, 'error': "Invalid scores generated"
                    })
                    continue

                # Reconstruct: model scores at kept positions, 0 at lost anomalies
                full_score = np.full(n, np.nan)
                full_score[~nan_mask] = padded_score
                full_score[masked_anomaly] = 0.0

                # Evaluate: exclude masked normal points
                eval_mask = ~masked_normal
                eval_scores = full_score[eval_mask]
                eval_labels = labels[eval_mask]

                metrics = get_metrics(eval_scores, eval_labels, metric="all", slidingWindow=sw)

                row = {
                    'file': file_name, 'fraction': fraction,
                    'burst_ratio': burst_ratio, 'burst_length': burst_length,
                    'num_bursts': num_bursts,
                    'actual_missing_rate': round(actual_missing_rate, 4),
                    'n_lost_anomalies': n_lost_anomalies,
                    'n_original': n, 'n_kept': n_kept,
                    'seed': seed, 'model': model_name,
                    'condition': condition_name, 'error': None
                }
                for key in SELECTED_METRICS:
                    row[key] = round(metrics.get(key, 0.0), 4)
                results.append(row)

            except Exception as e:
                results.append({
                    'file': file_name, 'fraction': fraction,
                    'burst_ratio': burst_ratio, 'burst_length': burst_length,
                    'num_bursts': num_bursts,
                    'actual_missing_rate': round(actual_missing_rate, 4),
                    'n_lost_anomalies': n_lost_anomalies,
                    'n_original': n, 'n_kept': n_kept,
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

    grouped = df_success.groupby(['fraction', 'burst_ratio', 'model'])

    summary_data = []
    for name, group in grouped:
        frac, br, model = name
        row = {
            'fraction': frac, 'burst_ratio': br,
            'mean_burst_length': round(group['burst_length'].mean(), 1),
            'mean_num_bursts': round(group['num_bursts'].mean(), 1),
            'model': model, 'n_runs': len(group),
            'mean_actual_missing_rate': round(group['actual_missing_rate'].mean(), 4),
            'mean_n_lost_anomalies': round(group['n_lost_anomalies'].mean(), 1)
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
    parser.add_argument('--models', nargs='+', default=['IForest'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(SUBSET_CSV):
        print(f"[ERROR] Subset CSV not found: {SUBSET_CSV}")
        return

    df_files_metadata = pd.read_csv(SUBSET_CSV)
    df_files = df_files_metadata['filepath'].tolist()

    fractions = FRACTIONS
    burst_ratios = BURST_RATIOS

    if args.test:
        df_files = df_files[:3]
        fractions = [0.05, 0.20]
        burst_ratios = [0.005, 0.05]
        print("!!! RUNNING IN TEST MODE !!!")

    # Count valid conditions (burst_ratio <= fraction)
    valid_conditions = [(f, br) for f in fractions for br in burst_ratios if br <= f]
    total_conditions = len(valid_conditions)

    n_workers = args.workers or os.cpu_count() or 1
    print(f"\n{'=' * 60}")
    print(f"  Missing Values — Burst Length Effect Experiment")
    print(f"{'=' * 60}")
    print(f"  burst_length = burst_ratio * n (scales with series)")
    print(f"  num_bursts = fraction / burst_ratio")
    print(f"  Evaluation: masking + score=0 for lost anomalies")
    print(f"{'=' * 60}")
    print(f"Fractions:    {fractions}")
    print(f"Burst ratios: {burst_ratios}")
    print(f"Valid conditions: {total_conditions}")
    print()
    print(f"{'fraction':>10} {'burst_ratio':>12} {'num_bursts':>12}")
    print("-" * 36)
    for f, br in valid_conditions:
        nb = max(1, round(f / br))
        print(f"{f:>10.2f} {br:>12.3f} {nb:>12}")
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
        for frac, br in valid_conditions:
            for seed in range(N_SEEDS):
                condition = f"frac_{frac}_br_{br}"
                needed_models = []
                for m in args.models:
                    job_id = f"{file_name}_{condition}_{seed}_{m}"
                    if job_id not in completed_jobs:
                        needed_models.append(m)
                if needed_models:
                    jobs.append((file_path, frac, br, seed, needed_models))

    print(f"Jobs Scheduled: {len(jobs)}")

    if len(jobs) > 0:
        new_results_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_job, job): job for job in jobs}

            with tqdm(total=len(jobs), desc="burst length") as pbar:
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

            for br in model_df['burst_ratio'].unique():
                out_dir = os.path.join(RESULTS_DIR, model_name, f"br_{br}")
                os.makedirs(out_dir, exist_ok=True)
                sub_df = model_df[model_df['burst_ratio'] == br]
                sub_df.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)

    print(f"\n[Done] All burst length experiments complete.")


if __name__ == "__main__":
    main()
