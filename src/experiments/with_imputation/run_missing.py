"""
Missing Values Robustness Experiment

Tests the impact of missing data (point + burst) on AD models.
Since AD models cannot handle NaN, imputation is applied before evaluation.
here we dont check the impact of the noise but the ,how strong are the imputation techn
Usage:
    python run_missing.py --models IForest PCA LOF MP
    python run_missing.py --test
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
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "missing")

# Parameter Grid
FRACTIONS = [0.01, 0.05, 0.10, 0.20]
NUM_BURSTS = [1, 3, 5, 10]     # dynamic burst: burst_length = fraction * n / num_bursts
IMPUTATION_TYPES = ['linear', 'ffill']
N_SEEDS = 1

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]


def apply_imputation(data, method):
    """Apply imputation to fill NaN values."""
    s = pd.Series(data)
    if method == 'linear':
        s = s.interpolate(method='linear').bfill().ffill()
    elif method == 'ffill':
        s = s.ffill().bfill()
    return s.to_numpy('float')


def process_single_job(job_args):
    (file_path, missing_type, fraction, num_bursts, imputation, seed, model_names) = job_args

    file_name = os.path.basename(file_path)
    if missing_type == 'point':
        condition_name = f"point_frac_{fraction}_{imputation}"
    else:
        condition_name = f"burst_frac_{fraction}_nb_{num_bursts}_{imputation}"

    try:
        # 1. Load Data
        df, file_name = load_tsb_dataframe(file_path)
        labels = df['is_anomaly'].to_numpy('int')

        # 2. Inject Missing Values
        corruptor = TSCorruptor(df, value_col='value', label_col='is_anomaly', seed=seed)

        if missing_type == 'point':
            ts_corruptor.injectors.inject_point_missing(corruptor, fraction=fraction)
        else:
            n = len(df)
            burst_length = max(1, int(fraction * n / num_bursts))
            ts_corruptor.injectors.inject_burst_missing(
                corruptor, num_bursts=num_bursts, burst_length=burst_length
            )

        df_corrupted = corruptor.get_corrupted_df()
        corrupted_data = df_corrupted['value'].to_numpy('float')

        nan_count = int(np.isnan(corrupted_data).sum())
        actual_missing_rate = nan_count / len(corrupted_data)

        # 3. Imputation
        data = apply_imputation(corrupted_data, imputation)

        if np.isnan(data).all():
            return {'status': 'error', 'file': file_name, 'error': 'All NaN after imputation'}

        # 4. Preprocessing
        clean_scaled = StandardScaler().fit_transform(
            df['value'].to_numpy('float').reshape(-1, 1)
        ).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        scaled_data = StandardScaler().fit_transform(
            data.reshape(-1, 1)
        ).flatten()
        X = Window(window=sliding_window).convert(scaled_data).to_numpy()

        # 5. Modeling & Evaluation
        results = []
        for model_name in model_names:
            try:
                score = None

                if model_name == 'IForest':
                    clf = IForest(n_estimators=100, random_state=42); clf.fit(X); score = clf.decision_scores_
                elif model_name == 'PCA':
                    clf = PCA(n_components=10); clf.fit(X); score = clf.decision_scores_
                elif model_name == 'MP':
                    clf = MatrixProfile(window=sliding_window); clf.fit(scaled_data); score = clf.decision_scores_
                elif model_name == 'LOF':
                    clf = LOF(n_neighbors=20); clf.fit(X); score = clf.decision_scores_
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                if score is not None:
                    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

                    # Symmetric padding (TSB-UAD convention)
                    full_score = np.array([score[0]] * math.ceil((sliding_window - 1) / 2) + list(score) + [score[-1]] * ((sliding_window - 1) // 2))

                    if np.isnan(full_score).any() or len(np.unique(full_score)) <= 1:
                        results.append({
                            'file': file_name, 'missing_type': missing_type,
                            'fraction': fraction, 'num_bursts': num_bursts,
                            'burst_length': burst_length if missing_type == 'burst' else 0,
                            'imputation': imputation, 'actual_missing_rate': actual_missing_rate,
                            'seed': seed, 'model': model_name, 'condition': condition_name,
                            'error': "Invalid scores generated"
                        })
                        continue

                    metrics = get_metrics(full_score, labels, metric="all", slidingWindow=sliding_window)

                    row = {
                        'file': file_name, 'missing_type': missing_type,
                        'fraction': fraction, 'num_bursts': num_bursts,
                        'burst_length': burst_length if missing_type == 'burst' else 0,
                        'imputation': imputation, 'actual_missing_rate': round(actual_missing_rate, 4),
                        'seed': seed, 'model': model_name,
                        'condition': condition_name, 'error': None
                    }
                    for key in SELECTED_METRICS:
                        row[key] = round(metrics.get(key, 0.0), 4)

                    results.append(row)

            except Exception as e:
                results.append({
                    'file': file_name, 'missing_type': missing_type,
                    'fraction': fraction, 'num_bursts': num_bursts,
                    'burst_length': burst_length if missing_type == 'burst' else 0,
                    'imputation': imputation, 'actual_missing_rate': round(actual_missing_rate, 4),
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

    grouped = df_success.groupby(['missing_type', 'fraction', 'num_bursts', 'imputation', 'model'])

    summary_data = []
    for name, group in grouped:
        mtype, frac, nb, imp, model = name
        row = {
            'missing_type': mtype, 'fraction': frac, 'num_bursts': nb,
            'mean_burst_length': round(group['burst_length'].mean(), 1) if mtype == 'burst' else 0,
            'imputation': imp, 'model': model, 'n_runs': len(group),
            'mean_actual_missing_rate': round(group['actual_missing_rate'].mean(), 4)
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
    num_bursts_list = NUM_BURSTS
    imputations = IMPUTATION_TYPES

    if args.test:
        df_files = df_files[:3]
        fractions = [0.05, 0.20]
        num_bursts_list = [3]
        imputations = ['linear']
        print("!!! RUNNING IN TEST MODE !!!")

    # Count conditions
    point_conditions = len(fractions) * len(imputations)
    burst_conditions = len(fractions) * len(num_bursts_list) * len(imputations)
    total_conditions = point_conditions + burst_conditions

    n_workers = args.workers or os.cpu_count() or 1
    print(f"\n{'=' * 60}")
    print(f"  Missing Values Robustness Experiment")
    print(f"{'=' * 60}")
    print(f"Point conditions: {point_conditions}")
    print(f"Burst conditions: {burst_conditions}")
    print(f"Total conditions: {total_conditions}")
    print(f"Imputation: {imputations}")
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
        for frac in fractions:
            for imp in imputations:
                for seed in range(N_SEEDS):
                    # Point missing
                    condition = f"point_frac_{frac}_{imp}"
                    needed_models = []
                    for m in args.models:
                        job_id = f"{file_name}_{condition}_{seed}_{m}"
                        if job_id not in completed_jobs:
                            needed_models.append(m)
                    if needed_models:
                        jobs.append((file_path, 'point', frac, 0, imp, seed, needed_models))

                    # Burst missing
                    for nb in num_bursts_list:
                        condition = f"burst_frac_{frac}_nb_{nb}_{imp}"
                        needed_models = []
                        for m in args.models:
                            job_id = f"{file_name}_{condition}_{seed}_{m}"
                            if job_id not in completed_jobs:
                                needed_models.append(m)
                        if needed_models:
                            jobs.append((file_path, 'burst', frac, nb, imp, seed, needed_models))

    print(f"Jobs Scheduled: {len(jobs)}")

    if len(jobs) > 0:
        new_results_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_job, job): job for job in jobs}

            with tqdm(total=len(jobs), desc="missing") as pbar:
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

        # Global summary
        compute_summary(df_all, os.path.join(RESULTS_DIR, "summary.csv"))

        # Save per model -> point/burst
        for model_name in args.models:
            model_df = df_all[df_all['model'] == model_name]
            if model_df.empty:
                continue

            # Point missing
            point_df = model_df[model_df['missing_type'] == 'point']
            if not point_df.empty:
                for imp in point_df['imputation'].unique():
                    out_dir = os.path.join(RESULTS_DIR, model_name, "point", imp)
                    os.makedirs(out_dir, exist_ok=True)
                    imp_df = point_df[point_df['imputation'] == imp]
                    imp_df.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)

            # Burst missing
            burst_df = model_df[model_df['missing_type'] == 'burst']
            for nb in burst_df['num_bursts'].unique():
                for imp in burst_df['imputation'].unique():
                    out_dir = os.path.join(RESULTS_DIR, model_name, "burst", f"nb_{int(nb)}", imp)
                    os.makedirs(out_dir, exist_ok=True)
                    sub_df = burst_df[(burst_df['num_bursts'] == nb) & (burst_df['imputation'] == imp)]
                    sub_df.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)

    print(f"\n[Done] All missing value experiments complete.")


if __name__ == "__main__":
    main()
