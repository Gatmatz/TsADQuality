"""
Recovery for: Missing Values — True Impact (MCAR point + burst)

Same corruption as run_missing_true_impact.py, but instead of true impact
evaluation, applies imputation and re-runs the model.

Compare recovered_AUC with corrupted AUC from:
  results/experiments/missing_true_impact/checkpoint.csv

Usage:
    python run_recovery_missing.py --models IForest
    python run_recovery_missing.py --test
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
# MUST MATCH run_missing_true_impact.py
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "recovery_missing")
CORRUPTION_CHECKPOINT = os.path.join(project_root, "results", "experiments", "missing_true_impact", "checkpoint.csv")
BASELINE_CSV = os.path.join(project_root, "results", "tables", "baseline_141_iforest.csv")

FRACTIONS = [0.01, 0.05, 0.10, 0.20]
NUM_BURSTS = [1, 3, 5, 10]
IMPUTATION_METHODS = ['linear', 'ffill', 'nearest']
N_SEEDS = 1

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
]


def apply_imputation(data, method):
    s = pd.Series(data)
    if method == 'linear':
        s = s.interpolate(method='linear').bfill().ffill()
    elif method == 'ffill':
        s = s.ffill().bfill()
    elif method == 'nearest':
        s = s.interpolate(method='nearest').bfill().ffill()
    return s.to_numpy('float')


def process_single_job(job_args):
    (file_path, missing_type, fraction, num_bursts, seed, model_names, baseline_dict) = job_args

    file_name = os.path.basename(file_path)
    results = []

    try:
        df, file_name = load_tsb_dataframe(file_path)
        labels = df['is_anomaly'].to_numpy('int')
        n = len(df)

        # Sliding window from clean data
        clean_data = df['value'].to_numpy('float')
        clean_scaled = StandardScaler().fit_transform(clean_data.reshape(-1, 1)).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # === SAME CORRUPTION AS run_missing_true_impact.py ===
        corruptor = TSCorruptor(df.copy(), value_col='value', label_col='is_anomaly', seed=seed)

        if missing_type == 'point':
            ts_corruptor.injectors.inject_point_missing(corruptor, fraction=fraction)
            burst_length = 0
        else:
            burst_length = max(1, int(fraction * n / num_bursts))
            ts_corruptor.injectors.inject_burst_missing(
                corruptor, num_bursts=num_bursts, burst_length=burst_length
            )

        df_corrupted = corruptor.get_corrupted_df()
        corrupted_data = df_corrupted['value'].to_numpy('float')
        nan_mask = np.isnan(corrupted_data)
        actual_missing_rate = nan_mask.sum() / n

        # === IMPUTATION + MODEL ===
        for imp_method in IMPUTATION_METHODS:
            for model_name in model_names:
                try:
                    imputed_data = apply_imputation(corrupted_data.copy(), imp_method)

                    if np.isnan(imputed_data).any():
                        results.append({
                            'file': file_name, 'model': model_name,
                            'missing_type': missing_type, 'fraction': fraction,
                            'num_bursts': num_bursts, 'burst_length': burst_length,
                            'imputation': imp_method,
                            'actual_missing_rate': round(actual_missing_rate, 4),
                            'seed': seed, 'error': 'NaN after imputation',
                        })
                        continue

                    scaled_data = StandardScaler().fit_transform(imputed_data.reshape(-1, 1)).flatten()

                    if model_name == 'MP':
                        clf = MatrixProfile(window=sliding_window)
                        clf.fit(scaled_data)
                        score = clf.decision_scores_
                    else:
                        X = Window(window=sliding_window).convert(scaled_data).to_numpy()
                        if model_name == 'IForest':
                            clf = IForest(n_estimators=100, random_state=42)
                        elif model_name == 'PCA':
                            clf = PCA(n_components=10)
                        elif model_name == 'LOF':
                            clf = LOF(n_neighbors=20)
                        else:
                            raise ValueError(f"Unknown model: {model_name}")
                        clf.fit(X)
                        score = clf.decision_scores_

                    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
                    full_score = np.array(
                        [score[0]] * math.ceil((sliding_window - 1) / 2) +
                        list(score) +
                        [score[-1]] * ((sliding_window - 1) // 2)
                    )

                    if np.isnan(full_score).any() or len(np.unique(full_score)) <= 1:
                        results.append({
                            'file': file_name, 'model': model_name,
                            'missing_type': missing_type, 'fraction': fraction,
                            'num_bursts': num_bursts, 'burst_length': burst_length,
                            'imputation': imp_method,
                            'actual_missing_rate': round(actual_missing_rate, 4),
                            'seed': seed, 'error': 'Invalid scores',
                        })
                        continue

                    metrics = get_metrics(full_score, labels, metric="all", slidingWindow=sliding_window)
                    baseline_auc = baseline_dict.get(file_name, {}).get(model_name, None)

                    row = {
                        'file': file_name, 'model': model_name,
                        'missing_type': missing_type, 'fraction': fraction,
                        'num_bursts': num_bursts, 'burst_length': burst_length,
                        'imputation': imp_method,
                        'actual_missing_rate': round(actual_missing_rate, 4),
                        'seed': seed,
                        'baseline_AUC_ROC': round(baseline_auc, 4) if baseline_auc else None,
                        'recovered_AUC_ROC': round(metrics.get('AUC_ROC', 0.0), 4),
                        'error': None,
                    }
                    for key in SELECTED_METRICS:
                        row[f'recovered_{key}'] = round(metrics.get(key, 0.0), 4)

                    results.append(row)

                except Exception as e:
                    results.append({
                        'file': file_name, 'model': model_name,
                        'missing_type': missing_type, 'fraction': fraction,
                        'num_bursts': num_bursts, 'burst_length': burst_length,
                        'imputation': imp_method,
                        'actual_missing_rate': round(actual_missing_rate, 4),
                        'seed': seed, 'error': str(e),
                    })

        return {'status': 'success', 'results': results}

    except Exception as e:
        return {'status': 'error', 'file': file_name, 'error': traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(description="Recovery for Missing True Impact")
    parser.add_argument('--models', nargs='+', default=['IForest'],
                        choices=['IForest', 'PCA', 'LOF', 'MP'])
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(RESULTS_DIR, "checkpoint.csv")

    # Load baseline AUC
    baseline_df = pd.read_csv(BASELINE_CSV)
    baseline_dict = {}
    for _, r in baseline_df.iterrows():
        baseline_dict.setdefault(r['file'], {})[r['model']] = r['AUC_ROC']

    # Load subset
    subset_df = pd.read_csv(SUBSET_CSV)
    file_paths = subset_df['filepath'].tolist()

    if args.test:
        file_paths = file_paths[:3]
        print(f"[TEST MODE] Using {len(file_paths)} files")

    # Resume — skip jobs where ALL requested (model, imputation) combos are done
    done_results = set()
    if os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        for _, r in existing.iterrows():
            key = (r.get('file'), r.get('missing_type'), r.get('fraction'),
                   r.get('num_bursts'), int(r.get('seed', 0)),
                   r.get('model'), r.get('imputation'))
            done_results.add(key)
        print(f"[Resume] Loaded {len(existing)} rows from checkpoint")

    # Build jobs — skip only if ALL model×imputation combos are done
    jobs = []
    skipped = 0
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        for seed in range(N_SEEDS):
            # Point missing
            for fraction in FRACTIONS:
                if done_results and all(
                    (file_name, 'point', fraction, 0, seed, m, imp) in done_results
                    for m in args.models for imp in IMPUTATION_METHODS
                ):
                    skipped += 1
                    continue
                jobs.append((file_path, 'point', fraction, 0, seed, args.models, baseline_dict))
            # Burst missing
            for fraction in FRACTIONS:
                for nb in NUM_BURSTS:
                    if done_results and all(
                        (file_name, 'burst', fraction, nb, seed, m, imp) in done_results
                        for m in args.models for imp in IMPUTATION_METHODS
                    ):
                        skipped += 1
                        continue
                    jobs.append((file_path, 'burst', fraction, nb, seed, args.models, baseline_dict))
    if skipped:
        print(f"[Resume] Skipped {skipped} completed jobs")

    print(f"\n{'='*60}")
    print(f"Recovery — Missing True Impact (MCAR)")
    print(f"{'='*60}")
    print(f"Files: {len(file_paths)}")
    print(f"Models: {args.models}")
    print(f"Imputation: {IMPUTATION_METHODS}")
    print(f"Total jobs: {len(jobs)}")
    print(f"{'='*60}\n")

    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_job, j): j for j in jobs}
        with tqdm(total=len(futures), desc="recovery (missing)") as pbar:
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


if __name__ == '__main__':
    main()
