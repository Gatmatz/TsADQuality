"""
Gilbert-Elliott Channel Model — True Impact Experiment

Tests how Markov-chain burst errors affect anomaly detection using
the true impact evaluation approach (no imputation).

Gilbert-Elliott model:
  - Two states: Good (no corruption) and Bad (inject NaN)
  - p_good_to_bad (alpha): probability of entering a burst
  - p_bad_to_good (beta): probability of leaving a burst
  - Expected corruption rate ~ alpha / (alpha + beta)
  - Expected avg burst length ~ 1 / beta

Evaluation (true impact):
  - Remove NaN points, run model on shorter series
  - Lost anomalies → score=0 (false negatives)
  - Lost normal points → excluded from evaluation

Key question: "How do bursty missing patterns (vs random missing)
              affect anomaly detection?"

Usage:
    python run_gilbert_elliott_true_impact.py --models IForest
    python run_gilbert_elliott_true_impact.py --test
"""
import os
import sys
import math
import argparse

os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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
from data_loader import load_tsb_dataframe, remap_filepath, load_pretrained_ae

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "gilbert_elliott_true_impact")

ALPHAS = [0.01, 0.05, 0.10]    # p(Good→Bad) — burst frequency
BETAS = [0.10, 0.25, 0.50]     # p(Bad→Good) — burst duration (1/beta)
N_SEEDS = 1

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]


def run_model(model_name, X, scaled_data, sw, clf_ae=None):
    if model_name == 'IForest':
        clf = IForest(n_estimators=100, random_state=42); clf.fit(X); return clf.decision_scores_, False
    elif model_name == 'PCA':
        clf = PCA(n_components=10); clf.fit(X); return clf.decision_scores_, False
    elif model_name == 'MP':
        clf = MatrixProfile(window=sw); clf.fit(scaled_data); return clf.decision_scores_, False
    elif model_name == 'LOF':
        clf = LOF(n_neighbors=20); clf.fit(X); return clf.decision_scores_, False
    elif model_name == 'AE':
        clf_ae.predict(scaled_data)
        return clf_ae.decision_scores_, True  # True = already full length
    else:
        raise ValueError(f"Unknown model: {model_name}")


def process_single_job(job_args):
    (file_path, alpha, beta, seed, model_names) = job_args

    file_name = os.path.basename(file_path)
    condition_name = f"a_{alpha}_b_{beta}"
    expected_rate = alpha / (alpha + beta)
    expected_burst_len = 1.0 / beta

    try:
        # 1. Load Data
        df, file_name = load_tsb_dataframe(file_path)
        labels = df['is_anomaly'].to_numpy('int')
        n = len(df)

        # 2. Sliding window from clean data
        clean_scaled = StandardScaler().fit_transform(
            df['value'].to_numpy('float').reshape(-1, 1)
        ).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # 3. Inject Gilbert-Elliott burst missing
        corruptor = TSCorruptor(df, value_col='value', label_col='is_anomaly', seed=seed)
        ts_corruptor.injectors.inject_gilbert_elliott(
            corruptor,
            p_good_to_bad=alpha,
            p_bad_to_good=beta,
            noise_type='missing'
        )

        df_corrupted = corruptor.get_corrupted_df()
        corrupted_data = df_corrupted['value'].to_numpy('float')

        # 4. Compute missing stats
        nan_mask = np.isnan(corrupted_data)
        nan_count = int(nan_mask.sum())
        actual_missing_rate = nan_count / n

        masked_normal = nan_mask & (labels == 0)
        masked_anomaly = nan_mask & (labels == 1)
        n_lost_anomalies = int(masked_anomaly.sum())

        # 5. Run model on non-NaN data
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

        # 6. Modeling & Evaluation
        results = []
        # Load AE once if needed
        clf_ae = None
        if 'AE' in model_names:
            clf_ae, _ = load_pretrained_ae(file_name, project_root)

        for model_name in model_names:
            try:
                score, is_full_length = run_model(model_name, X, scaled_data, sw, clf_ae)
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

                if is_full_length:
                    padded_score = score
                    if len(padded_score) > n_kept:
                        padded_score = padded_score[:n_kept]
                    elif len(padded_score) < n_kept:
                        padded_score = np.pad(padded_score, (0, n_kept - len(padded_score)), mode='edge')
                else:
                    padded_score = np.array(
                        [score[0]] * math.ceil((sw - 1) / 2) +
                        list(score) +
                        [score[-1]] * ((sw - 1) // 2)
                    )

                if np.isnan(padded_score).any() or len(np.unique(padded_score)) <= 1:
                    results.append({
                        'file': file_name, 'alpha': alpha, 'beta': beta,
                        'expected_rate': round(expected_rate, 4),
                        'expected_burst_len': round(expected_burst_len, 1),
                        'actual_missing_rate': round(actual_missing_rate, 4),
                        'n_lost_anomalies': n_lost_anomalies,
                        'n_original': n, 'n_kept': n_kept,
                        'seed': seed, 'model': model_name,
                        'condition': condition_name,
                        'error': "Invalid scores generated"
                    })
                    continue

                # True impact: score=0 for lost anomalies, exclude lost normals
                full_score = np.full(n, np.nan)
                full_score[~nan_mask] = padded_score
                full_score[masked_anomaly] = 0.0

                eval_mask = ~masked_normal
                eval_scores = full_score[eval_mask]
                eval_labels = labels[eval_mask]

                metrics = get_metrics(eval_scores, eval_labels, metric="all", slidingWindow=sw)

                row = {
                    'file': file_name, 'alpha': alpha, 'beta': beta,
                    'expected_rate': round(expected_rate, 4),
                    'expected_burst_len': round(expected_burst_len, 1),
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
                    'file': file_name, 'alpha': alpha, 'beta': beta,
                    'expected_rate': round(expected_rate, 4),
                    'expected_burst_len': round(expected_burst_len, 1),
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

    grouped = df_success.groupby(['alpha', 'beta', 'model'])

    summary_data = []
    for name, group in grouped:
        a, b, model = name
        row = {
            'alpha': a, 'beta': b,
            'expected_rate': round(a / (a + b), 4),
            'expected_burst_len': round(1.0 / b, 1),
            'mean_actual_missing_rate': round(group['actual_missing_rate'].mean(), 4),
            'mean_n_lost_anomalies': round(group['n_lost_anomalies'].mean(), 1),
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
    df_files = [remap_filepath(fp, project_root) for fp in df_files]

    alphas = ALPHAS
    betas = BETAS

    if args.test:
        df_files = df_files[:3]
        alphas = [0.05]
        betas = [0.10, 0.50]
        print("!!! RUNNING IN TEST MODE !!!")

    n_conditions = len(alphas) * len(betas)
    n_workers = args.workers or os.cpu_count() or 1
    print(f"\n{'=' * 60}")
    print(f"  Gilbert-Elliott — True Impact (No Imputation)")
    print(f"{'=' * 60}")
    print(f"  Lost anomalies → score=0 (false negatives)")
    print(f"  Lost normal points → excluded from evaluation")
    print(f"{'=' * 60}")
    print(f"Alphas (p G→B): {alphas}")
    print(f"Betas  (p B→G): {betas}")
    print(f"Conditions: {n_conditions}")
    print()
    print(f"{'alpha':>8} {'beta':>8} {'exp_rate':>10} {'exp_burst':>10}")
    print("-" * 40)
    for a in alphas:
        for b in betas:
            print(f"{a:>8.2f} {b:>8.2f} {a/(a+b):>10.2%} {1/b:>10.1f}")
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
        for a in alphas:
            for b in betas:
                for seed in range(N_SEEDS):
                    condition = f"a_{a}_b_{b}"
                    needed_models = []
                    for m in args.models:
                        job_id = f"{file_name}_{condition}_{seed}_{m}"
                        if job_id not in completed_jobs:
                            needed_models.append(m)
                    if needed_models:
                        jobs.append((file_path, a, b, seed, needed_models))

    print(f"Jobs Scheduled: {len(jobs)}")

    if len(jobs) > 0:
        new_results_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_job, job): job for job in jobs}

            with tqdm(total=len(jobs), desc="gilbert-elliott (true impact)") as pbar:
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

            for a in model_df['alpha'].unique():
                for b in model_df['beta'].unique():
                    out_dir = os.path.join(RESULTS_DIR, model_name, f"a_{a}_b_{b}")
                    os.makedirs(out_dir, exist_ok=True)
                    sub_df = model_df[(model_df['alpha'] == a) & (model_df['beta'] == b)]
                    if not sub_df.empty:
                        sub_df.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)

    print(f"\n[Done] All Gilbert-Elliott true impact experiments complete.")


if __name__ == "__main__":
    main()
