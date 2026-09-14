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
from data_loader import load_tsb_dataframe, load_pretrained_ae

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "freeze")
NOISE_TYPE = 'freeze'

# Parameter Grid
FRACTIONS = [0.01, 0.05, 0.10, 0.20]
NUM_STUCKS = [1, 3, 5, 10, 20]    # dynamic: stuck_length = fraction * n / num_stucks
N_SEEDS = 1

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]


def process_single_job(job_args):
    (file_path, fraction, num_stucks, seed, model_names) = job_args

    file_name = os.path.basename(file_path)
    condition_name = f"frac_{fraction}_ns_{num_stucks}"

    try:
        # 1. Load Data
        df, file_name = load_tsb_dataframe(file_path)

        # 2. Compute stuck_length dynamically from fraction and series length
        n = len(df)
        stuck_length = max(1, int(fraction * n / num_stucks))

        # 3. Add Freeze Corruption
        corruptor = TSCorruptor(df, value_col='value', label_col='is_anomaly', seed=seed)
        ts_corruptor.injectors.inject_sensor_stuck(
            corruptor,
            num_stucks=num_stucks,
            stuck_length=stuck_length
        )

        df_corrupted = corruptor.get_corrupted_df()

        corrupted_data = df_corrupted['value'].to_numpy('float')
        labels = df_corrupted['is_anomaly'].to_numpy('int')

        if np.isnan(corrupted_data).all():
            return {'status': 'error', 'file': file_name, 'error': 'All data is NaN after corruption'}

        # 4. Preprocessing
        clean_scaled = StandardScaler().fit_transform(
            df['value'].to_numpy('float').reshape(-1, 1)
        ).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        scaled_data = StandardScaler().fit_transform(
            corrupted_data.reshape(-1, 1)
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
                elif model_name == 'AE':
                    clf, _ = load_pretrained_ae(file_name, project_root)
                    clf.predict(scaled_data)
                    score = clf.decision_scores_
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                if score is not None:
                    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

                    # AE produces full-length scores, others need symmetric padding
                    if model_name == 'AE':
                        full_score = score
                    else:
                        full_score = np.array([score[0]] * math.ceil((sliding_window - 1) / 2) + list(score) + [score[-1]] * ((sliding_window - 1) // 2))

                    if np.isnan(full_score).any() or len(np.unique(full_score)) <= 1:
                        results.append({
                            'file': file_name, 'fraction': fraction,
                            'num_stucks': num_stucks, 'stuck_length': stuck_length,
                            'seed': seed, 'model': model_name, 'condition': condition_name,
                            'error': "Invalid scores generated"
                        })
                        continue

                    metrics = get_metrics(full_score, labels, metric="all", slidingWindow=sliding_window)

                    row = {
                        'file': file_name, 'fraction': fraction,
                        'num_stucks': num_stucks, 'stuck_length': stuck_length,
                        'seed': seed, 'model': model_name,
                        'condition': condition_name, 'error': None
                    }
                    for key in SELECTED_METRICS:
                        row[key] = round(metrics.get(key, 0.0), 4)

                    results.append(row)

            except Exception as e:
                results.append({
                    'file': file_name, 'fraction': fraction,
                    'num_stucks': num_stucks, 'stuck_length': stuck_length,
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

    grouped = df_success.groupby(['fraction', 'num_stucks', 'model'])

    summary_data = []
    for name, group in grouped:
        frac, ns, model = name
        row = {
            'fraction': frac, 'num_stucks': ns,
            'mean_stuck_length': round(group['stuck_length'].mean(), 1),
            'model': model, 'n_runs': len(group)
        }
        for col in SELECTED_METRICS:
            if col in group.columns:
                row[f'mean_{col}'] = round(group[col].mean(), 4)
                row[f'std_{col}'] = round(group[col].std(), 4)

        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_path, index=False)
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

    fractions = FRACTIONS
    num_stucks_list = NUM_STUCKS

    if args.test:
        df_files = df_files[:3]
        fractions = [0.05, 0.20]
        num_stucks_list = [3]
        print("!!! RUNNING IN TEST MODE !!!")

    n_workers = args.workers or os.cpu_count() or 1
    print(f"\n{'=' * 60}")
    print(f"  Freeze (Sensor Stuck) Robustness Experiment")
    print(f"{'=' * 60}")
    print(f"Fractions: {fractions}")
    print(f"Num Stucks: {num_stucks_list}")
    print(f"Total Conditions: {len(fractions) * len(num_stucks_list)}")
    print(f"{'=' * 60}\n")

    # Build all jobs
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
                job_id = f"{r['file']}_{r['fraction']}_{r['num_stucks']}_{r['seed']}_{r['model']}"
                completed_jobs.add(job_id)
            print(f"Loaded {len(all_results)} successful jobs from checkpoint.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint: {e}")

    jobs = []
    for file_path in df_files:
        file_name = os.path.basename(file_path)
        for frac in fractions:
            for ns in num_stucks_list:
                for seed in range(N_SEEDS):
                    needed_models = []
                    for m in args.models:
                        job_id = f"{file_name}_{frac}_{ns}_{seed}_{m}"
                        if job_id not in completed_jobs:
                            needed_models.append(m)

                    if needed_models:
                        jobs.append((file_path, frac, ns, seed, needed_models))

    print(f"Jobs Scheduled: {len(jobs)}")

    if len(jobs) > 0:
        new_results_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_job, job): job for job in jobs}

            with tqdm(total=len(jobs), desc="freeze") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        all_results.extend(res['results'])
                        new_results_count += len(res['results'])

                    pbar.update(1)

                    if new_results_count >= 500:
                        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
                        new_results_count = 0

        # Save global checkpoint
        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)

    if all_results:
        df_all = pd.DataFrame(all_results)

        # Global summary
        compute_summary(df_all, os.path.join(RESULTS_DIR, "summary.csv"))

        # Save per model -> stuck_length
        for model_name in args.models:
            model_df = df_all[df_all['model'] == model_name]
            if model_df.empty:
                continue

            for ns in model_df['num_stucks'].unique():
                out_dir = os.path.join(RESULTS_DIR, model_name, f"ns_{int(ns)}")
                os.makedirs(out_dir, exist_ok=True)
                ns_df = model_df[model_df['num_stucks'] == ns]
                ns_df.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)

    print(f"\n[Done] All freeze experiments complete.")


if __name__ == "__main__":
    main()
