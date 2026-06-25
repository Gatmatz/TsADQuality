"""
Spike experiment variant: NO z-score normalization at all.

Raw data is used directly (no StandardScaler). This shows the raw impact
of spikes on anomaly detection without any standardization step.
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
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "spikes_no_zscore")
NOISE_TYPE = 'spikes'

# Same parameter grid as run_spikes.py
FRACTIONS = [0.01, 0.05, 0.10, 0.20]
MULTIPLIERS = [3.0, 10.0]
MODES = [(False, 1), (True, 3), (True, 10)]
N_SEEDS = 1

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]


def process_single_job(job_args):
    (file_path, fraction, multiplier, sequential, seq_len, seed, model_names) = job_args

    file_name = os.path.basename(file_path)
    mode_str = "point" if not sequential else f"burst_{seq_len}"
    condition_name = f"frac_{fraction}_mult_{multiplier}_{mode_str}"

    try:
        # 1. Load Data
        df, file_name = load_tsb_dataframe(file_path)

        # 2. Compute sliding window on raw clean data (no z-score)
        clean_data = df['value'].to_numpy('float')
        sliding_window = max(int(find_length(clean_data)), 10)

        # 3. Add Spike Corruption
        corruptor = TSCorruptor(df, value_col='value', label_col='is_anomaly', seed=seed)
        ts_corruptor.injectors.inject_spikes(
            corruptor,
            fraction=fraction,
            multiplier=multiplier,
            sequential=sequential,
            sequence_length=seq_len
        )

        df_corrupted = corruptor.get_corrupted_df()

        corrupted_data = df_corrupted['value'].to_numpy('float')
        labels = df_corrupted['is_anomaly'].to_numpy('int')

        if np.isnan(corrupted_data).all():
            return {'status': 'error', 'file': file_name, 'error': 'All data is NaN after corruption'}

        # 4. Use raw corrupted data directly (NO z-score)
        X = Window(window=sliding_window).convert(corrupted_data).to_numpy()

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
                    clf = MatrixProfile(window=sliding_window); clf.fit(corrupted_data); score = clf.decision_scores_
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
                            'file': file_name, 'fraction': fraction, 'multiplier': multiplier,
                            'sequential': sequential, 'sequence_length': seq_len,
                            'seed': seed, 'model': model_name, 'condition': condition_name,
                            'error': "Invalid scores generated"
                        })
                        continue

                    metrics = get_metrics(full_score, labels, metric="all", slidingWindow=sliding_window)

                    row = {
                        'file': file_name, 'fraction': fraction, 'multiplier': multiplier,
                        'sequential': sequential, 'sequence_length': seq_len,
                        'seed': seed, 'model': model_name,
                        'condition': condition_name, 'error': None
                    }
                    for key in SELECTED_METRICS:
                        row[key] = round(metrics.get(key, 0.0), 4)

                    results.append(row)

            except Exception as e:
                results.append({
                    'file': file_name, 'fraction': fraction, 'multiplier': multiplier,
                    'sequential': sequential, 'sequence_length': seq_len,
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

    grouped = df_success.groupby(['fraction', 'multiplier', 'sequential', 'sequence_length', 'model'])

    summary_data = []
    for name, group in grouped:
        frac, mult, seq, seq_len, model = name
        row = {
            'fraction': frac, 'multiplier': mult,
            'sequential': seq, 'sequence_length': seq_len,
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
    multipliers = MULTIPLIERS
    modes = MODES

    if args.test:
        df_files = df_files[:3]
        fractions = [0.05, 0.20]
        multipliers = [5.0]
        modes = [(False, 1), (True, 3), (True, 10)]
        print("!!! RUNNING IN TEST MODE !!!")

    n_workers = args.workers or 4
    print(f"\n{'=' * 60}")
    print(f"  Spike Experiment (NO Z-Score)")
    print(f"{'=' * 60}")
    print(f"  No StandardScaler applied — raw data used directly")
    print(f"{'=' * 60}")
    print(f"Fractions: {fractions}")
    print(f"Multipliers: {multipliers}")
    print(f"Modes: {modes}")
    print(f"Total Conditions: {len(fractions) * len(multipliers) * len(modes)}")
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
                mode_str = "point" if not r['sequential'] else f"burst_{r['sequence_length']}"
                job_id = f"{r['file']}_{r['fraction']}_{r['multiplier']}_{mode_str}_{r['seed']}_{r['model']}"
                completed_jobs.add(job_id)
            print(f"Loaded {len(all_results)} successful jobs from checkpoint.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint: {e}")

    jobs = []
    for file_path in df_files:
        file_name = os.path.basename(file_path)
        for frac in fractions:
            for mult in multipliers:
                for sequential, seq_len in modes:
                    for seed in range(N_SEEDS):
                        mode_str = "point" if not sequential else f"burst_{seq_len}"
                        needed_models = []
                        for m in args.models:
                            job_id = f"{file_name}_{frac}_{mult}_{mode_str}_{seed}_{m}"
                            if job_id not in completed_jobs:
                                needed_models.append(m)

                        if needed_models:
                            jobs.append((file_path, frac, mult, sequential, seq_len, seed, needed_models))

    print(f"Jobs Scheduled: {len(jobs)}")

    if len(jobs) > 0:
        new_results_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_job, job): job for job in jobs}

            with tqdm(total=len(jobs), desc="spikes (no z-score)") as pbar:
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

            point_df = model_df[model_df['sequential'] == False]
            if not point_df.empty:
                point_dir = os.path.join(RESULTS_DIR, model_name, "point")
                os.makedirs(point_dir, exist_ok=True)
                point_df.to_csv(os.path.join(point_dir, "raw_results.csv"), index=False)

            burst_df = model_df[model_df['sequential'] == True]
            for seq_len in burst_df['sequence_length'].unique():
                burst_dir = os.path.join(RESULTS_DIR, model_name, "burst", f"seq_{seq_len}")
                os.makedirs(burst_dir, exist_ok=True)
                seq_df = burst_df[burst_df['sequence_length'] == seq_len]
                seq_df.to_csv(os.path.join(burst_dir, "raw_results.csv"), index=False)

    print(f"\n[Done] All spike (no z-score) experiments complete.")


if __name__ == "__main__":
    main()
