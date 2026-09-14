import os
import math
import pandas as pd
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import sys
import warnings
import traceback
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# PATH CONFIGURATION
# ==========================================
# Ensure project root and TSB-UAD are in path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
tsb_uad_path = os.path.join(project_root, 'TSB-UAD')
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path: sys.path.insert(0, project_root)
if tsb_uad_path not in sys.path: sys.path.insert(0, tsb_uad_path)
if src_path not in sys.path: sys.path.insert(0, src_path)

# Optional imports for models - TSB_UAD must be in PYTHONPATH
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
import ts_corruptor.injectors  # Ensures injectors are registered
from data_loader import load_tsb_dataframe, load_pretrained_ae

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUBSET_CSV  = _PROJECT_ROOT / "results" / "tables" / "final_subset.csv"
RESULTS_DIR = _PROJECT_ROOT / "results" / "experiments" / "white_noise_snr"
NOISE_TYPE = 'white_noise_snr'

# Parameter Grid
SNRS_DB = [40, 30, 20, 10, 5, 0, -5, -10, -20]
N_SEEDS = 1

def process_single_job(job_args):
    """
    Worker function to process a single dataset for a specific SNR and seed.
    """
    (file_path, snr_db, seed, model_names) = job_args

    file_name = os.path.basename(file_path)  # fallback for error reporting
    condition_name = f"snr_{snr_db}dB"

    try:
        # 1. Load Data (handles NASA train+test concatenation)
        df, file_name = load_tsb_dataframe(file_path)

        # Compute sliding window on CLEAN signal (before noise injection)
        # so the window size is consistent with baseline and not confounded by noise
        clean_scaled = StandardScaler().fit_transform(
            df['value'].to_numpy('float').reshape(-1, 1)
        ).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # 2. Add Noise (SNR calculation using ts_corruptor)
        corruptor = TSCorruptor(df, value_col='value', label_col='is_anomaly', seed=seed)
        ts_corruptor.injectors.inject_white_noise_snr(corruptor, snr_db=snr_db)

        df_corrupted = corruptor.get_corrupted_df()

        # Extract corrupted signals
        corrupted_data = df_corrupted['value'].to_numpy('float')
        labels = df_corrupted['is_anomaly'].to_numpy('int')

        # Check for catastrophic NaNs (should not happen with SNR, but just in case)
        if np.isnan(corrupted_data).all():
            return {'status': 'error', 'file': file_name, 'error': 'All data is NaN after corruption'}

        # 3. Preprocessing (Standardization on corrupted data)
        scaled_data = StandardScaler().fit_transform(
            corrupted_data.reshape(-1, 1)
        ).flatten()

        # Use the clean-signal sliding window for all models
        X = Window(window=sliding_window).convert(scaled_data).to_numpy()

        # 4. Modeling & Evaluation
        results = []
        for model_name in model_names:
            try:
                score = None
                
                if model_name == 'IForest':
                    clf = IForest(n_estimators=100, random_state=42)
                    clf.fit(X)
                    score = clf.decision_scores_
                elif model_name == 'PCA':
                    clf = PCA(n_components=10)
                    clf.fit(X)
                    score = clf.decision_scores_
                elif model_name == 'MP':
                    clf = MatrixProfile(window=sliding_window)
                    clf.fit(scaled_data)
                    score = clf.decision_scores_
                elif model_name == 'LOF':
                    clf = LOF(n_neighbors=20)
                    clf.fit(X)
                    score = clf.decision_scores_
                elif model_name == 'AE':
                    # Load pre-trained AE — no retraining, predict only
                    clf, meta = load_pretrained_ae(file_name, str(_PROJECT_ROOT))
                    clf.predict(scaled_data)
                    score = clf.decision_scores_
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                # Reshape scores to match original length (TSB_UAD convention)
                if score is not None:
                    # AE already produces full-length scores, others need padding
                    if model_name == 'AE':
                        full_score = MinMaxScaler(feature_range=(0,1)).fit_transform(
                            score.reshape(-1, 1)).ravel()
                    else:
                        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1, 1)).ravel()

                        # Symmetric padding (TSB-UAD convention)
                        full_score = np.array([score[0]]*math.ceil((sliding_window-1)/2) + list(score) + [score[-1]]*((sliding_window-1)//2))
                    
                    # Prevent "Score must not be none" spam by catching NaNs early
                    if np.isnan(full_score).any() or len(np.unique(full_score)) <= 1:
                        results.append({
                           'file': file_name,
                           'snr_db': snr_db,
                           'seed': seed,
                           'model': model_name,
                           'condition': condition_name,
                           'error': "Invalid scores generated (all identical or NaNs)"
                        })
                        continue
                        
                    metrics = get_metrics(full_score, labels, metric="all", slidingWindow=sliding_window)
                    metrics['file'] = file_name
                    metrics['snr_db'] = snr_db
                    metrics['seed'] = seed
                    metrics['model'] = model_name
                    metrics['condition'] = condition_name
                    metrics['error'] = None
                    results.append(metrics)
                    
            except Exception as e:
                results.append({
                    'file': file_name,
                    'snr_db': snr_db,
                    'seed': seed,
                    'model': model_name,
                    'condition': condition_name,
                    'error': str(e)
                })
                
        return {'status': 'success', 'results': results}
        
    except Exception as e:
        return {'status': 'error', 'file': file_name, 'error': traceback.format_exc()}


def compute_summary(df_results, output_path):
    """Computes mean and std of metrics grouped by condition, SNR and model."""
    metrics_cols = [
        'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F', 
        'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR', 
        'Affiliation_Precision', 'Affiliation_Recall'
    ]
    
    # Keep only successful runs
    df_success = df_results[df_results['error'].isnull()]
    
    if df_success.empty:
        print("No successful runs to summarize.")
        return
        
    grouped = df_success.groupby(['snr_db', 'condition', 'model'])
    
    summary_data = []
    for name, group in grouped:
        snr_db, condition, model = name
        row = {
            'snr_db': snr_db,
            'condition': condition,
            'model': model,
            'n_runs': len(group)
        }
        for col in metrics_cols:
            if col in group.columns:
                row[f'mean_{col}'] = round(group[col].mean(), 4)
                row[f'std_{col}'] = round(group[col].std(), 4)
                
        summary_data.append(row)
        
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_path, index=False)
    print(f"Summary computed and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='White Gaussian Noise SNR Robustness Experiment')
    parser.add_argument('--test', action='store_true',
                        help='Smoke test: 3 datasets, 2 conditions, 2 seeds')
    parser.add_argument('--models', nargs='+', default=['AE'],
                        help='Models to evaluate. Available: IForest, PCA, MP, LOF, AE (requires pre-trained models)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel workers (default: CPU count)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not os.path.exists(SUBSET_CSV):
        print(f"[ERROR] Subset CSV not found: {os.path.abspath(SUBSET_CSV)}")
        return
        
    df_files_metadata = pd.read_csv(SUBSET_CSV)
    df_files = df_files_metadata['filepath'].tolist()
    
    snrs = SNRS_DB
    n_seeds = N_SEEDS

    if args.test:
        df_files = df_files[:3]
        snrs = [40, 10]
        n_seeds = 2
        print("!!! RUNNING IN TEST MODE !!!")

    n_conditions = len(snrs)
    n_workers = args.workers or 4
    print(f"\n{'='*60}")
    print(f"  White Noise (SNR) Robustness Experiment")
    print(f"{'='*60}")
    print(f"Datasets: {len(df_files)}")
    print(f"SNR Levels (dB): {snrs}")
    print(f"Conditions: {n_conditions}")
    print(f"Seeds: {n_seeds}")
    print(f"Models: {args.models}")
    print(f"Workers: {n_workers}")
    print(f"Total Jobs per model: {len(df_files) * n_conditions * n_seeds}")
    print(f"{'='*60}\n")

    # Load checkpointing
    checkpoint_file = os.path.join(RESULTS_DIR, "checkpoint.csv")
    completed_jobs = set()
    all_results = []
    
    if os.path.exists(checkpoint_file):
        try:
            df_checkpoint = pd.read_csv(checkpoint_file)
            
            # Filter out previous errors so we retry them
            if 'error' in df_checkpoint.columns:
                df_success = df_checkpoint[df_checkpoint['error'].isna()]
            else:
                df_success = df_checkpoint
                
            all_results = df_success.to_dict('records')
            
            for r in all_results:
                job_id = f"{r['file']}_{r['snr_db']}_{r['seed']}_{r['model']}"
                completed_jobs.add(job_id)
            print(f"Loaded checkpoint with {len(all_results)} SUCCESSFUL previous results.")
            print(f"Any previously failed datasets will be automatically retried.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint file: {e}")

    # Build job list
    jobs = []
    for file_path in df_files:
        file_name = os.path.basename(file_path)
        for snr_db in snrs:
            for seed in range(n_seeds):
                needed_models = []
                for m in args.models:
                    job_id = f"{file_name}_{snr_db}_{seed}_{m}"
                    if job_id not in completed_jobs:
                        needed_models.append(m)
                
                if needed_models:
                    jobs.append((file_path, snr_db, seed, needed_models))

    print(f"Jobs Scheduled: {len(jobs)} (Skipped full jobs already in checkpoint)")
    
    if len(jobs) == 0:
        print("All jobs are already completed!")
    else:
        new_results_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_job, job): job for job in jobs}
            
            with tqdm(total=len(jobs), desc="Evaluating") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        all_results.extend(res['results'])
                        new_results_count += len(res['results'])
                    else:
                        print(f"Error in {res.get('file', 'unknown')}: {res.get('error', '')}")
                    
                    pbar.update(1)
                    
                    # Autosave checkpoint every 5000 new model results
                    if new_results_count >= 5000:
                        df_temp = pd.DataFrame(all_results)
                        df_temp.to_csv(checkpoint_file, index=False)
                        new_results_count = 0
                        print("  [CHECKPOINT] Autosaving progress...")

        # Final save
        df_final = pd.DataFrame(all_results)
        df_final.to_csv(checkpoint_file, index=False)
        print(f"\nFinal results saved to {checkpoint_file}")

    # Compute Summary
    if all_results:
        df_all = pd.DataFrame(all_results)
        summary_path = os.path.join(RESULTS_DIR, "summary.csv")
        compute_summary(df_all, summary_path)

        # Save per-SNR level, then per-model
        for snr_val in df_all['snr_db'].unique():
            snr_df = df_all[df_all['snr_db'] == snr_val]
            snr_dir = os.path.join(RESULTS_DIR, f"snr_{int(snr_val)}dB")
            os.makedirs(snr_dir, exist_ok=True)

            for model_name in args.models:
                model_df = snr_df[snr_df['model'] == model_name]
                if not model_df.empty:
                    model_dir = os.path.join(snr_dir, model_name)
                    os.makedirs(model_dir, exist_ok=True)
                    model_df.to_csv(os.path.join(model_dir, "raw_results.csv"), index=False)

        # Also save flat per-model (all SNRs)
        for model_name in args.models:
            model_df = df_all[df_all['model'] == model_name]
            if not model_df.empty:
                model_dir = os.path.join(RESULTS_DIR, model_name)
                os.makedirs(model_dir, exist_ok=True)
                model_df.to_csv(os.path.join(model_dir, "raw_results_all.csv"), index=False)

if __name__ == "__main__":
    main()
