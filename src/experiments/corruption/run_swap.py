import os
import sys
import math
import argparse
import time
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
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "swap")
NOISE_TYPE = 'swap'

# Parameter Grid
FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30]
SWAP_LENGTHS = [1, 10, 50, 100]   # 1 = point swap, 10/50/100 = segment swap (fixed length)
# MAX_DISTANCES = [500, None]       # TODO: local (500) vs global — requires max_distance > swap_length
MAX_DISTANCES = [None]              # Global swap only for now
N_SEEDS = 1

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F', 
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR', 
    'Affiliation_Precision', 'Affiliation_Recall'
]

def process_single_job(job_args):
    (file_path, fraction, max_dist, swap_length, seed, model_names) = job_args

    file_name = os.path.basename(file_path)  # fallback for error reporting
    dist_str = "Global" if max_dist is None else str(max_dist)
    condition_name = f"frac_{fraction}_dist_{dist_str}_len_{swap_length}"

    try:
        # 1. Load Data (handles NASA train+test concatenation)
        df, file_name = load_tsb_dataframe(file_path)

        n = len(df)

        # Skip invalid combinations where swap is meaningless
        num_swaps = int(fraction * n) // (2 * swap_length)
        if num_swaps < 1:
            return {'status': 'skipped', 'file': file_name,
                    'reason': f'Invalid combo: fraction={fraction}, swap_length={swap_length}, n={n}'}
        
        # 3. Add Swap Corruption
        corruptor = TSCorruptor(df, value_col='value', label_col='is_anomaly', seed=seed)
        ts_corruptor.injectors.inject_swap(
            corruptor, 
            fraction=fraction, 
            swap_length=swap_length, 
            max_distance=max_dist
        )
        
        df_corrupted = corruptor.get_corrupted_df()
        
        corrupted_data = df_corrupted['value'].to_numpy('float')
        labels = df_corrupted['is_anomaly'].to_numpy('int')
        
        if np.isnan(corrupted_data).all():
            return {'status': 'error', 'file': file_name, 'error': 'All data is NaN after corruption'}
            
        # 3. Preprocessing — compute sliding_window on CLEAN signal before swap
        clean_scaled = StandardScaler().fit_transform(
            df['value'].to_numpy('float').reshape(-1, 1)
        ).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        scaled_data = StandardScaler().fit_transform(
            corrupted_data.reshape(-1, 1)
        ).flatten()
        X = Window(window=sliding_window).convert(scaled_data).to_numpy()

        # 4. Modeling & Evaluation
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
                    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1, 1)).ravel()
                    
                    # Symmetric padding (TSB-UAD convention)
                    full_score = np.array([score[0]]*math.ceil((sliding_window-1)/2) + list(score) + [score[-1]]*((sliding_window-1)//2))
                    
                    if np.isnan(full_score).any() or len(np.unique(full_score)) <= 1:
                        results.append({
                           'file': file_name, 'fraction': fraction, 'max_distance': dist_str,
                           'seed': seed, 'model': model_name, 'condition': condition_name,
                           'error': "Invalid scores generated"
                        })
                        continue
                        
                    metrics = get_metrics(full_score, labels, metric="all", slidingWindow=sliding_window)
                    
                    row = {
                        'file': file_name, 'fraction': fraction, 'max_distance': dist_str,
                        'swap_length': swap_length,
                        'seed': seed, 'model': model_name,
                        'condition': condition_name, 'error': None
                    }
                    for key in SELECTED_METRICS:
                        row[key] = round(metrics.get(key, 0.0), 4)
                        
                    results.append(row)
                    
            except Exception as e:
                results.append({
                    'file': file_name, 'fraction': fraction, 'max_distance': dist_str,
                    'swap_length': swap_length,
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
        
    grouped = df_success.groupby(['fraction', 'max_distance', 'swap_length', 'condition', 'model'])
    
    summary_data = []
    for name, group in grouped:
        frac, dist, swap_len, condition, model = name
        row = {
            'fraction': frac, 'max_distance': dist,
            'swap_length': swap_len, 'condition': condition, 'model': model,
            'n_runs': len(group)
        }
        for col in SELECTED_METRICS:
            if col in group.columns:
                row[f'mean_{col}'] = round(group[col].mean(), 4)
                row[f'std_{col}'] = round(group[col].std(), 4)
                
        summary_data.append(row)
        
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_path, index=False)
    print(f"Summary computed and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--models', nargs='+', default=['IForest'])
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not os.path.exists(SUBSET_CSV):
        print(f"[ERROR] Subset CSV not found: {SUBSET_CSV}")
        return
        
    df_files_metadata = pd.read_csv(SUBSET_CSV)
    df_files = df_files_metadata['filepath'].tolist()
    
    fractions = FRACTIONS
    distances = MAX_DISTANCES
    swap_lengths = SWAP_LENGTHS

    if args.test:
        df_files = df_files[:3]
        fractions = [0.05, 0.20]
        distances = [None]
        swap_lengths = [1, 50]
        print("!!! RUNNING IN TEST MODE !!!")

    n_workers = args.workers or os.cpu_count() or 1
    print(f"\n{'='*60}")
    print(f"  Swap Error Robustness Experiment")
    print(f"{'='*60}")
    print(f"Fractions: {fractions}")
    print(f"Max Distances: {distances}")
    print(f"Swap Lengths: {swap_lengths}")
    print(f"Total Conditions: {len(fractions) * len(distances) * len(swap_lengths)}")
    print(f"{'='*60}\n")

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
                job_id = f"{r['file']}_{r['fraction']}_{r['max_distance']}_{r['swap_length']}_{r['seed']}_{r['model']}"
                completed_jobs.add(job_id)
            print(f"Loaded {len(all_results)} successful jobs from checkpoint.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint: {e}")

    jobs = []
    for file_path in df_files:
        file_name = os.path.basename(file_path)
        for frac in fractions:
            for dist in distances:
                dist_str = "Global" if dist is None else str(dist)
                for swap_len in swap_lengths:
                    for seed in range(N_SEEDS):
                        needed_models = []
                        for m in args.models:
                            job_id = f"{file_name}_{frac}_{dist_str}_{swap_len}_{seed}_{m}"
                            if job_id not in completed_jobs:
                                needed_models.append(m)

                        if needed_models:
                            jobs.append((file_path, frac, dist, swap_len, seed, needed_models))

    print(f"Jobs Scheduled: {len(jobs)}")

    if len(jobs) > 0:
        new_results_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_job, job): job for job in jobs}

            with tqdm(total=len(jobs), desc="swap") as pbar:
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

        # Save per model → point/segment → (ratio)
        for model_name in args.models:
            model_df = df_all[df_all['model'] == model_name]
            if model_df.empty:
                continue

            # Point swap (length == 1)
            point_df = model_df[model_df['swap_length'] == 1]
            if not point_df.empty:
                point_dir = os.path.join(RESULTS_DIR, model_name, "point")
                os.makedirs(point_dir, exist_ok=True)
                point_df.to_csv(os.path.join(point_dir, "raw_results.csv"), index=False)

            # Segment swap (length > 1)
            segment_df = model_df[model_df['swap_length'] > 1]
            for slen in segment_df['swap_length'].unique():
                seg_dir = os.path.join(RESULTS_DIR, model_name, "segment", f"len_{int(slen)}")
                os.makedirs(seg_dir, exist_ok=True)
                len_df = segment_df[segment_df['swap_length'] == slen]
                len_df.to_csv(os.path.join(seg_dir, "raw_results.csv"), index=False)

    print(f"\n[Done] All swap experiments complete.")

if __name__ == "__main__":
    main()
