
import os
import sys
import gc
import math
import json
import time
import argparse
import concurrent.futures
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

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
import ts_corruptor.injectors
from data_loader import load_tsb_dataframe

# ============================
# Configuration
# ============================
ALPHAS = [0.01, 0.05, 0.10]          # p(Good→Bad) - Error Probability
BETAS = [0.10, 0.25, 0.50]           # p(Bad→Good) - Recovery Probability
NOISE_TYPE = 'missing'               # Missing Values (NaN)
N_SEEDS = 1                          # Consistent with other experiments (144 files provide variance)
IMPUTATION_TYPES = ['linear', 'ffill']
#εδω κανυμε και zero imputation για να δουμε αν ειναι πιο ανθεκτικο το μοντελο σε διαφορετικες τεχνικες αντιμετωπισης των NaN

SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
OUTPUT_DIR = os.path.join(project_root, "results", "experiments", "gilbert_elliott")

# Metrics to compute (VUS and Range-based are essential for TSB-UAD)
SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F', 
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR', 
    'Affiliation_Precision', 'Affiliation_Recall'
]

def process_single_job(job):
    """
    Single job: load dataset → corrupt with Gilbert-Elliott → evaluate models → return metrics.
    """
    filepath = job['filepath']
    alpha = job['alpha']
    beta = job['beta']
    seed = job['seed']
    models_to_run = job['models']
    folder = job['folder']
    baseline_name = job['baseline_name']
    imputation_type = job.get('imputation_type', 'ffill')

    results = []

    try:
        # 1. Load (handles NASA train+test concatenation)
        df, canonical_name = load_tsb_dataframe(filepath)
        label = df['is_anomaly'].to_numpy('int')

        if label.sum() == 0:
            return []

        # 2. Corrupt (deterministic via seed)
        corruptor = TSCorruptor(df, value_col='value', label_col='is_anomaly', seed=seed)
        ts_corruptor.injectors.inject_gilbert_elliott(
            corruptor,
            p_good_to_bad=alpha,
            p_bad_to_good=beta,
            noise_type=NOISE_TYPE
        )

        corrupted_df = corruptor.get_corrupted_df()
        report = corruptor.get_corruption_report()
        actual_corruption_rate = report['summary']['corruption_percentage'] / 100

        # 3. Handle Imputation
        data = corrupted_df['value'].copy().to_numpy('float')
        nan_count = int(np.isnan(data).sum())
        
        if nan_count > 0:
            s = pd.Series(data)
            if imputation_type == 'ffill':
                s = s.ffill().bfill()
            elif imputation_type == 'linear':
                s = s.interpolate(method='linear').bfill().ffill()
            elif imputation_type == 'zero':
                s = s.fillna(0.0)
            data = s.to_numpy('float')
            
            if np.isnan(data).all():
                return []

        # 4. Scale & Prepare for TSB-UAD
        # Compute sliding window on CLEAN signal (before corruption)
        # so the window size is consistent with baseline
        clean_scaled = StandardScaler().fit_transform(
            df['value'].dropna().to_numpy('float').reshape(-1, 1)
        ).ravel()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1)).ravel()

        X = Window(window=sliding_window).convert(scaled_data).to_numpy()
        ratio = 0.30 if folder == "YAHOO" else 0.10

        # 5. Build models
        models = {}
        if "IForest" in models_to_run: models["IForest"] = IForest(n_estimators=100, random_state=42)
        if "MP" in models_to_run: models["MP"] = MatrixProfile(window=sliding_window)
        if "PCA" in models_to_run: models["PCA"] = PCA(n_components=10)
        if "LOF" in models_to_run: models["LOF"] = LOF(n_neighbors=20)

        for model_name, clf in models.items():
            try:
                if model_name == "MP":
                    clf.fit(scaled_data)
                    score = clf.decision_scores_
                else:
                    clf.fit(X)
                    score = clf.decision_scores_

                if np.isnan(score).any(): continue

                # Normalize & Pad (symmetric padding — TSB-UAD convention)
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
                full_score = np.array([score[0]]*math.ceil((sliding_window-1)/2) + list(score) + [score[-1]]*((sliding_window-1)//2))

                # Compute metrics
                metrics = get_metrics(full_score, label, metric="all", slidingWindow=sliding_window)

                row = {
                    'baseline_name': baseline_name,
                    'folder': folder,
                    'alpha': alpha,
                    'beta': beta,
                    'seed': seed,
                    'imputation_type': imputation_type,
                    'condition': f"a{alpha:.2f}_b{beta:.2f}_{imputation_type}",
                    'actual_corruption_rate': round(actual_corruption_rate, 4),
                    'model': model_name,
                }
                # Add metrics
                for key in SELECTED_METRICS:
                    row[key] = round(metrics.get(key, 0.0), 4)

                results.append(row)
            except Exception:
                continue

        gc.collect()

    except Exception as e:
        return [{'baseline_name': baseline_name, 'error': str(e)}]

    return results

def compute_summary(df_results):
    """Aggregates results per condition and model."""
    agg_dict = {'actual_corruption_rate': 'mean'}
    for key in SELECTED_METRICS:
        agg_dict[key] = ['mean', 'std']
    
    summary = df_results.groupby(['alpha', 'beta', 'imputation_type', 'model']).agg(agg_dict).round(4)
    summary.columns = [f"{c[0]}_{c[1]}" if isinstance(c, tuple) and c[1] != "" else c[0] for c in summary.columns]
    return summary.reset_index()

def main():
    print("Starting Gilbert-Elliott experiment...")
    parser = argparse.ArgumentParser(description='Gilbert-Elliott Robustness Experiment')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--models', nargs='+', default=['IForest'])
    parser.add_argument('--workers', type=int, default=2) # Default 2 for Colab
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    args = parser.parse_args()

    print(f"Subset CSV: {SUBSET_CSV}")
    if not os.path.exists(SUBSET_CSV):
        print(f"[ERROR] Subset CSV not found!")
        return

    df_files = pd.read_csv(SUBSET_CSV)
    print(f"Loaded {len(df_files)} files from CSV.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_file = os.path.join(OUTPUT_DIR, "checkpoint.csv")
    print(f"Output dir: {OUTPUT_DIR}")

    # Load existing results
    completed_jobs = set()
    all_results = []
    if os.path.exists(checkpoint_file):
        try:
            df_check = pd.read_csv(checkpoint_file)
            all_results = df_check.to_dict('records')
            for r in all_results:
                job_id = (r['baseline_name'], r.get('alpha'), r.get('beta'), r.get('seed'), r.get('imputation_type'), r.get('model'))
                completed_jobs.add(job_id)
            print(f"Loaded {len(completed_jobs)} completed jobs from checkpoint.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint: {e}")

    # Build jobs
    print("Building job list...")
    jobs = []
    missing_files_count = 0
    
    # Πρώτο δείγμα για έλεγχο path
    first_path_check = True

    for alpha in ALPHAS:
        for beta in BETAS:
            for seed in range(args.seeds):
                for imp in IMPUTATION_TYPES:
                    for _, row in df_files.iterrows():
                        actual_filepath = str(row['filepath'])

                        if first_path_check:
                            print(f"Path Check: {actual_filepath}")
                            if os.path.exists(actual_filepath):
                                print("First file found OK.")
                            else:
                                print("First file NOT found. Check paths in subset CSV.")
                            first_path_check = False

                        if not os.path.exists(actual_filepath):
                            missing_files_count += 1
                            continue

                        needed_models = [m for m in args.models if (row['baseline_name'], alpha, beta, seed, imp, m) not in completed_jobs]
                        if needed_models:
                            jobs.append({
                                'filepath': actual_filepath, 'baseline_name': row['baseline_name'],
                                'folder': row['folder'], 'alpha': alpha, 'beta': beta,
                                'seed': seed, 'imputation_type': imp, 'models': needed_models
                            })

    if missing_files_count > 0:
        print(f"Warning: {missing_files_count} dataset files not found.")

    print(f"Total jobs to run: {len(jobs)}")
    if not jobs:
        print("No new jobs to run (all completed or files missing).")
        return

    # Run in parallel
    print(f"Starting parallel execution with {args.workers} workers...")
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_job, j): j for j in jobs}
        
        count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res and isinstance(res, list) and len(res) > 0 and 'model' in res[0]:
                    all_results.extend(res)
                count += 1
                if count % 100 == 0:
                    print(f"Progress: {count}/{len(jobs)} jobs done...")
                    pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
            except Exception as e:
                print(f"Error in job execution: {e}")

    if not all_results:
        print("[ERROR] No valid results generated. Stopping before summary.")
        return

    # Final Save
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(checkpoint_file, index=False)

    if df_all.empty or 'model' not in df_all.columns:
        print("[ERROR] Could not compute summary: results are empty or missing model column.")
        return

    # Global summary
    summary = compute_summary(df_all)
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    # Split per imputation type, then per model
    for imp_type in df_all['imputation_type'].unique():
        imp_df = df_all[df_all['imputation_type'] == imp_type]
        imp_dir = os.path.join(OUTPUT_DIR, f"imputation_{imp_type}")
        os.makedirs(imp_dir, exist_ok=True)

        # Per-imputation summary
        imp_summary = compute_summary(imp_df)
        imp_summary.to_csv(os.path.join(imp_dir, "summary.csv"), index=False)

        # Per-model raw results
        for model in args.models:
            model_df = imp_df[imp_df['model'] == model]
            if not model_df.empty:
                model_dir = os.path.join(imp_dir, model)
                os.makedirs(model_dir, exist_ok=True)
                model_df.to_csv(os.path.join(model_dir, "raw_results.csv"), index=False)

    print(f"Experiment finished in {time.time()-start_time:.1f}s. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
