import os
import sys
import gc
import json
import math
import time
import concurrent.futures
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np

# TSB-UAD models and metrics (assuming they are in path as they are in mass_experiment)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def process_corrupt_file(file_info):
    """
    Evaluates a single corrupted file for the specified models.
    """
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.config.set_visible_devices([], 'GPU')
    try:
        tf.keras.utils.disable_interactive_logging()
    except:
        pass

    from TSB_UAD.models.iforest import IForest
    from TSB_UAD.models.lof import LOF
    from TSB_UAD.models.pca import PCA
    from TSB_UAD.models.matrix_profile import MatrixProfile
    from TSB_UAD.models.AE import AE_MLP2
    from TSB_UAD.models.feature import Window
    from TSB_UAD.utils.slidingWindows import find_length
    from TSB_UAD.vus.metrics import get_metrics

    corrupt_path = file_info['filepath']
    original_basename = file_info['baseline_name']
    folder = file_info['folder']
    models_to_run = file_info['models']

    file_results = []
    
    try:
        df = pd.read_csv(corrupt_path, header=None)
        data = df[0].to_numpy('float')
        label = df[1].to_numpy('int')
    except Exception as e:
        return []

    # Handle NaNs (Some corruptors inject NaN. TSB models often fail on NaN)
    # If the corruptor injects NaN, we either impute them or drop them.
    # Here, we'll impute with the previous value (ffill) to preserve TS length
    # otherwise Matrix Profile shape drops out.
    if np.isnan(data).any():
        df[0] = df[0].ffill().bfill()
        data = df[0].to_numpy('float')

    if label.sum() == 0:
        return []
        
    # Scale Data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).ravel()
    
    sliding_window = find_length(scaled_data)
    sliding_window = max(int(sliding_window), 10) 
    
    X = Window(window=sliding_window).convert(scaled_data).to_numpy()
    ratio = 0.30 if folder == "YAHOO" else 0.10
    
    models = {}
    if "IForest" in models_to_run: models["IForest"] = IForest(n_estimators=100, random_state=42)
    if "MP" in models_to_run: models["MP"] = MatrixProfile(window=sliding_window)
    if "Autoencoder" in models_to_run: models["Autoencoder"] = AE_MLP2(slidingWindow=sliding_window, epochs=100, verbose=0)
    if "LOF" in models_to_run: models["LOF"] = LOF(n_neighbors=20)
    if "PCA" in models_to_run: models["PCA"] = PCA(n_components=10)

    for name, clf in models.items():
        try:
            if name == "MP":
                clf.fit(scaled_data)
                score = clf.decision_scores_
            elif name == "Autoencoder":
                train_data = scaled_data[:int(ratio*len(scaled_data))]
                if len(train_data) < sliding_window:
                    continue
                clf.fit(train_data, scaled_data)
                score = clf.decision_scores_
            else:
                clf.fit(X)
                score = clf.decision_scores_
            
            # Post-processing
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            
            # Symmetric padding (TSB-UAD convention)
            full_score = np.array([score[0]]*math.ceil((sliding_window-1)/2) + list(score) + [score[-1]]*((sliding_window-1)//2))
            
            metrics = get_metrics(full_score, label, metric="all", slidingWindow=sliding_window)
            
            file_results.append({
                "baseline_name": original_basename,
                "folder": folder,
                "model": name,
                "corrupted_AUC_ROC": round(metrics["AUC_ROC"], 4),
                "corrupted_AUC_PR": round(metrics["AUC_PR"], 4),
            })
        except Exception as e:
            continue
            
    del models, X, scaled_data, data, label
    gc.collect()
    return file_results

def run_evaluation(corrupt_dir, baseline_csv_path, models_to_run=["MP", "PCA", "IForest"], n_workers=4):
    print(f"--- Starting TS_Corruptor Batch Evaluation ---")
    log_path = os.path.join(corrupt_dir, "experiment_log.json")
    
    if not os.path.exists(log_path):
        print(f"[ERROR] Experiment log not found in {corrupt_dir}.")
        return

    # Load baseline
    if not os.path.exists(baseline_csv_path):
        print(f"[ERROR] Baseline CSV not found: {baseline_csv_path}")
        return
    
    df_base = pd.read_csv(baseline_csv_path)

    with open(log_path, 'r') as f:
        log_data = json.load(f)
        
    stats = log_data.get('dataset_stats', [])
    print(f"[INFO] Found {len(stats)} corrupted files to evaluate.")
    
    # Prepare payload for Multiprocessing
    to_process = []
    for s in stats:
        c_file = os.path.join(corrupt_dir, s['output_file'])
        if os.path.exists(c_file):
            to_process.append({
                'filepath': c_file,
                'baseline_name': s['baseline_name'],
                'folder': s['folder'],
                'models': models_to_run
            })

    print(f"[INFO] Initiating processing using {n_workers} Workers...")
    all_corrupt_results = []
    start_time = time.time()
    
    # Sort files by size to optimize pool scheduling
    to_process.sort(key=lambda x: os.path.getsize(x['filepath']))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_corrupt_file, payload): payload for payload in to_process}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res:
                    all_corrupt_results.extend(res)
                
                completed += 1
                if completed % 10 == 0:
                    print(f"Processed {completed} / {len(to_process)} files...")
            except Exception as e:
                print(f"[WARNING] Worker failed: {e}")
                
    # Create Corrupt DataFrame
    df_corrupt = pd.DataFrame(all_corrupt_results)
    
    # Merge with Baseline
    # Merge on baseline_name and model
    df_merged = pd.merge(df_corrupt, df_base, left_on=['baseline_name', 'model'], right_on=['file', 'model'], how='inner')
    
    # Calculate Degradation metrics
    df_merged['Drop_AUC_ROC'] = df_merged['AUC_ROC'] - df_merged['corrupted_AUC_ROC']
    df_merged['Drop_AUC_PR'] = df_merged['AUC_PR'] - df_merged['corrupted_AUC_PR']
    
    out_csv = os.path.join(corrupt_dir, "evaluation_report.csv")
    df_merged.to_csv(out_csv, index=False)
    
    print(f"\n[FINISH] Evaluation took {time.time() - start_time:.1f}s")
    print(f"[INFO] Saved Evaluation Report to: {out_csv}")
    
    # Output quick average summary
    summary = df_merged.groupby('model')[['AUC_ROC', 'corrupted_AUC_ROC', 'Drop_AUC_ROC']].mean().round(4)
    print("\n--- Average Performance Degradation ---")
    print(summary)
    
if __name__ == "__main__":
    baseline_csv = _PROJECT_ROOT / "results" / "tables" / "baseline_tsb_full_zscore.csv"
    corrupt_dir  = _PROJECT_ROOT / "results" / "corrupted_data" / "adv_combinatorial_test"

    run_evaluation(corrupt_dir, baseline_csv, models_to_run=["MP", "PCA", "IForest"], n_workers=6)
