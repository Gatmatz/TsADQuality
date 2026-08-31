"""
Run IForest & MP on the representative 350-file subset from TSB-UAD.
Uses TSB-UAD data format (.out files with 2 columns: data, label).
3 parallel processes, computes VUS-PR + all metrics.
"""
import warnings
warnings.filterwarnings('ignore')
import math
import os
import gc
import sys
import numpy as np
import pandas as pd
import csv
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import defaultdict

# ── Paths ──
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
tsb_ad_path = os.path.join(project_root, 'TSB-AD')
tsb_uad_path = os.path.join(project_root, 'TSB-UAD')
src_path = os.path.join(project_root, 'src')

for path in [tsb_ad_path, src_path]:
    if path not in sys.path:
        sys.path.insert(0, path)


def get_subset_files():
    """Load the representative 350-file subset with resolved paths."""
    resolved_file = os.path.join(tsb_uad_path, 'result', 'representative_subset_350_resolved.csv')
    files = []
    with open(resolved_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            files.append(row['path'])
    return files


def run_single_job(file_path, model_names=None, out_put_path=None):
    """Run IForest and MP on a single TSB-UAD file."""
    
    # Resume: skip if already done
    if out_put_path:
        os.makedirs(out_put_path, exist_ok=True)
        result_filename = os.path.basename(file_path).replace('.out', '_results.csv').replace('.txt', '_results.csv')
        result_file = os.path.join(out_put_path, result_filename)
        if os.path.exists(result_file):
            try:
                print(f"Skipping {os.path.basename(file_path)}, already processed.")
                return pd.read_csv(result_file).to_dict('records')
            except Exception:
                pass
    else:
        result_file = None
    
    try:
        # TSB-UAD format: .out files with no header, 2 columns (data, label)
        # Some may be CSV with headers
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
            data.columns = [str(col).lower().strip() for col in data.columns]
            if 'value' in data.columns:
                ts_data = data['value'].values.astype(float)
            elif 'data' in data.columns:
                ts_data = data['data'].values.astype(float)
            else:
                ts_data = data.iloc[:, 0].values.astype(float)
            if 'label' in data.columns:
                labels = data['label'].values.astype(int)
            else:
                labels = data.iloc[:, -1].values.astype(int)
        else:
            # .out or .txt format: no header, space/comma separated
            raw = pd.read_csv(file_path, header=None, sep=r'\s+|,', engine='python')
            ts_data = raw.iloc[:, 0].values.astype(float)
            labels = raw.iloc[:, -1].values.astype(int)
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
    
    file_results = []
    
    from TSB_AD.models.IForest import IForest
    from TSB_AD.models.MatrixProfile import MatrixProfile
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_UAD.models.feature import Window
    from TSB_UAD.utils.slidingWindows import find_length
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(ts_data.reshape(-1, 1)).ravel()
    
    sliding_window = find_length(scaled_data)
    sliding_window = max(int(sliding_window), 10)
    
    all_models = {
        'IForest': IForest(n_estimators=100, random_state=42, slidingWindow=sliding_window),
        'MP': MatrixProfile(window=sliding_window)
    }
    
    models = {k: v for k, v in all_models.items() if model_names is None or k in model_names}
    
    for name, model in models.items():
        try:
            model.fit(scaled_data.reshape(-1, 1))
            
            scores = getattr(model, 'decision_scores_', None)
            if scores is None:
                scores = model.decision_function(scaled_data.reshape(-1, 1))
            
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
            
            # Padding if needed
            if len(score) < len(labels):
                full_score = np.array(
                    [score[0]] * math.ceil((sliding_window - 1) / 2) +
                    list(score) +
                    [score[-1]] * ((sliding_window - 1) // 2)
                )
            else:
                full_score = score
            
            # Ensure same length
            min_len = min(len(full_score), len(labels))
            full_score = full_score[:min_len]
            cur_labels = labels[:min_len]
            
            metrics = get_metrics(full_score, cur_labels, slidingWindow=sliding_window)
            
            file_results.append({
                "file": os.path.basename(file_path),
                "dataset": os.path.basename(os.path.dirname(file_path)),
                "model": name,
                "n_anomalies": int(cur_labels.sum()),
                "anomaly_ratio": round(cur_labels.sum() / len(cur_labels), 4),
                "AUC_ROC": metrics.get("AUC-ROC", 0),
                "AUC_PR": metrics.get("AUC-PR", 0),
                "VUS_ROC": metrics.get("VUS-ROC", 0),
                "VUS_PR": metrics.get("VUS-PR", 0),
                "Standard-F1": metrics.get("Standard-F1", 0),
                "PA-F1": metrics.get("PA-F1", 0),
                "Event-based-F1": metrics.get("Event-based-F1", 0),
                "R-based-F1": metrics.get("R-based-F1", 0),
                "Affiliation-F": metrics.get("Affiliation-F", 0),
            })
        except Exception as e:
            import traceback
            print(f"Error with model {name} on {os.path.basename(file_path)}: {e}")
            print(traceback.format_exc())
    
    if result_file and file_results:
        pd.DataFrame(file_results).to_csv(result_file, index=False)
    
    del ts_data, labels, scaled_data
    try:
        del models
    except:
        pass
    gc.collect()
    
    return file_results


def run_jobs_in_parallel(max_workers=3, files_paths=None, model_names=None, out_put_path=None):
    if not files_paths:
        return []
    
    os.makedirs(out_put_path, exist_ok=True)
    checkpoint_file = os.path.join(out_put_path, 'results_checkpoint.csv')
    
    all_results = []
    total_files = len(files_paths)
    completed_files = 0
    
    print(f"Starting parallel execution for {total_files} files with {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_job, path, model_names, out_put_path): path
            for path in files_paths
        }
        
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                res = future.result()
                if res:
                    all_results.extend(res)
            except Exception as exc:
                print(f"\n[Error] {path}: {exc}")
            
            completed_files += 1
            print(f"Progress: {completed_files}/{total_files} files processed.", end='\r', flush=True)
            
            if completed_files % 10 == 0 and all_results:
                pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
                print(f"\n[Checkpoint] Saved {len(all_results)} results")
    
    print(f"\nFinished processing all {total_files} files.")
    
    if all_results:
        final_file = os.path.join(out_put_path, 'results_final.csv')
        pd.DataFrame(all_results).to_csv(final_file, index=False)
        print(f"Saved final results to {final_file}")
        
        # Print summary
        df = pd.DataFrame(all_results)
        print(f"\n{'='*60}")
        print("SUMMARY: Mean VUS-PR & AUC-PR per algorithm")
        print(f"{'='*60}")
        for model in df['model'].unique():
            mask = df['model'] == model
            vus = df.loc[mask, 'VUS_PR'].mean()
            auc = df.loc[mask, 'AUC_PR'].mean()
            n = mask.sum()
            print(f"  {model}: VUS-PR = {vus:.6f}, AUC-PR = {auc:.6f} (n={n})")
    
    return all_results


if __name__ == "__main__":
    files = get_subset_files()
    print(f"Loaded {len(files)} files from representative subset.")
    
    out_dir = os.path.join(project_root, 'eval_subset_350')
    
    run_jobs_in_parallel(
        max_workers=3,
        files_paths=files,
        model_names=['IForest', 'MP'],
        out_put_path=out_dir
    )
