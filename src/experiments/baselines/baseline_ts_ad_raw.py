import warnings

from sympy import im
warnings.filterwarnings('ignore')
import math
import os

# Βελτιστοποίηση: Περιορισμός threads για αποφυγή CPU Thrashing
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import time
import sys
import numpy as np
import pandas as pd
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# this function will load all the files from a directory 350 TS-ad 

current_file_path = os.path.abspath(__file__) 
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
tsb_ad_path = os.path.join(project_root, 'TSB-AD')
src_path = os.path.join(project_root, 'src')

def  add_path_to_sys():
    
    for path in [tsb_ad_path, src_path]:
        if path not in sys.path:
            sys.path.insert(0, path)

add_path_to_sys() # Κλήση για τα worker processes
def get_tsb_ad_eval():
    """ Get TSB-AD evalaution files"""
    file_path = os.path.join(tsb_ad_path, 'Datasets', 'File_List', 'TSB-AD-U-Eva.csv')
    data_dir = os.path.join(tsb_ad_path, 'Datasets', 'TSB-AD-U')
    
    files = [] 
    with open(file_path, 'r') as f:
        # Αγνοούμε την πρώτη γραμμή αν είναι header ('file_name')
        first_line = True
        for line in f:
            filename = line.strip()
            if first_line and filename == 'file_name':
                first_line = False
                continue
            first_line = False
            
            # Ενώνουμε το σωστό data directory με το όνομα του αρχείου
            path = os.path.join(data_dir, filename)
            files.append(path)
    
    return files

def get_tsb_ad_tuning():
    """ Get TSB-AD tuning files"""
    file_path = os.path.join(tsb_ad_path, 'Datasets', 'File_List', 'TSB-AD-U-Tuning.csv')
    data_dir = os.path.join(tsb_ad_path, 'Datasets', 'TSB-AD-U')
    
    files = []
    with open(file_path, 'r') as f:
        first_line = True
        for line in f:
            filename = line.strip()
            if first_line and filename == 'file_name':
                first_line = False
                continue
            first_line = False
            
            path = os.path.join(data_dir, filename)
            files.append(path)
    
    return files

def run_single_job(file_path=None, model_names=None, out_put_path=None, mode='eval'):
    
    # Επιλογή Resume: Αν υπάρχει το αποτέλεσμα, το παραλείπουμε
    if out_put_path is not None and file_path is not None:
        os.makedirs(out_put_path, exist_ok=True)
        result_filename = os.path.basename(file_path).replace('.csv', '_results.csv')
        result_file = os.path.join(out_put_path, result_filename)
        if os.path.exists(result_file):
            try:
                print(f"Skipping {file_path}, already processed.")
                return pd.read_csv(result_file).to_dict('records')
            except Exception:
                pass
    else:
        result_file = None
        
    try:
        if file_path is None or out_put_path is None:
            raise ValueError("file_path and out_put_path must be provided.")
            
        data = pd.read_csv(file_path)
        
        # Κανονικοποίηση των ονομάτων των στηλών σε μικρά γράμματα χωρίς κενά (π.χ. 'Label' -> 'label', 'Data' -> 'data')
        data.columns = [str(col).lower().strip() for col in data.columns]
        
        # Ανάγνωση του timeseries (αποφεύγουμε το DataFrame reshape error)
        if 'value' in data.columns:
            ts_data = data['value'].values
        elif 'data' in data.columns:
            ts_data = data['data'].values
        else:
            ts_data = data.iloc[:, 0].values
            
        if 'label' in data.columns:
            labels = data['label'].values
        else:
            labels = data.iloc[:, -1].values # Fallback στην τελευταία στήλη
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
    
    file_results = []
    
    if mode == 'eval':
        from TSB_AD.models.IForest import IForest
        from TSB_AD.models.MatrixProfile import MatrixProfile
        from TSB_AD.evaluation.metrics import get_metrics
        from TSB_UAD.models.feature import Window
        from TSB_UAD.utils.slidingWindows import find_length
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        
        sliding_window = find_length(ts_data)
        sliding_window = max(int(sliding_window), 10)

        X = Window(window=sliding_window).convert(ts_data).to_numpy()
             
        all_models = {
            'IForest': IForest(n_estimators=100, random_state=42, slidingWindow=sliding_window),
            'MP': MatrixProfile(window=sliding_window)
        }
        
        models = {k: v for k, v in all_models.items() if model_names is None or k in model_names}
        
        for name, model in models.items():
            try:
                if name in ['IForest', 'MP']:
                    # Το IForest και το MP κάνουν windowing εσωτερικά. Τους δίνουμε τα αρχικά 1D δεδομένα 
                    # μορφοποιημένα ως στήλη (N, 1) για να μη σκάσουν τα `.shape`.
                    model.fit(ts_data.reshape(-1, 1))
                else:
                    model.fit(X)
                    
                scores = getattr(model, 'decision_scores_', None)
                if scores is None:
                    if name in ['IForest', 'MP']:
                        scores = model.decision_function(ts_data.reshape(-1, 1))
                    else:
                        scores = model.decision_function(X)
                
                # Post-processing: κανονικοποίηση
                score = MinMaxScaler(feature_range=(0,1)).fit_transform(scores.reshape(-1,1)).ravel() 
                
                # Εάν το μοντέλο επέστρεψε μικρότερο πίνακα (π.χ. IForest λόγω window), κάνουμε padding
                if len(score) < len(labels):
                    full_score = np.array([score[0]]*math.ceil((sliding_window-1)/2) + list(score) + [score[-1]]*((sliding_window-1)//2)) 
                else:
                    full_score = score
                                
                metrics = get_metrics(full_score, labels, slidingWindow=sliding_window)
                file_results.append({
                    "file": os.path.basename(file_path),
                    "folder": os.path.basename(os.path.dirname(file_path)),
                    "model": name,
                    "n_anomalies": int(labels.sum()),
                    "anomaly_ratio" : round(labels.sum() / len(labels), 4),
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
                print(f"Error with model {name} on {file_path}: {e}")
                print(traceback.format_exc())
                
        if result_file and file_results:
            pd.DataFrame(file_results).to_csv(result_file, index=False)

    # Καθαρισμός μνήμης για να αποφύγουμε διαρροές
    del data, labels, ts_data
    if mode == 'eval':
        try:
            del X, models
        except Exception:
            pass
    gc.collect()
    
    return file_results

# Concurrency execution (με ProcessPoolExecutor)
def run_jobs_in_parallel(max_workers=2, files_paths=None, model_names=None, out_put_path=None, mode='eval'):
    if not files_paths:
        return []
    
    checkpoint_file = None
    if out_put_path:
        os.makedirs(out_put_path, exist_ok=True)
        checkpoint_file = os.path.join(out_put_path, 'results_checkpoint.csv')
        
    all_results = []
    total_files = len(files_paths)
    completed_files = 0
    
    print(f"Starting parallel execution for {total_files} files with {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_job, path, model_names, out_put_path, mode): path 
            for path in files_paths
        }
        
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                res = future.result()
                if res:
                    all_results.extend(res)
            except Exception as exc:
                print(f"\n[Error] {path} generated an exception: {exc}")
                
            completed_files += 1
            
            # Ενημέρωση προόδου (progress) στην ίδια γραμμή
            print(f"Progress: {completed_files}/{total_files} files processed.", end='\r', flush=True)
            
            # Checkpoint κάθε 10 αρχεία
            if checkpoint_file and completed_files % 10 == 0:
                if all_results:
                    pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
                    # Χρησιμοποιούμε end='\n' για να μην σβηστεί από το επόμενο progress
                    print(f"\n[Checkpoint] Saved intermediate results ({len(all_results)} metrics) to {checkpoint_file}")

    print(f"\nFinished processing all {total_files} files.")
    
    if out_put_path and all_results:
        final_file = os.path.join(out_put_path, 'results_final.csv')
        pd.DataFrame(all_results).to_csv(final_file, index=False)
        print(f"Saved final results to {final_file}")
                
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run TSB-AD experiments')
    
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'tuning'], help='Mode to run: eval or tuning')
    parser.add_argument('--workers', type=int, default=5, help='Number of max workers for parallel execution')
    parser.add_argument('--out_dir', type=str, default='results', help='Output directory for the results')
    
    args = parser.parse_args()
    
    if args.mode == 'eval':
        files_paths = get_tsb_ad_eval()
    else:
        files_paths = get_tsb_ad_tuning()
        
    print(f"Loaded {len(files_paths)} files for {args.mode} mode.")
    
    # Εκκίνηση των παράλληλων jobs
    run_jobs_in_parallel(max_workers=args.workers, files_paths=files_paths, model_names=['IForest', 'MP'], out_put_path=args.out_dir, mode=args.mode)