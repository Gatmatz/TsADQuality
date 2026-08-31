''' 
this file is used to run and obtain the baselines results for TSB_AD dataset.
'''

import os
import sys
import math
import numpy as np
import pandas as pd
import concurrent.futures
from TSB_AD.model_wrapper import run_Unsupervise_AD
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Αγνοούμε τα warnings του sklearn όταν το precision είναι 0 επειδή δεν προβλέφθηκαν ανωμαλίες
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # Για κάθε ενδεχόμενο διαίρεσης με το μηδέν

# Λίστα με τα μοντέλα που θέλουμε να τρέξουμε
# Μπορείς να προσθέσεις/αφαιρέσεις μοντέλα από εδώ (π.χ. 'IForest', 'IForest_Wrapper', 'MP', 'Sub_LOF', 'Sub_PCA', 'AE', 'KShapeAD')
MODELS_TO_RUN = ['Sub_PCA']

# Seeding exactly as paper
import random
import torch
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Βελτιστοποίηση: Περιορισμός threads για αποφυγή CPU Thrashing
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


current_file_path = os.path.abspath(__file__) 
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
tsb_ad_path = os.path.join(project_root, 'TSB-AD')
src_path = os.path.join(project_root, 'src')

'''
προσθήκη των διαδρομών tsb_ad_path και src_path στο sys.path για να εξασφαλιστεί ότι τα modules μπορούν να βρεθούν και να εισαχθούν σωστά.
'''
def add_path_to_sys():
    for path in [tsb_ad_path, src_path]:
        if path not in sys.path:
            sys.path.insert(0, path)
            
add_path_to_sys()
    
 
def get_tsb_ad_eval()-> list:
    # Ανάγνωση του αρχείου CSV που περιέχει τη λίστα των αρχείων αξιολόγησης για το TSB-AD dataset.
    file_path = os.path.join(tsb_ad_path, 'Datasets', 'File_List', 'TSB-AD-U-Eva.csv')
    data_dir = os.path.join(tsb_ad_path, 'Datasets', 'TSB-AD-U')  # Καθορισμός του καταλόγου δεδομένων για το TSB-AD dataset.
    
    files = []
    with open(file_path, 'r') as f:
        # κανουμε skip την πρώτη γραμμή (header) του αρχείου CSV
        next(f)
        for line in f: 
            line = line.strip()
            path = os.path.join(data_dir, line)
            files.append(path)
        
    return files

def run_single_job(file_path):
    from TSB_AD.models.IForest import IForest
    from TSB_AD.models.MatrixProfile import MatrixProfile
    from TSB_AD.models.LOF import LOF
    from TSB_AD.models.AE import AutoEncoder
    from TSB_AD.models.PCA import PCA
    from TSB_AD.models.SAND import SAND
    from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank
    from sklearn.preprocessing import MinMaxScaler
    import gc
    
    try:
        data = pd.read_csv(file_path)
        data.columns = [str(col).lower().strip() for col in data.columns]
        ts_data = data['value'].values if 'value' in data.columns else (data['data'].values if 'data' in data.columns else data.iloc[:, 0].values)
        labels = data['label'].values if 'label' in data.columns else data.iloc[:, -1].values
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    # Σταθερό παράθυρο 100 (χρησιμοποιείται από IForest και AutoEncoder)
    fixed_window = 100
    
    file_results = []

    # Φόρτωση των Optimal HPs
    # Για το IForest παίρνουμε το απλό 'IForest' (που πετυχαίνει τη max απόδοση 0.298) 
    # και ΟΧΙ το 'Sub_IForest' (0.223).
    iforest_hp = Optimal_Uni_algo_HP_dict.get('IForest', {}).copy()
    
    # Για τα υπόλοιπα κρατάμε τις 'Sub_' εκδόσεις (γιατί π.χ. στο LOF το Sub_LOF κερδίζει κατά κράτος)
    lof_hp     = Optimal_Uni_algo_HP_dict.get('Sub_LOF', {}).copy()
    pca_hp     = Optimal_Uni_algo_HP_dict.get('Sub_PCA', {}).copy()
    ae_hp      = Optimal_Uni_algo_HP_dict.get('AutoEncoder', {}).copy()
    
    # Εξαγωγή του periodicity (μόνο για τα Sub- μοντέλα)
    lof_period     = lof_hp.pop('periodicity', 2)
    pca_period     = pca_hp.pop('periodicity', 1)
    
    # Καθαρισμός παραμέτρων
    ae_hp.pop('window_size', None)
         
    # Υπολογισμός του slidingWindow δυναμικά μόνο για όσα το χρειάζονται
    windows = {
        'IForest':     fixed_window,  # Σταθερό 100 για max απόδοση!
        'IForest_Wrapper': fixed_window,
        'Sub_LOF':     find_length_rank(ts_data, rank=lof_period),
        'Sub_PCA':     find_length_rank(ts_data, rank=pca_period),
        'MP':          find_length_rank(ts_data, rank=1),
        'KShapeAD':    find_length_rank(ts_data, rank=1),
        'AE':          fixed_window   # Σταθερό 100
    }
         
    # Αρχικοποίηση Μοντέλων (μόνο όσα επιλέχθηκαν)
    all_models = {
        'IForest':     IForest(**iforest_hp, slidingWindow=windows['IForest'], random_state=42),
        'IForest_Wrapper': 'Wrapper',
        'MP':          MatrixProfile(window=windows['MP']),
        'Sub_LOF':     LOF(**lof_hp, slidingWindow=windows['Sub_LOF']),
        'Sub_PCA':     PCA(**pca_hp, slidingWindow=windows['Sub_PCA']),
        'AE':          AutoEncoder(**ae_hp, slidingWindow=windows['AE'], epochs=50, batch_size=128),
        'KShapeAD':    SAND(pattern_length=windows['KShapeAD'], subsequence_length=4 * windows['KShapeAD'])
    }

    # Χρησιμοποιούμε τη global μεταβλητή
    active_models = {k: v for k, v in all_models.items() if k in MODELS_TO_RUN}

    for name, model in active_models.items():
        w = windows[name] # Το δυναμικό window του συγκεκριμένου μοντέλου
        
        try:
            if name == 'IForest_Wrapper':
                # Κάνουμε απευθείας κλήση στον αυτούσιο wrapper κώδικα της βιβλιοθήκης (TSB-AD/model_wrapper.py)
                scores = run_Unsupervise_AD('IForest', ts_data.reshape(-1, 1), **iforest_hp)
                full_score = scores
            elif name == 'KShapeAD':
                model.fit(ts_data.squeeze(), overlaping_rate=int(1.5 * w))
                scores = model.decision_scores_
            elif name == 'AE':
                model.fit(ts_data.reshape(-1, 1))
                scores = model.decision_function(ts_data.reshape(-1, 1))
            else:
                model.fit(ts_data.reshape(-1, 1))
                scores = getattr(model, 'decision_scores_', None)
                if scores is None:
                    scores = model.decision_function(ts_data.reshape(-1, 1))
            
            # 100% πιστό στο paper: Τα raw scores πάνε ΑΠΕΥΘΕΙΑΣ στο get_metrics
            # ΧΩΡΙΣ MinMaxScaler, ΧΩΡΙΣ extra padding (το LOF κάνει ήδη padding εσωτερικά)
            full_score = scores.ravel()
                            
            # Το paper χρησιμοποιεί ΠΑΝΤΑ δυναμικό window (rank=1) για την αξιολόγηση (get_metrics) ασχέτως μοντέλου!
            metrics_window = find_length_rank(ts_data, rank=1)
            metrics = get_metrics(full_score, labels, slidingWindow=metrics_window)
            
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
            })
        except Exception as e:
            pass

    del ts_data, labels
    gc.collect()

    return file_results

def run_jobs_in_parallel(max_workers=3, files_paths=None, out_dir='results'):
    if not files_paths:
        return
        
    os.makedirs(out_dir, exist_ok=True)
    final_file = os.path.join(out_dir, 'results_final_tsb_ad.csv')
    
    header_written = False
    completed_files = 0
    
    # ------------------ LOGIC ΓΙΑ RESUME ------------------
    if os.path.exists(final_file):
        try:
            # Διαβάζουμε ποια αρχεία έχουν ήδη ολοκληρωθεί
            df_existing = pd.read_csv(final_file)
            if not df_existing.empty and 'file' in df_existing.columns:
                header_written = True
                
                # Βρίσκουμε πόσα μοντέλα τρέξαμε ανά αρχείο
                counts = df_existing.groupby('file')['model'].nunique()
                # Ένα αρχείο θεωρείται ολοκληρωμένο αν έχει τρέξει για όλα τα ζητούμενα μοντέλα
                completed_file_names = set(counts[counts >= len(MODELS_TO_RUN)].index)
                
                # Φιλτράρουμε τη λίστα ώστε να τρέξουμε ΜΟΝΟ όσα απομένουν!
                files_paths = [p for p in files_paths if os.path.basename(p) not in completed_file_names]
                completed_files = len(completed_file_names)
                print(f"Resuming... Found {completed_files} already processed files.")
        except Exception as e:
            print(f"Could not read existing CSV for resume: {e}")
            
    total_files_to_run = len(files_paths)
    
    if total_files_to_run == 0:
        print("All files are already processed for these models!")
        return
    
    print(f"Starting parallel execution for remaining {total_files_to_run} files with {max_workers} workers...")
    print(f"Results will be continuously saved (checkpointed) to: {final_file}")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_job, path): path for path in files_paths}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                # Μετατροπή των αποτελεσμάτων του ΕΝΟΣ αρχείου σε DataFrame
                df_res = pd.DataFrame(res)
                
                # Checkpoint: Αποθήκευση/Append απευθείας στο CSV 
                df_res.to_csv(final_file, mode='a', header=not header_written, index=False)
                header_written = True
                
            completed_files += 1
            # Εκτυπώνουμε την πρόοδο (αφαιρέσαμε το \r γιατί σε κάποια terminals δεν φαίνεται)
            print(f"Progress: {completed_files} / {total_files_to_run} files processed.", flush=True)

    print(f"\nFinished processing all {completed_files} files.")
    print(f"Final results are fully saved at {final_file}")

if __name__ == "__main__":
    from datetime import datetime
    
    eval_files = get_tsb_ad_eval()
    print(f"Number of evaluation files: {len(eval_files)}")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Φτιάχνουμε φάκελο με βάση τα μοντέλα που επιλέχθηκαν
    folder_name = "run_" + "_".join(MODELS_TO_RUN) + "_v2"
    out_directory = os.path.join(project_root, 'results', 'baselines', folder_name)
    
    # Εκτέλεση 
    run_jobs_in_parallel(max_workers=10, files_paths=eval_files, out_dir=out_directory)