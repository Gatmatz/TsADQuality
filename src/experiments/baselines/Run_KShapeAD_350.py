import pandas as pd
import numpy as np
import os
import sys
import concurrent.futures

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_root, 'TSB-AD'))

from TSB_AD.model_wrapper import run_Unsupervise_AD
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

import torch
import random
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Seeding ακριβώς όπως στο paper
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

dataset_dir = os.path.join(project_root, 'TSB-AD', 'Datasets', 'TSB-AD-U')
Optimal_Det_HP = Optimal_Uni_algo_HP_dict['KShapeAD']

def process_file(filename):
    file_path = os.path.join(dataset_dir, filename)
    
    try:
        # 1. Φόρτωση δεδομένων
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        
        # 2. Εύρεση παραθύρου
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        
        # 3. Εκτέλεση KShapeAD από τον wrapper
        output = run_Unsupervise_AD('KShapeAD', data, **Optimal_Det_HP)
        
        # 4. Υπολογισμός Metrics
        metrics = get_metrics(output, label, slidingWindow=slidingWindow)
        
        return {
            'file': filename,
            'VUS_PR': metrics.get('VUS-PR', 0),
            'VUS_ROC': metrics.get('VUS-ROC', 0),
            'AUC_PR': metrics.get('AUC-PR', 0),
            'AUC_ROC': metrics.get('AUC-ROC', 0),
            'Standard-F1': metrics.get('Standard-F1', 0),
            'PA-F1': metrics.get('PA-F1', 0)
        }
    except Exception as e:
        return None

def main():
    # 350 αρχεία (πλήρες benchmark eval list)
    file_list_path = os.path.join(project_root, 'TSB-AD', 'Datasets', 'File_List', 'TSB-AD-U-Eva.csv')
    file_list = pd.read_csv(file_list_path)['file_name'].values
    
    print(f'Starting KShapeAD for {len(file_list)} files...')
    print('HPs:', Optimal_Det_HP)
    
    results = []
    completed = 0
    
    # Το KShapeAD μπορεί να είναι πιο αργό, οπότε η παραλληλοποίηση βοηθάει πάρα πολύ
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_file, f): f for f in file_list}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
            
            completed += 1
            print(f"Progress: {completed}/{len(file_list)} files completed.", flush=True)

    df_results = pd.DataFrame(results)
    
    out_dir = os.path.join(project_root, 'results', 'baselines', 'run_KShapeAD_350')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'results_final_tsb_ad.csv')
    df_results.to_csv(out_file, index=False)
    
    print('\n=======================================')
    print(f'ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ KShapeAD ΓΙΑ {len(df_results)} ΑΡΧΕΙΑ')
    print('=======================================')
    print(df_results[['AUC_ROC', 'AUC_PR', 'VUS_ROC', 'VUS_PR', 'Standard-F1', 'PA-F1']].mean().round(4).to_string())
    print(f'\nΑποθηκεύτηκαν στο: {out_file}')

if __name__ == '__main__':
    main()
