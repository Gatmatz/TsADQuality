import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Fix sys.path to easily import the whole project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.pca import PCA
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

def evaluate_model(X, y, sw, target_model_name):
    Xc = np.ascontiguousarray(X, dtype=np.float64)
    Xf = Window(window=sw).convert(Xc).to_numpy()
    m = IForest(n_jobs=1) if target_model_name == 'IForest' else PCA()
    m.fit(Xf)
    sc = np.pad(np.array(m.decision_scores_, dtype=np.float64),
               (sw-1,0), 'constant', constant_values=(m.decision_scores_[0],))
    sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
    return get_metrics(sc, y, metric="all", slidingWindow=sw)

if __name__ == "__main__":
    target_csv = "results/tables/robust_subset_TSB.csv"
    df_registry = pd.read_csv(target_csv).head(2)
    
    print("=== PER-MODEL ADVERSARIAL DROPS ===\n")
    
    targets = ['IForest', 'PCA']
    
    for target in targets:
        print(f"--- TARGET: {target} ---")
        adv_dir = Path(f"outputs/data_corrupted/Distillation_{target}_a0.05")
        
        for idx, row in df_registry.iterrows():
            dataset_name = row['baseline_name']
            orig_file_path = row['filepath']
            adv_file_path = adv_dir / f"{dataset_name}.csv"
            
            if not adv_file_path.exists():
                continue
                
            df_orig = pd.read_csv(orig_file_path, header=None)
            X_orig = df_orig[0].to_numpy('float')
            y_orig = df_orig[1].to_numpy('int')
            
            mask = ~np.isnan(X_orig)
            X_orig = X_orig[mask]
            y_orig = y_orig[mask]
            sw = max(int(find_length(X_orig)), 10)
            
            df_adv = pd.read_csv(adv_file_path, header=None)
            X_adv = df_adv[0].to_numpy('float')
            
            m_orig = evaluate_model(X_orig, y_orig, sw, target)
            m_adv = evaluate_model(X_adv, y_orig, sw, target)
            
            auc_orig = m_orig['AUC_ROC']
            auc_adv = m_adv['AUC_ROC']
            drop = auc_orig - auc_adv
            
            print(f"Dataset:     {dataset_name}")
            print(f"  Clean AUC: {auc_orig:.4f}")
            print(f"  Adv AUC:   {auc_adv:.4f}")
            print(f"  Drop:      {drop:+.4f}\n")
