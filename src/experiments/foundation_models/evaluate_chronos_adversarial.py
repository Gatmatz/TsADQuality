import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from pathlib import Path

# Fix sys.path to easily import the whole project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

try:
    from chronos import BaseChronosPipeline
except ImportError:
    print("[ERROR] 'chronos-forecasting' library not found. Please install chronos package.")
    sys.exit(1)

from src.utils.logger import log_experiment
from TSB_UAD.vus.metrics import get_metrics
from TSB_UAD.utils.slidingWindows import find_length

def get_chronos_anomaly_scores(pipeline, X, seq_len=512, batch_size=16):
    """
    Runs Chronos prediction over the whole time series using sliding windows and calculating uncertainty-normalized scores.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X.reshape(-1, 1)).ravel()

    limit = 5000 # Evaluate up to limit points to save time for huge sequences
    start_i = seq_len
    end_i = min(len(data_scaled), seq_len + limit)
    
    contexts = []
    targets = []
    for i in range(start_i, end_i):
        contexts.append(data_scaled[i-seq_len : i])
        targets.append(data_scaled[i])
    
    if len(contexts) == 0:
        return np.zeros(len(X))

    prediction_errors = []
    
    for b in tqdm(range(0, len(contexts), batch_size), desc="Chronos Inference", leave=False):
        batch_x = contexts[b : b + batch_size]
        batch_y = targets[b : b + batch_size]
        
        context_tensor = torch.tensor(np.array(batch_x), dtype=torch.float32)
        if len(context_tensor.shape) == 2:
            context_tensor = context_tensor.unsqueeze(1)
            
        with torch.no_grad():
            forecast = pipeline.predict(context_tensor, prediction_length=1)
            if isinstance(forecast, list):
                forecast = torch.stack(forecast)
                
            batch_size_val = len(batch_y)
            batch_dim = -1
            for d in range(forecast.ndim):
                if forecast.shape[d] == batch_size_val:
                    batch_dim = d
                    break
                    
            samples_dim = 1 if batch_dim == 0 else 0 
            
            median_tensor = torch.quantile(forecast, 0.5, dim=samples_dim)
            q05_tensor = torch.quantile(forecast, 0.05, dim=samples_dim)
            q95_tensor = torch.quantile(forecast, 0.95, dim=samples_dim)
            
            median = median_tensor.reshape(batch_size_val, -1).mean(dim=1).cpu().numpy()
            q05 = q05_tensor.reshape(batch_size_val, -1).mean(dim=1).cpu().numpy()
            q95 = q95_tensor.reshape(batch_size_val, -1).mean(dim=1).cpu().numpy()
            
            iqr = q95 - q05
            norm_error = np.abs(np.array(batch_y) - median) / (iqr + 1e-6)
            prediction_errors.extend(norm_error)

    full_scores = np.zeros(len(X))
    full_scores[start_i : end_i] = prediction_errors
    
    if prediction_errors:
        full_scores[:start_i] = prediction_errors[0]
        if end_i < len(X):
            full_scores[end_i:] = prediction_errors[-1]

    # Normalize scores to 0-1
    full_scores = MinMaxScaler().fit_transform(full_scores.reshape(-1, 1)).ravel()
    return full_scores

def evaluate_chronos_on_dataset(dataset_name, orig_path, adv_path, pipeline, exp_name):
    """
    Evaluates Chronos on both original and adversarial data and logs the results.
    """
    print(f"\n--- Evaluating {dataset_name} | {exp_name} ---")
    
    # 1. Load Original Data
    df_orig = pd.read_csv(orig_path, header=None)
    X_orig = df_orig[0].to_numpy('float')
    y_orig = df_orig[1].to_numpy('int')
    
    mask = ~np.isnan(X_orig)
    X_orig = X_orig[mask]
    y_orig = y_orig[mask]
    
    sw = max(int(find_length(X_orig)), 10)
    
    print("  [Chronos] Running on Clean Original Data...")
    scores_orig = get_chronos_anomaly_scores(pipeline, X_orig)
    m_orig = get_metrics(scores_orig, y_orig, metric="all", slidingWindow=sw)
    print(f"    -> Original AUC-ROC: {m_orig.get('AUC_ROC', 0.0):.3f}")

    # 2. Load Adversarial Data
    if not os.path.exists(adv_path):
        print(f"  [ERROR] Adversarial file {adv_path} not found. Skipping.")
        return
        
    df_adv = pd.read_csv(adv_path, header=None)
    X_adv = df_adv[0].to_numpy('float')
    y_adv = df_adv[1].to_numpy('int')
    
    mask = ~np.isnan(X_adv)
    X_adv = X_adv[mask]
    y_adv = y_adv[mask]
    
    print(f"  [Chronos] Running on {exp_name} Adversarial Data...")
    scores_adv = get_chronos_anomaly_scores(pipeline, X_adv)
    m_adv = get_metrics(scores_adv, y_adv, metric="all", slidingWindow=sw)
    print(f"    -> Adversarial AUC-ROC: {m_adv.get('AUC_ROC', 0.0):.3f}")
    
    # Calculate drops manually for immediate display, logger calculates internally too
    auc_drop = m_orig.get('AUC_ROC', 0.0) - m_adv.get('AUC_ROC', 0.0)
    print(f"  [RESULT] Chronos AUC Drop: {auc_drop:.3f}")
    
    # 3. Log using the unified central logger!
    final_metrics = {
        "AUC_ROC_Original": m_orig.get('AUC_ROC', 0.0),
        "AUC_ROC_Corrupted": m_adv.get('AUC_ROC', 0.0),
        "AUC_Drop": auc_drop,
        "VUS_ROC_Original": m_orig.get('VUS_ROC', 0.0),
        "VUS_ROC_Corrupted": m_adv.get('VUS_ROC', 0.0)
    }
    
    log_experiment(
        model_name="Chronos",
        experiment_name=exp_name,
        dataset_name=dataset_name,
        parameters={"alpha_factor": 0.05},
        metrics=final_metrics
    )

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Loading amazon/chronos-2 model on {device}...")
    try:
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2", 
            device_map=device,
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(f"[ERROR] Could not load Chronos model: {e}")
        sys.exit(1)

    target_csv = "results/tables/robust_subset_TSB.csv"
    if not os.path.exists(target_csv):
        print(f"[ERROR] Registry {target_csv} not found.")
        sys.exit(1)
        
    df = pd.read_csv(target_csv).head(2) # Limit to 2 for the initial test run
    
    adv_dirs = [
        ("Ensemble_SGM_a0.05", Path("outputs/data_corrupted/Ensemble_SGM_a0.05")),
        ("Distillation_IForest_a0.05", Path("outputs/data_corrupted/Distillation_IForest_a0.05")),
        ("Distillation_PCA_a0.05", Path("outputs/data_corrupted/Distillation_PCA_a0.05"))
    ]
    
    for idx, row in df.iterrows():
        dataset_name = row['baseline_name']
        orig_file_path = row['filepath']
        
        for exp_name, adv_dir in adv_dirs:
            adv_file_path = adv_dir / f"{dataset_name}.csv"
            if adv_file_path.exists():
                evaluate_chronos_on_dataset(dataset_name, orig_file_path, str(adv_file_path), pipeline, exp_name)
    
    print("\n[Done] All Chronos adversarial evaluations complete.")
