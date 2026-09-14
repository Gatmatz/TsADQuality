import torch
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import time
import math

from TSB_UAD.vus.metrics import get_metrics

# Try to import Chronos from the official library
try:
    from chronos import BaseChronosPipeline
except ImportError:
    print("[ERROR] 'chronos-forecasting' library not found.")
    sys.exit(1)

def run_chronos2_normalized_evaluation(subset_csv, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load Chronos-2 Model
    print("[INFO] Loading amazon/chronos-2 model...")
    try:
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2", 
            device_map=device,
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    # 2. Load Subset
    if not os.path.exists(subset_csv):
        print(f"[ERROR] Subset file {subset_csv} not found.")
        return
    
    df_subset = pd.read_csv(subset_csv)
    print(f"[INFO] Running Normalized Chronos-2 on {len(df_subset)} proportional files...")

    all_results = []
    
    # Check for existing results to resume
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        all_results = existing_df.to_dict('records')
        done_files = set(existing_df['file'].unique())
        print(f"[INFO] Resuming. {len(done_files)} files already processed.")
    else:
        done_files = set()

    seq_len = 512 # Context length
    batch_size = 16
    
    for idx, row in df_subset.iterrows():
        filename = row['file']
        folder = row['folder']
        path = row['path']
        
        if filename in done_files:
            continue
            
        print(f"[{idx+1}/{len(df_subset)}] Processing: {filename} ({folder})")
        
        try:
            # Load Data
            df_data = pd.read_csv(path, header=None)
            data = df_data[0].to_numpy('float')
            label = df_data[1].to_numpy('int')
            
            mask = ~np.isnan(data)
            data = data[mask]
            label = label[mask]
            
            if len(data) < seq_len + 10:
                continue

            # Preprocessing: Z-score
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.reshape(-1, 1)).ravel()

            # Prediction Loop
            limit = 5000
            start_i = seq_len
            end_i = min(len(data_scaled), seq_len + limit)
            
            contexts = []
            targets = []
            for i in range(start_i, end_i):
                contexts.append(data_scaled[i-seq_len : i])
                targets.append(data_scaled[i])
            
            prediction_errors = []
            for b in tqdm(range(0, len(contexts), batch_size), desc="Normalized Inference"):
                batch_x = contexts[b : b + batch_size]
                batch_y = targets[b : b + batch_size]
                
                # Keep on CPU for pipeline pinning efficiency
                context_tensor = torch.tensor(np.array(batch_x), dtype=torch.float32)
                if len(context_tensor.shape) == 2:
                    context_tensor = context_tensor.unsqueeze(1)
                
                with torch.no_grad():
                    # forecast shape is usually (Samples, Batch, Variates, Pred_Len)
                    forecast = pipeline.predict(context_tensor, prediction_length=1)
                    if isinstance(forecast, list):
                        forecast = torch.stack(forecast)
                    
                    # Robustly find the batch and samples dimensions
                    # We know batch_size is len(batch_y)
                    batch_size_val = len(batch_y)
                    
                    # Find which dimension is the batch
                    batch_dim = -1
                    for d in range(forecast.ndim):
                        if forecast.shape[d] == batch_size_val:
                            batch_dim = d
                            break
                    
                    # The samples dimension is typically the other large dimension (usually 0 or 1)
                    if batch_dim == 0:
                        samples_dim = 1
                    else:
                        samples_dim = 0 # Default to 0 as it's the standard for Chronos-2
                    
                    # Calculate Quantiles over the samples dimension
                    median_tensor = torch.quantile(forecast, 0.5, dim=samples_dim)
                    q05_tensor = torch.quantile(forecast, 0.05, dim=samples_dim)
                    q95_tensor = torch.quantile(forecast, 0.95, dim=samples_dim)
                    
                    # Now we must ensure the result is exactly (Batch,)
                    # We flatten any remaining singleton dimensions (Variates, Pred_Len)
                    # but keep the Batch dimension intact.
                    median = median_tensor.reshape(batch_size_val, -1).mean(dim=1).cpu().numpy()
                    q05 = q05_tensor.reshape(batch_size_val, -1).mean(dim=1).cpu().numpy()
                    q95 = q95_tensor.reshape(batch_size_val, -1).mean(dim=1).cpu().numpy()
                    
                    # Uncertainty-Normalized Anomaly Score
                    iqr = q95 - q05
                    norm_error = np.abs(np.array(batch_y) - median) / (iqr + 1e-6)
                    prediction_errors.extend(norm_error)

            # Alignment and Padding
            full_scores = np.zeros(len(data))
            full_scores[start_i : end_i] = prediction_errors
            full_scores[:start_i] = prediction_errors[0] if prediction_errors else 0
            if end_i < len(data):
                full_scores[end_i:] = prediction_errors[-1]

            full_scores = MinMaxScaler().fit_transform(full_scores.reshape(-1, 1)).ravel()

            # Metrics
            from TSB_UAD.utils.slidingWindows import find_length
            sw = find_length(data_scaled)
            sw = max(int(sw), 10)
            
            metrics = get_metrics(full_scores, label, metric="all", slidingWindow=sw)
            
            res = {
                "file": filename, "folder": folder, "model": "Chronos-2-Normalized",
                "AUC_ROC": metrics["AUC_ROC"], "VUS_ROC": metrics["VUS_ROC"], "F1": metrics["F"]
            }
            all_results.append(res)
            print(f"  AUC-ROC (Normalized): {res['AUC_ROC']:.4f}")
            pd.DataFrame(all_results).to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

    print(f"[FINISH] Normalized Chronos-2 complete. Saved to: {output_path}")

if __name__ == "__main__":
    subset_csv = "representative_subset_proportional_200.csv"
    output_results = "results/tables/chronos2_normalized_results_200.csv"
    os.makedirs(os.path.dirname(output_results), exist_ok=True)
    run_chronos2_normalized_evaluation(subset_csv, output_results)
