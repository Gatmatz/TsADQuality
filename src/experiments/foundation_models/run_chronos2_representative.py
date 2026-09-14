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
    print("[ERROR] 'chronos-forecasting' library not found. Install with: pip install chronos-forecasting")
    sys.exit(1)

def run_chronos2_evaluation(subset_csv, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load Chronos-2 Model (Pre-trained)
    print("[INFO] Loading amazon/chronos-2 model...")
    try:
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2", 
            device_map=device,
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(f"[ERROR] Could not load amazon/chronos-2: {e}")
        print("[INFO] Trying variation 'amazon/chronos-v2-base'...")
        try:
            pipeline = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-v2-base",
                device_map=device,
                torch_dtype=torch.float32,
            )
        except:
            print("[ERROR] Failed to load any Chronos-2 variant. Please check your internet or model ID.")
            return

    # 2. Load Subset
    if not os.path.exists(subset_csv):
        print(f"[ERROR] Subset file {subset_csv} not found.")
        return
    
    df_subset = pd.read_csv(subset_csv)
    print(f"[INFO] Running Chronos-2 on {len(df_subset)} proportional files (200 subset)...")

    all_results = []
    
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        all_results = existing_df.to_dict('records')
        done_files = set(existing_df['file'].unique())
        print(f"[INFO] Resuming. {len(done_files)} files already processed.")
    else:
        done_files = set()

    seq_len = 512 
    
    for idx, row in df_subset.iterrows():
        filename = row['file']
        folder = row['folder']
        path = row['path']
        
        if filename in done_files:
            continue
            
        print(f"\n[{idx+1}/{len(df_subset)}] Processing: {filename} ({folder})")
        
        try:
            df_data = pd.read_csv(path, header=None)
            data = df_data[0].to_numpy('float')
            label = df_data[1].to_numpy('int')
            
            mask = ~np.isnan(data)
            data = data[mask]
            label = label[mask]
            
            if len(data) < seq_len + 10:
                continue

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.reshape(-1, 1)).ravel()

            limit = 5000
            start_i = seq_len
            end_i = min(len(data_scaled), seq_len + limit)
            
            contexts = []
            targets = []
            for i in range(start_i, end_i):
                contexts.append(data_scaled[i-seq_len : i])
                targets.append(data_scaled[i])
            
            batch_size = 16
            prediction_errors = []
            for b in tqdm(range(0, len(contexts), batch_size), desc="Inferring"):
                batch_x = contexts[b : b + batch_size]
                batch_y = targets[b : b + batch_size]
                
                # Keep tensor on CPU. Chronos pipeline handles moving to device internally.
                context_tensor = torch.tensor(np.array(batch_x), dtype=torch.float32)
                
                # Reshape to 3D: (Batch, 1, Time) - Required by Chronos-2
                if len(context_tensor.shape) == 2:
                    context_tensor = context_tensor.unsqueeze(1)
                
                with torch.no_grad():
                    forecast = pipeline.predict(context_tensor, prediction_length=1)
                    
                    # Chronos-2 returns (Samples, Batch, Variates, Pred_Len) 
                    # or (Samples, Batch, Pred_Len)
                    if isinstance(forecast, list):
                        forecast = torch.stack(forecast)
                    
                    # We want to take the median over the 'Samples' dimension.
                    # Usually it's the first dimension (dim=0) in Chronos-2
                    # We'll check which dimension is NOT our batch_size or pred_len
                    if forecast.shape[1] == len(batch_y):
                        # Shape is (Samples, Batch, ...) -> reduce dim 0
                        pred_tensor = torch.median(forecast, dim=0).values
                    elif forecast.shape[0] == len(batch_y):
                        # Shape is (Batch, Samples, ...) -> reduce dim 1
                        pred_tensor = torch.median(forecast, dim=1).values
                    else:
                        # Fallback: assume first dim is samples
                        pred_tensor = torch.median(forecast, dim=0).values

                    # Flatten to (Batch,)
                    pred = pred_tensor.cpu().numpy().ravel()
                    # Ensure pred matches batch_y size (handling potential variates/pred_len dims)
                    if len(pred) > len(batch_y):
                        pred = pred[:len(batch_y)]
                    
                errors = np.abs(pred - np.array(batch_y))
                prediction_errors.extend(errors)

            full_scores = np.zeros(len(data))
            full_scores[start_i : end_i] = prediction_errors
            full_scores[:start_i] = prediction_errors[0] if prediction_errors else 0
            if end_i < len(data):
                full_scores[end_i:] = prediction_errors[-1]

            full_scores = MinMaxScaler().fit_transform(full_scores.reshape(-1, 1)).ravel()

            from TSB_UAD.utils.slidingWindows import find_length
            sw = find_length(data_scaled)
            sw = max(int(sw), 10)
            
            metrics = get_metrics(full_scores, label, metric="all", slidingWindow=sw)
            
            res = {
                "file": filename, "folder": folder, "model": "Chronos-2",
                "AUC_ROC": metrics["AUC_ROC"], "VUS_ROC": metrics["VUS_ROC"], "F1": metrics["F"]
            }
            all_results.append(res)
            print(f"  AUC-ROC: {res['AUC_ROC']:.4f}")
            pd.DataFrame(all_results).to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

    print(f"\n[FINISH] Chronos-2 complete. Saved to: {output_path}")

if __name__ == "__main__":
    subset_csv = "representative_subset_proportional_200.csv"
    output_results = "results/tables/chronos2_proportional_results_200.csv"
    os.makedirs(os.path.dirname(output_results), exist_ok=True)
    run_chronos2_evaluation(subset_csv, output_results)
