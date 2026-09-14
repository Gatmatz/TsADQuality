import torch
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time
import traceback
from scipy.interpolate import interp1d

from TSB_UAD.vus.metrics import get_metrics

try:
    from chronos import BaseChronosPipeline
except ImportError:
    print("[ERROR] 'chronos-forecasting' library not found.")
    sys.exit(1)

def extract_embeddings(pipeline, context_tensors, device):
    """Extracts pooled hidden states from the Chronos model with extreme robustness."""
    # Correctly identify the model object
    model = pipeline.model if hasattr(pipeline, "model") else pipeline
    
    with torch.no_grad():
        # Input must be 2D (Batch, Context_Len) for .encode()
        if len(context_tensors.shape) == 3:
            context_tensors = context_tensors.squeeze(1)
            
        context_tensors = context_tensors.to(device)
        
        # Call encode
        outputs = model.encode(context_tensors)
        
        # Safety check on outputs
        if outputs is None:
            raise ValueError("Model.encode() returned None.")
        
        # Chronos-2 typically returns a tuple where index 0 is the EncoderOutput
        try:
            encoder_output = outputs[0]
        except (IndexError, TypeError):
            encoder_output = outputs # Fallback if it's not a tuple
            
        hidden = None
        # Try finding the hidden states in various common attributes
        for attr in ["last_hidden_state", "embeddings", "hidden_states"]:
            if hasattr(encoder_output, attr):
                hidden = getattr(encoder_output, attr)
                # If it's a list (like hidden_states often is), take the last one
                if isinstance(hidden, (list, tuple)) and len(hidden) > 0:
                    hidden = hidden[-1]
                break
        
        # If still not found, check if encoder_output itself is the tensor or a list
        if hidden is None:
            if isinstance(encoder_output, torch.Tensor):
                hidden = encoder_output
            elif isinstance(encoder_output, (list, tuple)) and len(encoder_output) > 0:
                hidden = encoder_output[0]
                
        if hidden is None:
            raise AttributeError(f"Could not extract hidden states from {type(encoder_output)}")
            
        # Pooling: Mean over the tokens dimension (dim 1)
        # Expected shape [Batch, Tokens, Hidden_Dim]
        if hidden.ndim == 3:
            pooled = torch.mean(hidden, dim=1)
        elif hidden.ndim == 2:
            pooled = hidden # Already pooled
        else:
            # Fallback for weird shapes: flatten everything except batch
            pooled = hidden.view(hidden.size(0), -1)
            
    return pooled.cpu().numpy()

def run_chronos2_embeddings_evaluation(subset_csv, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load Model
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
    all_results = []
    
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        all_results = existing_df.to_dict('records')
        done_files = set(existing_df['file'].unique())
        print(f"[INFO] Resuming. {len(done_files)} files already processed.")
    else:
        done_files = set()

    seq_len = 512
    batch_size = 64
    stride = 5 
    
    for idx, row in df_subset.iterrows():
        filename = row['file']
        folder = row['folder']
        path = row['path']
        
        if filename in done_files: continue
            
        print(f"\n[{idx+1}/{len(df_subset)}] Embedding Extraction: {filename} ({folder})")
        
        try:
            df_data = pd.read_csv(path, header=None)
            data = df_data[0].to_numpy('float')
            label = df_data[1].to_numpy('int')
            
            mask = ~np.isnan(data)
            data = data[mask]
            label = label[mask]
            
            # Smart Slicing: Locate the first anomaly and take a 5000-point window around it
            anomaly_indices = np.where(label == 1)[0]
            limit = 5000
            
            if len(anomaly_indices) > 0:
                first_anom = anomaly_indices[0]
                # Start 1000 points before the anomaly to have "normal" context
                start_idx = max(0, first_anom - 1000)
                end_idx = min(len(data), start_idx + limit + seq_len)
                
                # Adjust start if end was capped to maintain 5000 points
                if end_idx - start_idx < limit and start_idx > 0:
                    start_idx = max(0, end_idx - limit - seq_len)
                
                data = data[start_idx:end_idx]
                label = label[start_idx:end_idx]
            else:
                data = data[:limit + seq_len]
                label = label[:limit + seq_len]

            if label.sum() == 0:
                print(f"  [SKIP] {filename}: Truly no anomalies found in file.")
                continue

            if len(data) < seq_len + 10: continue

            windows = []
            window_indices = []
            for i in range(0, len(data) - seq_len + 1, stride):
                windows.append(data[i : i + seq_len])
                window_indices.append(i + seq_len - 1)
            
            train_ratio = 0.30 if folder == "YAHOO" else 0.10
            n_train_windows = max(1, int(len(windows) * train_ratio))
            
            print(f"  Encoding {len(windows)} windows...")
            
            all_embeddings = []
            for b in tqdm(range(0, len(windows), batch_size), desc="GPU Encoding"):
                batch_x = torch.tensor(np.array(windows[b : b + batch_size]), dtype=torch.float32)
                emb = extract_embeddings(pipeline, batch_x, device)
                all_embeddings.append(emb)
            
            if not all_embeddings:
                raise ValueError("No embeddings extracted.")
                
            all_embeddings = np.vstack(all_embeddings) 
            reference_centroid = np.mean(all_embeddings[:n_train_windows], axis=0).reshape(1, -1)
            
            similarities = cosine_similarity(all_embeddings, reference_centroid).ravel()
            scores_sampled = 1.0 - similarities
            
            full_scores = np.zeros(len(data))
            if len(window_indices) > 1:
                f_interp = interp1d(window_indices, scores_sampled, kind='linear', fill_value="extrapolate")
                start_idx, end_idx = window_indices[0], window_indices[-1]
                full_scores[start_idx:end_idx+1] = f_interp(np.arange(start_idx, end_idx+1))
                full_scores[:start_idx] = scores_sampled[0]
                full_scores[end_idx+1:] = scores_sampled[-1]
            else:
                full_scores[:] = scores_sampled[0]

            full_scores = MinMaxScaler().fit_transform(full_scores.reshape(-1, 1)).ravel()

            from TSB_UAD.utils.slidingWindows import find_length
            sw = find_length(data)
            sw = max(int(sw), 10)
            
            metrics = get_metrics(full_scores, label, metric="all", slidingWindow=sw)
            
            res = {
                "file": filename, "folder": folder, "model": "Chronos-2-Embeddings",
                "AUC_ROC": metrics["AUC_ROC"], "VUS_ROC": metrics["VUS_ROC"], "F1": metrics["F"]
            }
            all_results.append(res)
            print(f"  AUC-ROC: {res['AUC_ROC']:.4f}")
            pd.DataFrame(all_results).to_csv(output_results, index=False)
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            continue

    print(f"\n[FINISH] Embedding extraction complete.")

if __name__ == "__main__":
    subset_csv = "representative_subset_proportional_200.csv"
    output_results = "results/tables/chronos2_embeddings_results_200.csv"
    os.makedirs(os.path.dirname(output_results), exist_ok=True)
    run_chronos2_embeddings_evaluation(subset_csv, output_results)
