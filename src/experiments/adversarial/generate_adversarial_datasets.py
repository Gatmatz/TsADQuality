import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Fix sys.path to easily import the whole project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.pca import PCA
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

class SurrogateScoreNet(nn.Module):
    def __init__(self, ws):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ws, ws), nn.ReLU(),
            nn.Linear(ws, ws//2), nn.ReLU(),
            nn.Linear(ws//2, max(2, ws//4)), nn.ReLU(),
            nn.Linear(max(2, ws//4), 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def apply_smoothing(perturbation, smooth_window):
    if smooth_window <= 1: return perturbation
    kernel = torch.ones(1, 1, smooth_window) / smooth_window
    p_reshaped = perturbation.view(1, 1, -1)
    pad_left = smooth_window // 2
    pad_right = smooth_window - 1 - pad_left
    p_padded = torch.nn.functional.pad(p_reshaped, (pad_left, pad_right), mode='replicate')
    return torch.nn.functional.conv1d(p_padded, kernel).view(-1)

def run_per_model_attack(csv_path, dataset_name, target_model_name, alpha_factor=0.02, epsilon_factor=0.05, iterations=150, smooth_window=7, contrast_weight=0.5):
    """
    Args:
        alpha_factor: Step size as fraction of std (original domain). 0.02 = 2% of std per step.
        epsilon_factor: Max perturbation as fraction of std (original domain). 0.05 = 5% of std.
        contrast_weight: Weight for normal window penalty in contrastive loss.
    """
    print(f"\n[+] Starting {target_model_name} Distillation Attack on {dataset_name}")
    
    df = pd.read_csv(csv_path, header=None, names=['value', 'is_anomaly'])
    X = df['value'].values
    y = df['is_anomaly'].values
    sw = max(int(find_length(X)), 10)
    
    # 1. Get True Scores (z-score normalize data first)
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X.reshape(-1,1)).ravel()
    Xc = np.ascontiguousarray(X_scaled, dtype=np.float64)
    Xf = Window(window=sw).convert(Xc).to_numpy()
    det = IForest(n_jobs=1) if target_model_name == 'IForest' else PCA()
    det.fit(Xf)
    scores = np.pad(np.array(det.decision_scores_, dtype=np.float64),
                    (sw-1,0), 'constant', constant_values=(det.decision_scores_[0],))
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    s_min, s_max = scores.min(), scores.max()+1e-8
    scores_n = (scores - s_min)/(s_max - s_min)
    
    # 2. Train Distillation Surrogate
    Xm, Xs = X.mean(), X.std()+1e-8
    Xn = (X - Xm)/Xs
    T_orig = torch.tensor(Xn, dtype=torch.float32)
    ws = min(50, len(Xn)//4)
    
    wins = T_orig.unfold(0, ws, 1)
    st = torch.tensor(scores_n, dtype=torch.float32)
    # Use max score per window to preserve anomaly peaks (not mean which flattens them)
    ws_scores = st.unfold(0, ws, 1).max(dim=1).values
    
    model = SurrogateScoreNet(ws)
    opt = optim.Adam(model.parameters(), lr=0.005)
    crit = nn.MSELoss()
    print(f"    -> Training Surrogate for {target_model_name}...")
    for _ in range(150): # 150 epochs
        opt.zero_grad()
        loss = crit(model(wins), ws_scores)
        loss.backward()
        opt.step()
        
    # 3. Generating Attack
    print(f"    -> Generating Adversarial Perturbation...")
    # epsilon/alpha are fractions of std — in normalized space (divided by Xs), they equal the factor directly
    alpha = alpha_factor      # step size in normalized space (= alpha_factor * Xs / Xs)
    epsilon = epsilon_factor   # max perturbation in normalized space (= epsilon_factor * Xs / Xs)
    
    anomaly_indices = np.where(y == 1)[0]
    
    # Pre-compute window mask indices outside attack loop (vectorized)
    n_windows = len(T_orig) - ws + 1
    window_mask = torch.zeros(n_windows, dtype=torch.bool)
    starts = np.clip(anomaly_indices - ws + 1, 0, n_windows)
    ends = np.clip(anomaly_indices + 1, 0, n_windows)
    for s, e in zip(starts, ends):
        window_mask[s:e] = True
    normal_mask = ~window_mask
    has_normal = normal_mask.any()
    
    delta_data = torch.zeros_like(T_orig)
    
    for _ in range(iterations):
        delta = delta_data.detach().requires_grad_(True)
        T_adv = T_orig + delta
        windows = T_adv.unfold(0, ws, 1)
        predictions = model(windows)
        
        # Contrastive loss: minimize anomaly scores, penalize lowering normal scores
        loss_anomaly = torch.mean(predictions[window_mask]) if window_mask.any() else torch.tensor(0.0)
        loss_normal = torch.mean(predictions[normal_mask]) if has_normal else torch.tensor(0.0)
        loss = loss_anomaly - contrast_weight * loss_normal
        
        if loss_anomaly.item() == 0: break
        
        loss.backward()
        with torch.no_grad():
            grad_sign = delta.grad.sign()
            perturbation = -grad_sign * alpha
            if smooth_window > 1:
                perturbation = apply_smoothing(perturbation, smooth_window)
            delta_data = (delta_data + perturbation).clamp_(min=-epsilon, max=epsilon)
            
    final_noise_n = delta_data.detach().numpy()
    X_adv = X + (final_noise_n * Xs)  # Rescale noise back to original domain
    
    l2_norm = np.linalg.norm((final_noise_n * Xs))
    print(f"       Attack complete. Global L2 Norm: {l2_norm:.2f}")
    
    # 4. Evaluate attack effectiveness — score adversarial data with the SAME fitted detector
    try:
        X_adv_c = np.ascontiguousarray(X_adv, dtype=np.float64)
        Xf_adv = Window(window=sw).convert(X_adv_c).to_numpy()
        # Use the SAME detector (already fitted on original X) — true adversarial eval
        scores_adv_raw = det.decision_function(Xf_adv)
        scores_adv = np.pad(np.array(scores_adv_raw, dtype=np.float64),
                            (sw-1, 0), 'constant', constant_values=(scores_adv_raw[0],))
        scores_adv = np.nan_to_num(scores_adv, nan=0.0, posinf=0.0, neginf=0.0)
        
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        # Z-score first (handle outlier scores), then MinMax to [0,1] for metrics
        scores_orig_z = StandardScaler().fit_transform(scores.reshape(-1,1)).ravel()
        scores_adv_z = StandardScaler().fit_transform(scores_adv.reshape(-1,1)).ravel()
        scores_orig_norm = MinMaxScaler().fit_transform(scores_orig_z.reshape(-1,1)).ravel()
        scores_adv_norm = MinMaxScaler().fit_transform(scores_adv_z.reshape(-1,1)).ravel()
        
        m_orig = get_metrics(scores_orig_norm, y.astype(int), metric="all", slidingWindow=sw)
        m_adv = get_metrics(scores_adv_norm, y.astype(int), metric="all", slidingWindow=sw)
        
        auc_orig = m_orig.get('AUC_ROC', 0.0)
        auc_adv = m_adv.get('AUC_ROC', 0.0)
        
        score_orig_anom = scores[anomaly_indices].mean() if len(anomaly_indices) > 0 else 0
        score_adv_anom = scores_adv[anomaly_indices].mean() if len(anomaly_indices) > 0 else 0
        
        print(f"    [EVAL] Anomaly scores (same det): {score_orig_anom:.3f} → {score_adv_anom:.3f} "
              f"(Δ={score_adv_anom - score_orig_anom:+.3f})")
        print(f"    [EVAL] AUC-ROC (same det): {auc_orig:.3f} → {auc_adv:.3f} "
              f"(Drop={auc_orig - auc_adv:+.3f})")
    except Exception as e:
        print(f"    [EVAL] Could not evaluate attack: {e}")
    
    # 5. Save to Outputs
    out_dir_data = Path(f"outputs/data_corrupted/Distillation_{target_model_name}_a{alpha_factor}")
    out_dir_data.mkdir(parents=True, exist_ok=True)
    out_file = out_dir_data / f"{dataset_name}.csv"
    
    df_out = pd.DataFrame({'value': X_adv, 'is_anomaly': y})
    df_out.to_csv(out_file, index=False, header=False)
    print(f"       Saved to: {out_file}")
    
    return str(out_file)

if __name__ == "__main__":
    target_csv = "results/tables/robust_subset_TSB.csv"
    if not os.path.exists(target_csv):
        print(f"Error: Could not find {target_csv}.")
        sys.exit(1)
        
    df_registry = pd.read_csv(target_csv).head(2) # Test on 2 initially
    
    targets = ['IForest', 'PCA']
    
    for target_model in targets:
        for idx, row in df_registry.iterrows():
            run_per_model_attack(
                csv_path=row['filepath'], 
                dataset_name=row['baseline_name'],
                target_model_name=target_model,
                alpha_factor=0.02,
                epsilon_factor=0.05,
                iterations=150,
                smooth_window=7,
                contrast_weight=0.5
            )
