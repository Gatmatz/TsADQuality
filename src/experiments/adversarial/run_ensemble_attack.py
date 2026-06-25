import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

# Fix sys.path to easily import the whole project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ts_corruptor.core import TSCorruptor
from src.utils.logger import log_experiment
from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.pca import PCA
from TSB_UAD.models.lof import LOF
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

# -------------------------------------------------------------------------------------
# ENSEMBLE SURROGATE ARCHITECTURE
# -------------------------------------------------------------------------------------
class EnsembleSurrogateRegression(nn.Module):
    """
    A Neural Network that predicts the anomaly scores of *multiple* Traditional ML Anomaly Detectors.
    Output dim = 3 (IForest Score, PCA Score, LOF Score).
    """
    def __init__(self, window_size):
        super(EnsembleSurrogateRegression, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(window_size, window_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(window_size, window_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(window_size // 2, 3), # Predicts exactly 3 scores
            nn.Sigmoid() # Normalize scores between 0 and 1
        )

    def forward(self, x):
        return self.network(x)

def train_ensemble_surrogate(T_tensor, y_targets, window_size, epochs=150, lr=0.005):
    """Trains the surrogate to mimic all 3 target algorithms' anomaly scores."""
    model = EnsembleSurrogateRegression(window_size)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # We use MSE to match the target scores exactly
    criterion = nn.MSELoss()

    T_tensor = T_tensor.view(-1)
    
    if len(T_tensor) < window_size:
        return model

    windows = T_tensor.unfold(0, window_size, 1)
    
    # Align the targets with the windows
    # Window w ends at index w + window_size - 1
    # We assign the anomaly score of the last point in the window as the target for the window
    n_windows = windows.shape[0]
    
    # y_targets is shape (len(T_tensor), 3) -> (n_windows, 3)
    target_scores = y_targets[window_size-1 : window_size-1+n_windows]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(windows) # Shape: (n_windows, 3)
        loss = criterion(predictions, target_scores)
        loss.backward()
        optimizer.step()
        
    return model

def apply_smoothing(perturbation, smooth_window):
    """Applies a moving average filter to make the attack look completely natural (Pialla et al. SGM)"""
    if smooth_window <= 1: return perturbation
    kernel = torch.ones(1, 1, smooth_window) / smooth_window
    p_reshaped = perturbation.view(1, 1, -1)
    pad_left = smooth_window // 2
    pad_right = smooth_window - 1 - pad_left
    p_padded = torch.nn.functional.pad(p_reshaped, (pad_left, pad_right), mode='replicate')
    return torch.nn.functional.conv1d(p_padded, kernel).view(-1)

# -------------------------------------------------------------------------------------
# THE ENSEMBLE ATTACK MANIFESTATION
# -------------------------------------------------------------------------------------
def run_ensemble_distillation_attack(csv_path, dataset_name, alpha_factor=0.01, epsilon_factor=0.5, iterations=40, smooth_window=5):
    print(f"\n[+] Starting Ensemble Surrogate Attack on {dataset_name}")
    print(f"    a={alpha_factor} | e={epsilon_factor} | Iters={iterations} | Smooth={smooth_window}")
    
    df = pd.read_csv(csv_path, header=None, names=['value', 'is_anomaly'])
    X = df['value'].values
    y = df['is_anomaly'].values
    window_size = find_length(X)
    
    # Skip if dataset is incredibly small or completely devoid of anomalies
    if sum(y) < 10 or len(X) < window_size * 2:
        print("    [!] Dataset too small or lacks anomalies. Skipping.")
        return None
        
    print("    -> 1. Getting True Target Scores from Baselines...")
    # Get scores from IForest
    clf_iforest = IForest()
    clf_iforest.fit(X)
    score_iforest = clf_iforest.decision_scores_
    
    # Get scores from PCA
    clf_pca = PCA()
    clf_pca.fit(X)
    score_pca = clf_pca.decision_scores_
    
    # Get scores from LOF
    clf_lof = LOF()
    clf_lof.fit(X)
    score_lof = clf_lof.decision_scores_
    
    # Min-Max Scale scores so the Neural Network can learn them uniformly via Sigmoid
    def minmax(s):
        return (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-8)
        
    score_iforest = minmax(score_iforest)
    score_pca = minmax(score_pca)
    score_lof = minmax(score_lof)
    
    # Stack them into shape (N, 3)
    y_targets = np.column_stack((score_iforest, score_pca, score_lof))
    y_targets_tensor = torch.tensor(y_targets, dtype=torch.float32)
    
    print("    -> 2. Distilling Multiple Models into Ensemble Surrogate...")
    T_tensor = torch.tensor(X, dtype=torch.float32)
    model = train_ensemble_surrogate(T_tensor, y_targets_tensor, window_size, epochs=150)
    model.eval()
    
    print("    -> 3. Generating Transferable 'Universal' Adversarial Perturbation...")
    std = np.std(X)
    epsilon = epsilon_factor * std
    alpha = alpha_factor * std
    
    # Track the pure perturbation delta rather than replacing the whole tensor to avoid vanishing gradients
    T_tensor = torch.tensor(X, dtype=torch.float32)
    delta = torch.zeros_like(T_tensor, requires_grad=True)
    
    anomaly_indices = np.where(y == 1)[0]
    
    for _ in range(iterations):
        T_adv = T_tensor + delta
        windows = T_adv.unfold(0, window_size, 1)
        predictions = model(windows) # Shape: (n_windows, 3)
        
        max_scores_per_window, _ = torch.max(predictions, dim=1) 
        n_windows = windows.shape[0]
        window_mask = torch.zeros(n_windows, dtype=torch.bool, device=T_adv.device)
        
        for t in anomaly_indices:
            start_w = max(0, t - window_size + 1)
            end_w = min(n_windows, t + 1)
            if start_w < end_w:
                window_mask[start_w:end_w] = True
                
        loss = torch.sum(max_scores_per_window[window_mask])
        
        if loss.item() == 0:
            break
            
        loss.backward()
        
        with torch.no_grad():
            grad_sign = delta.grad.sign()
            perturbation = -grad_sign * alpha
            
            if smooth_window > 1:
                perturbation = apply_smoothing(perturbation, smooth_window)
                
            delta.add_(perturbation)
            delta.clamp_(min=-epsilon, max=epsilon)
            delta.grad.zero_()
            
    final_noise = delta.detach().numpy()
    X_adv = X + final_noise
    
    # Calculate Stealh Metrics
    l2_norm = np.linalg.norm(final_noise)
    print(f"       Attack complete. Global L2 Norm generated: {l2_norm:.2f}")
    
    # Evaluate Impact on Baselines
    # IForest Original vs Attack
    m_if_orig = get_metrics(score_iforest, y)
    auc_if_orig = m_if_orig.get('AUC_ROC', 0.0) if hasattr(m_if_orig, 'get') else (m_if_orig[9] if len(m_if_orig)>9 else 0)
    
    clf_iforest_adv = IForest()
    clf_iforest_adv.fit(X_adv)
    m_if_adv = get_metrics(clf_iforest_adv.decision_scores_, y)
    auc_if_adv = m_if_adv.get('AUC_ROC', 0.0) if hasattr(m_if_adv, 'get') else (m_if_adv[9] if len(m_if_adv)>9 else 0)
    
    # PCA Original vs Attack
    m_pca_orig = get_metrics(score_pca, y)
    auc_pca_orig = m_pca_orig.get('AUC_ROC', 0.0) if hasattr(m_pca_orig, 'get') else (m_pca_orig[9] if len(m_pca_orig)>9 else 0)
    
    clf_pca_adv = PCA()
    clf_pca_adv.fit(X_adv)
    m_pca_adv = get_metrics(clf_pca_adv.decision_scores_, y)
    auc_pca_adv = m_pca_adv.get('AUC_ROC', 0.0) if hasattr(m_pca_adv, 'get') else (m_pca_adv[9] if len(m_pca_adv)>9 else 0)
    
    print(f"       IForest AUC: {auc_if_orig:.3f} -> {auc_if_adv:.3f} (Drop: {auc_if_orig - auc_if_adv:.3f})")
    print(f"       PCA AUC: {auc_pca_orig:.3f} -> {auc_pca_adv:.3f} (Drop: {auc_pca_orig - auc_pca_adv:.3f})")
    
    # --- SAVE TO OUTPUTS ---
    out_dir_data = Path(f"outputs/data_corrupted/Ensemble_SGM_a{alpha_factor}")
    out_dir_data.mkdir(parents=True, exist_ok=True)
    out_file = out_dir_data / f"{dataset_name}.csv"
    
    # Save exact array + labels
    df_out = pd.DataFrame({'value': X_adv, 'is_anomaly': y})
    df_out.to_csv(out_file, index=False, header=False)
    print(f"    [Logger] Universal Dataset Saved to {out_file}")
    
    # We could log metrics here, but let's decouple evaluation for real pipelines.
    return out_file

if __name__ == "__main__":
    # Test on a singular file to prove the concept works.
    target_csv = "results/tables/robust_subset_TSB.csv"
    
    if not os.path.exists(target_csv):
        print(f"Error: Could not find {target_csv}.")
        sys.exit(1)
        
    df_registry = pd.read_csv(target_csv).head(2) # Just test on 2
    
    for idx, row in df_registry.iterrows():
        run_ensemble_distillation_attack(
            csv_path=row['filepath'], 
            dataset_name=row['baseline_name'],
            alpha_factor=0.05,
            epsilon_factor=0.8,
            iterations=150,
            smooth_window=7
        )
