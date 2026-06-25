"""
Adversarial Attack Test — BIM, GM, SGM (faithful to Pialla et al. 2025)
Adapted for anomaly detection via surrogate AutoEncoder.

Methods:
  1. BIM  — Basic Iterative Method with ε-clipping (Kurakin et al.)
  2. GM   — Gradient Method: maximize surrogate loss + L2 regularization
  3. SGM  — Smooth GM: GM + fused lasso regularization for smooth perturbations

Key adaptation: The paper attacks a classifier (white-box), we attack 
anomaly detectors (IForest, PCA) via a black-box surrogate approach.
Instead of KL-divergence on class probabilities, we minimize the surrogate
AE reconstruction error (to hide anomalies).

Usage:
    python -u scripts/test_adversarial_attacks.py
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.pca import PCA
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

# ---- Config ----
SUBSET_CSV = PROJECT_ROOT / "results" / "tables" / "robust_subset_TSB.csv"
PLOT_DIR   = PROJECT_ROOT / "results" / "plots" / "adversarial"
os.makedirs(PLOT_DIR, exist_ok=True)

WINDOW_SIZE = 50
N_DATASETS = 8

# Paper hyper-parameters
BIM_EPSILON = 0.1      # Paper: ε = 0.1
BIM_ALPHA = 0.01       # Small step size
BIM_ITERATIONS = 200   # Paper: 1000, we use 200 for speed

GM_MU = 1.0            # Paper: μ = 1
GM_ALPHA = 1.0         # Paper: α = 1 (L2 regularization)
GM_LR = 0.01           # Learning rate for optimizer
GM_ITERATIONS = 200    # Optimization steps

SGM_LAMBDA = 1.0       # Paper: λ = 1 (fused lasso penalty)


# ===========================================================
# Surrogate AutoEncoder
# ===========================================================
class SurrogateAE(nn.Module):
    def __init__(self, window_size):
        super(SurrogateAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, window_size // 2), nn.ReLU(),
            nn.Linear(window_size // 2, max(2, window_size // 4)), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(max(2, window_size // 4), window_size // 2), nn.ReLU(),
            nn.Linear(window_size // 2, window_size)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_surrogate(T_data, window_size, epochs=50):
    model = SurrogateAE(window_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    windows = T_data.unfold(0, window_size, 1)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(windows), windows)
        loss.backward()
        optimizer.step()
    return model


def surrogate_anomaly_score(model, T, window_size):
    """Per-window MSE reconstruction error (acts as anomaly score)."""
    windows = T.unfold(0, window_size, 1)
    reconstruction = model(windows)
    return torch.mean((reconstruction - windows) ** 2, dim=1)


# ===========================================================
# Attack Methods (faithful to Pialla et al.)
# ===========================================================

def attack_bim(model, T, window_size, epsilon=0.1, alpha=0.01, iterations=200,
               target_indices=None):
    """
    Basic Iterative Method (BIM) — Eq. 1 from the paper.
    
    x_{N+1} = Clip_{x,ε} { x_N - α · sign(∇_x Loss) }
    
    We minimize the loss (reconstruction error) to hide anomalies.
    The clipping ensures ‖x_adv - x‖_∞ ≤ ε.
    """
    model.eval()
    T_adv = T.clone().detach().requires_grad_(True)

    for _ in range(iterations):
        scores = surrogate_anomaly_score(model, T_adv, window_size)

        if target_indices is not None and len(target_indices) > 0:
            total_loss = 0
            for idx in target_indices:
                start_w = max(0, idx - window_size + 1)
                end_w = min(len(scores), idx + 1)
                if start_w < end_w:
                    total_loss = total_loss + torch.sum(scores[start_w:end_w])
        else:
            total_loss = torch.sum(scores)

        total_loss.backward()

        with torch.no_grad():
            # Gradient descent: minimize anomaly score to hide anomalies
            T_adv_new = T_adv - alpha * T_adv.grad.sign()
            # Clip to ε-neighborhood (paper Eq. 1)
            eta = torch.clamp(T_adv_new - T, min=-epsilon, max=epsilon)
            T_adv = (T + eta).detach().requires_grad_(True)

    return T_adv.detach()


def attack_gm(model, T, window_size, mu=1.0, alpha_l2=1.0, lr=0.01, iterations=200,
              target_indices=None):
    """
    Gradient Method (GM) — Eq. 7 from the paper.
    
    Optimize r directly:
        min { -μ · Loss(x+r) + α · ‖r‖² }
    
    Since we want to MINIMIZE anomaly score (hide anomalies),
    the loss sign is already what we want: minimize reconstruction error
    while keeping r small.
    """
    model.eval()
    # Initialize perturbation r ~ N(0, 0.01)
    r = torch.randn_like(T) * 0.01
    r.requires_grad_(True)

    optimizer = optim.Adam([r], lr=lr)

    for _ in range(iterations):
        optimizer.zero_grad()
        T_adv = T + r
        scores = surrogate_anomaly_score(model, T_adv, window_size)

        if target_indices is not None and len(target_indices) > 0:
            anomaly_loss = 0
            for idx in target_indices:
                start_w = max(0, idx - window_size + 1)
                end_w = min(len(scores), idx + 1)
                if start_w < end_w:
                    anomaly_loss = anomaly_loss + torch.mean(scores[start_w:end_w])
        else:
            anomaly_loss = torch.mean(scores)

        # GM objective: minimize anomaly_score + α·‖r‖²
        # (We minimize anomaly score to hide anomalies)
        l2_penalty = alpha_l2 * torch.sum(r ** 2)
        loss = mu * anomaly_loss + l2_penalty

        loss.backward()
        optimizer.step()

    return (T + r).detach()


def attack_sgm(model, T, window_size, mu=1.0, alpha_l2=1.0, lam=1.0,
               lr=0.01, iterations=200, target_indices=None):
    """
    Smooth Gradient Method (SGM) — Eq. 8 from the paper.
    
    Optimize r directly:
        min { -μ · Loss(x+r) + α · ‖r‖² + λ · Σ|r_i - r_{i+1}| }
    
    The fused lasso term λ·Σ|r_i - r_{i+1}| enforces smooth perturbations.
    """
    model.eval()
    r = torch.randn_like(T) * 0.01
    r.requires_grad_(True)

    optimizer = optim.Adam([r], lr=lr)

    for _ in range(iterations):
        optimizer.zero_grad()
        T_adv = T + r
        scores = surrogate_anomaly_score(model, T_adv, window_size)

        if target_indices is not None and len(target_indices) > 0:
            anomaly_loss = 0
            for idx in target_indices:
                start_w = max(0, idx - window_size + 1)
                end_w = min(len(scores), idx + 1)
                if start_w < end_w:
                    anomaly_loss = anomaly_loss + torch.mean(scores[start_w:end_w])
        else:
            anomaly_loss = torch.mean(scores)

        # L2 regularization
        l2_penalty = alpha_l2 * torch.sum(r ** 2)

        # Fused lasso: enforce smoothness (Eq. 8)
        fused_lasso = lam * torch.sum(torch.abs(r[1:] - r[:-1]))

        loss = mu * anomaly_loss + l2_penalty + fused_lasso

        loss.backward()
        optimizer.step()

    return (T + r).detach()


# ===========================================================
# Smoothness Metric (Paper Eq. 11)
# ===========================================================
def smoothness_metric(x):
    """
    Paper Eq. 11:  s(x) = (1/T) * Σ (x_{t-1} - 2·x_t + x_{t+1})²
    Lower = smoother.
    """
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    if T < 3:
        return 0.0
    second_diff = x[:-2] - 2 * x[1:-1] + x[2:]
    return np.sum(second_diff ** 2) / T


# ===========================================================
# Evaluation
# ===========================================================
def evaluate_signal(X, y, slidingWindow):
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    results = {}
    for model_name in ['IForest', 'PCA']:
        try:
            X_feat = Window(window=slidingWindow).convert(X).to_numpy()
            if model_name == 'IForest':
                m = IForest(n_jobs=1)
            else:
                m = PCA()
            m.fit(X_feat)
            score = np.array(m.decision_scores_, dtype=np.float64)
            score = np.pad(score, (slidingWindow - 1, 0), 'constant',
                           constant_values=(score[0],))
            score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            metrics = get_metrics(score, y)
            results[model_name] = metrics['AUC_ROC']
        except Exception as e:
            print(f"    Error {model_name}: {e}")
            results[model_name] = None
    return results


def get_anomaly_indices(y):
    return np.where(y > 0)[0]


# ===========================================================
# Main Test
# ===========================================================
def run_adversarial_test():
    print("=" * 70)
    print("ADVERSARIAL ATTACK TEST — BIM / GM / SGM (Pialla et al.)")
    print("Adapted for anomaly detection via surrogate AE")
    print("=" * 70)
    print(f"BIM: ε={BIM_EPSILON}, α={BIM_ALPHA}, iter={BIM_ITERATIONS}")
    print(f"GM:  μ={GM_MU}, α={GM_ALPHA}, lr={GM_LR}, iter={GM_ITERATIONS}")
    print(f"SGM: μ={GM_MU}, α={GM_ALPHA}, λ={SGM_LAMBDA}, lr={GM_LR}, iter={GM_ITERATIONS}")

    df_subset = pd.read_csv(SUBSET_CSV)
    test_datasets = df_subset[df_subset['data_len'] > 500].head(N_DATASETS)

    all_results = []

    for i, (_, row) in enumerate(test_datasets.iterrows()):
        name = row['baseline_name']
        filepath = row['filepath']
        folder = row['folder']
        print(f"\n{'='*60}")
        print(f"[{i+1}/{N_DATASETS}] {folder}/{name}")
        print(f"{'='*60}")

        # Load
        df = pd.read_csv(filepath, header=None)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_orig = df.iloc[:, 0].values.astype(np.float64)
        y = df.iloc[:, 1].values.astype(np.float64)

        X_mean, X_std = X_orig.mean(), X_orig.std() + 1e-8
        X_norm = (X_orig - X_mean) / X_std
        T = torch.tensor(X_norm, dtype=torch.float32)
        slidingWindow = find_length(X_orig)
        ws = min(WINDOW_SIZE, len(X_norm) // 4)

        # Train surrogate
        print("  Training surrogate AE...")
        surrogate = train_surrogate(T, ws, epochs=50)

        # Baseline
        print("  Evaluating baseline...")
        auc_orig = evaluate_signal(X_orig, y, slidingWindow)
        smooth_orig = smoothness_metric(X_norm)
        print(f"  Baseline -> IF: {auc_orig['IForest']:.3f} | PCA: {auc_orig['PCA']:.3f} | Smoothness: {smooth_orig:.4f}")

        # Attack targets: anomaly regions
        anomaly_idx = get_anomaly_indices(y)
        target = anomaly_idx if len(anomaly_idx) > 0 else None

        attacks = {
            'BIM': lambda: attack_bim(surrogate, T, ws,
                                      epsilon=BIM_EPSILON, alpha=BIM_ALPHA,
                                      iterations=BIM_ITERATIONS, target_indices=target),
            'GM':  lambda: attack_gm(surrogate, T, ws,
                                     mu=GM_MU, alpha_l2=GM_ALPHA, lr=GM_LR,
                                     iterations=GM_ITERATIONS, target_indices=target),
            'SGM': lambda: attack_sgm(surrogate, T, ws,
                                      mu=GM_MU, alpha_l2=GM_ALPHA, lam=SGM_LAMBDA,
                                      lr=GM_LR, iterations=GM_ITERATIONS, target_indices=target),
        }

        attack_data = {}
        for attack_name, attack_fn in attacks.items():
            print(f"\n  [{attack_name}] Running...")
            T_adv = attack_fn()
            X_adv = T_adv.numpy() * X_std + X_mean
            perturbation = X_adv - X_orig

            # Evaluate
            auc_adv = evaluate_signal(X_adv, y, slidingWindow)
            l2_norm = np.linalg.norm(perturbation) / len(X_orig)
            smooth_adv = smoothness_metric(T_adv.numpy())
            smooth_pert = smoothness_metric((T_adv - T).numpy())

            print(f"    IF: {auc_orig['IForest']:.3f} -> {auc_adv['IForest']:.3f} (Δ={auc_orig['IForest'] - auc_adv['IForest']:+.3f})")
            print(f"    PCA: {auc_orig['PCA']:.3f} -> {auc_adv['PCA']:.3f} (Δ={auc_orig['PCA'] - auc_adv['PCA']:+.3f})")
            print(f"    L2 norm: {l2_norm:.4f} | Perturbation smoothness: {smooth_pert:.4f}")

            attack_data[attack_name] = {
                'X_adv': X_adv, 'perturbation': perturbation,
                'auc_adv': auc_adv, 'l2_norm': l2_norm,
                'smooth_pert': smooth_pert,
            }

            for model_name in ['IForest', 'PCA']:
                all_results.append({
                    'dataset': name, 'folder': folder,
                    'model': model_name, 'attack': attack_name,
                    'AUC_Original': auc_orig.get(model_name),
                    'AUC_Attacked': auc_adv.get(model_name),
                    'AUC_Drop': (auc_orig.get(model_name) or 0) - (auc_adv.get(model_name) or 0),
                    'L2_norm': l2_norm,
                    'Smoothness_perturbation': smooth_pert,
                    'Smoothness_original': smooth_orig,
                })

        # Plot detail for first 3 datasets
        if i < 3:
            plot_attack_detail(X_orig, y, auc_orig, attack_data, name, folder, i)

    # Save and plot summary
    df_res = pd.DataFrame(all_results)
    csv_out = os.path.join(PLOT_DIR, "adversarial_results_paper.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\nResults saved to {csv_out}")

    plot_summary(df_res)
    plot_smoothness_comparison(df_res)
    return df_res


def plot_attack_detail(X_orig, y, auc_orig, attack_data, name, folder, idx):
    """4-panel plot: Original + BIM + GM + SGM with perturbation overlay."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    x_axis = np.arange(len(X_orig))

    colors = {'BIM': '#FF5722', 'GM': '#FF9800', 'SGM': '#9C27B0'}

    # Panel 0: Original
    ax = axes[0]
    ax.plot(x_axis, X_orig, color='#2196F3', linewidth=0.5, label='Original')
    anomaly_mask = y > 0
    if anomaly_mask.any():
        ax.fill_between(x_axis, X_orig.min(), X_orig.max(),
                        where=anomaly_mask, alpha=0.15, color='red', label='Anomaly regions')
    ax.set_title(f'Original Signal | IF: {auc_orig["IForest"]:.3f} | PCA: {auc_orig["PCA"]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylabel('Value', fontsize=9)

    # Panels 1-3: BIM, GM, SGM
    for panel, attack_name in enumerate(['BIM', 'GM', 'SGM'], start=1):
        ax = axes[panel]
        data = attack_data[attack_name]
        auc = data['auc_adv']

        ax.plot(x_axis, X_orig, color='#2196F3', linewidth=0.3, alpha=0.3, label='Original')
        ax.plot(x_axis, data['X_adv'], color=colors[attack_name], linewidth=0.5,
                label=f'{attack_name} Perturbed')
        if anomaly_mask.any():
            ax.fill_between(x_axis, X_orig.min(), X_orig.max(),
                            where=anomaly_mask, alpha=0.08, color='red')

        drop_if = auc_orig['IForest'] - auc['IForest']
        drop_pca = auc_orig['PCA'] - auc['PCA']
        ax.set_title(f'{attack_name} Attack | IF: {auc["IForest"]:.3f} (Δ{drop_if:+.3f}) | '
                     f'PCA: {auc["PCA"]:.3f} (Δ{drop_pca:+.3f}) | '
                     f'L2: {data["l2_norm"]:.4f} | Smooth: {data["smooth_pert"]:.2f}',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylabel('Value', fontsize=9)

    axes[3].set_xlabel('Time Step', fontsize=10)

    fig.suptitle(f'Adversarial Attack Comparison: {folder}/{name}\n'
                 f'(BIM: ε={BIM_EPSILON}, GM/SGM: μ={GM_MU}, α={GM_ALPHA}, λ_sgm={SGM_LAMBDA})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(PLOT_DIR, f"paper_detail_{idx+1}_{folder}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved detail plot: {out}")


def plot_summary(df):
    """Side-by-side bar chart: AUC Drop per attack method for IForest and PCA."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    attacks = ['BIM', 'GM', 'SGM']
    x = np.arange(len(attacks))
    colors = ['#FF5722', '#FF9800', '#9C27B0']

    for ax_idx, model in enumerate(['IForest', 'PCA']):
        ax = axes[ax_idx]
        mask = df['model'] == model

        means = [df.loc[mask & (df['attack'] == a), 'AUC_Drop'].mean() for a in attacks]
        stds = [df.loc[mask & (df['attack'] == a), 'AUC_Drop'].std() for a in attacks]

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, width=0.6)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Attack Method', fontsize=12)
        ax.set_ylabel('Mean AUC-ROC Drop (+/- std)', fontsize=11)
        ax.set_title(f'{model}: Attack Effectiveness (Targeted)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, fontsize=11)
        ax.grid(axis='y', alpha=0.2)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            if mean != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{mean:+.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "paper_attack_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attack comparison: {out}")


def plot_smoothness_comparison(df):
    """Compare L2 norm vs smoothness for each attack (like Paper Fig. 2 win/draw/loss)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    attacks = ['BIM', 'GM', 'SGM']
    colors = ['#FF5722', '#FF9800', '#9C27B0']

    # Left: L2 norm comparison
    ax = axes[0]
    for a, c in zip(attacks, colors):
        vals = df.loc[df['attack'] == a, 'L2_norm'].values
        ax.bar(attacks.index(a), np.mean(vals), yerr=np.std(vals), capsize=5,
               color=c, alpha=0.85, width=0.6, label=a)
    ax.set_xlabel('Attack Method', fontsize=12)
    ax.set_ylabel('Mean L2 Norm (per sample)', fontsize=11)
    ax.set_title('Perturbation Magnitude (Lower = stealthier)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attacks, fontsize=11)
    ax.grid(axis='y', alpha=0.2)

    # Right: Smoothness comparison (Paper Eq. 11)
    ax = axes[1]
    for a, c in zip(attacks, colors):
        vals = df.loc[df['attack'] == a, 'Smoothness_perturbation'].values
        ax.bar(attacks.index(a), np.mean(vals), yerr=np.std(vals), capsize=5,
               color=c, alpha=0.85, width=0.6, label=a)
    ax.set_xlabel('Attack Method', fontsize=12)
    ax.set_ylabel('Smoothness s(r) — Paper Eq. 11', fontsize=11)
    ax.set_title('Perturbation Smoothness (Lower = smoother)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attacks, fontsize=11)
    ax.grid(axis='y', alpha=0.2)

    fig.suptitle('Attack Stealth Analysis: L2 Norm vs Smoothness\n(Paper-style metrics)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(PLOT_DIR, "paper_stealth_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved stealth analysis: {out}")

    # Print final summary
    print("\n" + "=" * 70)
    print("ADVERSARIAL ATTACK SUMMARY — BIM / GM / SGM")
    print("=" * 70)
    for model in ['IForest', 'PCA']:
        print(f"\n{model}:")
        for a in attacks:
            mask = (df['model'] == model) & (df['attack'] == a)
            drop = df.loc[mask, 'AUC_Drop'].mean()
            l2 = df.loc[mask, 'L2_norm'].mean()
            smooth = df.loc[mask, 'Smoothness_perturbation'].mean()
            print(f"  {a:4s} | Drop: {drop:+.4f} | L2: {l2:.4f} | Smoothness: {smooth:.4f}")


if __name__ == "__main__":
    run_adversarial_test()
