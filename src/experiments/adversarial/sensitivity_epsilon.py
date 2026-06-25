"""
Adversarial Epsilon Sensitivity Analysis.

Varies the perturbation budget (epsilon) and measures how attack
effectiveness scales. Generates a sensitivity curve with confidence bands.

Usage:
    python scripts/test_adversarial_epsilon.py
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

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.pca import PCA
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

# ---- Config ----
SUBSET_CSV = "c:/Users/gkost/thesis_timeseries/results/tables/robust_subset_TSB.csv"
PLOT_DIR = "c:/Users/gkost/thesis_timeseries/results/plots/adversarial"
os.makedirs(PLOT_DIR, exist_ok=True)

WINDOW_SIZE = 50
ALPHA = 0.02
ITERATIONS = 30
SMOOTH_WINDOW = 15

# Epsilon values to test
EPSILONS = [0.05, 0.1, 0.3, 0.5, 0.8]
N_DATASETS = 8


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


def train_surrogate(T_normal, window_size, epochs=50):
    model = SurrogateAE(window_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    windows = T_normal.unfold(0, window_size, 1)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(windows), windows)
        loss.backward()
        optimizer.step()
    return model


def apply_smoothing(perturbation, smooth_window):
    if smooth_window <= 1:
        return perturbation
    kernel = torch.ones(1, 1, smooth_window) / smooth_window
    p = perturbation.view(1, 1, -1)
    pad_left = smooth_window // 2
    pad_right = smooth_window - 1 - pad_left
    p_padded = torch.nn.functional.pad(p, (pad_left, pad_right), mode='replicate')
    return torch.nn.functional.conv1d(p_padded, kernel).view(-1)


def adversarial_attack(model, T, window_size, epsilon, alpha, iterations,
                       smooth_window=1, target_indices=None, minimize=True):
    """minimize=True → gradient DESCENT to hide anomalies, False → ASCENT for false positives."""
    model.eval()
    T_adv = T.clone().detach().requires_grad_(True)
    direction = -1.0 if minimize else 1.0
    for _ in range(iterations):
        windows = T_adv.unfold(0, window_size, 1)
        reconstruction = model(windows)
        loss_per_window = torch.mean((reconstruction - windows) ** 2, dim=1)
        total_loss = 0
        if target_indices is not None and len(target_indices) > 0:
            for idx in target_indices:
                start_w = max(0, idx - window_size + 1)
                end_w = min(len(loss_per_window), idx + 1)
                if start_w < end_w:
                    total_loss += torch.sum(loss_per_window[start_w:end_w])
        else:
            total_loss = torch.sum(loss_per_window)
        total_loss.backward()
        gradients = T_adv.grad
        with torch.no_grad():
            perturbation = gradients.sign()
            if smooth_window > 1:
                perturbation = apply_smoothing(perturbation, smooth_window)
            T_adv_update = T_adv + direction * alpha * perturbation
            eta = torch.clamp(T_adv_update - T, min=-epsilon, max=epsilon)
            T_adv = (T + eta).detach().requires_grad_(True)
    return T_adv.detach()


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
            score = np.pad(score, (slidingWindow - 1, 0), 'constant', constant_values=(score[0],))
            score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            metrics = get_metrics(score, y)
            results[model_name] = metrics['AUC_ROC']
        except Exception as e:
            print(f"    Error {model_name}: {e}")
            results[model_name] = None
    return results


# ===========================================================
# Main
# ===========================================================
def run_epsilon_sensitivity():
    print("=" * 70)
    print("ADVERSARIAL EPSILON SENSITIVITY ANALYSIS")
    print(f"Epsilons: {EPSILONS}")
    print(f"Datasets: {N_DATASETS}")
    print("=" * 70)

    df_subset = pd.read_csv(SUBSET_CSV)
    test_datasets = df_subset[df_subset['data_len'] > 500].head(N_DATASETS)

    all_results = []

    for i, (_, row) in enumerate(test_datasets.iterrows()):
        name = row['baseline_name']
        filepath = row['filepath']
        folder = row['folder']
        print(f"\n[{i+1}/{N_DATASETS}] {folder}/{name}")

        # Load & normalize
        df = pd.read_csv(filepath, header=None)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_orig = df.iloc[:, 0].values.astype(np.float64)
        y = df.iloc[:, 1].values.astype(np.float64)
        X_mean, X_std = X_orig.mean(), X_orig.std() + 1e-8
        X_norm = (X_orig - X_mean) / X_std
        T = torch.tensor(X_norm, dtype=torch.float32)
        slidingWindow = find_length(X_orig)
        ws = min(WINDOW_SIZE, len(X_norm) // 4)

        # Train surrogate once per dataset
        print("  Training surrogate AE...")
        surrogate = train_surrogate(T, ws, epochs=40)

        # Baseline evaluation
        auc_orig = evaluate_signal(X_orig, y, slidingWindow)
        print(f"  Baseline -> IF: {auc_orig['IForest']:.3f} | PCA: {auc_orig['PCA']:.3f}")

        # Sweep epsilon
        for eps in EPSILONS:
            print(f"  epsilon={eps}...", end=" ")

            # Global BIM
            T_bim = adversarial_attack(surrogate, T, ws, eps, ALPHA, ITERATIONS,
                                       smooth_window=1, target_indices=None)
            X_bim = T_bim.numpy() * X_std + X_mean
            auc_bim = evaluate_signal(X_bim, y, slidingWindow)

            # Global Smooth
            T_smooth = adversarial_attack(surrogate, T, ws, eps, ALPHA, ITERATIONS,
                                          smooth_window=SMOOTH_WINDOW, target_indices=None)
            X_smooth = T_smooth.numpy() * X_std + X_mean
            auc_smooth = evaluate_signal(X_smooth, y, slidingWindow)

            # L2 norm of perturbation (stealth metric)
            l2_bim = np.linalg.norm(X_bim - X_orig) / len(X_orig)
            l2_smooth = np.linalg.norm(X_smooth - X_orig) / len(X_orig)

            for model_name in ['IForest', 'PCA']:
                all_results.append({
                    'dataset': name, 'folder': folder, 'model': model_name,
                    'epsilon': eps,
                    'AUC_Original': auc_orig.get(model_name),
                    'AUC_BIM': auc_bim.get(model_name),
                    'AUC_Smooth': auc_smooth.get(model_name),
                    'Drop_BIM': (auc_orig.get(model_name) or 0) - (auc_bim.get(model_name) or 0),
                    'Drop_Smooth': (auc_orig.get(model_name) or 0) - (auc_smooth.get(model_name) or 0),
                    'L2_BIM': l2_bim,
                    'L2_Smooth': l2_smooth,
                })

            print(f"IF: BIM={auc_bim['IForest']:.3f} Sm={auc_smooth['IForest']:.3f} | "
                  f"PCA: BIM={auc_bim['PCA']:.3f} Sm={auc_smooth['PCA']:.3f}")

    df_res = pd.DataFrame(all_results)
    csv_out = os.path.join(PLOT_DIR, "epsilon_sensitivity_results.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\nResults saved to {csv_out}")

    plot_epsilon_sensitivity(df_res)
    return df_res


def plot_epsilon_sensitivity(df):
    """2x2 plot: AUC Drop + Absolute AUC for each model, BIM vs Smooth."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'BIM': '#FF5722', 'Smooth': '#9C27B0'}

    for col, model in enumerate(['IForest', 'PCA']):
        mask = df['model'] == model

        # Row 0: AUC Drop vs epsilon
        ax = axes[0, col]
        for attack, drop_col in [('BIM', 'Drop_BIM'), ('Smooth', 'Drop_Smooth')]:
            grouped = df.loc[mask].groupby('epsilon')[drop_col].agg(['mean', 'std'])
            x = grouped.index.values
            y_mean = grouped['mean'].values
            y_std = grouped['std'].values

            ax.plot(x, y_mean, marker='o', linewidth=2.5, markersize=8,
                    color=colors[attack], label=f'{attack} Attack')
            ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                            alpha=0.15, color=colors[attack])

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epsilon (Perturbation Budget)', fontsize=11)
        ax.set_ylabel('Mean AUC-ROC Drop', fontsize=11)
        ax.set_title(f'{model}: Attack Effectiveness vs Epsilon', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # Row 1: Absolute AUC vs epsilon
        ax = axes[1, col]
        for attack, auc_col in [('BIM', 'AUC_BIM'), ('Smooth', 'AUC_Smooth')]:
            grouped = df.loc[mask].groupby('epsilon')[auc_col].agg(['mean', 'std'])
            x = grouped.index.values
            y_mean = grouped['mean'].values
            y_std = grouped['std'].values

            ax.plot(x, y_mean, marker='s', linewidth=2.5, markersize=8,
                    color=colors[attack], label=f'{attack} Attack')
            ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                            alpha=0.15, color=colors[attack])

        # Baseline line
        baseline_auc = df.loc[mask, 'AUC_Original'].mean()
        ax.axhline(baseline_auc, color='#2196F3', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Baseline ({baseline_auc:.3f})')

        ax.set_xlabel('Epsilon (Perturbation Budget)', fontsize=11)
        ax.set_ylabel('AUC-ROC', fontsize=11)
        ax.set_title(f'{model}: Absolute AUC vs Epsilon', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle('Adversarial Attack Sensitivity to Perturbation Budget (ε)\n'
                 'Global Attack (Paper-style) — Mean ± 1 std over 8 datasets',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(PLOT_DIR, "epsilon_sensitivity_curve.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved epsilon sensitivity plot: {out}")

    # L2 norm plot (stealth analysis)
    plot_stealth_analysis(df)

    # Print summary
    print("\n" + "=" * 70)
    print("EPSILON SENSITIVITY SUMMARY")
    print("=" * 70)
    for model in ['IForest', 'PCA']:
        print(f"\n{model}:")
        for eps in EPSILONS:
            mask = (df['model'] == model) & (df['epsilon'] == eps)
            bim = df.loc[mask, 'Drop_BIM'].mean()
            smooth = df.loc[mask, 'Drop_Smooth'].mean()
            print(f"  ε={eps:.2f} | BIM: {bim:+.4f} | Smooth: {smooth:+.4f}")


def plot_stealth_analysis(df):
    """Plot L2 perturbation norm vs epsilon (how 'visible' is the attack)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'BIM': '#FF5722', 'Smooth': '#9C27B0'}

    for attack, l2_col in [('BIM', 'L2_BIM'), ('Smooth', 'L2_Smooth')]:
        grouped = df.groupby('epsilon')[l2_col].agg(['mean', 'std'])
        x = grouped.index.values
        y_mean = grouped['mean'].values
        y_std = grouped['std'].values

        ax.plot(x, y_mean, marker='o', linewidth=2.5, markersize=8,
                color=colors[attack], label=f'{attack} Attack')
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                        alpha=0.15, color=colors[attack])

    ax.set_xlabel('Epsilon (Perturbation Budget)', fontsize=12)
    ax.set_ylabel('Mean L2 Perturbation Norm (per sample)', fontsize=12)
    ax.set_title('Stealth Analysis: Perturbation Visibility vs Epsilon\n'
                 '(Lower = harder to detect the attack)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "stealth_l2_analysis.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved stealth analysis plot: {out}")


if __name__ == "__main__":
    run_epsilon_sensitivity()
