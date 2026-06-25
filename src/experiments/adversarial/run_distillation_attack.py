"""
Adversarial Attack via Surrogate Distillation.

Instead of a generic AutoEncoder, we train a neural network to MIMIC
the target detector's anomaly scores (IForest/PCA). This creates a
differentiable proxy whose gradients directly point toward reducing
the target's scores.

Generalizable to ANY anomaly detector that produces scores.

Usage:
    python -u scripts/test_distillation_attack.py
"""
import os, sys, torch, torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.pca import PCA
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

SUBSET_CSV = "c:/Users/gkost/thesis_timeseries/results/tables/robust_subset_TSB.csv"
PLOT_DIR = "c:/Users/gkost/thesis_timeseries/results/plots/adversarial"
os.makedirs(PLOT_DIR, exist_ok=True)

WINDOW_SIZE = 50
N_DATASETS = 8
ATTACK_ITERS = 200
ATTACK_LR = 0.01
SGM_LAMBDA = 1.0


# ===========================================================
# Surrogate Distillation Model
# ===========================================================
class SurrogateScoreNet(nn.Module):
    """Neural network that learns to predict a target detector's anomaly scores."""
    def __init__(self, window_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_size, window_size),
            nn.ReLU(),
            nn.Linear(window_size, window_size // 2),
            nn.ReLU(),
            nn.Linear(window_size // 2, max(2, window_size // 4)),
            nn.ReLU(),
            nn.Linear(max(2, window_size // 4), 1)  # Single score output
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SurrogateAE(nn.Module):
    """Old-style generic AutoEncoder (for comparison)."""
    def __init__(self, ws):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(ws, ws//2), nn.ReLU(),
                                     nn.Linear(ws//2, max(2, ws//4)), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(max(2, ws//4), ws//2), nn.ReLU(),
                                     nn.Linear(ws//2, ws))
    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_distillation_surrogate(X_signal, y_labels, target_model_name, window_size,
                                  sliding_window, epochs=100):
    """
    Train a NN to mimic the target detector's anomaly scores.
    
    1. Fit the target detector on the data
    2. Get its per-point anomaly scores (ground truth for the surrogate)
    3. Train NN: input=window -> output=predicted_score
    """
    X = np.ascontiguousarray(X_signal, dtype=np.float64)

    # Step 1: Fit the target detector and get its scores
    X_feat = Window(window=sliding_window).convert(X).to_numpy()
    if target_model_name == 'IForest':
        detector = IForest(n_jobs=1)
    else:
        detector = PCA()
    detector.fit(X_feat)
    raw_scores = np.array(detector.decision_scores_, dtype=np.float64)
    # Pad to match signal length
    scores_full = np.pad(raw_scores, (sliding_window - 1, 0),
                         'constant', constant_values=(raw_scores[0],))
    scores_full = np.nan_to_num(scores_full, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize scores to [0, 1]
    s_min, s_max = scores_full.min(), scores_full.max() + 1e-8
    scores_norm = (scores_full - s_min) / (s_max - s_min)

    # Step 2: Create training windows + target scores
    X_norm = (X - X.mean()) / (X.std() + 1e-8)
    T = torch.tensor(X_norm, dtype=torch.float32)
    windows = T.unfold(0, window_size, 1)  # [N_windows, window_size]

    # Target: mean score for each window
    scores_tensor = torch.tensor(scores_norm, dtype=torch.float32)
    window_scores = scores_tensor.unfold(0, window_size, 1).mean(dim=1)

    # Step 3: Train the surrogate
    model = SurrogateScoreNet(window_size)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted = model(windows)
        loss = criterion(predicted, window_scores)
        loss.backward()
        optimizer.step()

    # Check fit quality
    with torch.no_grad():
        pred_final = model(windows).numpy()
    correlation = np.corrcoef(pred_final, window_scores.numpy())[0, 1]

    return model, scores_full, correlation


def train_generic_ae(X_signal, window_size, epochs=50):
    """Train generic AE (old approach, for comparison)."""
    X_norm = (X_signal - X_signal.mean()) / (X_signal.std() + 1e-8)
    T = torch.tensor(X_norm, dtype=torch.float32)
    model = SurrogateAE(window_size)
    opt = optim.Adam(model.parameters(), lr=0.01)
    crit = nn.MSELoss()
    wins = T.unfold(0, window_size, 1)
    for _ in range(epochs):
        opt.zero_grad(); loss = crit(model(wins), wins); loss.backward(); opt.step()
    return model


# ===========================================================
# Attack Methods
# ===========================================================

def attack_distillation(surrogate, T, window_size, lr=0.01, iters=200,
                        lam_smooth=0.0, target_indices=None):
    """
    Attack using distilled surrogate.
    Minimize the surrogate's predicted anomaly score.
    Optional: lam_smooth > 0 adds SGM-style fused lasso.
    """
    surrogate.eval()
    r = torch.randn_like(T) * 0.01
    r.requires_grad_(True)
    optimizer = optim.Adam([r], lr=lr)

    for _ in range(iters):
        optimizer.zero_grad()
        T_adv = T + r
        windows = T_adv.unfold(0, window_size, 1)
        predicted_scores = surrogate(windows)

        if target_indices is not None and len(target_indices) > 0:
            # Target specific windows overlapping anomaly regions
            loss_score = 0
            for idx in target_indices:
                start_w = max(0, idx - window_size + 1)
                end_w = min(len(predicted_scores), idx + 1)
                if start_w < end_w:
                    loss_score = loss_score + torch.mean(predicted_scores[start_w:end_w])
        else:
            loss_score = torch.mean(predicted_scores)

        # L2 regularization
        l2_reg = torch.sum(r ** 2)

        # Optional fused lasso for smooth perturbations
        if lam_smooth > 0:
            fused_lasso = lam_smooth * torch.sum(torch.abs(r[1:] - r[:-1]))
        else:
            fused_lasso = 0

        loss = loss_score + 1.0 * l2_reg + fused_lasso
        loss.backward()
        optimizer.step()

    return (T + r).detach()


def attack_ae_baseline(model, T, window_size, lr=0.01, iters=200,
                       lam_smooth=0.0, target_indices=None):
    """Attack using generic AE (old approach, for comparison)."""
    model.eval()
    r = torch.randn_like(T) * 0.01
    r.requires_grad_(True)
    optimizer = optim.Adam([r], lr=lr)

    for _ in range(iters):
        optimizer.zero_grad()
        T_adv = T + r
        windows = T_adv.unfold(0, window_size, 1)
        recon = model(windows)
        scores = torch.mean((recon - windows) ** 2, dim=1)

        if target_indices is not None and len(target_indices) > 0:
            loss_score = 0
            for idx in target_indices:
                start_w = max(0, idx - window_size + 1)
                end_w = min(len(scores), idx + 1)
                if start_w < end_w:
                    loss_score = loss_score + torch.mean(scores[start_w:end_w])
        else:
            loss_score = torch.mean(scores)

        l2_reg = torch.sum(r ** 2)
        fused_lasso = lam_smooth * torch.sum(torch.abs(r[1:] - r[:-1])) if lam_smooth > 0 else 0
        loss = loss_score + 1.0 * l2_reg + fused_lasso
        loss.backward()
        optimizer.step()

    return (T + r).detach()


# ===========================================================
# Evaluation & Metrics
# ===========================================================

def smoothness(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3: return 0.0
    d2 = x[:-2] - 2*x[1:-1] + x[2:]
    return np.sum(d2**2) / len(x)


def evaluate_signal(X, y, sw):
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    results = {}
    for name in ['IForest', 'PCA']:
        try:
            Xf = Window(window=sw).convert(X).to_numpy()
            m = IForest(n_jobs=1) if name == 'IForest' else PCA()
            m.fit(Xf)
            sc = np.pad(np.array(m.decision_scores_, dtype=np.float64),
                        (sw-1, 0), 'constant', constant_values=(m.decision_scores_[0],))
            sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
            results[name] = get_metrics(sc, y)['AUC_ROC']
        except:
            results[name] = None
    return results


# ===========================================================
# Main
# ===========================================================
def main():
    print("=" * 70)
    print("ADVERSARIAL ATTACK: Distillation vs Generic AutoEncoder")
    print("=" * 70)

    df_sub = pd.read_csv(SUBSET_CSV)
    datasets = df_sub[df_sub['data_len'] > 500].head(N_DATASETS)
    all_results = []

    for i, (_, row) in enumerate(datasets.iterrows()):
        name, fp, folder = row['baseline_name'], row['filepath'], row['folder']
        print(f"\n{'='*60}")
        print(f"[{i+1}/{N_DATASETS}] {folder}/{name}")
        print(f"{'='*60}")

        df = pd.read_csv(fp, header=None).apply(pd.to_numeric, errors='coerce').fillna(0)
        X_orig = df.iloc[:, 0].values.astype(np.float64)
        y = df.iloc[:, 1].values.astype(np.float64)
        Xm, Xs = X_orig.mean(), X_orig.std() + 1e-8
        Xn = (X_orig - Xm) / Xs
        T = torch.tensor(Xn, dtype=torch.float32)
        sw = find_length(X_orig)
        ws = min(WINDOW_SIZE, len(Xn) // 4)
        target = np.where(y > 0)[0]
        target = target if len(target) > 0 else None

        # Baseline AUC
        auc_orig = evaluate_signal(X_orig, y, sw)
        print(f"  Baseline  IF:{auc_orig['IForest']:.3f}  PCA:{auc_orig['PCA']:.3f}")

        # --- Train surrogates ---
        print("  Training generic AE surrogate...")
        ae_model = train_generic_ae(X_orig, ws, epochs=50)

        for target_model in ['IForest', 'PCA']:
            print(f"\n  === Target: {target_model} ===")

            # Train distillation surrogate for this specific target
            print(f"  Training distillation surrogate for {target_model}...")
            dist_model, _, corr = train_distillation_surrogate(
                X_orig, y, target_model, ws, sw, epochs=100)
            print(f"  Surrogate-to-{target_model} correlation: {corr:.3f}")

            # Attack 1: Generic AE (old approach)
            print(f"  Attacking with Generic AE...")
            T_ae = attack_ae_baseline(ae_model, T, ws, lr=ATTACK_LR, iters=ATTACK_ITERS,
                                       target_indices=target)
            X_ae = T_ae.numpy() * Xs + Xm
            auc_ae = evaluate_signal(X_ae, y, sw)
            r_ae = (T_ae - T).numpy()
            l2_ae = np.linalg.norm(r_ae) / len(r_ae)

            # Attack 2: Distilled surrogate (new approach)
            print(f"  Attacking with Distilled surrogate ({target_model})...")
            T_dist = attack_distillation(dist_model, T, ws, lr=ATTACK_LR, iters=ATTACK_ITERS,
                                          target_indices=target)
            X_dist = T_dist.numpy() * Xs + Xm
            auc_dist = evaluate_signal(X_dist, y, sw)
            r_dist = (T_dist - T).numpy()
            l2_dist = np.linalg.norm(r_dist) / len(r_dist)

            # Attack 3: Distilled + SGM smoothing
            print(f"  Attacking with Distilled SGM ({target_model})...")
            T_dist_sgm = attack_distillation(dist_model, T, ws, lr=ATTACK_LR, iters=ATTACK_ITERS,
                                              lam_smooth=SGM_LAMBDA, target_indices=target)
            X_dist_sgm = T_dist_sgm.numpy() * Xs + Xm
            auc_dist_sgm = evaluate_signal(X_dist_sgm, y, sw)
            r_dist_sgm = (T_dist_sgm - T).numpy()
            l2_dist_sgm = np.linalg.norm(r_dist_sgm) / len(r_dist_sgm)

            drop_ae = (auc_orig[target_model] or 0) - (auc_ae[target_model] or 0)
            drop_dist = (auc_orig[target_model] or 0) - (auc_dist[target_model] or 0)
            drop_dist_sgm = (auc_orig[target_model] or 0) - (auc_dist_sgm[target_model] or 0)

            print(f"  Results on {target_model}:")
            print(f"    Generic AE:     {auc_orig[target_model]:.3f} → {auc_ae[target_model]:.3f} (Δ={drop_ae:+.3f}) L2:{l2_ae:.5f}")
            print(f"    Distilled:      {auc_orig[target_model]:.3f} → {auc_dist[target_model]:.3f} (Δ={drop_dist:+.3f}) L2:{l2_dist:.5f}")
            print(f"    Distilled+SGM:  {auc_orig[target_model]:.3f} → {auc_dist_sgm[target_model]:.3f} (Δ={drop_dist_sgm:+.3f}) L2:{l2_dist_sgm:.5f}")

            for attack_name, drop, l2, r_vec in [
                ('Generic_AE', drop_ae, l2_ae, r_ae),
                ('Distilled', drop_dist, l2_dist, r_dist),
                ('Distilled_SGM', drop_dist_sgm, l2_dist_sgm, r_dist_sgm)
            ]:
                all_results.append({
                    'dataset': name, 'folder': folder,
                    'target_model': target_model, 'attack': attack_name,
                    'AUC_Original': auc_orig[target_model],
                    'AUC_Drop': drop, 'L2': l2,
                    'Smoothness': smoothness(r_vec),
                    'Surrogate_Correlation': corr if 'Distill' in attack_name else 0,
                })

    df_res = pd.DataFrame(all_results)
    csv_out = os.path.join(PLOT_DIR, "distillation_results.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\nResults saved to {csv_out}")

    plot_distillation_comparison(df_res)


def plot_distillation_comparison(df):
    """Bar chart: Generic AE vs Distilled vs Distilled+SGM for each target model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    attacks = ['Generic_AE', 'Distilled', 'Distilled_SGM']
    labels = ['Generic AE\n(old)', 'Distilled\n(new)', 'Distilled\n+ SGM']
    colors = ['#FF9800', '#2196F3', '#9C27B0']

    for ax_idx, model in enumerate(['IForest', 'PCA']):
        ax = axes[ax_idx]
        mask = df['target_model'] == model

        means = [df.loc[mask & (df['attack'] == a), 'AUC_Drop'].mean() for a in attacks]
        stds = [df.loc[mask & (df['attack'] == a), 'AUC_Drop'].std() for a in attacks]

        bars = ax.bar(range(len(attacks)), means, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, width=0.6)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Mean AUC-ROC Drop', fontsize=11)
        ax.set_title(f'Target: {model}', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(axis='y', alpha=0.2)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{mean:+.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.suptitle('Adversarial Attack: Surrogate Distillation vs Generic AE\n'
                 'Distillation mimics the target model → stronger attacks',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out = os.path.join(PLOT_DIR, "distillation_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison: {out}")

    # Summary
    print("\n" + "=" * 70)
    print("DISTILLATION vs GENERIC AE — SUMMARY")
    print("=" * 70)
    for model in ['IForest', 'PCA']:
        print(f"\nTarget: {model}")
        for a, label in zip(attacks, ['Generic AE', 'Distilled', 'Distilled+SGM']):
            mask = (df['target_model'] == model) & (df['attack'] == a)
            drop = df.loc[mask, 'AUC_Drop'].mean()
            l2 = df.loc[mask, 'L2'].mean()
            sm = df.loc[mask, 'Smoothness'].mean()
            print(f"  {label:<16s} | Drop: {drop:+.4f} | L2: {l2:.5f} | Smooth: {sm:.6f}")

    # Improvement factor
    for model in ['IForest', 'PCA']:
        mask_ae = (df['target_model'] == model) & (df['attack'] == 'Generic_AE')
        mask_d = (df['target_model'] == model) & (df['attack'] == 'Distilled')
        ae_drop = df.loc[mask_ae, 'AUC_Drop'].mean()
        d_drop = df.loc[mask_d, 'AUC_Drop'].mean()
        if ae_drop > 0:
            print(f"\n  {model}: Distillation is {d_drop/ae_drop:.1f}x more effective than Generic AE")


if __name__ == "__main__":
    main()
