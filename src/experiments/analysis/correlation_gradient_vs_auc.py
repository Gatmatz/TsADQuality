"""
Correlation Experiment: Does Gradient Cancellation actually cause the defense?

For each dataset we:
1. Train a 2-head surrogate (IForest + PCA)
2. Measure the Cosine Similarity of the gradients (already done, reuse CSV)
3. Run an SGM attack using that surrogate
4. Measure IForest AUC on clean vs adversarial data
5. Correlate Cosine Similarity with AUC Drop

If correlation is strong & positive => Gradient Cancellation IS the cause
If no correlation => it's just optimization difficulty
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.pca import PCA
from TSB_UAD.utils.slidingWindows import find_length
from sklearn.metrics import roc_auc_score


class Surrogate2(nn.Module):
    def __init__(self, ws):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ws, ws), nn.LeakyReLU(),
            nn.Linear(ws, ws // 2), nn.LeakyReLU(),
            nn.Linear(ws // 2, 2), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


def apply_smoothing(perturbation, smooth_window=5):
    kernel = torch.ones(1, 1, smooth_window) / smooth_window
    p = perturbation.unsqueeze(0).unsqueeze(0)
    smoothed = F.conv1d(p, kernel, padding=smooth_window // 2)
    return smoothed.squeeze()


def minmax(s):
    return (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-8)


def run_experiment():
    df_reg = pd.read_csv("results/tables/robust_subset_TSB.csv")
    # Load pre-computed cosine similarities
    cos_df = pd.read_csv("outputs/metrics/gradient_cancellations.csv")
    cos_lookup = dict(zip(cos_df["dataset"], cos_df["cosine_sim"]))

    print(f"=== Correlation Experiment: {len(df_reg)} datasets ===\n")

    results = []
    alpha = 0.01
    epsilon = 0.15
    iterations = 30
    smooth_window = 5

    for idx, row in df_reg.iterrows():
        csv_path = row["filepath"]
        dataset_name = row["baseline_name"]

        # Skip datasets we don't have cosine similarity for
        if dataset_name not in cos_lookup:
            continue

        try:
            df = pd.read_csv(csv_path, header=None)
            X = df[0].to_numpy("float")
            y = df[1].to_numpy("int")
        except Exception:
            continue

        mask = ~np.isnan(X)
        X = X[mask]
        y = y[mask]

        if sum(y) < 10 or len(X) < 100:
            continue

        window_size = find_length(X)

        # --- Step 1: Get baseline scores ---
        clf_if = IForest(); clf_if.fit(X); s_if = clf_if.decision_scores_
        clf_pca = PCA(); clf_pca.fit(X); s_pca = clf_pca.decision_scores_

        # Baseline AUC for IForest (on clean data)
        try:
            auc_clean = roc_auc_score(y, minmax(s_if))
        except ValueError:
            continue

        # --- Step 2: Train surrogate ---
        y_targets = np.column_stack((minmax(s_if), minmax(s_pca)))
        y_targets_tensor = torch.tensor(y_targets, dtype=torch.float32)
        T_tensor = torch.tensor(X, dtype=torch.float32)

        model = Surrogate2(window_size)
        opt = optim.Adam(model.parameters(), lr=0.005)
        crit = nn.MSELoss()

        windows = T_tensor.unfold(0, window_size, 1)
        target_scores = y_targets_tensor[window_size - 1:]

        for _ in range(50):
            opt.zero_grad()
            loss = crit(model(windows), target_scores)
            loss.backward()
            opt.step()

        # --- Step 3: Generate adversarial perturbation (SGM) ---
        model.eval()
        delta = torch.zeros_like(T_tensor, requires_grad=False)

        for _ in range(iterations):
            T_adv = (T_tensor + delta).detach().requires_grad_(True)
            windows_adv = T_adv.unfold(0, window_size, 1)
            preds = model(windows_adv)
            # Maximize the maximum predicted score (ensemble loss)
            loss = -torch.mean(torch.max(preds, dim=1).values)
            loss.backward()

            grad = T_adv.grad.detach()
            grad_sign = grad.sign()
            raw_perturbation = -grad_sign * alpha
            smoothed = apply_smoothing(raw_perturbation, smooth_window)
            delta = (delta + smoothed).clamp(-epsilon, epsilon)

        # --- Step 4: Evaluate IForest on adversarial data ---
        X_adv = (T_tensor + delta).detach().numpy()
        clf_if_adv = IForest(); clf_if_adv.fit(X_adv); s_if_adv = clf_if_adv.decision_scores_

        try:
            auc_adv = roc_auc_score(y, minmax(s_if_adv))
        except ValueError:
            continue

        auc_drop = auc_clean - auc_adv
        cosine_sim = cos_lookup[dataset_name]

        results.append({
            "dataset": dataset_name,
            "cosine_sim": cosine_sim,
            "auc_clean": round(auc_clean, 4),
            "auc_adv": round(auc_adv, 4),
            "auc_drop": round(auc_drop, 4),
        })

        print(f"[{len(results)}] {dataset_name}: CosSim={cosine_sim:.3f}, AUC Drop={auc_drop:.4f}")

    # --- Step 5: Correlation Analysis ---
    res_df = pd.DataFrame(results)
    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "correlation_gradient_vs_auc.csv", index=False)

    cos_arr = res_df["cosine_sim"].values
    drop_arr = res_df["auc_drop"].values

    pearson_r, pearson_p = stats.pearsonr(cos_arr, drop_arr)
    spearman_r, spearman_p = stats.spearmanr(cos_arr, drop_arr)

    print("\n\n========================================")
    print("   CORRELATION ANALYSIS RESULTS")
    print("========================================")
    print(f"Datasets analyzed: {len(res_df)}")
    print(f"Pearson  r = {pearson_r:.4f}  (p-value = {pearson_p:.6f})")
    print(f"Spearman r = {spearman_r:.4f}  (p-value = {spearman_p:.6f})")
    print("========================================")

    if pearson_p < 0.05 and pearson_r > 0.2:
        print("\n=> [CONFIRMED]: Statistically significant POSITIVE correlation!")
        print("   Gradient Cancellation IS the cause of the ensemble defense.")
        print("   Datasets where gradients conflict more => less AUC drop.")
    elif pearson_p < 0.05 and pearson_r < -0.2:
        print("\n=> [SURPRISING]: Significant NEGATIVE correlation!")
        print("   Higher gradient similarity => LESS AUC drop (unexpected).")
    else:
        print("\n=> [NO EFFECT]: No significant correlation found.")
        print("   The defense is due to optimization difficulty, NOT gradient cancellation.")


if __name__ == "__main__":
    run_experiment()
