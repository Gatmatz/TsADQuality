"""
Adversarial vs Random Noise Comparison at Equal Perturbation Budget.

For each dataset and epsilon value, applies:
  1. Random uniform noise with ||perturbation||_inf <= epsilon
  2. BIM adversarial attack  with ||perturbation||_inf <= epsilon

Then evaluates IForest and PCA on both and compares AUC drop.
This demonstrates that adversarial perturbations cause significantly
more damage than random noise of equal magnitude.

Based on: Pialla et al. (2022, 2025), Kurakin et al. (2017)

Usage:
    python run_adversarial_vs_random.py
    python run_adversarial_vs_random.py --test
    python run_adversarial_vs_random.py --workers 2 --epsilons 0.05 0.1 0.2
"""
import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import warnings
import traceback

# ==========================================
# PATH CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "TSB-UAD"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    from TSB_UAD.models.iforest import IForest
    from TSB_UAD.models.pca import PCA
    from TSB_UAD.models.matrix_profile import MatrixProfile
    from TSB_UAD.models.feature import Window
    from TSB_UAD.utils.slidingWindows import find_length
    from TSB_UAD.vus.metrics import get_metrics
except ImportError as e:
    warnings.warn(f"TSB_UAD components could not be imported: {e}")

from data_loader import load_tsb_file

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
SUBSET_CSV  = PROJECT_ROOT / "results" / "tables" / "final_subset.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "adversarial_vs_random"

EPSILONS = [0.01, 0.05, 0.1, 0.2, 0.5]
N_RANDOM_SEEDS = 3  # Multiple seeds for random noise to get stable estimates

# BIM hyper-parameters (same as run_bim_gm_sgm.py)
BIM_ALPHA = 0.01
BIM_ITERATIONS = 200
SURROGATE_EPOCHS = 50
SURROGATE_WINDOW = 50  # Window size for surrogate AE


# ===========================================================
# Surrogate AutoEncoder (same as run_bim_gm_sgm.py)
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
    windows = T.unfold(0, window_size, 1)
    reconstruction = model(windows)
    return torch.mean((reconstruction - windows) ** 2, dim=1)


# ===========================================================
# Attack Methods
# ===========================================================
def attack_bim(model, T, window_size, epsilon, alpha=0.01, iterations=200,
               target_indices=None):
    """BIM attack: minimize anomaly score subject to ||perturbation||_inf <= epsilon."""
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
            T_adv_new = T_adv - alpha * T_adv.grad.sign()
            eta = torch.clamp(T_adv_new - T, min=-epsilon, max=epsilon)
            T_adv = (T + eta).detach().requires_grad_(True)

    return T_adv.detach()


def apply_random_noise(T, epsilon, seed=0):
    """Random uniform noise with ||perturbation||_inf <= epsilon."""
    rng = np.random.RandomState(seed)
    perturbation = rng.uniform(-epsilon, epsilon, size=len(T))
    return T + torch.tensor(perturbation, dtype=torch.float32)


def apply_random_noise_targeted(T, epsilon, target_indices, seed=0):
    """Random uniform noise ONLY at anomaly regions, ||perturbation||_inf <= epsilon."""
    rng = np.random.RandomState(seed)
    perturbation = np.zeros(len(T))
    if target_indices is not None and len(target_indices) > 0:
        perturbation[target_indices] = rng.uniform(-epsilon, epsilon, size=len(target_indices))
    return T + torch.tensor(perturbation, dtype=torch.float32)


# ===========================================================
# Evaluation (same models as run_bim_gm_sgm.py)
# ===========================================================
def evaluate_signal(X, y, slidingWindow, model_names=None):
    """Evaluate selected models on a signal, return dict of AUC_ROC values."""
    if model_names is None:
        model_names = ['IForest', 'PCA']
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    results = {}
    for model_name in model_names:
        try:
            if model_name == 'MP':
                m = MatrixProfile(window=slidingWindow)
                m.fit(X)
            else:
                X_feat = Window(window=slidingWindow).convert(X).to_numpy()
                if model_name == 'IForest':
                    m = IForest(n_jobs=1)
                elif model_name == 'PCA':
                    m = PCA()
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                m.fit(X_feat)
            score = np.array(m.decision_scores_, dtype=np.float64)
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                score.reshape(-1, 1)
            ).ravel()
            # Symmetric padding (TSB-UAD convention)
            full_score = np.array(
                [score[0]] * math.ceil((slidingWindow - 1) / 2) +
                list(score) +
                [score[-1]] * ((slidingWindow - 1) // 2)
            )
            score_clean = np.nan_to_num(full_score, nan=0.0, posinf=0.0, neginf=0.0)
            if len(np.unique(score_clean)) <= 1:
                results[model_name] = None
                continue
            metrics = get_metrics(score_clean, y, metric="all", slidingWindow=slidingWindow)
            results[model_name] = metrics.get('AUC_ROC', None)
        except Exception as e:
            results[model_name] = None
    return results


# ===========================================================
# Worker Function
# ===========================================================
def process_single_dataset(job_args):
    """Process one dataset across all epsilon values."""
    (file_path, epsilons, n_random_seeds, model_names) = job_args
    file_name = os.path.basename(file_path)

    try:
        # Load data
        data, label, canonical_name = load_tsb_file(file_path)

        # Standardize
        X_mean, X_std = data.mean(), data.std() + 1e-8
        X_norm = (data - X_mean) / X_std
        T = torch.tensor(X_norm, dtype=torch.float32)

        # Sliding window for evaluation
        slidingWindow = max(int(find_length(X_norm)), 10)

        # Surrogate window size
        ws = min(SURROGATE_WINDOW, len(X_norm) // 4)
        if ws < 4:
            return {'status': 'error', 'file': canonical_name,
                    'error': 'Series too short for surrogate AE'}

        # Train surrogate AE (once per dataset)
        surrogate = train_surrogate(T, ws, epochs=SURROGATE_EPOCHS)

        # Baseline evaluation (clean signal)
        auc_baseline = evaluate_signal(data, label, slidingWindow, model_names)

        # Anomaly indices for targeted attack
        anomaly_idx = np.where(label > 0)[0]
        target = anomaly_idx if len(anomaly_idx) > 0 else None

        results = []

        for epsilon in epsilons:
            # --- BIM adversarial attack ---
            T_adv = attack_bim(surrogate, T, ws, epsilon=epsilon,
                               alpha=BIM_ALPHA, iterations=BIM_ITERATIONS,
                               target_indices=target)
            X_adv = T_adv.numpy() * X_std + X_mean
            auc_adv = evaluate_signal(X_adv, label, slidingWindow, model_names)

            linf_adv = np.max(np.abs(X_adv - data))
            l2_adv = np.linalg.norm(X_adv - data) / len(data)

            for model_name in model_names:
                baseline_val = auc_baseline.get(model_name)
                adv_val = auc_adv.get(model_name)
                results.append({
                    'file': canonical_name,
                    'epsilon': epsilon,
                    'method': 'BIM',
                    'seed': 0,
                    'model': model_name,
                    'AUC_baseline': round(baseline_val, 4) if baseline_val else None,
                    'AUC_perturbed': round(adv_val, 4) if adv_val else None,
                    'AUC_drop': round(baseline_val - adv_val, 4) if (baseline_val and adv_val) else None,
                    'Linf_norm': round(float(linf_adv), 6),
                    'L2_norm': round(float(l2_adv), 6),
                    'error': None,
                })

            # --- Random noise (multiple seeds) ---
            for rand_seed in range(n_random_seeds):
                T_rand = apply_random_noise(T, epsilon, seed=rand_seed)
                X_rand = T_rand.numpy() * X_std + X_mean

                auc_rand = evaluate_signal(X_rand, label, slidingWindow, model_names)
                linf_rand = np.max(np.abs(X_rand - data))
                l2_rand = np.linalg.norm(X_rand - data) / len(data)

                for model_name in model_names:
                    baseline_val = auc_baseline.get(model_name)
                    rand_val = auc_rand.get(model_name)
                    results.append({
                        'file': canonical_name,
                        'epsilon': epsilon,
                        'method': 'Random',
                        'seed': rand_seed,
                        'model': model_name,
                        'AUC_baseline': round(baseline_val, 4) if baseline_val else None,
                        'AUC_perturbed': round(rand_val, 4) if rand_val else None,
                        'AUC_drop': round(baseline_val - rand_val, 4) if (baseline_val and rand_val) else None,
                        'Linf_norm': round(float(linf_rand), 6),
                        'L2_norm': round(float(l2_rand), 6),
                        'error': None,
                    })

            # --- Random-Targeted noise (only at anomaly regions) ---
            for rand_seed in range(n_random_seeds):
                T_rtarg = apply_random_noise_targeted(T, epsilon, target, seed=rand_seed)
                X_rtarg = T_rtarg.numpy() * X_std + X_mean

                auc_rtarg = evaluate_signal(X_rtarg, label, slidingWindow, model_names)
                linf_rtarg = np.max(np.abs(X_rtarg - data))
                l2_rtarg = np.linalg.norm(X_rtarg - data) / len(data)

                for model_name in model_names:
                    baseline_val = auc_baseline.get(model_name)
                    rtarg_val = auc_rtarg.get(model_name)
                    results.append({
                        'file': canonical_name,
                        'epsilon': epsilon,
                        'method': 'Random-Targeted',
                        'seed': rand_seed,
                        'model': model_name,
                        'AUC_baseline': round(baseline_val, 4) if baseline_val else None,
                        'AUC_perturbed': round(rtarg_val, 4) if rtarg_val else None,
                        'AUC_drop': round(baseline_val - rtarg_val, 4) if (baseline_val and rtarg_val) else None,
                        'Linf_norm': round(float(linf_rtarg), 6),
                        'L2_norm': round(float(l2_rtarg), 6),
                        'error': None,
                    })

        return {'status': 'success', 'results': results}

    except Exception as e:
        return {'status': 'error', 'file': file_name, 'error': traceback.format_exc()}


# ===========================================================
# Summary
# ===========================================================
def compute_summary(df_results, output_path):
    """Mean and std of AUC drop grouped by epsilon, method, model."""
    df_ok = df_results[df_results['error'].isnull()].copy()
    if df_ok.empty:
        print("No successful runs to summarize.")
        return

    grouped = df_ok.groupby(['epsilon', 'method', 'model'])
    rows = []
    for (eps, method, model), grp in grouped:
        rows.append({
            'epsilon': eps,
            'method': method,
            'model': model,
            'n_runs': len(grp),
            'mean_AUC_baseline': round(grp['AUC_baseline'].mean(), 4),
            'mean_AUC_perturbed': round(grp['AUC_perturbed'].mean(), 4),
            'mean_AUC_drop': round(grp['AUC_drop'].mean(), 4),
            'std_AUC_drop': round(grp['AUC_drop'].std(), 4),
            'median_AUC_drop': round(grp['AUC_drop'].median(), 4),
            'mean_L2_norm': round(grp['L2_norm'].mean(), 6),
        })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")

    # Print table
    print("\n" + "=" * 80)
    print("ADVERSARIAL vs RANDOM — SUMMARY")
    print("=" * 80)
    for model in sorted(df_summary['model'].unique()):
        print(f"\n{model}:")
        print(f"  {'Epsilon':>8s} | {'BIM Drop':>10s} | {'Random Drop':>12s} | {'Ratio':>8s}")
        print(f"  {'-'*8} | {'-'*10} | {'-'*12} | {'-'*8}")
        for eps in sorted(df_summary['epsilon'].unique()):
            bim_row = df_summary[(df_summary['epsilon'] == eps) &
                                  (df_summary['method'] == 'BIM') &
                                  (df_summary['model'] == model)]
            rand_row = df_summary[(df_summary['epsilon'] == eps) &
                                   (df_summary['method'] == 'Random') &
                                   (df_summary['model'] == model)]
            bim_drop = bim_row['mean_AUC_drop'].values[0] if len(bim_row) else 0
            rand_drop = rand_row['mean_AUC_drop'].values[0] if len(rand_row) else 0
            ratio = bim_drop / rand_drop if abs(rand_drop) > 1e-6 else float('inf')
            print(f"  {eps:>8.3f} | {bim_drop:>+10.4f} | {rand_drop:>+12.4f} | {ratio:>8.1f}x")


# ===========================================================
# Main
# ===========================================================
def main():
    parser = argparse.ArgumentParser(
        description='Adversarial (BIM) vs Random Noise at Equal Perturbation Budget')
    parser.add_argument('--test', action='store_true',
                        help='Smoke test: 3 datasets, 2 epsilons')
    parser.add_argument('--workers', type=int, default=2,
                        help='Parallel workers (default: 2, keep low — PyTorch overhead)')
    parser.add_argument('--epsilons', nargs='+', type=float, default=None,
                        help='Override epsilon values')
    parser.add_argument('--models', nargs='+', default=['IForest', 'PCA'],
                        choices=['IForest', 'PCA', 'MP'],
                        help='Models to evaluate (default: IForest PCA)')
    parser.add_argument('--random-seeds', type=int, default=None,
                        help='Number of random seeds (default: 3)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not SUBSET_CSV.exists():
        print(f"[ERROR] Subset CSV not found: {SUBSET_CSV}")
        return

    df_files = pd.read_csv(SUBSET_CSV)
    file_paths = df_files['filepath'].tolist()

    epsilons = args.epsilons or EPSILONS
    n_random_seeds = args.random_seeds or N_RANDOM_SEEDS

    if args.test:
        file_paths = file_paths[:3]
        epsilons = [0.05, 0.2]
        n_random_seeds = 1
        print("!!! RUNNING IN TEST MODE !!!")

    n_workers = args.workers

    print(f"\n{'='*60}")
    print(f"  Adversarial (BIM) vs Random Noise Comparison")
    print(f"{'='*60}")
    print(f"Datasets:       {len(file_paths)}")
    print(f"Epsilons:       {epsilons}")
    print(f"Random seeds:   {n_random_seeds}")
    print(f"Models:         {args.models}")
    print(f"Workers:        {n_workers}")
    print(f"BIM iterations: {BIM_ITERATIONS}")
    print(f"Surrogate AE:   window={SURROGATE_WINDOW}, epochs={SURROGATE_EPOCHS}")
    n_total = len(file_paths) * len(epsilons) * (1 + 2 * n_random_seeds) * len(args.models)
    print(f"Total results:  ~{n_total}")
    print(f"{'='*60}\n")

    # Checkpointing
    checkpoint_file = RESULTS_DIR / "checkpoint.csv"
    completed_files = set()
    all_results = []

    if checkpoint_file.exists():
        try:
            df_ckpt = pd.read_csv(checkpoint_file)
            df_ok = df_ckpt[df_ckpt['error'].isna()] if 'error' in df_ckpt.columns else df_ckpt
            all_results = df_ok.to_dict('records')
            completed_files = set(df_ok['file'].unique())
            print(f"Loaded checkpoint: {len(completed_files)} datasets already done.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint: {e}")

    # Build job list (one job per dataset)
    jobs = []
    for fp in file_paths:
        fname = os.path.basename(fp)
        if fname not in completed_files:
            jobs.append((fp, epsilons, n_random_seeds, args.models))

    print(f"Jobs to run: {len(jobs)} (skipped {len(completed_files)} from checkpoint)")

    if len(jobs) == 0:
        print("All jobs already completed!")
    else:
        new_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_dataset, j): j for j in jobs}

            with tqdm(total=len(jobs), desc="Adversarial vs Random") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        all_results.extend(res['results'])
                        new_count += len(res['results'])
                    else:
                        err_msg = res.get('error', '')
                        if len(err_msg) > 200:
                            err_msg = err_msg[:200] + '...'
                        print(f"\n  Error in {res.get('file', '?')}: {err_msg}")

                    pbar.update(1)

                    # Checkpoint every 10 datasets
                    if new_count >= 100:
                        df_temp = pd.DataFrame(all_results)
                        df_temp.to_csv(checkpoint_file, index=False)
                        new_count = 0

        # Final save
        df_final = pd.DataFrame(all_results)
        df_final.to_csv(checkpoint_file, index=False)
        print(f"\nResults saved to {checkpoint_file}")

    # Summary
    if all_results:
        df_all = pd.DataFrame(all_results)
        summary_path = RESULTS_DIR / "summary.csv"
        compute_summary(df_all, summary_path)


if __name__ == "__main__":
    main()
