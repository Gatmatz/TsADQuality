"""
Missing Values — MCAR vs MNAR (Burst version)

Same comparison as run_missing_mnar.py but with burst missing instead of point.
Bursts are contiguous blocks of NaN.

Mechanisms:
  1. mcar_burst:         burst positions chosen randomly
  2. mnar_extreme_burst: bursts placed where |z-score| is highest
  3. mnar_high_burst:    bursts placed where values are highest

burst_length = fraction * n / num_bursts (dynamic, same as freeze/missing_true_impact)

Usage:
    python run_missing_mnar_burst.py --models IForest
    python run_missing_mnar_burst.py --test
"""
import os
import sys
import math
import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import traceback

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
tsb_uad_path = os.path.join(project_root, 'TSB-UAD')
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path: sys.path.insert(0, project_root)
if tsb_uad_path not in sys.path: sys.path.insert(0, tsb_uad_path)
if src_path not in sys.path: sys.path.insert(0, src_path)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    from TSB_UAD.models.iforest import IForest
    from TSB_UAD.models.pca import PCA
    from TSB_UAD.models.matrix_profile import MatrixProfile
    from TSB_UAD.models.lof import LOF
    from TSB_UAD.models.feature import Window
    from TSB_UAD.utils.slidingWindows import find_length
    from TSB_UAD.vus.metrics import get_metrics
except ImportError as e:
    warnings.warn(f"TSB_UAD components could not be imported: {e}")

from data_loader import load_tsb_dataframe, load_pretrained_ae

SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "missing_mnar_burst")

FRACTIONS = [0.01, 0.05, 0.10, 0.20]
NUM_BURSTS = [1, 3, 5, 10]
MECHANISMS = ['mnar_extreme_burst', 'mnar_high_burst']
N_SEEDS = 1

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]


def select_burst_starts(values, n, burst_length, num_bursts, mechanism, rng):
    """Select burst starting positions based on mechanism.

    Returns list of start indices for each burst. Bursts don't overlap.
    """
    if burst_length >= n:
        return [0]

    # Compute scores for each possible start position
    max_start = n - burst_length
    if max_start <= 0:
        return [0]

    if mechanism == 'mcar_burst':
        # Random positions
        candidates = np.arange(max_start)
        rng.shuffle(candidates)
        starts = []
        used = set()
        for c in candidates:
            if len(starts) >= num_bursts:
                break
            # Check no overlap with existing bursts
            burst_range = set(range(c, c + burst_length))
            if not burst_range & used:
                starts.append(c)
                used |= burst_range
        return starts

    else:
        # MNAR: score each position by the mean |z-score| (or z-score) in the window
        z = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-10)

        position_scores = np.zeros(max_start)
        # Use cumsum for efficiency
        if mechanism == 'mnar_extreme_burst':
            abs_z = np.abs(z)
            cumsum = np.concatenate([[0], np.cumsum(abs_z)])
            position_scores = (cumsum[burst_length:max_start + burst_length] - cumsum[:max_start]) / burst_length
        elif mechanism == 'mnar_high_burst':
            high_z = np.maximum(z, 0)
            cumsum = np.concatenate([[0], np.cumsum(high_z)])
            position_scores = (cumsum[burst_length:max_start + burst_length] - cumsum[:max_start]) / burst_length

        # Select top positions greedily (no overlap)
        sorted_idx = np.argsort(-position_scores)  # highest first
        starts = []
        used = set()
        for idx in sorted_idx:
            if len(starts) >= num_bursts:
                break
            burst_range = set(range(idx, idx + burst_length))
            if not burst_range & used:
                starts.append(idx)
                used |= burst_range

        return starts


def run_model(model_name, X, scaled_data, sw, file_name=None):
    if model_name == 'IForest':
        clf = IForest(n_estimators=100, random_state=42); clf.fit(X); return clf.decision_scores_
    elif model_name == 'PCA':
        n_comp = min(10, X.shape[1] - 1) if X.shape[1] > 1 else 1
        clf = PCA(n_components=n_comp); clf.fit(X); return clf.decision_scores_
    elif model_name == 'MP':
        clf = MatrixProfile(window=sw); clf.fit(scaled_data); return clf.decision_scores_
    elif model_name == 'LOF':
        n_neigh = min(20, len(X) - 1)
        clf = LOF(n_neighbors=n_neigh); clf.fit(X); return clf.decision_scores_
    elif model_name == 'AE':
        clf, _ = load_pretrained_ae(file_name, project_root)
        clf.predict(scaled_data)
        return clf.decision_scores_
    else:
        raise ValueError(f"Unknown model: {model_name}")


def process_single_job(job_args):
    (file_path, fraction, num_bursts, mechanism, seed, model_names) = job_args

    file_name = os.path.basename(file_path)
    condition_name = f"frac_{fraction}_nb_{num_bursts}_{mechanism}"

    try:
        df, file_name = load_tsb_dataframe(file_path)
        values = df['value'].to_numpy('float')
        labels = df['is_anomaly'].to_numpy('int')
        n = len(df)

        clean_scaled = StandardScaler().fit_transform(values.reshape(-1, 1)).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # Dynamic burst length
        burst_length = max(1, int(fraction * n / num_bursts))

        # Select burst positions
        rng = np.random.default_rng(seed)
        starts = select_burst_starts(values, n, burst_length, num_bursts, mechanism, rng)

        # Create NaN mask
        nan_mask = np.zeros(n, dtype=bool)
        for s in starts:
            nan_mask[s:s + burst_length] = True

        actual_missing = nan_mask.sum()
        masked_normal = nan_mask & (labels == 0)
        masked_anomaly = nan_mask & (labels == 1)
        n_lost_anomalies = int(masked_anomaly.sum())
        pct_anomalies_lost = n_lost_anomalies / max(labels.sum(), 1)

        # Masking: run on kept data
        model_data = values[~nan_mask]
        n_kept = len(model_data)

        if n_kept < sliding_window + 10:
            return {'status': 'skipped', 'file': file_name,
                    'reason': f'Too few points: {n_kept}'}

        sw = min(sliding_window, n_kept // 4)
        sw = max(sw, 10)

        scaled_data = StandardScaler().fit_transform(model_data.reshape(-1, 1)).flatten()
        X = Window(window=sw).convert(scaled_data).to_numpy()

        results = []
        for model_name in model_names:
            try:
                score = run_model(model_name, X, scaled_data, sw, file_name=file_name)
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

                # AE already produces full-length scores, others need padding
                if model_name == 'AE':
                    padded_score = score
                else:
                    padded_score = np.array(
                        [score[0]] * math.ceil((sw - 1) / 2) +
                        list(score) +
                        [score[-1]] * ((sw - 1) // 2)
                    )

                if np.isnan(padded_score).any() or len(np.unique(padded_score)) <= 1:
                    results.append({
                        'file': file_name, 'fraction': fraction,
                        'num_bursts': num_bursts, 'mechanism': mechanism,
                        'burst_length': burst_length,
                        'actual_missing': actual_missing,
                        'n_lost_anomalies': n_lost_anomalies,
                        'pct_anomalies_lost': round(pct_anomalies_lost, 4),
                        'seed': seed, 'model': model_name,
                        'condition': condition_name, 'error': "Invalid scores"
                    })
                    continue

                full_score = np.full(n, np.nan)
                full_score[~nan_mask] = padded_score
                full_score[masked_anomaly] = 0.0

                eval_mask = ~masked_normal
                eval_scores = full_score[eval_mask]
                eval_labels = labels[eval_mask]

                metrics = get_metrics(eval_scores, eval_labels, metric="all", slidingWindow=sw)

                row = {
                    'file': file_name, 'fraction': fraction,
                    'num_bursts': num_bursts, 'mechanism': mechanism,
                    'burst_length': burst_length,
                    'actual_missing': actual_missing,
                    'n_lost_anomalies': n_lost_anomalies,
                    'pct_anomalies_lost': round(pct_anomalies_lost, 4),
                    'seed': seed, 'model': model_name,
                    'condition': condition_name, 'error': None
                }
                for key in SELECTED_METRICS:
                    row[key] = round(metrics.get(key, 0.0), 4)
                results.append(row)

            except Exception as e:
                results.append({
                    'file': file_name, 'fraction': fraction,
                    'num_bursts': num_bursts, 'mechanism': mechanism,
                    'burst_length': burst_length,
                    'actual_missing': actual_missing,
                    'n_lost_anomalies': n_lost_anomalies,
                    'pct_anomalies_lost': round(pct_anomalies_lost, 4),
                    'seed': seed, 'model': model_name,
                    'condition': condition_name, 'error': str(e)
                })

        return {'status': 'success', 'results': results}

    except Exception as e:
        return {'status': 'error', 'file': file_name, 'error': traceback.format_exc()}


def compute_summary(df_results, output_path):
    df_success = df_results[df_results['error'].isnull()]
    if df_success.empty:
        return
    summary_data = []
    for name, group in df_success.groupby(['fraction', 'num_bursts', 'mechanism', 'model']):
        frac, nb, mech, model = name
        row = {
            'fraction': frac, 'num_bursts': nb, 'mechanism': mech,
            'mean_burst_length': round(group['burst_length'].mean(), 1),
            'model': model, 'n_runs': len(group),
            'mean_n_lost_anomalies': round(group['n_lost_anomalies'].mean(), 1),
            'mean_pct_anomalies_lost': round(group['pct_anomalies_lost'].mean(), 4),
        }
        for col in SELECTED_METRICS:
            if col in group.columns:
                row[f'mean_{col}'] = round(group[col].mean(), 4)
                row[f'std_{col}'] = round(group[col].std(), 4)
        summary_data.append(row)
    pd.DataFrame(summary_data).to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--models', nargs='+', default=['AE'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(SUBSET_CSV):
        print(f"[ERROR] Subset CSV not found: {SUBSET_CSV}")
        return

    df_files = pd.read_csv(SUBSET_CSV)['filepath'].tolist()
    fractions = FRACTIONS
    num_bursts_list = NUM_BURSTS
    mechanisms = MECHANISMS

    if args.test:
        df_files = df_files[:3]
        fractions = [0.05, 0.20]
        num_bursts_list = [1, 5]
        mechanisms = ['mnar_extreme_burst']
        print("!!! TEST MODE !!!")

    total = len(fractions) * len(num_bursts_list) * len(mechanisms)
    print(f"\n{'='*60}")
    print(f"  Missing Values — MCAR vs MNAR (Burst)")
    print(f"{'='*60}")
    print(f"  Fractions:  {fractions}")
    print(f"  Num bursts: {num_bursts_list}")
    print(f"  Mechanisms: {mechanisms}")
    print(f"  Conditions: {total}")
    print(f"{'='*60}\n")

    checkpoint_file = os.path.join(RESULTS_DIR, "checkpoint.csv")
    completed_jobs = set()
    all_results = []

    if os.path.exists(checkpoint_file):
        try:
            df_checkpoint = pd.read_csv(checkpoint_file)
            df_success = df_checkpoint[df_checkpoint['error'].isna()] if 'error' in df_checkpoint.columns else df_checkpoint
            all_results = df_success.to_dict('records')
            for r in all_results:
                completed_jobs.add(f"{r['file']}_{r['condition']}_{r['seed']}_{r['model']}")
            print(f"Loaded {len(all_results)} from checkpoint.")
        except Exception as e:
            print(f"Warning: {e}")

    jobs = []
    for file_path in df_files:
        file_name = os.path.basename(file_path)
        for frac in fractions:
            for nb in num_bursts_list:
                for mech in mechanisms:
                    for seed in range(N_SEEDS):
                        condition = f"frac_{frac}_nb_{nb}_{mech}"
                        needed = [m for m in args.models if f"{file_name}_{condition}_{seed}_{m}" not in completed_jobs]
                        if needed:
                            jobs.append((file_path, frac, nb, mech, seed, needed))

    print(f"Jobs: {len(jobs)}")

    if jobs:
        new_count = 0
        with ProcessPoolExecutor(max_workers=args.workers or 4) as executor:
            futures = {executor.submit(process_single_job, j): j for j in jobs}
            with tqdm(total=len(jobs), desc="missing MNAR burst") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        all_results.extend(res['results'])
                        new_count += len(res['results'])
                    pbar.update(1)
                    if new_count >= 500:
                        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
                        new_count = 0
        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)

    if all_results:
        compute_summary(pd.DataFrame(all_results), os.path.join(RESULTS_DIR, "summary.csv"))

    print(f"\n[Done] MNAR burst experiments complete.")


if __name__ == "__main__":
    main()
