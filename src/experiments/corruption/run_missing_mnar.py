"""
Missing Values — MCAR vs MNAR Comparison

Compares two missing data mechanisms at the SAME actual missing rate:

  1. MCAR: each point has equal probability of being missing (random)
  2. MNAR-extreme: points with extreme values (high |z-score|) are more
     likely to be missing — simulates sensor overload/saturation
  3. MNAR-high: only high values go missing — sensor caps at upper limit

Real-world motivation:
  - Sensors often fail when readings are extreme (overload, saturation)
  - This means anomalies (which are often extreme) are disproportionately lost
  - MNAR is the most dangerous missing mechanism for anomaly detection

Key question: "Given the same % of missing data, does the mechanism matter?"

Usage:
    python run_missing_mnar.py --models IForest
    python run_missing_mnar.py --test
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

# ==========================================
# PATH CONFIGURATION
# ==========================================
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

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "missing_mnar")

FRACTIONS = [0.01, 0.05, 0.10, 0.20]
MECHANISMS = ['mcar', 'mnar_extreme', 'mnar_high']
N_SEEDS = 1

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]


def select_missing_indices(values, labels, fraction, mechanism, rng):
    """Select which indices go missing based on mechanism.

    All mechanisms produce exactly the same number of missing points.
    """
    n = len(values)
    n_missing = max(1, int(fraction * n))

    if mechanism == 'mcar':
        # Equal probability for all points
        indices = rng.choice(n, size=n_missing, replace=False)

    elif mechanism == 'mnar_extreme':
        # Probability proportional to |z-score|
        # Extreme values (both high and low) more likely to be missing
        z = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-10)
        weights = np.abs(z)
        weights = weights / weights.sum()
        indices = rng.choice(n, size=n_missing, replace=False, p=weights)

    elif mechanism == 'mnar_high':
        # Probability proportional to z-score (only positive direction)
        # High values more likely to be missing (sensor caps out)
        z = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-10)
        weights = np.maximum(z, 0) + 0.01  # small floor so all points have some chance
        weights = weights / weights.sum()
        indices = rng.choice(n, size=n_missing, replace=False, p=weights)

    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    return indices


def run_model(model_name, X, scaled_data, sw, file_name=None):
    """Return anomaly scores. AE returns full-length scores (len == len(scaled_data));
    the other models return sliding-window scores that the caller pads."""
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
    (file_path, fraction, mechanism, seed, model_names) = job_args

    file_name = os.path.basename(file_path)
    condition_name = f"frac_{fraction}_{mechanism}"

    try:
        # 1. Load Data
        df, file_name = load_tsb_dataframe(file_path)
        values = df['value'].to_numpy('float')
        labels = df['is_anomaly'].to_numpy('int')
        n = len(df)

        # 2. Sliding window from clean data
        clean_scaled = StandardScaler().fit_transform(values.reshape(-1, 1)).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # 3. Select missing indices based on mechanism
        rng = np.random.default_rng(seed)
        missing_idx = select_missing_indices(values, labels, fraction, mechanism, rng)

        # Create NaN mask
        nan_mask = np.zeros(n, dtype=bool)
        nan_mask[missing_idx] = True

        # Track what was lost
        masked_normal = nan_mask & (labels == 0)
        masked_anomaly = nan_mask & (labels == 1)
        n_lost_anomalies = int(masked_anomaly.sum())
        n_lost_normal = int(masked_normal.sum())
        n_missing_total = int(nan_mask.sum())
        pct_anomalies_lost = n_lost_anomalies / max(labels.sum(), 1)

        # 4. Masking: run model on non-NaN data only
        model_data = values[~nan_mask]
        n_kept = len(model_data)

        if n_kept < sliding_window + 10:
            return {'status': 'skipped', 'file': file_name,
                    'reason': f'Too few points after masking: {n_kept}'}

        sw = min(sliding_window, n_kept // 4)
        sw = max(sw, 10)

        scaled_data = StandardScaler().fit_transform(model_data.reshape(-1, 1)).flatten()
        X = Window(window=sw).convert(scaled_data).to_numpy()

        # 5. Modeling & Evaluation
        results = []
        for model_name in model_names:
            try:
                score = run_model(model_name, X, scaled_data, sw, file_name=file_name)
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

                if len(score) == len(scaled_data):
                    # AE already returns full-length scores
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
                        'mechanism': mechanism,
                        'n_missing': n_missing_total,
                        'n_lost_anomalies': n_lost_anomalies,
                        'n_lost_normal': n_lost_normal,
                        'pct_anomalies_lost': round(pct_anomalies_lost, 4),
                        'seed': seed, 'model': model_name,
                        'condition': condition_name, 'error': "Invalid scores"
                    })
                    continue

                # Reconstruct full scores: model scores at kept positions, 0 at lost anomalies
                full_score = np.full(n, np.nan)
                full_score[~nan_mask] = padded_score
                full_score[masked_anomaly] = 0.0  # lost anomalies = missed

                # Evaluate: exclude masked normal points
                eval_mask = ~masked_normal
                eval_scores = full_score[eval_mask]
                eval_labels = labels[eval_mask]

                metrics = get_metrics(eval_scores, eval_labels, metric="all", slidingWindow=sw)

                row = {
                    'file': file_name, 'fraction': fraction,
                    'mechanism': mechanism,
                    'n_missing': n_missing_total,
                    'n_lost_anomalies': n_lost_anomalies,
                    'n_lost_normal': n_lost_normal,
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
                    'mechanism': mechanism,
                    'n_missing': n_missing_total,
                    'n_lost_anomalies': n_lost_anomalies,
                    'n_lost_normal': n_lost_normal,
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
        print("No successful runs to summarize.")
        return

    grouped = df_success.groupby(['fraction', 'mechanism', 'model'])

    summary_data = []
    for name, group in grouped:
        frac, mech, model = name
        row = {
            'fraction': frac, 'mechanism': mech,
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
    parser.add_argument('--models', nargs='+', default=['MP'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(SUBSET_CSV):
        print(f"[ERROR] Subset CSV not found: {SUBSET_CSV}")
        return

    df_files_metadata = pd.read_csv(SUBSET_CSV)
    df_files = df_files_metadata['filepath'].tolist()

    fractions = FRACTIONS
    mechanisms = MECHANISMS

    if args.test:
        df_files = df_files[:3]
        fractions = [0.05, 0.20]
        mechanisms = ['mcar', 'mnar_extreme']
        print("!!! RUNNING IN TEST MODE !!!")

    total_conditions = len(fractions) * len(mechanisms)
    n_workers = args.workers or os.cpu_count() or 1

    print(f"\n{'='*60}")
    print(f"  Missing Values — MCAR vs MNAR Comparison")
    print(f"{'='*60}")
    print(f"  Mechanisms: {mechanisms}")
    print(f"  Fractions:  {fractions}")
    print(f"  Total conditions: {total_conditions}")
    print(f"  Same actual missing rate for fair comparison")
    print(f"{'='*60}\n")

    # Checkpoint
    checkpoint_file = os.path.join(RESULTS_DIR, "checkpoint.csv")
    completed_jobs = set()
    all_results = []

    if os.path.exists(checkpoint_file):
        try:
            df_checkpoint = pd.read_csv(checkpoint_file)
            if 'error' in df_checkpoint.columns:
                df_success = df_checkpoint[df_checkpoint['error'].isna()]
            else:
                df_success = df_checkpoint
            all_results = df_success.to_dict('records')
            for r in all_results:
                job_id = f"{r['file']}_{r['condition']}_{r['seed']}_{r['model']}"
                completed_jobs.add(job_id)
            print(f"Loaded {len(all_results)} successful jobs from checkpoint.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint: {e}")

    # Build jobs
    jobs = []
    for file_path in df_files:
        file_name = os.path.basename(file_path)
        for frac in fractions:
            for mech in mechanisms:
                for seed in range(N_SEEDS):
                    condition = f"frac_{frac}_{mech}"
                    needed_models = []
                    for m in args.models:
                        job_id = f"{file_name}_{condition}_{seed}_{m}"
                        if job_id not in completed_jobs:
                            needed_models.append(m)
                    if needed_models:
                        jobs.append((file_path, frac, mech, seed, needed_models))

    print(f"Jobs Scheduled: {len(jobs)}")

    if len(jobs) > 0:
        new_results_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_job, job): job for job in jobs}

            with tqdm(total=len(jobs), desc="missing (MCAR vs MNAR)") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        all_results.extend(res['results'])
                        new_results_count += len(res['results'])

                    pbar.update(1)

                    if new_results_count >= 500:
                        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
                        new_results_count = 0

        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)

    if all_results:
        df_all = pd.DataFrame(all_results)
        compute_summary(df_all, os.path.join(RESULTS_DIR, "summary.csv"))

    print(f"\n[Done] MCAR vs MNAR experiments complete.")


if __name__ == "__main__":
    main()
