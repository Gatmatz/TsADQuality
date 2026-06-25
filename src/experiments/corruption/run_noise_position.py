"""
Noise Position vs Anomaly Detection Experiment

Tests how the POSITION of noise relative to anomalies affects detection.

Position modes:
  - only_normal         — noise only on non-anomaly points
  - overlapping_anomaly — noise directly on anomaly points
  - before_anomaly      — noise on normal points BEFORE anomalies
  - after_anomaly       — noise on normal points AFTER anomalies
  - far_from_anomaly    — noise far from any anomaly

Optimization: jobs grouped by (file, model) so data loads once and AE loads once.

Usage:
    python run_noise_position.py                              # full run
    python run_noise_position.py --test                       # 3 files
    python run_noise_position.py --models IForest LOF         # select models
    python run_noise_position.py --workers 4                  # parallel
"""
import os
import sys
import math
import argparse

os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import traceback
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.matrix_profile import MatrixProfile
from TSB_UAD.models.lof import LOF
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors
from data_loader import load_tsb_dataframe, remap_filepath, load_pretrained_ae

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "noise_position")

POSITION_MODES = [
    'only_normal',
    'overlapping_anomaly',
    'before_anomaly',
    'after_anomaly',
    'far_from_anomaly',
]
SNRS_DB = [30, 20, 10, 5, 0]
NEAR_WINDOW_MULTIPLIERS = [0.5, 1.0, 2.0]
WINDOWED_MODES = ('before_anomaly', 'after_anomaly', 'far_from_anomaly')
N_SEEDS = 1

ALL_MODELS = ['IForest', 'LOF', 'MP', 'AE']

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall',
]


# ==========================================
# BUILD CONDITIONS LIST
# ==========================================
def build_conditions(positions, snrs, n_seeds):
    """Build list of all (position, snr, window_mult, seed) conditions."""
    conditions = []
    for pos in positions:
        mults = NEAR_WINDOW_MULTIPLIERS if pos in WINDOWED_MODES else [1.0]
        for wm in mults:
            for snr in snrs:
                for seed in range(n_seeds):
                    if pos in WINDOWED_MODES:
                        cond_name = f"pos_{pos}_wm{wm}_snr_{snr}dB"
                    else:
                        cond_name = f"pos_{pos}_snr_{snr}dB"
                    conditions.append({
                        'position': pos, 'snr_db': snr,
                        'window_mult': wm, 'seed': seed,
                        'condition': cond_name,
                    })
    return conditions


# ==========================================
# MODEL SCORING
# ==========================================
def run_model_score(model_name, scaled_data, sliding_window, clf_ae=None):
    """Run model and return full-length MinMax-scaled scores."""
    if model_name == 'AE':
        clf_ae.predict(scaled_data)
        score = clf_ae.decision_scores_
        full_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            score.reshape(-1, 1)).ravel()
        n = len(scaled_data)
        if len(full_score) > n:
            full_score = full_score[:n]
        elif len(full_score) < n:
            full_score = np.pad(full_score, (0, n - len(full_score)), mode='edge')
        return full_score

    if model_name == 'MP':
        clf = MatrixProfile(window=sliding_window)
        clf.fit(scaled_data)
        score = clf.decision_scores_
    else:
        X = Window(window=sliding_window).convert(scaled_data).to_numpy()
        if model_name == 'IForest':
            clf = IForest(n_estimators=100, random_state=42)
        elif model_name == 'LOF':
            clf = LOF(n_neighbors=20)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        clf.fit(X)
        score = clf.decision_scores_

    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        score.reshape(-1, 1)).ravel()
    full_score = np.array(
        [score[0]] * math.ceil((sliding_window - 1) / 2)
        + list(score)
        + [score[-1]] * ((sliding_window - 1) // 2)
    )
    n = len(scaled_data)
    if len(full_score) > n:
        full_score = full_score[:n]
    elif len(full_score) < n:
        full_score = np.pad(full_score, (0, n - len(full_score)), mode='edge')
    return full_score


# ==========================================
# CORE WORKER — one (file, model) pair
# ==========================================
def process_file_model(job_args):
    """Process all conditions for one (file, model) pair.

    Loads data once, loads AE once, iterates over all conditions.
    """
    file_path, model_name, conditions, skip_keys = job_args
    file_name = os.path.basename(file_path)

    try:
        # 1. Load data once
        df, canonical_name = load_tsb_dataframe(file_path)
        data = df['value'].to_numpy('float')
        labels = df['is_anomaly'].to_numpy('int')
        n = len(data)

        clean_scaled = StandardScaler().fit_transform(data.reshape(-1, 1)).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # 2. Load AE once if needed
        clf_ae = None
        if model_name == 'AE':
            clf_ae, _ = load_pretrained_ae(canonical_name, project_root)

        # 3. Iterate conditions
        all_rows = []
        for cond in conditions:
            cond_key = f"{canonical_name}_{cond['snr_db']}_{cond['position']}_{cond['window_mult']}_{cond['seed']}_{model_name}"
            if cond_key in skip_keys:
                continue

            try:
                near_window = max(1, int(sliding_window * cond['window_mult']))

                # Inject noise at specified position
                corruptor = TSCorruptor(
                    df, value_col='value', label_col='is_anomaly',
                    seed=cond['seed'], corruption_target=cond['position'],
                    window_size=near_window,
                )
                ts_corruptor.injectors.inject_white_noise_snr(corruptor, snr_db=cond['snr_db'])

                df_corrupted = corruptor.get_corrupted_df()
                report = corruptor.get_corruption_report()
                corrupted_data = df_corrupted['value'].to_numpy('float')

                if np.isnan(corrupted_data).all():
                    continue

                avg_dist = report['summary'].get('avg_distance_to_anomaly', np.nan)

                # Preprocessing
                scaled_data = StandardScaler().fit_transform(
                    corrupted_data.reshape(-1, 1)).flatten()

                # Run model
                full_score = run_model_score(model_name, scaled_data, sliding_window, clf_ae)

                if np.isnan(full_score).any() or len(np.unique(full_score)) <= 1:
                    all_rows.append({
                        'file': canonical_name, 'snr_db': cond['snr_db'],
                        'position_mode': cond['position'],
                        'window_mult': cond['window_mult'],
                        'near_window': near_window,
                        'seed': cond['seed'], 'model': model_name,
                        'condition': cond['condition'],
                        'corrupted_points': report['summary']['corrupted_points'],
                        'corruption_pct': report['summary']['corruption_percentage'],
                        'avg_dist_to_anomaly': avg_dist,
                        'error': 'Invalid scores',
                    })
                    continue

                metrics = get_metrics(full_score, labels, metric="all",
                                      slidingWindow=sliding_window)

                row = {
                    'file': canonical_name, 'snr_db': cond['snr_db'],
                    'position_mode': cond['position'],
                    'window_mult': cond['window_mult'],
                    'near_window': near_window,
                    'seed': cond['seed'], 'model': model_name,
                    'condition': cond['condition'],
                    'corrupted_points': report['summary']['corrupted_points'],
                    'corruption_pct': round(report['summary']['corruption_percentage'], 2),
                    'avg_dist_to_anomaly': round(avg_dist, 2) if not np.isnan(avg_dist) else np.nan,
                    'error': None,
                }
                for key in SELECTED_METRICS:
                    row[key] = round(metrics.get(key, 0.0), 4)
                all_rows.append(row)

            except Exception as e:
                all_rows.append({
                    'file': canonical_name, 'snr_db': cond['snr_db'],
                    'position_mode': cond['position'],
                    'window_mult': cond['window_mult'],
                    'near_window': 0,
                    'seed': cond['seed'], 'model': model_name,
                    'condition': cond['condition'],
                    'corrupted_points': 0, 'corruption_pct': 0,
                    'avg_dist_to_anomaly': np.nan,
                    'error': str(e)[:300],
                })

        return {'status': 'success', 'rows': all_rows,
                'job_id': f"{canonical_name}_{model_name}"}

    except Exception as e:
        return {'status': 'error', 'job_id': f"{file_name}_{model_name}",
                'error': traceback.format_exc()[:500]}


# ==========================================
# SUMMARY
# ==========================================
def compute_summary(df_results, output_path):
    df_success = df_results[df_results['error'].isnull()]
    if df_success.empty:
        print("No successful runs to summarize.")
        return

    group_keys = ['snr_db', 'position_mode', 'condition', 'model']
    metric_cols = [c for c in SELECTED_METRICS if c in df_success.columns]
    extra_cols = [c for c in ['corruption_pct', 'avg_dist_to_anomaly'] if c in df_success.columns]

    agg_dict = {c: ['mean', 'std'] for c in metric_cols}
    for c in extra_cols:
        agg_dict[c] = 'mean'
    agg_dict['file'] = 'count'

    grouped = df_success.groupby(group_keys).agg(agg_dict).round(4)
    grouped.columns = [
        f'mean_{c}' if stat == 'mean' and c in metric_cols
        else f'std_{c}' if stat == 'std'
        else f'mean_{c}' if c in extra_cols
        else 'n_runs'
        for c, stat in grouped.columns
    ]
    grouped = grouped.reset_index()

    grouped.to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Noise Position vs Anomaly Detection Experiment')
    parser.add_argument('--test', action='store_true', help='Quick test: 3 files, 2 SNRs')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--positions', nargs='+', default=None)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_files_metadata = pd.read_csv(SUBSET_CSV)
    df_files = df_files_metadata['filepath'].tolist()
    df_files = [remap_filepath(fp, project_root) for fp in df_files]

    snrs = SNRS_DB
    positions = args.positions if args.positions else POSITION_MODES
    n_seeds = N_SEEDS

    if args.test:
        indices = np.linspace(0, len(df_files) - 1, 3, dtype=int)
        df_files = [df_files[i] for i in indices]
        snrs = [20, 5]
        positions = ['overlapping_anomaly', 'far_from_anomaly']
        n_seeds = 1
        print("!!! TEST MODE !!!")

    conditions = build_conditions(positions, snrs, n_seeds)
    n_conditions = len(conditions)

    # Checkpoint
    checkpoint_file = os.path.join(RESULTS_DIR, "checkpoint.csv")
    completed_keys = set()
    all_results = []

    if os.path.exists(checkpoint_file):
        try:
            df_checkpoint = pd.read_csv(checkpoint_file)
            df_ok = df_checkpoint[df_checkpoint['error'].isna()] if 'error' in df_checkpoint.columns else df_checkpoint
            all_results = df_ok.to_dict('records')
            for r in all_results:
                wm = r.get('window_mult', 1.0)
                key = f"{r['file']}_{r['snr_db']}_{r['position_mode']}_{wm}_{r['seed']}_{r['model']}"
                completed_keys.add(key)
            print(f"[Resume] {len(completed_keys)} results from checkpoint")
        except Exception as e:
            print(f"Warning: checkpoint error: {e}")

    # Build jobs: one per (file, model)
    jobs = []
    for file_path in df_files:
        file_name = os.path.basename(file_path)
        for model in args.models:
            # Check if any condition still needed
            has_remaining = False
            for cond in conditions:
                key = f"{file_name}_{cond['snr_db']}_{cond['position']}_{cond['window_mult']}_{cond['seed']}_{model}"
                if key not in completed_keys:
                    has_remaining = True
                    break
            if has_remaining:
                jobs.append((file_path, model, conditions, completed_keys))

    n_total_jobs = len(df_files) * len(args.models)
    total_evals = len(jobs) * n_conditions

    print(f"\n{'=' * 60}")
    print(f"  Noise Position vs Anomaly Detection Experiment")
    print(f"{'=' * 60}")
    print(f"  Datasets:         {len(df_files)}")
    print(f"  Positions:        {positions}")
    print(f"  SNRs (dB):        {snrs}")
    print(f"  Models:           {args.models}")
    print(f"  Conditions/job:   {n_conditions}")
    print(f"  Total jobs:       {n_total_jobs} (file x model)")
    print(f"  Already done:     {n_total_jobs - len(jobs)}")
    print(f"  Remaining jobs:   {len(jobs)}")
    print(f"  Total evaluations:{total_evals}")
    print(f"  Workers:          {args.workers}")
    print(f"{'=' * 60}\n")

    if not jobs:
        print("All jobs done!")
    else:
        new_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_file_model, job): job for job in jobs}

            with tqdm(total=len(jobs), desc="Position experiment") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        all_results.extend(res['rows'])
                        new_count += 1
                    elif res['status'] == 'error':
                        print(f"  [ERROR] {res['job_id']}: {res.get('error', '')[:100]}")
                    pbar.update(1)

                    if new_count % 50 == 0 and new_count > 0:
                        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)

        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
        print(f"\nCheckpoint saved: {len(all_results)} results")

    # Summary
    if all_results:
        df_all = pd.DataFrame(all_results)

        compute_summary(df_all, os.path.join(RESULTS_DIR, "summary_all.csv"))

        for pos_mode in positions:
            pos_df = df_all[df_all['position_mode'] == pos_mode]
            if pos_df.empty:
                continue
            pos_dir = os.path.join(RESULTS_DIR, pos_mode)
            os.makedirs(pos_dir, exist_ok=True)
            compute_summary(pos_df, os.path.join(pos_dir, "summary.csv"))

            for model_name in args.models:
                model_df = pos_df[pos_df['model'] == model_name]
                if not model_df.empty:
                    model_dir = os.path.join(pos_dir, model_name)
                    os.makedirs(model_dir, exist_ok=True)
                    model_df.to_csv(os.path.join(model_dir, "raw_results.csv"), index=False)

    print(f"\n[Done]")


if __name__ == "__main__":
    main()
