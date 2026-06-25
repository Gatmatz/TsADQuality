"""
Foundation Model Robustness Experiment — MOMENT

Evaluates MOMENT (reconstruction-based anomaly detection) on clean and corrupted
data using the same corruption types as classical model experiments.

MOMENT uses MSE between input and reconstruction as anomaly score (zero-shot).
This is a direct anomaly detection approach, unlike Chronos which uses forecasting.

Usage:
    python run_moment_robustness.py                  # full run
    python run_moment_robustness.py --test            # quick test (3 files)
    python run_moment_robustness.py --batch-size 32   # adjust for GPU memory
"""
import os
import sys
import argparse
import math
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

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

try:
    from momentfm import MOMENTPipeline
except ImportError:
    print("[ERROR] Install momentfm: pip install momentfm")
    sys.exit(1)

from TSB_UAD.vus.metrics import get_metrics
from TSB_UAD.utils.slidingWindows import find_length

from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors
from data_loader import load_tsb_dataframe

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "foundation_models", "moment")

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]

CORRUPTION_GRID = [
    ('clean', {}),

    # White noise SNR
    ('noise', {'snr_db': 20.0}),
    ('noise', {'snr_db': 5.0}),
    ('noise', {'snr_db': -5.0}),

    # Spikes
    ('spikes', {'fraction': 0.05, 'multiplier': 5.0, 'sequential': False, 'sequence_length': 1}),
    ('spikes', {'fraction': 0.20, 'multiplier': 5.0, 'sequential': False, 'sequence_length': 1}),

    # Missing (true impact — not in Chronos script)
    ('missing', {'fraction': 0.05}),
    ('missing', {'fraction': 0.20}),

    # Swap — point
    ('swap', {'fraction': 0.05, 'swap_length': 1, 'max_distance': None}),
    ('swap', {'fraction': 0.20, 'swap_length': 1, 'max_distance': None}),

    # Freeze
    ('freeze', {'fraction': 0.05, 'stuck_length': 50}),
    ('freeze', {'fraction': 0.20, 'stuck_length': 50}),
]

CONTEXT_LENGTH = 512  # MOMENT's expected input length


def make_condition_name(corruption_type, params):
    if corruption_type == 'clean':
        return 'clean'
    parts = [corruption_type]
    for k, v in sorted(params.items()):
        parts.append(f"{k}={v}")
    return "_".join(parts)


def apply_corruption(df, corruption_type, params, seed=42):
    """Apply corruption and return corrupted DataFrame."""
    if corruption_type == 'clean':
        return df.copy()

    corruptor = TSCorruptor(df, value_col='value', label_col='is_anomaly', seed=seed)

    if corruption_type == 'noise':
        ts_corruptor.injectors.inject_white_noise_snr(corruptor, snr_db=params['snr_db'])

    elif corruption_type == 'spikes':
        ts_corruptor.injectors.inject_spikes(
            corruptor,
            fraction=params['fraction'],
            multiplier=params['multiplier'],
            sequential=params['sequential'],
            sequence_length=params['sequence_length']
        )

    elif corruption_type == 'missing':
        ts_corruptor.injectors.inject_point_missing(
            corruptor, fraction=params['fraction']
        )

    elif corruption_type == 'swap':
        fraction = params['fraction']
        max_distance = params.get('max_distance', None)
        swap_length = params.get('swap_length', 1)
        ts_corruptor.injectors.inject_swap(
            corruptor, fraction=fraction, swap_length=swap_length, max_distance=max_distance
        )

    elif corruption_type == 'freeze':
        n = len(df)
        stuck_length = params['stuck_length']
        num_stucks = max(1, int(params['fraction'] * n / stuck_length))
        ts_corruptor.injectors.inject_sensor_stuck(
            corruptor, num_stucks=num_stucks, stuck_length=stuck_length
        )

    return corruptor.get_corrupted_df()


def moment_anomaly_scores(model, data, batch_size, device):
    """
    Reconstruction-based anomaly scores using MOMENT.

    Slides a window of CONTEXT_LENGTH over the series, reconstructs each window,
    and uses MSE as anomaly score. Overlapping regions are averaged.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).ravel()
    n = len(data_scaled)

    if n <= CONTEXT_LENGTH:
        # Pad short series
        padded = np.zeros(CONTEXT_LENGTH)
        padded[:n] = data_scaled
        mask = np.zeros(CONTEXT_LENGTH)
        mask[:n] = 1.0

        x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 512]
        input_mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, 512]

        x = x.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            output = model(x_enc=x, input_mask=input_mask)
            recon = output.reconstruction.squeeze().cpu().numpy()

        mse = (padded - recon) ** 2
        scores = mse[:n]
        scores = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()
        return scores

    # Sliding window for long series
    step = CONTEXT_LENGTH  # non-overlapping windows
    score_sum = np.zeros(n)
    score_count = np.zeros(n)

    # Collect all windows
    windows = []
    starts = []
    for start in range(0, n - CONTEXT_LENGTH + 1, step):
        windows.append(data_scaled[start:start + CONTEXT_LENGTH])
        starts.append(start)

    # Add last window if not covered
    if starts[-1] + CONTEXT_LENGTH < n:
        starts.append(n - CONTEXT_LENGTH)
        windows.append(data_scaled[n - CONTEXT_LENGTH:n])

    # Process in batches
    for b in range(0, len(windows), batch_size):
        batch_windows = windows[b:b + batch_size]
        batch_starts = starts[b:b + batch_size]

        x = torch.tensor(np.array(batch_windows), dtype=torch.float32).unsqueeze(1)  # [B, 1, 512]
        input_mask = torch.ones(x.shape[0], CONTEXT_LENGTH, dtype=torch.float32)

        x = x.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            output = model(x_enc=x, input_mask=input_mask)
            recon = output.reconstruction.squeeze(1).cpu().numpy()  # [B, 512]

        for i, s in enumerate(batch_starts):
            window_scores = (batch_windows[i] - recon[i]) ** 2
            score_sum[s:s + CONTEXT_LENGTH] += window_scores
            score_count[s:s + CONTEXT_LENGTH] += 1

    # Average overlapping regions
    score_count[score_count == 0] = 1
    scores = score_sum / score_count

    scores = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()
    return scores


def process_file(model, file_path, corruption_type, params, device, batch_size):
    """Process a single file with a single corruption condition."""
    file_name = os.path.basename(file_path)
    condition = make_condition_name(corruption_type, params)

    try:
        df, file_name = load_tsb_dataframe(file_path)

        # Apply corruption
        df_corrupted = apply_corruption(df, corruption_type, params)

        data = df_corrupted['value'].to_numpy('float')
        labels = df_corrupted['is_anomaly'].to_numpy('int')

        # Handle missing: replace NaN with 0 for model input, keep labels
        nan_mask = np.isnan(data)
        if nan_mask.any():
            data_for_model = data.copy()
            data_for_model[nan_mask] = 0.0
        else:
            data_for_model = data

        if len(data_for_model) < 50:
            return {'file': file_name, 'condition': condition, 'corruption': corruption_type,
                    'model': 'MOMENT', 'error': 'Too short'}

        # Compute sliding window from clean signal
        clean_data = df['value'].to_numpy('float')
        sw = max(int(find_length(StandardScaler().fit_transform(clean_data.reshape(-1, 1)).ravel())), 10)

        # Get anomaly scores
        scores = moment_anomaly_scores(model, data_for_model, batch_size, device)

        if np.isnan(scores).any() or len(np.unique(scores)) <= 1:
            return {'file': file_name, 'condition': condition, 'corruption': corruption_type,
                    'model': 'MOMENT', 'error': 'Invalid scores'}

        metrics = get_metrics(scores, labels, metric="all", slidingWindow=sw)

        row = {
            'file': file_name, 'condition': condition,
            'corruption': corruption_type, 'model': 'MOMENT', 'error': None
        }
        for k, v in params.items():
            row[k] = v
        for key in SELECTED_METRICS:
            row[key] = round(metrics.get(key, 0.0), 4)

        return row

    except Exception as e:
        return {'file': file_name, 'condition': condition, 'corruption': corruption_type,
                'model': 'MOMENT', 'error': str(e)[:300]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--model-id', type=str, default='AutonLab/MOMENT-1-large')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Device setup ----
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("[CPU] No GPU detected, running on CPU")

    # ---- Load model ----
    print(f"[INFO] Loading {args.model_id}...")
    try:
        model = MOMENTPipeline.from_pretrained(
            args.model_id,
            model_kwargs={"task_name": "reconstruction"},
        )
        model.init()
        model = model.to(device).float()
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    # ---- Load files ----
    if not os.path.exists(SUBSET_CSV):
        print(f"[ERROR] {SUBSET_CSV} not found")
        return

    df_files = pd.read_csv(SUBSET_CSV)['filepath'].tolist()

    corruption_grid = CORRUPTION_GRID

    if args.test:
        df_files = df_files[:3]
        corruption_grid = [
            ('clean', {}),
            ('noise', {'snr_db': 5.0}),
            ('spikes', {'fraction': 0.10, 'multiplier': 5.0, 'sequential': False, 'sequence_length': 1}),
            ('missing', {'fraction': 0.10}),
        ]
        print("!!! TEST MODE !!!")

    # ---- Checkpoint ----
    checkpoint_file = os.path.join(RESULTS_DIR, "checkpoint.csv")
    completed_jobs = set()
    all_results = []

    if os.path.exists(checkpoint_file):
        try:
            df_cp = pd.read_csv(checkpoint_file)
            if 'error' in df_cp.columns:
                df_success = df_cp[df_cp['error'].isna()]
            else:
                df_success = df_cp
            all_results = df_success.to_dict('records')
            for r in all_results:
                completed_jobs.add(f"{r['file']}_{r['condition']}")
            print(f"Loaded {len(all_results)} results from checkpoint.")
        except Exception as e:
            print(f"Warning: checkpoint read error: {e}")

    # ---- Build job list ----
    total_conditions = len(corruption_grid)
    total_jobs = len(df_files) * total_conditions
    print(f"\n{'=' * 60}")
    print(f"  MOMENT Robustness Experiment")
    print(f"{'=' * 60}")
    print(f"Files: {len(df_files)}")
    print(f"Conditions: {total_conditions}")
    print(f"Total jobs: {total_jobs}")
    print(f"Device: {device} | Batch size: {args.batch_size}")
    print(f"{'=' * 60}\n")

    # ---- Run ----
    new_count = 0
    pbar = tqdm(total=total_jobs, desc="MOMENT")

    # Skip already completed
    already_done = 0
    for file_path in df_files:
        file_name = os.path.basename(file_path)
        for corruption_type, params in corruption_grid:
            condition = make_condition_name(corruption_type, params)
            job_id = f"{file_name}_{condition}"
            if job_id in completed_jobs:
                already_done += 1
    pbar.update(already_done)

    for file_path in df_files:
        file_name = os.path.basename(file_path)

        for corruption_type, params in corruption_grid:
            condition = make_condition_name(corruption_type, params)
            job_id = f"{file_name}_{condition}"

            if job_id in completed_jobs:
                continue

            result = process_file(model, file_path, corruption_type, params, device, args.batch_size)
            all_results.append(result)
            new_count += 1
            pbar.update(1)

            if new_count % 50 == 0:
                pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)

        # Free GPU memory between files
        if device == "cuda":
            torch.cuda.empty_cache()

    pbar.close()

    # ---- Save final results ----
    pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)

    if all_results:
        df_all = pd.DataFrame(all_results)

        # Summary
        df_success = df_all[df_all['error'].isna()]
        if not df_success.empty:
            summary = df_success.groupby(['corruption', 'condition']).agg(
                n_runs=('file', 'count'),
                **{f'mean_{m}': (m, 'mean') for m in SELECTED_METRICS if m in df_success.columns},
                **{f'std_{m}': (m, 'std') for m in SELECTED_METRICS if m in df_success.columns},
            ).reset_index()
            summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)

            print(f"\n{'=' * 60}")
            print("  Summary (mean AUC-ROC per condition)")
            print(f"{'=' * 60}")
            for _, r in summary.iterrows():
                auc = r.get('mean_AUC_ROC', 0)
                print(f"  {r['condition']:45s}  AUC-ROC={auc:.4f}  (n={int(r['n_runs'])})")

        # Per corruption type
        for corr_type in df_all['corruption'].unique():
            corr_dir = os.path.join(RESULTS_DIR, corr_type)
            os.makedirs(corr_dir, exist_ok=True)
            corr_df = df_all[df_all['corruption'] == corr_type]
            corr_df.to_csv(os.path.join(corr_dir, "raw_results.csv"), index=False)

    print(f"\n[Done] Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
