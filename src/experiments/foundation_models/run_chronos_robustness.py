"""
Foundation Model Robustness Experiment — Chronos-2

Evaluates Chronos-2 (uncertainty-normalized forecasting) on clean and corrupted
data using the same corruption types as classical model experiments.

Usage:
    python run_chronos_robustness.py                  # full run
    python run_chronos_robustness.py --test            # quick test (3 files)
    python run_chronos_robustness.py --batch-size 32   # adjust for GPU memory
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
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

try:
    from chronos import BaseChronosPipeline
except ImportError:
    print("[ERROR] Install chronos: pip install chronos-forecasting")
    sys.exit(1)

from TSB_UAD.vus.metrics import get_metrics
from TSB_UAD.utils.slidingWindows import find_length

from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors
from data_loader import load_tsb_dataframe

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "robust_subset_TSB.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "foundation_models", "chronos")

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]

# Reduced parameter grid (Chronos is slow)
CORRUPTION_GRID = [
    # (corruption_type, params_dict)
    ('clean', {}),

    # White noise SNR (matching single experiments: 9 SNR values)
    ('noise', {'snr_db': 40.0}),
    ('noise', {'snr_db': 30.0}),
    ('noise', {'snr_db': 20.0}),
    ('noise', {'snr_db': 10.0}),
    ('noise', {'snr_db': 5.0}),
    ('noise', {'snr_db': 0.0}),
    ('noise', {'snr_db': -5.0}),
    ('noise', {'snr_db': -10.0}),
    ('noise', {'snr_db': -20.0}),

    # Spikes — point (matching single experiments: 4 fractions × 2 multipliers)
    ('spikes', {'fraction': 0.01, 'multiplier': 3.0, 'sequential': False, 'sequence_length': 1}),
    ('spikes', {'fraction': 0.01, 'multiplier': 10.0, 'sequential': False, 'sequence_length': 1}),
    ('spikes', {'fraction': 0.05, 'multiplier': 3.0, 'sequential': False, 'sequence_length': 1}),
    ('spikes', {'fraction': 0.05, 'multiplier': 10.0, 'sequential': False, 'sequence_length': 1}),
    ('spikes', {'fraction': 0.10, 'multiplier': 3.0, 'sequential': False, 'sequence_length': 1}),
    ('spikes', {'fraction': 0.10, 'multiplier': 10.0, 'sequential': False, 'sequence_length': 1}),
    ('spikes', {'fraction': 0.20, 'multiplier': 3.0, 'sequential': False, 'sequence_length': 1}),
    ('spikes', {'fraction': 0.20, 'multiplier': 10.0, 'sequential': False, 'sequence_length': 1}),

    # Swap — point (matching single experiments: 6 fractions, swap_length=1)
    ('swap', {'fraction': 0.01, 'swap_length': 1, 'max_distance': None}),
    ('swap', {'fraction': 0.05, 'swap_length': 1, 'max_distance': None}),
    ('swap', {'fraction': 0.10, 'swap_length': 1, 'max_distance': None}),
    ('swap', {'fraction': 0.20, 'swap_length': 1, 'max_distance': None}),
    ('swap', {'fraction': 0.30, 'swap_length': 1, 'max_distance': None}),
    ('swap', {'fraction': 0.40, 'swap_length': 1, 'max_distance': None}),

    # Swap — segment (matching single experiments: 5 fractions, num_swaps=5)
    ('swap', {'fraction': 0.01, 'num_swaps': 5, 'max_distance': None}),
    ('swap', {'fraction': 0.05, 'num_swaps': 5, 'max_distance': None}),
    ('swap', {'fraction': 0.10, 'num_swaps': 5, 'max_distance': None}),
    ('swap', {'fraction': 0.20, 'num_swaps': 5, 'max_distance': None}),
    ('swap', {'fraction': 0.30, 'num_swaps': 5, 'max_distance': None}),

    # Freeze
    ('freeze', {'fraction': 0.05, 'stuck_length': 50}),
    ('freeze', {'fraction': 0.20, 'stuck_length': 50}),
]

SEQ_LEN = 512
SCORE_LIMIT = 5000  # max points to score per file


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

    elif corruption_type == 'swap':
        fraction = params['fraction']
        max_distance = params.get('max_distance', None)
        n = len(df)
        if 'num_swaps' in params:
            # Segment swap: swap_length = fraction * n / (2 * num_swaps)
            num_swaps = params['num_swaps']
            swap_length = max(1, int(fraction * n / (2 * num_swaps)))
        elif 'swap_ratio' in params:
            swap_length = max(1, int(n * params['swap_ratio']))
        else:
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


def chronos_anomaly_scores(pipeline, data, seq_len, batch_size, device):
    """
    Uncertainty-normalized forecasting anomaly scores.
    score = |actual - median_prediction| / IQR(5%, 95%)
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).ravel()

    start_i = seq_len
    end_i = min(len(data_scaled), seq_len + SCORE_LIMIT)

    contexts = []
    targets = []
    for i in range(start_i, end_i):
        contexts.append(data_scaled[i - seq_len: i])
        targets.append(data_scaled[i])

    if len(contexts) == 0:
        return np.zeros(len(data))

    prediction_errors = []

    for b in range(0, len(contexts), batch_size):
        batch_x = contexts[b: b + batch_size]
        batch_y = targets[b: b + batch_size]

        context_tensor = torch.tensor(np.array(batch_x), dtype=torch.float32)
        if context_tensor.ndim == 2:
            context_tensor = context_tensor.unsqueeze(1)

        with torch.no_grad():
            forecast = pipeline.predict(context_tensor, prediction_length=1)
            if isinstance(forecast, list):
                forecast = torch.stack(forecast)

            batch_size_val = len(batch_y)

            # Find batch dimension
            batch_dim = -1
            for d in range(forecast.ndim):
                if forecast.shape[d] == batch_size_val:
                    batch_dim = d
                    break

            samples_dim = 1 if batch_dim == 0 else 0

            median = torch.quantile(forecast, 0.5, dim=samples_dim)
            q05 = torch.quantile(forecast, 0.05, dim=samples_dim)
            q95 = torch.quantile(forecast, 0.95, dim=samples_dim)

            median = median.reshape(batch_size_val, -1).mean(dim=1).cpu().numpy()
            q05 = q05.reshape(batch_size_val, -1).mean(dim=1).cpu().numpy()
            q95 = q95.reshape(batch_size_val, -1).mean(dim=1).cpu().numpy()

            iqr = q95 - q05
            norm_error = np.abs(np.array(batch_y) - median) / (iqr + 1e-6)
            prediction_errors.extend(norm_error)

    # Pad to full length
    full_scores = np.zeros(len(data))
    full_scores[start_i:end_i] = prediction_errors
    if prediction_errors:
        full_scores[:start_i] = prediction_errors[0]
        if end_i < len(data):
            full_scores[end_i:] = prediction_errors[-1]

    full_scores = MinMaxScaler().fit_transform(full_scores.reshape(-1, 1)).ravel()
    return full_scores


def process_file(pipeline, file_path, corruption_type, params, device, batch_size):
    """Process a single file with a single corruption condition."""
    file_name = os.path.basename(file_path)
    condition = make_condition_name(corruption_type, params)

    try:
        df, file_name = load_tsb_dataframe(file_path)

        # Apply corruption
        df_corrupted = apply_corruption(df, corruption_type, params)

        data = df_corrupted['value'].to_numpy('float')
        labels = df_corrupted['is_anomaly'].to_numpy('int')

        if np.isnan(data).all():
            return {'file': file_name, 'condition': condition, 'corruption': corruption_type,
                    'model': 'Chronos-2', 'error': 'All NaN'}

        # Compute sliding window from clean signal
        clean_data = df['value'].to_numpy('float')
        sw = max(int(find_length(StandardScaler().fit_transform(clean_data.reshape(-1, 1)).ravel())), 10)

        # Get anomaly scores
        scores = chronos_anomaly_scores(pipeline, data, SEQ_LEN, batch_size, device)

        if np.isnan(scores).any() or len(np.unique(scores)) <= 1:
            return {'file': file_name, 'condition': condition, 'corruption': corruption_type,
                    'model': 'Chronos-2', 'error': 'Invalid scores'}

        metrics = get_metrics(scores, labels, metric="all", slidingWindow=sw)

        row = {
            'file': file_name, 'condition': condition,
            'corruption': corruption_type, 'model': 'Chronos-2', 'error': None
        }
        # Add corruption params
        for k, v in params.items():
            row[k] = v

        for key in SELECTED_METRICS:
            row[key] = round(metrics.get(key, 0.0), 4)

        return row

    except Exception as e:
        return {'file': file_name, 'condition': condition, 'corruption': corruption_type,
                'model': 'Chronos-2', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--model-id', type=str, default='amazon/chronos-2',
                        help='HuggingFace model ID (e.g. amazon/chronos-2, amazon/chronos-bolt-base)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Device setup ----
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16  # faster on GPU
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("[CPU] No GPU detected, running on CPU (will be slow)")

    # ---- Load model ----
    print(f"[INFO] Loading {args.model_id}...")
    try:
        pipeline = BaseChronosPipeline.from_pretrained(
            args.model_id,
            device_map=device,
            torch_dtype=torch_dtype,
        )
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
    print(f"  Chronos-2 Robustness Experiment")
    print(f"{'=' * 60}")
    print(f"Files: {len(df_files)}")
    print(f"Conditions: {total_conditions}")
    print(f"Total jobs: {total_jobs}")
    print(f"Device: {device} | Batch size: {args.batch_size}")
    print(f"{'=' * 60}\n")

    # ---- Run ----
    new_count = 0
    pbar = tqdm(total=total_jobs, desc="Chronos-2")

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

            result = process_file(pipeline, file_path, corruption_type, params, device, args.batch_size)
            all_results.append(result)
            new_count += 1
            pbar.update(1)

            # Checkpoint every 50 results
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

        # Per corruption type
        for corr_type in df_all['corruption'].unique():
            corr_dir = os.path.join(RESULTS_DIR, corr_type)
            os.makedirs(corr_dir, exist_ok=True)
            corr_df = df_all[df_all['corruption'] == corr_type]
            corr_df.to_csv(os.path.join(corr_dir, "raw_results.csv"), index=False)

    print(f"\n[Done] Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
