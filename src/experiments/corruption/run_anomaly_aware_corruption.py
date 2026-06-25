"""
Anomaly-Aware Corruption Experiment

Key question:
    Is corruption more damaging when it lands on normal regions, on anomaly
    regions, or on a controlled mixture of both?

This experiment keeps the corruption budget fixed and changes only WHERE
corruption is applied relative to the ground-truth anomaly labels.

Target modes:
  - only_normal:   corruption only on normal points
  - only_anomaly:  corruption only on anomaly points
  - mixed:         approximately half the corrupted points on normal regions
                   and half on anomaly regions

Supported corruption types:
  - noise   : additive white noise controlled by SNR (dB)
  - spikes  : point spikes with magnitude in std units
  - missing : point missing values (true impact; no imputation)

Rationale:
  - only_normal  -> tests false positives / artificial anomalies
  - only_anomaly -> tests anomaly masking / false negatives
  - mixed        -> tests realistic mixed contamination

Usage:
    python run_anomaly_aware_corruption.py --models IForest LOF MP AE
    python run_anomaly_aware_corruption.py --corruptions noise spikes
    python run_anomaly_aware_corruption.py --test
"""
import os
import sys
import math
import argparse
import traceback

os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# PATH CONFIGURATION
# ==========================================
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
tsb_uad_path = os.path.join(project_root, 'TSB-UAD')
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if tsb_uad_path not in sys.path:
    sys.path.insert(0, tsb_uad_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.matrix_profile import MatrixProfile
from TSB_UAD.models.lof import LOF
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

from data_loader import load_tsb_dataframe, remap_filepath, load_pretrained_ae

# ==========================================
# CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "anomaly_aware_corruption")

TARGET_MODES = ['only_normal', 'only_anomaly', 'mixed']
CORRUPTION_TYPES = ['noise', 'spikes', 'missing']

FRACTIONS = [0.01, 0.05, 0.10]
SNRS_DB = [20, 10, 5]
SPIKE_MULTIPLIERS = [3.0, 5.0, 10.0]
N_SEEDS = 1

ALL_MODELS = ['IForest', 'LOF', 'MP', 'AE']

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]


# ==========================================
# CONDITION GRID
# ==========================================
def build_conditions(corruptions, target_modes):
    conditions = []
    for ctype in corruptions:
        for target_mode in target_modes:
            if ctype == 'noise':
                for fraction in FRACTIONS:
                    for snr_db in SNRS_DB:
                        conditions.append({
                            'corruption_type': ctype,
                            'target_mode': target_mode,
                            'fraction': fraction,
                            'snr_db': snr_db,
                            'multiplier': None,
                            'condition': f"noise_{target_mode}_frac_{fraction}_snr_{snr_db}dB",
                        })
            elif ctype == 'spikes':
                for fraction in FRACTIONS:
                    for multiplier in SPIKE_MULTIPLIERS:
                        conditions.append({
                            'corruption_type': ctype,
                            'target_mode': target_mode,
                            'fraction': fraction,
                            'snr_db': None,
                            'multiplier': multiplier,
                            'condition': f"spikes_{target_mode}_frac_{fraction}_mult_{multiplier}",
                        })
            elif ctype == 'missing':
                for fraction in FRACTIONS:
                    conditions.append({
                        'corruption_type': ctype,
                        'target_mode': target_mode,
                        'fraction': fraction,
                        'snr_db': None,
                        'multiplier': None,
                        'condition': f"missing_{target_mode}_frac_{fraction}",
                    })
    return conditions


# ==========================================
# TARGET SELECTION
# ==========================================
def choose_target_indices(labels, target_mode, count, rng):
    labels = np.asarray(labels).astype(int)
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    if target_mode == 'only_normal':
        if len(normal_idx) == 0:
            return np.array([], dtype=int)
        count = min(count, len(normal_idx))
        return np.sort(rng.choice(normal_idx, size=count, replace=False))

    if target_mode == 'only_anomaly':
        if len(anomaly_idx) == 0:
            return np.array([], dtype=int)
        count = min(count, len(anomaly_idx))
        return np.sort(rng.choice(anomaly_idx, size=count, replace=False))

    if target_mode == 'mixed':
        if len(normal_idx) == 0 and len(anomaly_idx) == 0:
            return np.array([], dtype=int)

        target_anom = count // 2
        target_norm = count - target_anom

        take_anom = min(target_anom, len(anomaly_idx))
        take_norm = min(target_norm, len(normal_idx))

        remaining = count - take_anom - take_norm
        if remaining > 0:
            # Fill from whichever pool still has capacity.
            spare_norm = max(0, len(normal_idx) - take_norm)
            add_norm = min(spare_norm, remaining)
            take_norm += add_norm
            remaining -= add_norm

        if remaining > 0:
            spare_anom = max(0, len(anomaly_idx) - take_anom)
            add_anom = min(spare_anom, remaining)
            take_anom += add_anom
            remaining -= add_anom

        picked = []
        if take_norm > 0:
            picked.extend(rng.choice(normal_idx, size=take_norm, replace=False).tolist())
        if take_anom > 0:
            picked.extend(rng.choice(anomaly_idx, size=take_anom, replace=False).tolist())
        return np.sort(np.array(picked, dtype=int))

    raise ValueError(f"Unknown target mode: {target_mode}")


# ==========================================
# CORRUPTION APPLICATION
# ==========================================
def apply_noise(values, target_indices, snr_db, rng):
    out = values.astype(float).copy()
    if len(target_indices) == 0:
        return out

    clean_signal = out[~np.isnan(out)]
    signal_power = max(np.mean(clean_signal ** 2), 1e-10)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)
    out[target_indices] = out[target_indices] + rng.normal(0, noise_std, size=len(target_indices))
    return out


def apply_spikes(values, target_indices, multiplier, std, rng):
    out = values.astype(float).copy()
    if len(target_indices) == 0:
        return out
    signs = rng.choice([-1, 1], size=len(target_indices))
    out[target_indices] = out[target_indices] + signs * multiplier * std
    return out


def apply_missing(values, target_indices):
    out = values.astype(float).copy()
    if len(target_indices) == 0:
        return out
    out[target_indices] = np.nan
    return out


# ==========================================
# MODEL SCORING
# ==========================================
def run_model_score(model_name, scaled_data, sliding_window, canonical_name):
    if model_name == 'AE':
        clf, _ = load_pretrained_ae(canonical_name, project_root)
        clf.predict(scaled_data)
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            clf.decision_scores_.reshape(-1, 1)
        ).ravel()
        n = len(scaled_data)
        if len(score) > n:
            score = score[:n]
        elif len(score) < n:
            score = np.pad(score, (0, n - len(score)), mode='edge')
        return score

    if model_name == 'MP':
        clf = MatrixProfile(window=sliding_window)
        clf.fit(scaled_data)
        score = clf.decision_scores_
    else:
        X = Window(window=sliding_window).convert(scaled_data).to_numpy()
        if model_name == 'IForest':
            clf = IForest(n_estimators=100, random_state=42)
        elif model_name == 'LOF':
            clf = LOF(n_neighbors=min(20, max(2, len(X) - 1)))
        else:
            raise ValueError(f"Unknown model: {model_name}")
        clf.fit(X)
        score = clf.decision_scores_

    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
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
# WORKER
# ==========================================
def process_file_model(job_args):
    file_path, model_name, conditions, skip_keys = job_args
    fallback_name = os.path.basename(file_path)

    try:
        df, canonical_name = load_tsb_dataframe(file_path)
        values = df['value'].to_numpy(float)
        labels = df['is_anomaly'].to_numpy(int)
        n = len(values)
        signal_std = float(np.nanstd(values))
        if signal_std == 0 or np.isnan(signal_std):
            signal_std = 1.0

        clean_scaled = StandardScaler().fit_transform(values.reshape(-1, 1)).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        rows = []
        for seed in range(N_SEEDS):
            for cond in conditions:
                cond_key = f"{canonical_name}_{cond['condition']}_{seed}_{model_name}"
                if cond_key in skip_keys:
                    continue

                try:
                    rng = np.random.RandomState(seed)
                    requested_count = max(1, int(n * cond['fraction']))
                    target_indices = choose_target_indices(labels, cond['target_mode'], requested_count, rng)

                    if len(target_indices) == 0:
                        rows.append({
                            'file': canonical_name,
                            'model': model_name,
                            'seed': seed,
                            'condition': cond['condition'],
                            'corruption_type': cond['corruption_type'],
                            'target_mode': cond['target_mode'],
                            'fraction': cond['fraction'],
                            'snr_db': cond['snr_db'],
                            'multiplier': cond['multiplier'],
                            'requested_points': requested_count,
                            'actual_corrupted_points': 0,
                            'corrupted_normal_points': 0,
                            'corrupted_anomaly_points': 0,
                            'error': 'No valid target points for this mode'
                        })
                        continue

                    if cond['corruption_type'] == 'noise':
                        corrupted = apply_noise(values, target_indices, cond['snr_db'], rng)
                    elif cond['corruption_type'] == 'spikes':
                        corrupted = apply_spikes(values, target_indices, cond['multiplier'], signal_std, rng)
                    elif cond['corruption_type'] == 'missing':
                        corrupted = apply_missing(values, target_indices)
                    else:
                        raise ValueError(f"Unsupported corruption type: {cond['corruption_type']}")

                    corrupted_normal = int(np.sum(labels[target_indices] == 0))
                    corrupted_anomaly = int(np.sum(labels[target_indices] == 1))

                    if cond['corruption_type'] == 'missing':
                        nan_mask = np.isnan(corrupted)
                        masked_normal = nan_mask & (labels == 0)
                        masked_anomaly = nan_mask & (labels == 1)
                        kept = corrupted[~nan_mask]
                        if len(kept) < sliding_window + 10:
                            raise ValueError(f"Too few points after masking: {len(kept)}")

                        sw = min(sliding_window, max(10, len(kept) // 4))
                        scaled = StandardScaler().fit_transform(kept.reshape(-1, 1)).flatten()
                        score_kept = run_model_score(model_name, scaled, sw, canonical_name)

                        full_score = np.full(n, np.nan)
                        full_score[~nan_mask] = score_kept
                        # Lost anomalies become guaranteed false negatives.
                        full_score[masked_anomaly] = 0.0
                        eval_mask = ~masked_normal
                        eval_scores = full_score[eval_mask]
                        eval_labels = labels[eval_mask]
                        metrics = get_metrics(eval_scores, eval_labels, metric="all", slidingWindow=sw)
                    else:
                        scaled = StandardScaler().fit_transform(corrupted.reshape(-1, 1)).flatten()
                        full_score = run_model_score(model_name, scaled, sliding_window, canonical_name)
                        if np.isnan(full_score).any() or len(np.unique(full_score)) <= 1:
                            raise ValueError("Invalid scores generated")
                        metrics = get_metrics(full_score, labels, metric="all", slidingWindow=sliding_window)

                    row = {
                        'file': canonical_name,
                        'model': model_name,
                        'seed': seed,
                        'condition': cond['condition'],
                        'corruption_type': cond['corruption_type'],
                        'target_mode': cond['target_mode'],
                        'fraction': cond['fraction'],
                        'snr_db': cond['snr_db'],
                        'multiplier': cond['multiplier'],
                        'requested_points': requested_count,
                        'actual_corrupted_points': int(len(target_indices)),
                        'corrupted_normal_points': corrupted_normal,
                        'corrupted_anomaly_points': corrupted_anomaly,
                        'error': None,
                    }
                    for key in SELECTED_METRICS:
                        row[key] = round(metrics.get(key, 0.0), 4)
                    rows.append(row)

                except Exception as e:
                    rows.append({
                        'file': canonical_name,
                        'model': model_name,
                        'seed': seed,
                        'condition': cond['condition'],
                        'corruption_type': cond['corruption_type'],
                        'target_mode': cond['target_mode'],
                        'fraction': cond['fraction'],
                        'snr_db': cond['snr_db'],
                        'multiplier': cond['multiplier'],
                        'requested_points': max(1, int(n * cond['fraction'])),
                        'actual_corrupted_points': 0,
                        'corrupted_normal_points': 0,
                        'corrupted_anomaly_points': 0,
                        'error': str(e),
                    })

        return {'status': 'success', 'rows': rows}

    except Exception:
        return {'status': 'error', 'file': fallback_name, 'error': traceback.format_exc()}


# ==========================================
# SUMMARY
# ==========================================
def compute_summary(df_results, out_csv):
    df_ok = df_results[df_results['error'].isna()].copy()
    if df_ok.empty:
        print("No successful runs to summarize.")
        return

    grouped = df_ok.groupby(['corruption_type', 'target_mode', 'fraction', 'snr_db', 'multiplier', 'model'], dropna=False)
    rows = []
    for key, g in grouped:
        ctype, target_mode, fraction, snr_db, multiplier, model = key
        row = {
            'corruption_type': ctype,
            'target_mode': target_mode,
            'fraction': fraction,
            'snr_db': snr_db,
            'multiplier': multiplier,
            'model': model,
            'n_runs': len(g),
            'mean_actual_corrupted_points': round(g['actual_corrupted_points'].mean(), 2),
            'mean_corrupted_normal_points': round(g['corrupted_normal_points'].mean(), 2),
            'mean_corrupted_anomaly_points': round(g['corrupted_anomaly_points'].mean(), 2),
        }
        for metric in SELECTED_METRICS:
            row[f'mean_{metric}'] = round(g[metric].mean(), 4)
            row[f'std_{metric}'] = round(g[metric].std(), 4)
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Summary saved to {out_csv}")


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Anomaly-aware corruption experiment')
    parser.add_argument('--test', action='store_true', help='Small smoke test')
    parser.add_argument('--models', nargs='+', default=['IForest','MP','LOF','AE'], choices=ALL_MODELS)
    parser.add_argument('--corruptions', nargs='+', default=CORRUPTION_TYPES, choices=CORRUPTION_TYPES)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_meta = pd.read_csv(SUBSET_CSV)
    file_paths = [remap_filepath(fp, project_root) for fp in df_meta['filepath'].tolist()]
    target_modes = TARGET_MODES
    corruptions = args.corruptions
    conditions = build_conditions(corruptions, target_modes)

    if args.test:
        file_paths = file_paths[:1]
        conditions = [
            {
                'corruption_type': 'noise',
                'target_mode': 'only_normal',
                'fraction': 0.05,
                'snr_db': 10,
                'multiplier': None,
                'condition': 'noise_only_normal_frac_0.05_snr_10dB',
            },
            {
                'corruption_type': 'noise',
                'target_mode': 'only_anomaly',
                'fraction': 0.05,
                'snr_db': 10,
                'multiplier': None,
                'condition': 'noise_only_anomaly_frac_0.05_snr_10dB',
            },
            {
                'corruption_type': 'spikes',
                'target_mode': 'mixed',
                'fraction': 0.05,
                'snr_db': None,
                'multiplier': 5.0,
                'condition': 'spikes_mixed_frac_0.05_mult_5.0',
            },
        ]
        print("!!! TEST MODE !!!")
    else:
        conditions = build_conditions(corruptions, target_modes)
    checkpoint_csv = os.path.join(RESULTS_DIR, "checkpoint.csv")
    completed_keys = set()

    if os.path.exists(checkpoint_csv):
        try:
            df_ckpt = pd.read_csv(checkpoint_csv)
            df_ok = df_ckpt[df_ckpt['error'].isna()] if 'error' in df_ckpt.columns else df_ckpt
            for _, r in df_ok.iterrows():
                completed_keys.add(f"{r['file']}_{r['condition']}_{int(r['seed'])}_{r['model']}")
            print(f"[Resume] {len(completed_keys)} completed runs")
        except Exception as e:
            print(f"[WARN] Could not parse checkpoint: {e}")

    jobs = []
    total_runs = 0
    for fp in file_paths:
        for model_name in args.models:
            jobs.append((fp, model_name, conditions, completed_keys))
            total_runs += sum(
                1 for seed in range(N_SEEDS)
                for cond in conditions
                if f"{os.path.basename(fp)}_{cond['condition']}_{seed}_{model_name}" not in completed_keys
            )

    print(f"\n{'=' * 72}")
    print("  Anomaly-Aware Corruption Experiment")
    print(f"{'=' * 72}")
    print(f"  Files:         {len(file_paths)}")
    print(f"  Models:        {args.models}")
    print(f"  Corruptions:   {corruptions}")
    print(f"  Target modes:  {target_modes}")
    print(f"  Conditions:    {len(conditions)} per file-model")
    print(f"  Workers:       {args.workers}")
    print(f"  Output:        {RESULTS_DIR}")
    print(f"{'=' * 72}\n")

    all_rows = []
    if os.path.exists(checkpoint_csv):
        try:
            all_rows = pd.read_csv(checkpoint_csv).to_dict('records')
        except Exception:
            all_rows = []

    total_expected = len(file_paths) * len(args.models) * len(conditions) * N_SEEDS
    processed_since_ckpt = 0

    with tqdm(total=total_expected, initial=len(completed_keys), desc="Anomaly-aware runs") as pbar:
        if args.workers <= 1:
            for job in jobs:
                res = process_file_model(job)
                if res['status'] == 'success':
                    all_rows.extend(res['rows'])
                    pbar.update(len(res['rows']))
                    processed_since_ckpt += len(res['rows'])
                else:
                    print(f"[ERROR] {res.get('file', '?')}: {res.get('error', 'unknown')[:160]}")

                if processed_since_ckpt >= max(20, len(conditions)):
                    pd.DataFrame(all_rows).to_csv(checkpoint_csv, index=False)
                    processed_since_ckpt = 0
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_file_model, job): job for job in jobs}
                for future in as_completed(futures):
                    res = future.result()
                    if res['status'] == 'success':
                        all_rows.extend(res['rows'])
                        pbar.update(len(res['rows']))
                        processed_since_ckpt += len(res['rows'])
                    else:
                        print(f"[ERROR] {res.get('file', '?')}: {res.get('error', 'unknown')[:160]}")

                    if processed_since_ckpt >= max(20, len(conditions)):
                        pd.DataFrame(all_rows).to_csv(checkpoint_csv, index=False)
                        processed_since_ckpt = 0

    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(checkpoint_csv, index=False)
    final_csv = os.path.join(RESULTS_DIR, "raw_results.csv")
    df_results.to_csv(final_csv, index=False)
    print(f"Saved raw results to {final_csv}")

    summary_csv = os.path.join(RESULTS_DIR, "summary.csv")
    compute_summary(df_results, summary_csv)


if __name__ == '__main__':
    main()
