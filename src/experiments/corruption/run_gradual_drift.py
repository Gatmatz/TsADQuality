"""
Gradual Drift Experiment — Progressive Sensor Degradation

Tests how GRADUALLY INCREASING corruption affects anomaly detection,
compared to the UNIFORM corruption of the same average intensity.

Drift types:
  1. Mean drift      — sensor calibration loss (value offset increases over time)
  2. Variance drift   — increasing noise (SNR decreases over time)
  3. Sensitivity decay — signal fading (amplitude shrinks over time)

For each drift type, we compare:
  - Gradual: corruption intensity increases linearly from 0 to max
  - Uniform: corruption at the average intensity (max/2) everywhere
  - Clean baseline (from existing baselines)

This lets us answer: is gradual drift more or less damaging than
uniform corruption of the same total energy?

We also split evaluation into first-half vs second-half to see
if anomalies in the degraded portion are harder to detect.

Usage:
    python run_gradual_drift.py --models IForest
    python run_gradual_drift.py --models IForest MP PCA LOF --workers 6
    python run_gradual_drift.py --test
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

from data_loader import load_tsb_dataframe, remap_filepath, load_pretrained_ae

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "gradual_drift")
N_SEEDS = 1
N_WINDOWS = 5  # number of temporal windows for tipping-point analysis (20% each)

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]

# ==========================================
# DRIFT CONFIGURATION
# ==========================================

# Severity: max_offset is in units of the signal's standard deviation.
# The gradual version ramps linearly from 0 to max; the uniform comparison
# applies max/2 everywhere, so both have the same mean corruption energy.
DRIFT_CONFIGS = {
    'mean_drift': {
        # max_offset: how many standard deviations the mean shifts by the end
        'low':    {'max_offset_std': 0.5},
        'med':    {'max_offset_std': 1.0},
        'high':   {'max_offset_std': 2.0},
        'severe': {'max_offset_std': 4.0},
    },
    'variance_drift': {
        # SNR ramps from start_snr down to end_snr over the series
        'low':    {'start_snr': 40, 'end_snr': 20},
        'med':    {'start_snr': 40, 'end_snr': 10},
        'high':   {'start_snr': 40, 'end_snr': 5},
        'severe': {'start_snr': 40, 'end_snr': 0},
    },
    'sensitivity_decay': {
        # min_sensitivity: fraction of original amplitude at the end
        # 1.0 = no change, 0.0 = completely flat
        'low':    {'min_sensitivity': 0.7},
        'med':    {'min_sensitivity': 0.4},
        'high':   {'min_sensitivity': 0.2},
        'severe': {'min_sensitivity': 0.05},
    },
}


# ==========================================
# DRIFT APPLICATION FUNCTIONS
# ==========================================

def apply_mean_drift_gradual(data, max_offset_std, rng):
    """Apply linearly increasing mean offset.

    At t=0: offset=0.  At t=n-1: offset = max_offset_std * std(data).
    This simulates gradual calibration loss.
    """
    n = len(data)
    std = np.std(data)
    if std < 1e-10:
        return data.copy()
    max_offset = max_offset_std * std
    drift = np.linspace(0, max_offset, n)
    return data + drift


def apply_mean_drift_uniform(data, max_offset_std, rng):
    """Apply UNIFORM mean offset equal to average of gradual version.

    Constant offset = max_offset / 2 everywhere (same total displacement).
    """
    n = len(data)
    std = np.std(data)
    if std < 1e-10:
        return data.copy()
    uniform_offset = (max_offset_std * std) / 2.0
    return data + uniform_offset


def apply_variance_drift_gradual(data, start_snr, end_snr, rng):
    """Apply noise whose intensity increases linearly (SNR decreases).

    At t=0: SNR = start_snr dB.  At t=n-1: SNR = end_snr dB.
    Each point gets independent Gaussian noise with locally-computed power.
    """
    n = len(data)
    signal_power = np.var(data)
    if signal_power < 1e-10:
        return data.copy()

    result = data.copy()
    # Pre-compute noise for all points at once for efficiency
    snr_profile = np.linspace(start_snr, end_snr, n)
    noise_power = signal_power / np.power(10, snr_profile / 10.0)
    noise_std = np.sqrt(np.maximum(noise_power, 0))
    noise = rng.normal(0, 1, n) * noise_std
    return result + noise


def apply_variance_drift_uniform(data, start_snr, end_snr, rng):
    """Apply UNIFORM noise with the same total noise energy as the gradual version.

    To ensure fair comparison, we match the total noise energy (integral of
    noise variance over time), NOT the average SNR.  For the gradual version,
    the noise power at time t is: P(t) = signal_var / 10^(snr(t)/10).
    The mean noise power is: mean_P = (1/n) * sum(P(t)).
    We apply uniform noise with this exact mean power.
    """
    n = len(data)
    signal_power = np.var(data)
    if signal_power < 1e-10:
        return data.copy()

    # Compute mean noise power matching the gradual version exactly
    snr_profile = np.linspace(start_snr, end_snr, n)
    noise_powers = signal_power / np.power(10, snr_profile / 10.0)
    mean_noise_power = np.mean(noise_powers)

    noise_std = np.sqrt(max(mean_noise_power, 0))
    noise = rng.normal(0, noise_std, n)
    return data + noise


def apply_sensitivity_decay_gradual(data, min_sensitivity, rng):
    """Apply linearly decreasing signal amplitude.

    At t=0: amplitude preserved (sensitivity=1.0).
    At t=n-1: amplitude *= min_sensitivity.
    This simulates a sensor losing sensitivity over time.
    """
    n = len(data)
    mean = np.mean(data)
    sensitivity = np.linspace(1.0, min_sensitivity, n)
    return mean + (data - mean) * sensitivity


def apply_sensitivity_decay_uniform(data, min_sensitivity, rng):
    """Apply UNIFORM sensitivity reduction at the average of gradual version.

    Average sensitivity = (1.0 + min_sensitivity) / 2.  Same mean attenuation.
    """
    mean = np.mean(data)
    avg_sensitivity = (1.0 + min_sensitivity) / 2.0
    return mean + (data - mean) * avg_sensitivity


# Map: (drift_type, mode) -> function
DRIFT_FUNCTIONS = {
    ('mean_drift', 'gradual'):          apply_mean_drift_gradual,
    ('mean_drift', 'uniform'):          apply_mean_drift_uniform,
    ('variance_drift', 'gradual'):      apply_variance_drift_gradual,
    ('variance_drift', 'uniform'):      apply_variance_drift_uniform,
    ('sensitivity_decay', 'gradual'):   apply_sensitivity_decay_gradual,
    ('sensitivity_decay', 'uniform'):   apply_sensitivity_decay_uniform,
}


# ==========================================
# CONDITION BUILDER
# ==========================================

def build_conditions():
    """Build all experiment conditions.

    For each drift type × severity, we create TWO conditions:
      - gradual: corruption increases linearly
      - uniform: same average corruption applied uniformly
    This allows direct comparison.
    """
    conditions = []

    for drift_type, severities in DRIFT_CONFIGS.items():
        for sev_name, params in severities.items():
            # Gradual version
            conditions.append({
                'name': f"{drift_type}_{sev_name}_gradual",
                'drift_type': drift_type,
                'severity': sev_name,
                'mode': 'gradual',
                'params': params.copy(),
            })
            # Uniform comparison
            conditions.append({
                'name': f"{drift_type}_{sev_name}_uniform",
                'drift_type': drift_type,
                'severity': sev_name,
                'mode': 'uniform',
                'params': params.copy(),
            })

    return conditions


# ==========================================
# MODEL RUNNER
# ==========================================

def run_model(model_name, X, scaled_data, sw, file_name=None):
    """Run a single model and return decision scores."""
    if model_name == 'IForest':
        clf = IForest(n_estimators=100, random_state=42)
        clf.fit(X)
        return clf.decision_scores_
    elif model_name == 'PCA':
        n_comp = min(10, X.shape[1] - 1) if X.shape[1] > 1 else 1
        clf = PCA(n_components=n_comp)
        clf.fit(X)
        return clf.decision_scores_
    elif model_name == 'MP':
        clf = MatrixProfile(window=sw)
        clf.fit(scaled_data)
        return clf.decision_scores_
    elif model_name == 'LOF':
        n_neigh = min(20, len(X) - 1)
        clf = LOF(n_neighbors=n_neigh)
        clf.fit(X)
        return clf.decision_scores_
    elif model_name == 'AE':
        ae_clf, meta = load_pretrained_ae(file_name, project_root)
        ae_clf.predict(scaled_data)
        return ae_clf.decision_scores_
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==========================================
# HALF-SERIES EVALUATION
# ==========================================

def evaluate_halves(full_score, labels, sliding_window):
    """Evaluate AUC-ROC on first half and second half of the series separately.

    Uses sklearn's roc_auc_score directly (simpler, more robust for sub-series).
    Only meaningful if both halves contain at least 1 anomaly and 1 normal point.
    Returns (auc_first_half, auc_second_half) or (None, None) if not computable.
    """
    from sklearn.metrics import roc_auc_score

    n = len(labels)
    mid = n // 2

    results = {}
    for half_name, start, end in [('first_half', 0, mid), ('second_half', mid, n)]:
        h_labels = labels[start:end]
        h_scores = full_score[start:end]

        # Need at least 1 anomaly and 1 normal point to compute AUC
        if h_labels.sum() < 1 or (h_labels == 0).sum() < 1:
            results[half_name] = None
            continue

        try:
            auc = roc_auc_score(h_labels, h_scores)
            results[half_name] = round(auc, 4)
        except Exception:
            results[half_name] = None

    return results.get('first_half'), results.get('second_half')


def evaluate_windows(full_score, labels, n_windows=N_WINDOWS):
    """Evaluate AUC-ROC in consecutive temporal windows (deciles by default).

    Splits the series into *n_windows* equal-length segments and computes
    AUC-ROC for each.  Returns a dict ``{0: auc_w0, 1: auc_w1, ...}``.
    Windows where AUC cannot be computed (too few anomalies / normals)
    are set to None.
    """
    from sklearn.metrics import roc_auc_score

    n = len(labels)
    boundaries = np.linspace(0, n, n_windows + 1, dtype=int)
    window_aucs = {}

    for i in range(n_windows):
        start, end = boundaries[i], boundaries[i + 1]
        w_labels = labels[start:end]
        w_scores = full_score[start:end]

        if w_labels.sum() < 1 or (w_labels == 0).sum() < 1:
            window_aucs[i] = None
            continue

        try:
            window_aucs[i] = round(roc_auc_score(w_labels, w_scores), 4)
        except Exception:
            window_aucs[i] = None

    return window_aucs


# ==========================================
# CORE WORKER
# ==========================================

def process_single_job(job_args):
    """Process one (file, condition, seed) -> results for all requested models."""
    (file_path, condition, seed, model_names) = job_args

    # Suppress TF logs in worker
    if 'AE' in model_names:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

    file_name = os.path.basename(file_path)
    condition_name = condition['name']
    drift_type = condition['drift_type']
    mode = condition['mode']
    params = condition['params']

    try:
        # 1. Load data
        df, file_name = load_tsb_dataframe(file_path)
        labels = df['is_anomaly'].to_numpy('int')
        data = df['value'].to_numpy('float')
        n = len(data)

        # 2. Sliding window from clean data
        clean_scaled = StandardScaler().fit_transform(
            data.reshape(-1, 1)
        ).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # 3. Apply drift
        rng = np.random.default_rng(seed)
        drift_fn = DRIFT_FUNCTIONS[(drift_type, mode)]

        # Filter out 'rng' parameter — pass only what the function needs
        fn_params = {k: v for k, v in params.items()}
        corrupted_data = drift_fn(data, **fn_params, rng=rng)

        # 4. Preprocessing
        # We scale using the CORRUPTED data's own statistics (StandardScaler),
        # consistent with all other experiments in this project.
        # Note: for mean_drift, StandardScaler will partially absorb the
        # offset (since it re-centers). This is realistic: in practice,
        # z-score normalization IS applied before AD, so a slow drift that
        # z-score absorbs is genuinely less harmful. The experiment measures
        # the NET effect after standard preprocessing.
        scaled_data = StandardScaler().fit_transform(
            corrupted_data.reshape(-1, 1)
        ).flatten()
        sw = sliding_window
        X = Window(window=sw).convert(scaled_data).to_numpy()

        # 5. Run models
        results = []
        for model_name in model_names:
            try:
                score = run_model(model_name, X, scaled_data, sw, file_name=file_name)
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                    score.reshape(-1, 1)).ravel()

                padded_score = np.array(
                    [score[0]] * math.ceil((sw - 1) / 2) +
                    list(score) +
                    [score[-1]] * ((sw - 1) // 2)
                )

                if np.isnan(padded_score).any() or len(np.unique(padded_score)) <= 1:
                    results.append(_build_row(
                        file_name, condition, n, seed, model_name,
                        error="Invalid scores"))
                    continue

                # Full-series metrics
                metrics = get_metrics(padded_score, labels,
                                      metric="all", slidingWindow=sw)

                # Half-series metrics (the key scientific output)
                auc_first, auc_second = evaluate_halves(
                    padded_score, labels, sw)

                # Tipping-point: per-window AUC-ROC
                window_aucs = evaluate_windows(padded_score, labels)

                row = _build_row(
                    file_name, condition, n, seed, model_name, error=None)
                for key in SELECTED_METRICS:
                    row[key] = round(metrics.get(key, 0.0), 4)
                row['AUC_ROC_first_half'] = auc_first
                row['AUC_ROC_second_half'] = auc_second
                for wi, auc_w in window_aucs.items():
                    row[f'AUC_ROC_w{wi}'] = auc_w
                results.append(row)

            except Exception as e:
                results.append(_build_row(
                    file_name, condition, n, seed, model_name,
                    error=str(e)))

        return {'status': 'success', 'results': results}

    except Exception as e:
        return {'status': 'error', 'file': file_name, 'error': traceback.format_exc()}


def _build_row(file_name, condition, n, seed, model_name, error=None):
    """Build a result row dict."""
    return {
        'file': file_name,
        'condition': condition['name'],
        'drift_type': condition['drift_type'],
        'severity': condition['severity'],
        'mode': condition['mode'],
        'n_points': n,
        'seed': seed,
        'model': model_name,
        'error': error,
        # Drift-specific params (for analysis)
        **{f'param_{k}': v for k, v in condition['params'].items()},
    }


# ==========================================
# SUMMARY & COMPARISON ANALYSIS
# ==========================================

def compute_summary(df_results, output_path):
    """Condition-level aggregated metrics."""
    df_ok = df_results[df_results['error'].isnull()]
    if df_ok.empty:
        print("No successful runs to summarize.")
        return

    grouped = df_ok.groupby(['condition', 'drift_type', 'severity', 'mode', 'model'])
    rows = []
    for name, group in grouped:
        cond, dtype, sev, mode, model = name
        row = {'condition': cond, 'drift_type': dtype, 'severity': sev,
               'mode': mode, 'model': model, 'n': len(group)}
        for m in SELECTED_METRICS:
            if m in group.columns:
                row[f'mean_{m}'] = round(group[m].mean(), 4)
                row[f'std_{m}'] = round(group[m].std(), 4)
        # Half-series means (excluding None)
        for half in ['AUC_ROC_first_half', 'AUC_ROC_second_half']:
            valid = group[half].dropna()
            if len(valid) > 0:
                row[f'mean_{half}'] = round(valid.mean(), 4)
                row[f'n_{half}'] = len(valid)
            else:
                row[f'mean_{half}'] = None
                row[f'n_{half}'] = 0
        # Per-window means (tipping-point)
        for wi in range(N_WINDOWS):
            col = f'AUC_ROC_w{wi}'
            if col in group.columns:
                valid = group[col].dropna()
                row[f'mean_{col}'] = round(valid.mean(), 4) if len(valid) > 0 else None
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


def compute_drift_comparison(df_results, output_dir):
    """Compare gradual vs uniform for each drift_type × severity × model.

    For each pair:
      - AUC_gradual vs AUC_uniform
      - delta = AUC_gradual - AUC_uniform
        (negative = gradual is worse = drift is more insidious)
      - Half-series degradation = AUC_first_half - AUC_second_half
        (positive = second half is worse, as expected under drift)
    """
    df_ok = df_results[df_results['error'].isnull()]
    if df_ok.empty:
        return

    mean_auc = df_ok.groupby(['drift_type', 'severity', 'mode', 'model'])[
        'AUC_ROC'].mean()

    # Half-series means
    half_cols = ['AUC_ROC_first_half', 'AUC_ROC_second_half']
    mean_halves = df_ok.groupby(['drift_type', 'severity', 'mode', 'model'])[
        half_cols].mean()

    rows = []
    for drift_type in DRIFT_CONFIGS:
        for sev in DRIFT_CONFIGS[drift_type]:
            for model in df_ok['model'].unique():
                key_g = (drift_type, sev, 'gradual', model)
                key_u = (drift_type, sev, 'uniform', model)

                auc_g = mean_auc.get(key_g)
                auc_u = mean_auc.get(key_u)
                if auc_g is None or auc_u is None:
                    continue

                # Half-series for gradual
                h1_g = mean_halves.loc[key_g, 'AUC_ROC_first_half'] if key_g in mean_halves.index else None
                h2_g = mean_halves.loc[key_g, 'AUC_ROC_second_half'] if key_g in mean_halves.index else None

                half_degradation = None
                if h1_g is not None and h2_g is not None:
                    if not (np.isnan(h1_g) or np.isnan(h2_g)):
                        half_degradation = round(h1_g - h2_g, 4)

                delta = auc_g - auc_u
                rows.append({
                    'drift_type': drift_type,
                    'severity': sev,
                    'model': model,
                    'AUC_gradual': round(auc_g, 4),
                    'AUC_uniform': round(auc_u, 4),
                    'delta_gradual_minus_uniform': round(delta, 4),
                    'gradual_worse': delta < -0.005,
                    'AUC_first_half_gradual': round(h1_g, 4) if h1_g is not None and not np.isnan(h1_g) else None,
                    'AUC_second_half_gradual': round(h2_g, 4) if h2_g is not None and not np.isnan(h2_g) else None,
                    'half_degradation': half_degradation,
                })

    if rows:
        df_comp = pd.DataFrame(rows)
        out = os.path.join(output_dir, "drift_comparison.csv")
        df_comp.to_csv(out, index=False)
        print(f"Drift comparison: {len(df_comp)} rows -> {out}")

        # Print summary
        print("\n=== Gradual vs Uniform Summary ===")
        print(f"{'Drift Type':<22} {'Severity':<8} {'Model':<10} "
              f"{'Gradual':>8} {'Uniform':>8} {'Delta':>8} {'Half-deg':>9}")
        print("-" * 85)
        for _, r in df_comp.iterrows():
            hd = f"{r['half_degradation']:>9.4f}" if r['half_degradation'] is not None else "     N/A"
            flag = " *" if r['gradual_worse'] else ""
            print(f"{r['drift_type']:<22} {r['severity']:<8} {r['model']:<10} "
                  f"{r['AUC_gradual']:>8.4f} {r['AUC_uniform']:>8.4f} "
                  f"{r['delta_gradual_minus_uniform']:>+8.4f}{flag} {hd}")

        # Count how often gradual is worse
        n_worse = df_comp['gradual_worse'].sum()
        n_total = len(df_comp)
        print(f"\nGradual worse than uniform: {n_worse}/{n_total} "
              f"({100*n_worse/n_total:.0f}%)")


def compute_tipping_point(df_results, output_dir):
    """Tipping-point analysis: mean AUC-ROC per temporal window.

    For each (drift_type, severity, mode, model) produces a degradation
    curve showing how performance evolves across the N_WINDOWS deciles
    of the time series.  Saves a long-format CSV ready for plotting.
    """
    df_ok = df_results[df_results['error'].isnull()]
    if df_ok.empty:
        return

    window_cols = [f'AUC_ROC_w{i}' for i in range(N_WINDOWS)]
    # Skip if window columns are missing (old checkpoint data)
    if not any(c in df_ok.columns for c in window_cols):
        print("[Tipping point] No window columns found — skipping.")
        return

    group_keys = ['drift_type', 'severity', 'mode', 'model']
    rows = []
    for name, group in df_ok.groupby(group_keys):
        dtype, sev, mode, model = name
        for wi in range(N_WINDOWS):
            col = f'AUC_ROC_w{wi}'
            if col not in group.columns:
                continue
            valid = group[col].dropna()
            if len(valid) == 0:
                continue
            rows.append({
                'drift_type': dtype,
                'severity': sev,
                'mode': mode,
                'model': model,
                'window': wi,
                'window_pct': round((wi + 0.5) / N_WINDOWS * 100, 1),  # midpoint %
                'mean_AUC_ROC': round(valid.mean(), 4),
                'std_AUC_ROC': round(valid.std(), 4),
                'n_valid': len(valid),
            })

    if not rows:
        return

    df_tp = pd.DataFrame(rows)
    out = os.path.join(output_dir, "tipping_point.csv")
    df_tp.to_csv(out, index=False)
    print(f"Tipping point: {len(df_tp)} rows -> {out}")

    # Print compact summary for gradual conditions only
    print("\n=== Tipping Point (gradual only, AUC-ROC per decile) ===")
    df_grad = df_tp[df_tp['mode'] == 'gradual']
    for (dtype, sev, model), grp in df_grad.groupby(['drift_type', 'severity', 'model']):
        grp = grp.sort_values('window')
        vals = grp['mean_AUC_ROC'].tolist()
        curve = " -> ".join(f"{v:.3f}" for v in vals)
        drop = vals[0] - vals[-1] if len(vals) >= 2 else 0
        print(f"  {dtype:<20} {sev:<8} {model:<8} | {curve}  (drop={drop:+.4f})")


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Gradual Drift Experiment")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--models', nargs='+', default=['IForest', 'LOF', 'MP', 'AE'],
                        choices=['IForest', 'PCA', 'LOF', 'MP', 'AE'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(SUBSET_CSV):
        print(f"[ERROR] Subset CSV not found: {SUBSET_CSV}")
        return

    # Load file list and remap paths for cross-platform compatibility
    df_files = pd.read_csv(SUBSET_CSV)['filepath'].tolist()
    df_files = [remap_filepath(fp, project_root) for fp in df_files]

    # Build conditions
    all_conditions = build_conditions()

    if args.test:
        df_files = df_files[:3]
        # Keep only low severity for testing
        all_conditions = [c for c in all_conditions if c['severity'] == 'low']
        print("!!! TEST MODE !!!")

    # Print experiment info
    n_gradual = sum(1 for c in all_conditions if c['mode'] == 'gradual')
    n_uniform = sum(1 for c in all_conditions if c['mode'] == 'uniform')

    print(f"\n{'=' * 60}")
    print(f"  Gradual Drift Experiment")
    print(f"{'=' * 60}")
    print(f"  Files:       {len(df_files)}")
    print(f"  Models:      {args.models}")
    print(f"  Conditions:  {len(all_conditions)} total")
    print(f"    Gradual:   {n_gradual}")
    print(f"    Uniform:   {n_uniform}")
    print(f"  Total evals: {len(df_files) * len(all_conditions) * len(args.models)}")
    print(f"{'=' * 60}")

    print(f"\n{'Condition':<40} {'Type':<20} {'Severity':<8} {'Mode':<10}")
    print("-" * 78)
    for c in all_conditions:
        print(f"{c['name']:<40} {c['drift_type']:<20} {c['severity']:<8} {c['mode']:<10}")
    print()

    # Checkpoint
    checkpoint_file = os.path.join(RESULTS_DIR, "checkpoint.csv")
    completed_jobs = set()
    all_results = []

    if os.path.exists(checkpoint_file):
        try:
            df_checkpoint = pd.read_csv(checkpoint_file)
            if 'error' in df_checkpoint.columns:
                df_ok = df_checkpoint[df_checkpoint['error'].isna()]
            else:
                df_ok = df_checkpoint
            all_results = df_ok.to_dict('records')
            for r in all_results:
                job_id = f"{r['file']}_{r['condition']}_{r['seed']}_{r['model']}"
                completed_jobs.add(job_id)
            print(f"[Resume] Loaded {len(all_results)} results, {len(completed_jobs)} jobs done")
        except Exception as e:
            print(f"Warning: checkpoint error: {e}")

    # Build jobs
    jobs = []
    for file_path in df_files:
        file_name = os.path.basename(file_path)
        for cond in all_conditions:
            for seed in range(N_SEEDS):
                needed_models = []
                for m in args.models:
                    job_id = f"{file_name}_{cond['name']}_{seed}_{m}"
                    if job_id not in completed_jobs:
                        needed_models.append(m)
                if needed_models:
                    jobs.append((file_path, cond, seed, needed_models))

    print(f"Jobs to run: {len(jobs)}")

    if jobs:
        new_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_job, job): job for job in jobs}

            with tqdm(total=len(jobs), desc="gradual drift") as pbar:
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

    # Summaries
    if all_results:
        df_all = pd.DataFrame(all_results)
        compute_summary(df_all, os.path.join(RESULTS_DIR, "summary.csv"))
        compute_drift_comparison(df_all, RESULTS_DIR)
        compute_tipping_point(df_all, RESULTS_DIR)

        # Per-model raw results in separate folders
        for model_name in df_all['model'].unique():
            model_df = df_all[df_all['model'] == model_name]
            if not model_df.empty:
                out_dir = os.path.join(RESULTS_DIR, model_name)
                os.makedirs(out_dir, exist_ok=True)
                model_df.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)
                print(f"  {model_name}: {len(model_df)} rows -> {out_dir}/raw_results.csv")

    print(f"\n[Done] Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
