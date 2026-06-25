"""
Compound Corruption Experiment — Multiple Simultaneous Data Quality Issues

Tests how COMBINATIONS of corruptions affect anomaly detection, compared
to each corruption alone. Computes interaction effects to determine if
compound damage is additive, synergistic, or sub-additive.

Combinations tested:
  1. Noise + Missing         (noisy sensor + transmission loss)
  2. Noise + Spikes          (sensor degradation + transient outliers)
  3. Spikes + Missing        (outliers + data gaps)
  4. Missing + Freeze        (data loss + stuck sensor)
  5. Noise + Spikes + Missing (worst realistic — triple)
  6. Noise + GE Missing      (noise + bursty missing via Gilbert-Elliott)

Application order (physically motivated):
  freeze → noise → spikes → missing

Evaluation:
  - If missing involved: true impact (mask NaN, score=0 for lost anomalies)
  - Otherwise: standard evaluation on full corrupted series

Usage:
    python run_compound_corruptions.py --models IForest
    python run_compound_corruptions.py --models IForest MP PCA LOF --workers 6
    python run_compound_corruptions.py --test
"""
import os
import sys
import math
import argparse
import itertools

os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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

from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors
from data_loader import load_tsb_dataframe, remap_filepath, load_pretrained_ae

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(project_root, "results", "experiments", "compound_corruptions")
N_SEEDS = 1

SELECTED_METRICS = [
    'AUC_ROC', 'AUC_PR', 'Precision', 'Recall', 'F',
    'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR',
    'Affiliation_Precision', 'Affiliation_Recall'
]

# Severity levels: 3 per corruption type (low, medium, high)
SEVERITY = {
    'noise':   {'low': {'snr_db': 20},
                'med': {'snr_db': 10},
                'high': {'snr_db': 5},
                'extreme': {'snr_db': 0}},
    'missing': {'low': {'fraction': 0.05},
                'med': {'fraction': 0.10},
                'high': {'fraction': 0.20}},
    'spikes':  {'low': {'fraction': 0.05, 'multiplier': 3, 'sequential': False, 'sequence_length': 1},
                'med': {'fraction': 0.10, 'multiplier': 3, 'sequential': False, 'sequence_length': 1},
                'high': {'fraction': 0.10, 'multiplier': 10, 'sequential': False, 'sequence_length': 1}},
    'freeze':  {'low': {'num_stucks': 3, 'freeze_fraction': 0.05},
                'med': {'num_stucks': 5, 'freeze_fraction': 0.10},
                'high': {'num_stucks': 5, 'freeze_fraction': 0.20}},
    'ge_missing': {'low': {'alpha': 0.01, 'beta': 0.25},    # ~3.8% missing, burst ~4
                   'med': {'alpha': 0.01, 'beta': 0.10},    # ~9.1% missing, burst ~10
                   'high': {'alpha': 0.05, 'beta': 0.10}},  # ~33% missing, burst ~10
}

# Combinations to test
COMBINATIONS = [
    ('noise_missing',        ['noise', 'missing']),
    ('noise_spikes',         ['noise', 'spikes']),
    ('spikes_missing',       ['spikes', 'missing']),
    ('missing_freeze',       ['missing', 'freeze']),
    ('noise_spikes_missing', ['noise', 'spikes', 'missing']),
    ('noise_ge_missing',     ['noise', 'ge_missing']),
]

# Physical order: sensor sticks → noise added → spikes occur → data lost
APPLICATION_ORDER = {'freeze': 0, 'noise': 1, 'spikes': 2, 'missing': 3, 'ge_missing': 3}


# ==========================================
# CONDITION BUILDER
# ==========================================

def build_conditions():
    """Build compound conditions only. Singles/baseline are imported from existing experiments."""
    conditions = []

    for combo_name, corruption_types in COMBINATIONS:
        # Per-type severity grid: noise now has 4 levels (extreme = 0 dB), others 3.
        per_type_levels = [list(SEVERITY[ct].keys()) for ct in corruption_types]

        # Compound: all severity crosses
        for sev_combo in itertools.product(*per_type_levels):
            corr_list = []
            name_parts = []
            has_missing = False

            for ctype, sev in zip(corruption_types, sev_combo):
                corr_list.append({'type': ctype, 'severity': sev,
                                  'params': SEVERITY[ctype][sev].copy()})
                name_parts.append(f"{ctype}_{sev}")
                if ctype in ('missing', 'ge_missing'):
                    has_missing = True

            conditions.append({
                'name': '+'.join(name_parts),
                'combination_name': combo_name,
                'condition_type': 'compound',
                'corruptions': corr_list,
                'has_missing': has_missing,
                'single_tag': None,
            })

    return conditions


# ==========================================
# CORRUPTION APPLICATION
# ==========================================

def apply_corruptions(corruptor, condition, series_length):
    """Apply corruptions in canonical physical order."""
    corruptions = sorted(condition['corruptions'],
                         key=lambda c: APPLICATION_ORDER[c['type']])

    for corr in corruptions:
        ctype = corr['type']
        params = corr['params'].copy()

        if ctype == 'noise':
            ts_corruptor.injectors.inject_white_noise_snr(corruptor, **params)
        elif ctype == 'missing':
            ts_corruptor.injectors.inject_point_missing(corruptor, **params)
        elif ctype == 'spikes':
            ts_corruptor.injectors.inject_spikes(corruptor, **params)
        elif ctype == 'freeze':
            freeze_frac = params.pop('freeze_fraction')
            num_stucks = params['num_stucks']
            stuck_length = max(1, int(freeze_frac * series_length / num_stucks))
            ts_corruptor.injectors.inject_sensor_stuck(
                corruptor, num_stucks=num_stucks, stuck_length=stuck_length)
        elif ctype == 'ge_missing':
            ts_corruptor.injectors.inject_gilbert_elliott(
                corruptor, p_good_to_bad=params['alpha'],
                p_bad_to_good=params['beta'], noise_type='missing')


# ==========================================
# MODEL RUNNER
# ==========================================

def run_model(model_name, X, scaled_data, sw, file_name=None):
    """Run a single model and return decision scores."""
    if model_name == 'IForest':
        clf = IForest(n_estimators=100, random_state=42)
        clf.fit(X)
        return clf.decision_scores_, False
    elif model_name == 'PCA':
        n_comp = min(10, X.shape[1] - 1) if X.shape[1] > 1 else 1
        clf = PCA(n_components=n_comp)
        clf.fit(X)
        return clf.decision_scores_, False
    elif model_name == 'MP':
        clf = MatrixProfile(window=sw)
        clf.fit(scaled_data)
        return clf.decision_scores_, False
    elif model_name == 'LOF':
        n_neigh = min(20, len(X) - 1)
        clf = LOF(n_neighbors=n_neigh)
        clf.fit(X)
        return clf.decision_scores_, False
    elif model_name == 'AE':
        clf, meta = load_pretrained_ae(file_name, project_root)
        clf.predict(scaled_data)
        return clf.decision_scores_, True  # True = already full length
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==========================================
# CORE WORKER
# ==========================================

def process_single_job(job_args):
    """Process one (file, condition, seed) → results for all requested models."""
    (file_path, condition, seed, model_names) = job_args

    file_name = os.path.basename(file_path)
    condition_name = condition['name']

    try:
        # 1. Load data
        df, file_name = load_tsb_dataframe(file_path)
        labels = df['is_anomaly'].to_numpy('int')
        n = len(df)

        # 2. Sliding window from clean data
        clean_scaled = StandardScaler().fit_transform(
            df['value'].to_numpy('float').reshape(-1, 1)
        ).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # 3. Apply corruptions
        corruptor = TSCorruptor(df, value_col='value', label_col='is_anomaly', seed=seed)
        apply_corruptions(corruptor, condition, n)

        df_corrupted = corruptor.get_corrupted_df()
        corrupted_data = df_corrupted['value'].to_numpy('float')

        # 4. Handle missing data
        nan_mask = np.isnan(corrupted_data)
        nan_count = int(nan_mask.sum())
        actual_missing_rate = nan_count / n

        masked_normal = nan_mask & (labels == 0)
        masked_anomaly = nan_mask & (labels == 1)
        n_lost_anomalies = int(masked_anomaly.sum())

        has_missing = nan_count > 0

        if has_missing:
            model_data = corrupted_data[~nan_mask]
            n_kept = len(model_data)
            if n_kept < sliding_window + 10:
                return {'status': 'skipped', 'file': file_name,
                        'reason': f'Too few points after masking: {n_kept}'}
            sw = max(min(sliding_window, n_kept // 4), 10)
        else:
            model_data = corrupted_data
            n_kept = n
            sw = sliding_window

        # 5. Preprocessing
        scaled_data = StandardScaler().fit_transform(
            model_data.reshape(-1, 1)
        ).flatten()
        X = Window(window=sw).convert(scaled_data).to_numpy()

        # 6. Extract corruption params for result columns
        noise_snr = None
        missing_frac = 0.0
        spikes_frac = 0.0
        spikes_mult = 0.0
        freeze_nstucks = 0
        freeze_frac = 0.0
        ge_alpha = None
        ge_beta = None
        for corr in condition['corruptions']:
            if corr['type'] == 'noise':
                noise_snr = corr['params'].get('snr_db')
            elif corr['type'] == 'missing':
                missing_frac = corr['params'].get('fraction', 0.0)
            elif corr['type'] == 'spikes':
                spikes_frac = corr['params'].get('fraction', 0.0)
                spikes_mult = corr['params'].get('multiplier', 0.0)
            elif corr['type'] == 'freeze':
                freeze_nstucks = corr['params'].get('num_stucks', 0)
                freeze_frac = corr['params'].get('freeze_fraction', 0.0)
            elif corr['type'] == 'ge_missing':
                ge_alpha = corr['params'].get('alpha')
                ge_beta = corr['params'].get('beta')

        # 7. Model evaluation
        results = []
        for model_name in model_names:
            try:
                score, is_full_length = run_model(model_name, X, scaled_data, sw, file_name)
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                    score.reshape(-1, 1)).ravel()

                if is_full_length:
                    padded_score = score
                else:
                    padded_score = np.array(
                        [score[0]] * math.ceil((sw - 1) / 2) +
                        list(score) +
                        [score[-1]] * ((sw - 1) // 2)
                    )

                if np.isnan(padded_score).any() or len(np.unique(padded_score)) <= 1:
                    results.append(_build_row(
                        file_name, condition, noise_snr, missing_frac,
                        spikes_frac, spikes_mult, freeze_nstucks, freeze_frac,
                        ge_alpha, ge_beta,
                        actual_missing_rate, n_lost_anomalies, n, n_kept,
                        seed, model_name, error="Invalid scores"))
                    continue

                if has_missing:
                    # True impact evaluation
                    full_score = np.full(n, np.nan)
                    full_score[~nan_mask] = padded_score
                    full_score[masked_anomaly] = 0.0
                    eval_mask = ~masked_normal
                    eval_scores = full_score[eval_mask]
                    eval_labels = labels[eval_mask]
                else:
                    eval_scores = padded_score
                    eval_labels = labels

                metrics = get_metrics(eval_scores, eval_labels,
                                      metric="all", slidingWindow=sw)

                row = _build_row(
                    file_name, condition, noise_snr, missing_frac,
                    spikes_frac, spikes_mult, freeze_nstucks, freeze_frac,
                    ge_alpha, ge_beta,
                    actual_missing_rate, n_lost_anomalies, n, n_kept,
                    seed, model_name, error=None)
                for key in SELECTED_METRICS:
                    row[key] = round(metrics.get(key, 0.0), 4)
                results.append(row)

            except Exception as e:
                results.append(_build_row(
                    file_name, condition, noise_snr, missing_frac,
                    spikes_frac, spikes_mult, freeze_nstucks, freeze_frac,
                    ge_alpha, ge_beta,
                    actual_missing_rate, n_lost_anomalies, n, n_kept,
                    seed, model_name, error=str(e)))

        return {'status': 'success', 'results': results}

    except Exception as e:
        return {'status': 'error', 'file': file_name, 'error': traceback.format_exc()}


def _build_row(file_name, condition, noise_snr, missing_frac,
               spikes_frac, spikes_mult, freeze_nstucks, freeze_frac,
               ge_alpha, ge_beta,
               actual_missing_rate, n_lost_anomalies, n, n_kept,
               seed, model_name, error=None):
    """Build a result row dict."""
    return {
        'file': file_name,
        'condition': condition['name'],
        'combination_name': condition['combination_name'],
        'condition_type': condition['condition_type'],
        'noise_snr_db': noise_snr,
        'missing_fraction': missing_frac,
        'spikes_fraction': spikes_frac,
        'spikes_multiplier': spikes_mult,
        'freeze_num_stucks': freeze_nstucks,
        'freeze_fraction': freeze_frac,
        'ge_alpha': ge_alpha,
        'ge_beta': ge_beta,
        'has_missing': condition['has_missing'],
        'actual_missing_rate': round(actual_missing_rate, 4),
        'n_lost_anomalies': n_lost_anomalies,
        'n_original': n,
        'n_kept': n_kept,
        'seed': seed,
        'model': model_name,
        'error': error,
    }


# ==========================================
# SUMMARY & INTERACTION ANALYSIS
# ==========================================

def compute_summary(df_results, output_path):
    """Compute condition-level aggregated metrics."""
    df_ok = df_results[df_results['error'].isnull()]
    if df_ok.empty:
        print("No successful runs to summarize.")
        return

    grouped = df_ok.groupby(['condition', 'combination_name', 'condition_type', 'model'])
    rows = []
    for name, group in grouped:
        cond, combo, ctype, model = name
        row = {'condition': cond, 'combination_name': combo,
               'condition_type': ctype, 'model': model, 'n': len(group)}
        for m in SELECTED_METRICS:
            if m in group.columns:
                row[f'mean_{m}'] = round(group[m].mean(), 4)
                row[f'std_{m}'] = round(group[m].std(), 4)
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


def compute_interaction_analysis(df_results, output_dir):
    """Compute interaction effects for compound corruptions."""
    df_ok = df_results[df_results['error'].isnull()]
    if df_ok.empty:
        return

    # Build single_tag for lookup
    all_conditions = build_conditions()
    tag_map = {}  # condition_name → single_tag
    for c in all_conditions:
        if c['single_tag']:
            tag_map[c['name']] = c['single_tag']

    # Per-file means by (condition, model)
    per_file = df_ok.groupby(['condition', 'model'])['AUC_ROC'].mean().to_dict()

    # Baseline AUC per model
    baseline = {}
    for model in df_ok['model'].unique():
        key = ('clean', model)
        if key in per_file:
            baseline[model] = per_file[key]

    if not baseline:
        print("Warning: no baseline results found for interaction analysis")
        return

    # Build interaction rows
    interaction_rows = []

    for combo_name, corruption_types in COMBINATIONS:
        if len(corruption_types) == 2:
            _compute_pair_interactions(
                corruption_types, combo_name, per_file, baseline,
                df_ok['model'].unique(), interaction_rows)
        elif len(corruption_types) == 3:
            _compute_triple_interactions(
                corruption_types, combo_name, per_file, baseline,
                df_ok['model'].unique(), interaction_rows)

    if interaction_rows:
        df_inter = pd.DataFrame(interaction_rows)
        df_inter.to_csv(os.path.join(output_dir, "interaction_analysis.csv"), index=False)
        print(f"Interaction analysis: {len(df_inter)} rows")

        # High-level summary: mean interaction per combination
        summary = df_inter.groupby(['combination_name', 'model']).agg(
            mean_interaction=('interaction', 'mean'),
            mean_interaction_pct=('interaction_pct', lambda x: x.mean()),
            n_synergistic=('interaction_type', lambda x: (x == 'synergistic').sum()),
            n_additive=('interaction_type', lambda x: (x == 'additive').sum()),
            n_subadditive=('interaction_type', lambda x: (x == 'sub-additive').sum()),
        ).reset_index()
        summary.to_csv(os.path.join(output_dir, "interaction_summary.csv"), index=False)
        print(f"Interaction summary saved.")


def _compute_pair_interactions(corruption_types, combo_name, per_file, baseline,
                               models, rows):
    """Compute interaction for a 2-corruption combination."""
    type_a, type_b = corruption_types

    for sev_a, sev_b in itertools.product(['low', 'med', 'high'], repeat=2):
        tag_a = f"{type_a}_{sev_a}"
        tag_b = f"{type_b}_{sev_b}"
        compound_name = f"{tag_a}+{tag_b}"
        single_a_name = f"{tag_a}_only"
        single_b_name = f"{tag_b}_only"

        for model in models:
            bl = baseline.get(model)
            auc_a = per_file.get((single_a_name, model))
            auc_b = per_file.get((single_b_name, model))
            auc_ab = per_file.get((compound_name, model))

            if any(v is None for v in [bl, auc_a, auc_b, auc_ab]):
                continue

            drop_a = bl - auc_a
            drop_b = bl - auc_b
            drop_ab = bl - auc_ab
            predicted = drop_a + drop_b
            interaction = drop_ab - predicted
            interaction_pct = (interaction / predicted * 100) if abs(predicted) > 1e-6 else 0.0

            if interaction > 0.01:
                itype = 'synergistic'
            elif interaction < -0.01:
                itype = 'sub-additive'
            else:
                itype = 'additive'

            rows.append({
                'combination_name': combo_name,
                'severity_A': f"{type_a}_{sev_a}",
                'severity_B': f"{type_b}_{sev_b}",
                'model': model,
                'baseline_auc': round(bl, 4),
                'auc_A': round(auc_a, 4),
                'auc_B': round(auc_b, 4),
                'auc_compound': round(auc_ab, 4),
                'drop_A': round(drop_a, 4),
                'drop_B': round(drop_b, 4),
                'drop_compound': round(drop_ab, 4),
                'predicted_additive': round(predicted, 4),
                'interaction': round(interaction, 4),
                'interaction_pct': round(interaction_pct, 1),
                'interaction_type': itype,
            })


def _compute_triple_interactions(corruption_types, combo_name, per_file, baseline,
                                  models, rows):
    """Compute interaction for a 3-corruption combination."""
    type_a, type_b, type_c = corruption_types

    for sev_a, sev_b, sev_c in itertools.product(['low', 'med', 'high'], repeat=3):
        tag_a = f"{type_a}_{sev_a}"
        tag_b = f"{type_b}_{sev_b}"
        tag_c = f"{type_c}_{sev_c}"
        compound_name = f"{tag_a}+{tag_b}+{tag_c}"
        single_a_name = f"{tag_a}_only"
        single_b_name = f"{tag_b}_only"
        single_c_name = f"{tag_c}_only"

        for model in models:
            bl = baseline.get(model)
            auc_a = per_file.get((single_a_name, model))
            auc_b = per_file.get((single_b_name, model))
            auc_c = per_file.get((single_c_name, model))
            auc_abc = per_file.get((compound_name, model))

            if any(v is None for v in [bl, auc_a, auc_b, auc_c, auc_abc]):
                continue

            drop_a = bl - auc_a
            drop_b = bl - auc_b
            drop_c = bl - auc_c
            drop_abc = bl - auc_abc
            predicted = drop_a + drop_b + drop_c
            interaction = drop_abc - predicted
            interaction_pct = (interaction / predicted * 100) if abs(predicted) > 1e-6 else 0.0

            if interaction > 0.01:
                itype = 'synergistic'
            elif interaction < -0.01:
                itype = 'sub-additive'
            else:
                itype = 'additive'

            rows.append({
                'combination_name': combo_name,
                'severity_A': f"{type_a}_{sev_a}",
                'severity_B': f"{type_b}_{sev_b}",
                'severity_C': f"{type_c}_{sev_c}",
                'model': model,
                'baseline_auc': round(bl, 4),
                'auc_A': round(auc_a, 4),
                'auc_B': round(auc_b, 4),
                'auc_C': round(auc_c, 4),
                'auc_compound': round(auc_abc, 4),
                'drop_A': round(drop_a, 4),
                'drop_B': round(drop_b, 4),
                'drop_C': round(drop_c, 4),
                'drop_compound': round(drop_abc, 4),
                'predicted_additive': round(predicted, 4),
                'interaction': round(interaction, 4),
                'interaction_pct': round(interaction_pct, 1),
                'interaction_type': itype,
            })


# ==========================================
# SHAPLEY VALUE ABLATION ANALYSIS
# ==========================================

def compute_shapley_analysis(df_results, output_dir):
    """Shapley value analysis for the triple corruption (noise+spikes+missing).

    For each corruption type, computes its average marginal contribution to
    total damage across all possible coalitions.  This answers: "which
    corruption should you fix FIRST for maximum AUC recovery?"

    Requires: baseline, all 3 singles, all 3 pairs, and the triple.
    """
    df_ok = df_results[df_results['error'].isnull()]
    if df_ok.empty:
        return

    # Mean AUC per (condition, model)
    mean_auc = df_ok.groupby(['condition', 'model'])['AUC_ROC'].mean().to_dict()

    # Baseline per model
    baseline = {}
    for model in df_ok['model'].unique():
        bl = mean_auc.get(('clean', model))
        if bl is not None:
            baseline[model] = bl

    if not baseline:
        print("[Shapley] No baseline found — skipping.")
        return

    rows = []
    corruption_types = ['noise', 'spikes', 'missing']
    n_players = len(corruption_types)

    for sev_n, sev_s, sev_m in itertools.product(['low', 'med', 'high'], repeat=3):
        sevs = {'noise': sev_n, 'spikes': sev_s, 'missing': sev_m}

        # Map each coalition to its condition name
        conditions = {
            frozenset():                         'clean',
            frozenset(['noise']):                f'noise_{sev_n}_only',
            frozenset(['spikes']):               f'spikes_{sev_s}_only',
            frozenset(['missing']):              f'missing_{sev_m}_only',
            frozenset(['noise', 'spikes']):      f'noise_{sev_n}+spikes_{sev_s}',
            frozenset(['noise', 'missing']):     f'noise_{sev_n}+missing_{sev_m}',
            frozenset(['spikes', 'missing']):    f'spikes_{sev_s}+missing_{sev_m}',
            frozenset(['noise', 'spikes', 'missing']):
                f'noise_{sev_n}+spikes_{sev_s}+missing_{sev_m}',
        }

        for model in baseline:
            # Collect AUC for every coalition
            aucs = {}
            skip = False
            for coalition, cond_name in conditions.items():
                auc = mean_auc.get((cond_name, model))
                if auc is None:
                    skip = True
                    break
                aucs[coalition] = auc

            if skip:
                continue

            # Damage function: v(S) = baseline - AUC(S)
            v = {s: baseline[model] - aucs[s] for s in aucs}
            total_damage = v[frozenset(corruption_types)]

            # Shapley value for each corruption
            for corr in corruption_types:
                others = [c for c in corruption_types if c != corr]
                shapley = 0.0

                for size in range(len(others) + 1):
                    for subset in itertools.combinations(others, size):
                        s = frozenset(subset)
                        s_with_i = s | {corr}
                        marginal = v[s_with_i] - v[s]
                        weight = (math.factorial(len(s))
                                  * math.factorial(n_players - len(s) - 1)
                                  / math.factorial(n_players))
                        shapley += weight * marginal

                # Recovery: what you gain by removing this corruption from the triple
                triple = frozenset(corruption_types)
                pair_without = triple - {corr}
                recovery = aucs[pair_without] - aucs[triple]

                rows.append({
                    'noise_severity': sev_n,
                    'spikes_severity': sev_s,
                    'missing_severity': sev_m,
                    'model': model,
                    'corruption': corr,
                    'corruption_severity': sevs[corr],
                    'shapley_value': round(shapley, 4),
                    'recovery_if_fixed': round(recovery, 4),
                    'total_damage': round(total_damage, 4),
                    'shapley_pct': (round(shapley / total_damage * 100, 1)
                                    if total_damage > 0.001 else 0.0),
                })

    if not rows:
        print("[Shapley] No complete coalitions found — skipping.")
        return

    df_sh = pd.DataFrame(rows)
    df_sh.to_csv(os.path.join(output_dir, "shapley_ablation.csv"), index=False)
    print(f"Shapley ablation: {len(df_sh)} rows -> {output_dir}/shapley_ablation.csv")

    # Summary: mean Shapley contribution per corruption × model
    summary = df_sh.groupby(['corruption', 'model']).agg(
        mean_shapley=('shapley_value', 'mean'),
        mean_recovery=('recovery_if_fixed', 'mean'),
        mean_pct=('shapley_pct', 'mean'),
    ).reset_index()

    print(f"\n=== Shapley Value Summary (mean contribution to damage) ===")
    print(f"{'Corruption':<12} {'Model':<10} {'Shapley':>9} {'Recovery':>10} {'% damage':>10}")
    print("-" * 55)
    for _, r in summary.sort_values(['model', 'mean_shapley'], ascending=[True, False]).iterrows():
        print(f"{r['corruption']:<12} {r['model']:<10} "
              f"{r['mean_shapley']:>9.4f} {r['mean_recovery']:>+10.4f} "
              f"{r['mean_pct']:>9.1f}%")

    summary.to_csv(os.path.join(output_dir, "shapley_summary.csv"), index=False)
    print(f"Shapley summary -> {output_dir}/shapley_summary.csv")


# ==========================================
# IMPORT EXISTING SINGLE RESULTS
# ==========================================

EXPERIMENTS_DIR = os.path.join(project_root, "results", "experiments")
BASELINE_CSV_PATH = os.path.join(project_root, "results", "tables", "baseline_final_subset.csv")

# Mapping: single_tag -> (checkpoint_file, filter_dict, has_missing)
SINGLE_SOURCES = {
    'noise_low':   ('white_noise_snr/checkpoint.csv',  {'snr_db': 20},  False),
    'noise_med':   ('white_noise_snr/checkpoint.csv',  {'snr_db': 10},  False),
    'noise_high':  ('white_noise_snr/checkpoint.csv',  {'snr_db': 5},   False),
    'missing_low': ('missing_true_impact/checkpoint.csv',
                    {'missing_type': 'point', 'fraction': 0.05}, True),
    'missing_med': ('missing_true_impact/checkpoint.csv',
                    {'missing_type': 'point', 'fraction': 0.10}, True),
    'missing_high':('missing_true_impact/checkpoint.csv',
                    {'missing_type': 'point', 'fraction': 0.20}, True),
    'spikes_low':  ('spikes_no_zscore/checkpoint.csv',
                    {'fraction': 0.05, 'multiplier': 3.0, 'sequential': False}, False),
    'spikes_med':  ('spikes_no_zscore/checkpoint.csv',
                    {'fraction': 0.10, 'multiplier': 3.0, 'sequential': False}, False),
    'spikes_high': ('spikes_no_zscore/checkpoint.csv',
                    {'fraction': 0.10, 'multiplier': 10.0, 'sequential': False}, False),
    'freeze_low':  ('freeze/checkpoint.csv', {'fraction': 0.05, 'num_stucks': 3}, False),
    'freeze_med':  ('freeze/checkpoint.csv', {'fraction': 0.10, 'num_stucks': 5}, False),
    'freeze_high': ('freeze/checkpoint.csv', {'fraction': 0.20, 'num_stucks': 5}, False),
    'ge_missing_low':  ('gilbert_elliott_true_impact/checkpoint.csv',
                        {'alpha': 0.01, 'beta': 0.25}, True),
    'ge_missing_med':  ('gilbert_elliott_true_impact/checkpoint.csv',
                        {'alpha': 0.01, 'beta': 0.10}, True),
    'ge_missing_high': ('gilbert_elliott_true_impact/checkpoint.csv',
                        {'alpha': 0.05, 'beta': 0.10}, True),
}


def load_existing_results():
    """Load baseline + single corruption results from existing experiments.

    Returns list of result dicts compatible with compound checkpoint format.
    """
    imported = []

    # --- Baseline ---
    if os.path.exists(BASELINE_CSV_PATH):
        bl = pd.read_csv(BASELINE_CSV_PATH)
        for _, r in bl.iterrows():
            if pd.notna(r.get('AUC_ROC')):
                row = _build_row(
                    r['file'], {'name': 'clean', 'combination_name': 'baseline',
                                'condition_type': 'baseline', 'corruptions': [],
                                'has_missing': False, 'single_tag': None},
                    None, 0.0, 0.0, 0.0, 0, 0.0,
                    None, None,
                    0.0, 0, len(bl), len(bl),
                    0, r['model'], error=None)
                for m in SELECTED_METRICS:
                    if m in r.index and pd.notna(r[m]):
                        row[m] = round(float(r[m]), 4)
                imported.append(row)
        print(f"  Baseline: {len(imported)} rows imported")

    # --- Singles ---
    for tag, (rel_path, filters, has_missing) in SINGLE_SOURCES.items():
        csv_path = os.path.join(EXPERIMENTS_DIR, rel_path)
        if not os.path.exists(csv_path):
            print(f"  {tag}: MISSING ({rel_path})")
            continue

        try:
            df = pd.read_csv(csv_path)
            # Filter errors
            if 'error' in df.columns:
                df = df[df['error'].isna()]
            # Apply filters
            for col, val in filters.items():
                if col in df.columns:
                    df = df[df[col] == val]

            count = 0
            for _, r in df.iterrows():
                cond_dict = {
                    'name': f"{tag}_only",
                    'combination_name': 'single',
                    'condition_type': 'single',
                    'corruptions': [],  # not needed for imported rows
                    'has_missing': has_missing,
                    'single_tag': tag,
                }
                row = _build_row(
                    r['file'], cond_dict,
                    None, 0.0, 0.0, 0.0, 0, 0.0,
                    None, None,
                    0.0, 0, 0, 0,
                    int(r.get('seed', 0)), r['model'], error=None)
                for m in SELECTED_METRICS:
                    if m in r.index and pd.notna(r[m]):
                        row[m] = round(float(r[m]), 4)
                imported.append(row)
                count += 1
            print(f"  {tag}: {count} rows imported")
        except Exception as e:
            print(f"  {tag}: ERROR ({e})")

    return imported


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Compound Corruption Experiment")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--models', nargs='+', default=['MP'],
                        choices=[ 'MP', ])
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--no-import', action='store_true',
                        help='Skip importing existing results, run everything fresh')
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
        # Keep: clean + 2 singles + 2 compounds
        test_names = {'clean', 'noise_low_only', 'missing_low_only',
                      'noise_low+missing_low', 'noise_high+missing_high'}
        all_conditions = [c for c in all_conditions if c['name'] in test_names]
        print("!!! TEST MODE !!!")

    # Print experiment info
    n_singles = sum(1 for c in all_conditions if c['condition_type'] == 'single')
    n_compounds = sum(1 for c in all_conditions if c['condition_type'] == 'compound')
    n_baseline = sum(1 for c in all_conditions if c['condition_type'] == 'baseline')

    print(f"\n{'=' * 60}")
    print(f"  Compound Corruption Experiment")
    print(f"{'=' * 60}")
    print(f"  Files:      {len(df_files)}")
    print(f"  Models:     {args.models}")
    print(f"  Conditions: {len(all_conditions)} total")
    print(f"    Baseline: {n_baseline}")
    print(f"    Singles:  {n_singles}")
    print(f"    Compounds:{n_compounds}")
    print(f"  Total evals: {len(df_files) * len(all_conditions) * len(args.models)}")
    print(f"  Order: freeze -> noise -> spikes -> missing")
    print(f"{'=' * 60}")

    print(f"\n{'Condition':<45} {'Type':<10} {'Combo':<20}")
    print("-" * 75)
    for c in all_conditions:
        print(f"{c['name']:<45} {c['condition_type']:<10} {c['combination_name']:<20}")
    print()

    # Import existing baseline + single results from other experiments
    if not args.test and not args.no_import:
        print("\nImporting existing results (baseline + singles)...")
        imported_results = load_existing_results()
    else:
        imported_results = []

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

    # Merge imported results (skip duplicates already in checkpoint)
    if imported_results:
        imported_count = 0
        for r in imported_results:
            job_id = f"{r['file']}_{r['condition']}_{r.get('seed', 0)}_{r['model']}"
            if job_id not in completed_jobs:
                all_results.append(r)
                completed_jobs.add(job_id)
                imported_count += 1
        if imported_count:
            print(f"[Import] Added {imported_count} results from existing experiments")

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

            with tqdm(total=len(jobs), desc="compound corruptions") as pbar:
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
        compute_interaction_analysis(df_all, RESULTS_DIR)
        compute_shapley_analysis(df_all, RESULTS_DIR)

        # Per-model raw results
        for model_name in args.models:
            model_df = df_all[df_all['model'] == model_name]
            if not model_df.empty:
                out_dir = os.path.join(RESULTS_DIR, model_name)
                os.makedirs(out_dir, exist_ok=True)
                model_df.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)

    print(f"\n[Done] Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
