"""
Compound Corruptions — Multiple Simultaneous Data Quality Issues, TSB-AD version.

Port of run_compound_corruptions.py to the TSB-AD benchmark, built on the same
skeleton as run_missing_true_impact_tsbad.py / run_freeze_tsbad.py.

Question: when several data quality problems hit the same series at once, is the
damage the SUM of the individual damages, or more (synergistic) / less
(sub-additive)? Answering it needs three things measured under identical
conditions: the clean anchor, each corruption alone, and the combination.

Combinations (identical to the original):
  1. noise + missing          (noisy sensor + transmission loss)
  2. noise + spikes           (sensor degradation + transient outliers)
  3. spikes + missing         (outliers + data gaps)
  4. missing + freeze         (data loss + stuck sensor)
  5. noise + spikes + missing (worst realistic — the triple)
  6. noise + ge_missing       (noise + bursty loss via Gilbert-Elliott)

Application order (physically motivated, byte-identical to the original):
      freeze -> noise -> spikes -> missing
A sensor sticks first, noise rides on whatever it outputs, spikes are transient
events on top, and transmission loss is the last thing that happens to the bytes.

Evaluation — chosen per condition by whether NaNs actually reached the series:
  * NaNs present -> "true impact", no imputation. Drop the NaN points, map the
    scores back onto the original timeline, give lost ANOMALY points the minimum
    score (they count as false negatives), and EXCLUDE lost normal points.
  * No NaNs -> standard evaluation on the full corrupted series.
The branch is decided from the realised nan_mask, not from the declared condition,
so a `missing` level that happens to drop nothing still takes the correct path.

Detection and metrics use the OFFICIAL TSB-AD pipeline (run_Unsupervise_AD /
run_Semisupervise_AD + find_length_rank + TSB_AD get_metrics) on the 350-series
TSB-AD-U eval list. The `clean` anchor runs that pipeline with no corruption at
all, so it reduces exactly to the official TSB-AD baseline.

DELIBERATE DEVIATIONS from the TSB-UAD original — all documented, none accidental:

  1. Singles and the clean anchor are GENERATED IN THIS RUN, not imported from
     other experiments' checkpoints (the original's SINGLE_SOURCES table).
     The interaction effect is a difference of drops; if the singles come from a
     run with a different window, HP or evaluation policy, that difference is
     contaminated by the mismatch rather than measuring interaction. Generating
     everything here makes one command reproduce the whole table and guarantees
     every term shares the same measuring stick. Condition NAMES are unchanged
     (`clean`, `<type>_<sev>_only`, `a+b`), so tables group exactly as before.
  2. No max(window, 10) floor on the metric window, and no MinMax on the scores —
     the same two deviations as every other TSB-AD port here, so the clean anchor
     reproduces the official baseline bit for bit. Rank-based metrics are
     invariant to the dropped MinMax.
  3. Lost anomalies get score.min(), not a literal 0.0. TSB-AD returns
     UNNORMALIZED scores (IForest sits around [-0.06, +0.03] with most points
     below zero), so a hardcoded 0.0 would rank a destroyed anomaly in the top
     ~10% and INFLATE the metrics as more anomalies are lost. score.min() carries
     the original's semantics — tied last with the least anomalous point.
  4. Interaction and Shapley analysis iterate the severity levels actually
     declared in SEVERITY instead of a hardcoded ['low','med','high'], so the
     noise 'extreme' (0 dB) arm is analysed too. This only ADDS rows; every row
     the original would have produced is still produced identically.

Usage:
    python src/experiments/corruption/run_compound_corruptions_tsbad.py --test
    python src/experiments/corruption/run_compound_corruptions_tsbad.py --models IForest --workers 4
    # the grid is large — start with one pair to size the run:
    python src/experiments/corruption/run_compound_corruptions_tsbad.py \
        --models IForest --combinations noise_missing
    # or run on a smaller validated file list:
    python src/experiments/corruption/run_compound_corruptions_tsbad.py \
        --models IForest --files-csv results/tables/representative_subset_tsb_ad_vuspr_n200.csv
"""
import os

# Pin every numeric backend to one thread BEFORE numpy/numba/stumpy are imported.
# We parallelise across files with ProcessPoolExecutor; without this each worker also grabs
# every core (stumpy.stump, used by MatrixProfile, is numba-parallel) and the oversubscription
# slows the run badly. Must stay above the numpy import to take effect.
for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import math
import random
import hashlib
import argparse
import warnings
import itertools
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm


def _silence_warnings():
    """Mute the expected, harmless noise from batch evaluation.

    sklearn raises UndefinedMetricWarning whenever a threshold yields no predicted positives
    (precision = 0/0) — routine once scores flatten, and it only touches the threshold-dependent
    F1s, never AUC/VUS. Called at import time so spawned workers inherit the filters.
    """
    from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rank_warning = getattr(getattr(np, 'exceptions', None), 'RankWarning', None) \
        or getattr(np, 'RankWarning', None)
    if rank_warning is not None:
        warnings.filterwarnings("ignore", category=rank_warning)


_silence_warnings()

# ==========================================
# PATHS
# ==========================================
_CUR = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_CUR))))
TSB_AD_PATH = os.path.join(PROJECT_ROOT, 'TSB-AD')
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
for p in (PROJECT_ROOT, TSB_AD_PATH, SRC_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors  # registers injectors

# ==========================================
# CONFIG
# ==========================================
_ROOT = Path(PROJECT_ROOT)
FILE_LIST_CSV = _ROOT / "TSB-AD" / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
DATA_DIR = _ROOT / "TSB-AD" / "Datasets" / "TSB-AD-U"
RESULTS_DIR = _ROOT / "results" / "experiments" / "compound_corruptions_tsbad"

N_SEEDS = 1

# The original passes no corruption_target, i.e. the default. Corruption may therefore land on
# anomaly points — deliberately: destroying anomalies is part of what this experiment measures.
CORRUPTION_TARGET = 'global'

# Severity levels per corruption type — byte-identical to the original.
# Noise carries a fourth 'extreme' level (0 dB) that the others do not.
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

COMBINATIONS = [
    ('noise_missing',        ['noise', 'missing']),
    ('noise_spikes',         ['noise', 'spikes']),
    ('spikes_missing',       ['spikes', 'missing']),
    ('missing_freeze',       ['missing', 'freeze']),
    ('noise_spikes_missing', ['noise', 'spikes', 'missing']),
    ('noise_ge_missing',     ['noise', 'ge_missing']),
]

# Physical order: sensor sticks -> noise added -> spikes occur -> data lost
APPLICATION_ORDER = {'freeze': 0, 'noise': 1, 'spikes': 2, 'missing': 3, 'ge_missing': 3}

# Corruption types that introduce NaNs (i.e. trigger the true-impact evaluation branch)
MISSING_TYPES = {'missing', 'ge_missing'}

METRIC_COLS = ['AUC_PR', 'AUC_ROC', 'VUS_PR', 'VUS_ROC', 'Standard_F1',
               'PA_F1', 'Event_based_F1', 'R_based_F1', 'Affiliation_F']

SEMISUP = {'AutoEncoder', 'AutoEncoder_2', 'StreamVAE', 'CNN', 'LSTMAD'}

# Models whose window is derived from the data via find_length_rank(periodicity). Under
# corruption the official wrapper would re-estimate it on the CORRUPTED signal; the original
# experiment instead fixed the window from the CLEAN signal. In window_mode='clean' we replicate
# each wrapper body verbatim but inject the clean window.
# (IForest is absent on purpose: it uses a fixed slidingWindow=100, so corruption cannot move it.)
WINDOW_MODELS = {'MatrixProfile', 'POLY', 'Sub_PCA', 'KShapeAD', 'KMeansAD_U'}

LEGACY_CKPT = None  # set in main()


# ==========================================
# CONDITION BUILDER
# ==========================================

def _tag(ctype, sev):
    """Canonical severity tag, e.g. 'noise_high'. Used inside every condition name."""
    return f"{ctype}_{sev}"


def build_conditions(combos=None, include_clean=True, include_singles=True):
    """Build clean anchor + singles + compounds.

    Unlike the original — which built compounds only and imported the rest from other
    experiments' checkpoints — every term of the interaction equation is produced here, under
    one window policy and one evaluation policy. Names are unchanged, so downstream tables and
    the interaction/Shapley code group exactly as before.
    """
    selected = COMBINATIONS if combos is None else [c for c in COMBINATIONS if c[0] in combos]
    conditions = []

    if include_clean:
        conditions.append({
            'name': 'clean',
            'combination_name': 'baseline',
            'condition_type': 'baseline',
            'corruptions': [],
            'has_missing': False,
            'single_tag': None,
        })

    # Singles: every (type, severity) pair that appears anywhere in the selected combinations.
    # Needed for the drop terms — a compound is only interpretable against its own parts.
    if include_singles:
        seen = set()
        for _, ctypes in selected:
            for ctype in ctypes:
                for sev in SEVERITY[ctype]:
                    if (ctype, sev) in seen:
                        continue
                    seen.add((ctype, sev))
                    conditions.append({
                        'name': f"{_tag(ctype, sev)}_only",
                        'combination_name': 'single',
                        'condition_type': 'single',
                        'corruptions': [{'type': ctype, 'severity': sev,
                                         'params': SEVERITY[ctype][sev].copy()}],
                        'has_missing': ctype in MISSING_TYPES,
                        'single_tag': _tag(ctype, sev),
                    })

    # Compounds: the full severity cross-product per combination
    for combo_name, ctypes in selected:
        per_type_levels = [list(SEVERITY[ct].keys()) for ct in ctypes]
        for sev_combo in itertools.product(*per_type_levels):
            corr_list, name_parts, has_missing = [], [], False
            for ctype, sev in zip(ctypes, sev_combo):
                corr_list.append({'type': ctype, 'severity': sev,
                                  'params': SEVERITY[ctype][sev].copy()})
                name_parts.append(_tag(ctype, sev))
                if ctype in MISSING_TYPES:
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
    """Apply every corruption of a condition, in canonical physical order.

    Order is by APPLICATION_ORDER, not by list position, so `noise+missing` and `missing+noise`
    would corrupt identically. Each injector mutates the same corruptor, i.e. later injectors see
    the output of the earlier ones — that stacking IS the compound effect being measured.
    """
    for corr in sorted(condition['corruptions'], key=lambda c: APPLICATION_ORDER[c['type']]):
        ctype = corr['type']
        params = corr['params'].copy()

        if ctype == 'noise':
            ts_corruptor.injectors.inject_white_noise_snr(corruptor, **params)
        elif ctype == 'missing':
            ts_corruptor.injectors.inject_point_missing(corruptor, **params)
        elif ctype == 'spikes':
            ts_corruptor.injectors.inject_spikes(corruptor, **params)
        elif ctype == 'freeze':
            # Same dynamic block length as run_freeze.py / run_freeze_tsbad.py: the requested
            # fraction of the series split evenly across num_stucks frozen blocks.
            freeze_frac = params.pop('freeze_fraction')
            num_stucks = params['num_stucks']
            stuck_length = max(1, int(freeze_frac * series_length / num_stucks))
            ts_corruptor.injectors.inject_sensor_stuck(
                corruptor, num_stucks=num_stucks, stuck_length=stuck_length)
        elif ctype == 'ge_missing':
            ts_corruptor.injectors.inject_gilbert_elliott(
                corruptor, p_good_to_bad=params['alpha'],
                p_bad_to_good=params['beta'], noise_type='missing')
        else:
            raise ValueError(f'unknown corruption type: {ctype}')


def _extract_params(condition):
    """Flatten a condition's corruption parameters into result columns."""
    out = {'noise_snr_db': None, 'missing_fraction': 0.0,
           'spikes_fraction': 0.0, 'spikes_multiplier': 0.0,
           'freeze_num_stucks': 0, 'freeze_fraction': 0.0,
           'ge_alpha': None, 'ge_beta': None}
    for corr in condition['corruptions']:
        p, ctype = corr['params'], corr['type']
        if ctype == 'noise':
            out['noise_snr_db'] = p.get('snr_db')
        elif ctype == 'missing':
            out['missing_fraction'] = p.get('fraction', 0.0)
        elif ctype == 'spikes':
            out['spikes_fraction'] = p.get('fraction', 0.0)
            out['spikes_multiplier'] = p.get('multiplier', 0.0)
        elif ctype == 'freeze':
            out['freeze_num_stucks'] = p.get('num_stucks', 0)
            out['freeze_fraction'] = p.get('freeze_fraction', 0.0)
        elif ctype == 'ge_missing':
            out['ge_alpha'] = p.get('alpha')
            out['ge_beta'] = p.get('beta')
    return out


# ==========================================
# MODELS (official TSB-AD pipeline)
# ==========================================

def _get_hp(model_name):
    from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
    key = 'AutoEncoder' if model_name in ('AutoEncoder', 'AutoEncoder_2') else model_name
    return dict(Optimal_Uni_algo_HP_dict.get(key, {}))


def _model_window(model_name, clean_data, hp, n_kept, clamp):
    """Window for periodicity-derived models, estimated on the CLEAN signal.

    `clamp` is on only when points were actually deleted: a window valid for the full series can
    exceed the survivor count. Same clamp as the original (`min(w, n_kept // 4)`, floor 10), and
    it is reported per row so a firing clamp stays visible.
    """
    from TSB_AD.utils.slidingWindows import find_length_rank
    w = int(find_length_rank(clean_data, rank=hp.get('periodicity', 1)))
    if not clamp:
        return w, False
    w_clamped = max(min(w, n_kept // 4), 10)
    return w_clamped, (w_clamped != w)


def _run_model(model_name, data, clean_data, hp, window_mode, file_name, nan_mask, clamp):
    """Run a TSB-AD detector on whatever points survived.

    `data` is the array the detector sees — the full corrupted series when nothing was dropped,
    the shortened one otherwise. `nan_mask` is over the ORIGINAL timeline and is needed to remap
    the semi-supervised train split.

    window_mode='native' -> pure official wrapper (re-estimates the window on the given data).
    window_mode='clean'  -> same model/HP, but periodicity-derived models use the window
                            estimated from the clean signal (matches the original experiment).
    On the clean condition both modes coincide exactly.

    Returns (scores, model_window, window_clamped).
    """
    from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD

    n_kept = len(data)

    if model_name in SEMISUP:
        # train_index counts points on the ORIGINAL timeline; after deletion the split moves.
        # Map it to the number of survivors before the original cut point.
        train_index = int(file_name.split('.')[0].split('_')[-3])
        train_kept = int((~nan_mask[:train_index]).sum())
        if train_kept < 10:
            raise ValueError(f'train split collapsed after data loss: {train_kept} points')
        return run_Semisupervise_AD(model_name, data[:train_kept, :], data, **hp), None, False

    if window_mode == 'native' or model_name not in WINDOW_MODELS:
        return run_Unsupervise_AD(model_name, data, **hp), None, False

    w, clamped = _model_window(model_name, clean_data, hp, n_kept, clamp)

    if model_name == 'MatrixProfile':
        from TSB_AD.models.MatrixProfile import MatrixProfile
        clf = MatrixProfile(window=w); clf.fit(data)
        return clf.decision_scores_.ravel(), w, clamped
    if model_name == 'POLY':
        from TSB_AD.models.POLY import POLY
        clf = POLY(power=hp.get('power', 3), window=w); clf.fit(data)
        return clf.decision_scores_.ravel(), w, clamped
    if model_name == 'Sub_PCA':
        from TSB_AD.models.PCA import PCA
        clf = PCA(slidingWindow=w, n_components=hp.get('n_components')); clf.fit(data)
        return clf.decision_scores_.ravel(), w, clamped
    if model_name == 'KShapeAD':
        from TSB_AD.models.SAND import SAND
        clf = SAND(pattern_length=w, subsequence_length=4 * w)
        clf.fit(data.squeeze(), overlaping_rate=int(1.5 * w))
        return clf.decision_scores_.ravel(), w, clamped
    if model_name == 'KMeansAD_U':
        from TSB_AD.models.KMeansAD import KMeansAD
        clf = KMeansAD(k=hp.get('n_clusters', 20), window_size=w, stride=1, n_jobs=1)
        return clf.fit_predict(data).ravel(), w, clamped
    raise ValueError(f'unhandled window model: {model_name}')


def _fit_length(score, n_kept):
    """Defensive: TSB-AD wrappers return full-length scores, but pad/truncate if one does not."""
    score = np.asarray(score).ravel()
    if len(score) > n_kept:
        return score[:n_kept]
    if len(score) < n_kept:
        return np.pad(score, (0, n_kept - len(score)), mode='edge')
    return score


# ==========================================
# CORE WORKER
# ==========================================

def process_single_job(job_args):
    _silence_warnings()   # defensive: ensure filters are active in this worker process
    (file_path, condition, seed, model_names, window_mode) = job_args
    file_name = os.path.basename(file_path)
    condition_name = condition['name']

    try:
        from TSB_AD.evaluation.metrics import get_metrics
        from TSB_AD.utils.slidingWindows import find_length_rank

        # 1. Load (official TSB-AD convention). reset_index because the injectors address rows
        # by LABEL (df.loc[...]) while generating POSITIONS — a dropna() gap would misalign them.
        # inject_sensor_stuck in particular walks a block with `.loc[start_idx + i]`.
        df = pd.read_csv(file_path).dropna().reset_index(drop=True)
        value_col = df.columns[0]           # 'Data'
        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]

        clean_data = df.iloc[:, 0:-1].values.astype(float)          # (N, feats)
        labels = df[label_col].astype(int).to_numpy()
        n = len(df)

        # Metric window from the CLEAN signal, rank=1 — the official formula, but fixed so the
        # measuring stick stays identical across every condition of the grid.
        sliding_window = int(find_length_rank(clean_data[:, 0].reshape(-1, 1), rank=1))

        # 2. Corrupt — every corruption of this condition, in physical order
        if not condition['corruptions']:
            data_full = clean_data
        else:
            corruptor = TSCorruptor(df.copy(), value_col=value_col, label_col=label_col,
                                    seed=seed, corruption_target=CORRUPTION_TARGET)
            apply_corruptions(corruptor, condition, n)
            df_work = corruptor.get_corrupted_df()
            data_full = df_work.iloc[:, 0:-1].values.astype(float)

        # The evaluation branch follows the REALISED NaNs, not the declared condition: a missing
        # level that happened to drop nothing must still take the standard path.
        nan_mask = np.isnan(data_full).any(axis=1)
        has_nan = bool(nan_mask.any())

        masked_anomaly = nan_mask & (labels == 1)
        masked_normal = nan_mask & (labels == 0)
        n_lost_anomalies = int(masked_anomaly.sum())
        actual_missing_rate = float(nan_mask.sum()) / n

        # 3. The detector sees only the survivors — no imputation
        model_data = data_full[~nan_mask] if has_nan else data_full
        n_kept = len(model_data)

        if n_kept < sliding_window + 10:
            return {'status': 'skipped', 'file': file_name, 'condition': condition_name,
                    'reason': f'too few points after corruption: {n_kept}'}
        if labels[~masked_normal].sum() == 0:
            return {'status': 'skipped', 'file': file_name, 'condition': condition_name,
                    'reason': 'no anomaly survives in the evaluation set'}

        params = _extract_params(condition)

        # 4. Model + eval via the OFFICIAL pipeline
        results = []
        for model_name in model_names:
            base = {'file': file_name,
                    'condition': condition_name,
                    'combination_name': condition['combination_name'],
                    'condition_type': condition['condition_type'],
                    **params,
                    'has_missing': condition['has_missing'],
                    'actual_missing_rate': round(actual_missing_rate, 4),
                    'n_lost_anomalies': n_lost_anomalies,
                    'n_original': n, 'n_kept': n_kept,
                    'metric_window': sliding_window,
                    'seed': seed, 'model': model_name}
            try:
                hp = _get_hp(model_name)
                # Deterministic per-job seeding: some TSB-AD models are stochastic (e.g. KMeansAD
                # builds sklearn KMeans without random_state). Stable md5, not Python's salted hash.
                _key = f"{file_name}|{condition_name}|{seed}|{model_name}".encode()
                _js = int(hashlib.md5(_key).hexdigest()[:8], 16) % (2**31 - 1)
                np.random.seed(_js)
                random.seed(_js)

                output, model_window, clamped = _run_model(
                    model_name, model_data, clean_data, hp, window_mode,
                    file_name, nan_mask, clamp=has_nan)

                if not isinstance(output, np.ndarray):
                    results.append({**base, 'model_window': model_window,
                                    'window_clamped': clamped,
                                    'error': f'wrapper returned: {str(output)[:120]}'})
                    continue

                score = _fit_length(output, n_kept)

                if has_nan:
                    # --- TRUE IMPACT remapping ---
                    # The original wrote a literal 0.0 for lost anomalies, valid there because it
                    # MinMax-scaled scores to [0, 1] first, making 0.0 the true minimum.
                    # TSB-AD wrappers return RAW scores, so a literal 0.0 would rank a destroyed
                    # anomaly near the TOP and inflate the metrics as more anomalies are lost.
                    # score.min() keeps the original semantics: tied last with the least
                    # anomalous point, scale-free, and rank metrics ignore the difference.
                    lost_anomaly_score = float(score.min())
                    full_score = np.full(n, np.nan)
                    full_score[~nan_mask] = score
                    full_score[masked_anomaly] = lost_anomaly_score  # destroyed anomaly = missed
                    eval_mask = ~masked_normal          # normals destroyed by the loss = excluded
                    eval_scores = full_score[eval_mask]
                    eval_labels = labels[eval_mask]
                else:
                    eval_scores = score
                    eval_labels = labels
                    eval_mask = np.ones(n, dtype=bool)

                if np.isnan(eval_scores).any():
                    results.append({**base, 'model_window': model_window,
                                    'window_clamped': clamped,
                                    'error': 'NaN left in evaluation scores'})
                    continue

                m = get_metrics(eval_scores, eval_labels, slidingWindow=sliding_window)
                row = {**base, 'model_window': model_window, 'window_clamped': clamped,
                       'n_evaluated': int(eval_mask.sum()), 'error': None}
                for k, v in m.items():          # store ALL metrics TSB-AD returns
                    row[k.replace('-', '_')] = v
                results.append(row)
            except Exception as e:
                results.append({**base, 'error': str(e)})
        return {'status': 'success', 'results': results}

    except Exception:
        return {'status': 'error', 'file': file_name, 'condition': condition_name,
                'error': traceback.format_exc()}


# ==========================================
# CHECKPOINTS (one file per model, so runs of different models never clobber each other)
# ==========================================
def _ckpt_path(model):
    return os.path.join(RESULTS_DIR, f"checkpoint_{model}.csv")


def _load_checkpoints(models):
    """Previous successful results for `models` only, plus any legacy combined file."""
    records, paths = [], [_ckpt_path(m) for m in models]
    if LEGACY_CKPT and os.path.exists(LEGACY_CKPT):
        paths.append(LEGACY_CKPT)
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
            if 'model' in df.columns:
                df = df[df['model'].isin(models)]
            ok = df[df['error'].isna()] if 'error' in df.columns else df
            records.extend(ok.to_dict('records'))
        except Exception as e:
            print(f"Warning: could not read {os.path.basename(p)}: {e}")
    seen, uniq, keys = set(), [], set()
    for r in records:
        # key on `condition` (a string): parameter columns are NaN for the clean anchor, so a
        # parameter-based key would never match on resume
        k = f"{r['file']}|{r['condition']}|{r['seed']}|{r['model']}"
        if k in seen:
            continue
        seen.add(k); keys.add(k); uniq.append(r)
    return uniq, keys


def _save_checkpoints(all_results, models):
    if not all_results:
        return
    df = pd.DataFrame(all_results)
    for m in models:
        sub = df[df['model'] == m]
        if not sub.empty:
            sub.to_csv(_ckpt_path(m), index=False)


def _load_all_results_for_summary():
    import glob
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "checkpoint_*.csv")))
    if LEGACY_CKPT and os.path.exists(LEGACY_CKPT):
        paths.append(LEGACY_CKPT)
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if {'file', 'condition', 'seed', 'model'}.issubset(df.columns):
        df = df.drop_duplicates(subset=['file', 'condition', 'seed', 'model'], keep='first')
    return df


# ==========================================
# SUMMARY
# ==========================================

def compute_summary(df_results, output_path):
    cols = [c for c in METRIC_COLS if c in df_results.columns]
    ok = df_results[df_results['error'].isnull()]
    if ok.empty:
        print("No successful runs to summarize.")
        return
    keys = [k for k in ['condition', 'combination_name', 'condition_type', 'model']
            if k in ok.columns]
    rows = []
    for name, g in ok.groupby(keys, dropna=False):
        row = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        row['n_runs'] = len(g)
        for extra in ('actual_missing_rate', 'n_lost_anomalies', 'n_kept'):
            if extra in g.columns:
                row[f'mean_{extra}'] = round(g[extra].mean(), 4)
        if 'window_clamped' in g.columns:
            row['n_window_clamped'] = int(g['window_clamped'].fillna(False).astype(bool).sum())
        for c in cols:
            row[f'mean_{c}'] = round(g[c].mean(), 4)
            row[f'std_{c}'] = round(g[c].std(), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


# ==========================================
# INTERACTION ANALYSIS
# ==========================================

def _classify(interaction):
    """Damage beyond the additive prediction is synergistic; below it, sub-additive.

    The +-0.01 AUC dead zone is the original's, kept so the three labels stay comparable.
    """
    if interaction > 0.01:
        return 'synergistic'
    if interaction < -0.01:
        return 'sub-additive'
    return 'additive'


def compute_interaction_analysis(df_results, output_dir, metric='AUC_ROC'):
    """Is compound damage additive, synergistic, or sub-additive?

    For each severity cross of each combination:
        drop(X)      = AUC(clean) - AUC(X alone)
        predicted    = sum of the individual drops
        interaction  = drop(compound) - predicted
    A positive interaction means the corruptions hurt each other's detectability more than
    their separate damages would suggest.
    """
    if metric not in df_results.columns:
        print(f"[Interaction] metric {metric} not in results — skipping.")
        return
    ok = df_results[df_results['error'].isnull()]
    if ok.empty:
        return

    # Mean over files, per (condition, model)
    mean_auc = ok.groupby(['condition', 'model'])[metric].mean().to_dict()
    models = sorted(ok['model'].dropna().unique())
    baseline = {m: mean_auc[('clean', m)] for m in models if ('clean', m) in mean_auc}

    if not baseline:
        print("[Interaction] no clean baseline found — skipping.")
        return

    rows = []
    for combo_name, ctypes in COMBINATIONS:
        # Severity levels come from SEVERITY, so the noise 'extreme' arm is covered too.
        for sev_combo in itertools.product(*[list(SEVERITY[ct].keys()) for ct in ctypes]):
            tags = [_tag(ct, sv) for ct, sv in zip(ctypes, sev_combo)]
            compound_name = '+'.join(tags)

            for model in models:
                bl = baseline.get(model)
                auc_singles = [mean_auc.get((f"{t}_only", model)) for t in tags]
                auc_compound = mean_auc.get((compound_name, model))
                if bl is None or auc_compound is None or any(a is None for a in auc_singles):
                    continue

                drops = [bl - a for a in auc_singles]
                drop_compound = bl - auc_compound
                predicted = sum(drops)
                interaction = drop_compound - predicted
                pct = (interaction / predicted * 100) if abs(predicted) > 1e-6 else 0.0

                row = {'combination_name': combo_name,
                       'condition': compound_name,
                       'n_corruptions': len(tags),
                       'model': model,
                       'baseline_auc': round(bl, 4),
                       'auc_compound': round(auc_compound, 4),
                       'drop_compound': round(drop_compound, 4),
                       'predicted_additive': round(predicted, 4),
                       'interaction': round(interaction, 4),
                       'interaction_pct': round(pct, 1),
                       'interaction_type': _classify(interaction)}
                for i, (t, a, d) in enumerate(zip(tags, auc_singles, drops)):
                    letter = chr(ord('A') + i)
                    row[f'severity_{letter}'] = t
                    row[f'auc_{letter}'] = round(a, 4)
                    row[f'drop_{letter}'] = round(d, 4)
                rows.append(row)

    if not rows:
        print("[Interaction] no complete compound/single/baseline triples found — skipping.")
        return

    df_inter = pd.DataFrame(rows)
    df_inter.to_csv(os.path.join(output_dir, "interaction_analysis.csv"), index=False)
    print(f"Interaction analysis: {len(df_inter)} rows -> interaction_analysis.csv")

    summary = df_inter.groupby(['combination_name', 'model']).agg(
        mean_interaction=('interaction', 'mean'),
        mean_interaction_pct=('interaction_pct', 'mean'),
        n_synergistic=('interaction_type', lambda x: (x == 'synergistic').sum()),
        n_additive=('interaction_type', lambda x: (x == 'additive').sum()),
        n_subadditive=('interaction_type', lambda x: (x == 'sub-additive').sum()),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "interaction_summary.csv"), index=False)

    print(f"\n=== Interaction summary ({metric}) ===")
    print(f"{'Combination':<24} {'Model':<14} {'mean int':>10} {'syn':>5} {'add':>5} {'sub':>5}")
    print("-" * 68)
    for _, r in summary.sort_values(['model', 'mean_interaction'], ascending=[True, False]).iterrows():
        print(f"{r['combination_name']:<24} {r['model']:<14} {r['mean_interaction']:>+10.4f} "
              f"{int(r['n_synergistic']):>5} {int(r['n_additive']):>5} {int(r['n_subadditive']):>5}")


# ==========================================
# SHAPLEY VALUE ABLATION
# ==========================================

def compute_shapley_analysis(df_results, output_dir, metric='AUC_ROC'):
    """Shapley values for the triple (noise + spikes + missing).

    Each corruption's average marginal contribution to total damage across all coalitions,
    answering the practical question: which problem should you fix FIRST for the biggest
    recovery? Needs the baseline, all 3 singles, all 3 pairs and the triple — every one of
    which this script produces, so no cross-experiment lookup can go stale.
    """
    if metric not in df_results.columns:
        return
    ok = df_results[df_results['error'].isnull()]
    if ok.empty:
        return

    mean_auc = ok.groupby(['condition', 'model'])[metric].mean().to_dict()
    models = sorted(ok['model'].dropna().unique())
    baseline = {m: mean_auc[('clean', m)] for m in models if ('clean', m) in mean_auc}
    if not baseline:
        print("[Shapley] no clean baseline found — skipping.")
        return

    players = ['noise', 'spikes', 'missing']
    n_players = len(players)
    rows = []

    for sev_combo in itertools.product(*[list(SEVERITY[p].keys()) for p in players]):
        sevs = dict(zip(players, sev_combo))
        tag = {p: _tag(p, sevs[p]) for p in players}

        # Coalition -> condition name. Pair names are spelled in the order the types appear in
        # COMBINATIONS, which is exactly how build_conditions() joins them.
        conditions = {
            frozenset():                    'clean',
            frozenset(['noise']):           f"{tag['noise']}_only",
            frozenset(['spikes']):          f"{tag['spikes']}_only",
            frozenset(['missing']):         f"{tag['missing']}_only",
            frozenset(['noise', 'spikes']):   f"{tag['noise']}+{tag['spikes']}",
            frozenset(['noise', 'missing']):  f"{tag['noise']}+{tag['missing']}",
            frozenset(['spikes', 'missing']): f"{tag['spikes']}+{tag['missing']}",
            frozenset(players):             f"{tag['noise']}+{tag['spikes']}+{tag['missing']}",
        }

        for model in baseline:
            aucs, skip = {}, False
            for coalition, cond_name in conditions.items():
                auc = mean_auc.get((cond_name, model))
                if auc is None:
                    skip = True
                    break
                aucs[coalition] = auc
            if skip:
                continue

            # Damage function v(S) = AUC(clean) - AUC(S)
            v = {s: baseline[model] - a for s, a in aucs.items()}
            total_damage = v[frozenset(players)]

            for corr in players:
                others = [c for c in players if c != corr]
                shapley = 0.0
                for size in range(len(others) + 1):
                    for subset in itertools.combinations(others, size):
                        s = frozenset(subset)
                        marginal = v[s | {corr}] - v[s]
                        weight = (math.factorial(len(s))
                                  * math.factorial(n_players - len(s) - 1)
                                  / math.factorial(n_players))
                        shapley += weight * marginal

                # Recovery: what you gain by removing just this corruption from the triple
                recovery = aucs[frozenset(players) - {corr}] - aucs[frozenset(players)]

                rows.append({
                    'noise_severity': sevs['noise'],
                    'spikes_severity': sevs['spikes'],
                    'missing_severity': sevs['missing'],
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
        print("[Shapley] no complete coalitions found — skipping.")
        return

    df_sh = pd.DataFrame(rows)
    df_sh.to_csv(os.path.join(output_dir, "shapley_ablation.csv"), index=False)
    print(f"Shapley ablation: {len(df_sh)} rows -> shapley_ablation.csv")

    summary = df_sh.groupby(['corruption', 'model']).agg(
        mean_shapley=('shapley_value', 'mean'),
        mean_recovery=('recovery_if_fixed', 'mean'),
        mean_pct=('shapley_pct', 'mean'),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "shapley_summary.csv"), index=False)

    print(f"\n=== Shapley summary ({metric}) — mean contribution to damage ===")
    print(f"{'Corruption':<12} {'Model':<14} {'Shapley':>9} {'Recovery':>10} {'% damage':>10}")
    print("-" * 60)
    for _, r in summary.sort_values(['model', 'mean_shapley'], ascending=[True, False]).iterrows():
        print(f"{r['corruption']:<12} {r['model']:<14} {r['mean_shapley']:>9.4f} "
              f"{r['mean_recovery']:>+10.4f} {r['mean_pct']:>9.1f}%")


# ==========================================
# MAIN
# ==========================================

def _read_file_list(path):
    """Accept either the official TSB-AD eval list ('file_name') or a subset table ('file')."""
    df = pd.read_csv(path)
    for col in ('file_name', 'file', 'filename'):
        if col in df.columns:
            return df[col].astype(str).tolist()
    raise ValueError(f"{path}: no 'file_name' or 'file' column")


def main():
    ap = argparse.ArgumentParser(description='Compound corruptions — TSB-AD (350)')
    ap.add_argument('--test', action='store_true', help='Smoke test: 2 files, tiny grid')
    ap.add_argument('--models', nargs='+', default=['IForest'],
                    help='TSB-AD models (IForest, MatrixProfile, Sub_PCA, POLY, KShapeAD, KMeansAD_U, ...)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--combinations', nargs='+', default=None,
                    choices=[c[0] for c in COMBINATIONS],
                    help='Run only these combinations (default: all six)')
    ap.add_argument('--files-csv', default=None,
                    help='Alternative file list (e.g. a validated subset table). '
                         'Reads a file_name/file column.')
    ap.add_argument('--max-files', type=int, default=None, help='Cap the number of files')
    ap.add_argument('--no-clean', action='store_true', help='Skip the clean anchor')
    ap.add_argument('--no-singles', action='store_true',
                    help='Skip the single-corruption arms (interaction analysis then needs them '
                         'to be already present in the checkpoints)')
    ap.add_argument('--interaction-metric', default='AUC_ROC',
                    help='Metric the interaction/Shapley analysis is computed on '
                         '(default AUC_ROC, as in the original; VUS_PR is TSB-AD\'s headline)')
    ap.add_argument('--window-mode', choices=['clean', 'native'], default='clean',
                    help="'clean' (default): periodicity models use the window from the clean "
                         "signal, as in the original experiment. 'native': pure TSB-AD wrapper.")
    ap.add_argument('--summary-only', action='store_true',
                    help='Recompute summary/interaction/Shapley from existing checkpoints')
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    global LEGACY_CKPT
    LEGACY_CKPT = os.path.join(RESULTS_DIR, "checkpoint.csv")

    if args.summary_only:
        df_all = _load_all_results_for_summary()
        if df_all.empty:
            print("No checkpoints found.")
            return
        print(f"Summarizing {len(df_all)} results across models: "
              f"{sorted(df_all['model'].dropna().unique())}")
        compute_summary(df_all, os.path.join(RESULTS_DIR, "summary.csv"))
        compute_interaction_analysis(df_all, RESULTS_DIR, args.interaction_metric)
        compute_shapley_analysis(df_all, RESULTS_DIR, args.interaction_metric)
        return

    list_path = Path(args.files_csv) if args.files_csv else FILE_LIST_CSV
    if not list_path.is_absolute():
        list_path = _ROOT / list_path
    if not list_path.exists():
        print(f"[ERROR] File list not found: {list_path}")
        return

    files = _read_file_list(list_path)
    file_paths = [str(DATA_DIR / f) for f in files]

    conditions = build_conditions(combos=args.combinations,
                                  include_clean=not args.no_clean,
                                  include_singles=not args.no_singles)

    if args.test:
        file_paths = file_paths[:2]
        keep = {'clean', 'noise_low_only', 'missing_low_only', 'freeze_low_only',
                'noise_low+missing_low', 'noise_high+missing_high', 'missing_low+freeze_low'}
        conditions = [c for c in conditions if c['name'] in keep]
        print("!!! TEST MODE !!!")
    elif args.max_files:
        file_paths = file_paths[:args.max_files]

    n_base = sum(1 for c in conditions if c['condition_type'] == 'baseline')
    n_single = sum(1 for c in conditions if c['condition_type'] == 'single')
    n_comp = sum(1 for c in conditions if c['condition_type'] == 'compound')
    n_jobs_total = len(file_paths) * len(conditions) * N_SEEDS

    print(f"\n{'='*66}\n  Compound corruptions — TSB-AD ({len(file_paths)} files)\n{'='*66}")
    print("  Application order: freeze -> noise -> spikes -> missing")
    print("  NaNs present -> true impact (lost anomaly = min score, lost normal excluded)")
    print("  No NaNs      -> standard evaluation on the full corrupted series")
    print(f"{'='*66}")
    print(f"File list: {list_path.name}")
    print(f"Conditions: {len(conditions)}  (baseline: {n_base}, singles: {n_single}, "
          f"compounds: {n_comp})")
    print(f"Combinations: {[c[0] for c in COMBINATIONS] if not args.combinations else args.combinations}")
    print(f"Models: {args.models} | Workers: {args.workers} | window_mode: {args.window_mode}")
    print(f"corruption_target: {CORRUPTION_TARGET} | interaction metric: {args.interaction_metric}")
    print(f"Total jobs: {n_jobs_total}")
    if n_jobs_total > 20000:
        print("  ^ large grid. --combinations / --files-csv / --max-files cut it down, and the "
              "run is resumable.")
    print(f"{'='*66}\n")

    all_results, completed = _load_checkpoints(args.models)
    if all_results:
        print(f"Loaded checkpoints: {len(all_results)} successful results for {args.models}.")

    jobs = []
    for fp in file_paths:
        fn = os.path.basename(fp)
        for cond in conditions:
            for seed in range(N_SEEDS):
                need = [m for m in args.models
                        if f"{fn}|{cond['name']}|{seed}|{m}" not in completed]
                if need:
                    jobs.append((fp, cond, seed, need, args.window_mode))

    # Longest-first scheduling: MatrixProfile costs ~O(n^2) and series span 18k-900k points,
    # so a few files carry most of the work. Ordering affects scheduling only, never results.
    try:
        sizes = {fp: os.path.getsize(fp) for fp in set(j[0] for j in jobs)}
        jobs.sort(key=lambda j: sizes.get(j[0], 0), reverse=True)
    except OSError:
        pass

    print(f"Jobs scheduled: {len(jobs)}")
    if jobs:
        new, skipped = 0, 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_single_job, j): j for j in jobs}
            for fut in tqdm(as_completed(futs), total=len(jobs), desc="compound"):
                res = fut.result()
                if res['status'] == 'success':
                    all_results.extend(res['results'])
                    new += len(res['results'])
                elif res['status'] == 'skipped':
                    skipped += 1
                else:
                    print(f"Error in {res.get('file','?')} / {res.get('condition','?')}: "
                          f"{str(res.get('error',''))[:200]}")
                if new >= 200:   # frequent autosave: a crash costs minutes, not hours
                    _save_checkpoints(all_results, args.models)
                    new = 0
        _save_checkpoints(all_results, args.models)
        if skipped:
            print(f"Skipped {skipped} job(s): too few points / no anomaly left after corruption.")
        print("\nFinal results saved to: "
              + ", ".join(os.path.basename(_ckpt_path(m)) for m in args.models))

    # Summary covers every model on disk, so it stays complete when models run one at a time.
    df_all = _load_all_results_for_summary()
    if not df_all.empty:
        print(f"\nSummarizing {len(df_all)} results across models: "
              f"{sorted(df_all['model'].dropna().unique())}")
        compute_summary(df_all, os.path.join(RESULTS_DIR, "summary.csv"))
        compute_interaction_analysis(df_all, RESULTS_DIR, args.interaction_metric)
        compute_shapley_analysis(df_all, RESULTS_DIR, args.interaction_metric)

    print(f"\n[Done] Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
