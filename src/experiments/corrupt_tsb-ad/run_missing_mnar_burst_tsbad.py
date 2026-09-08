"""
Missing Values — MCAR vs MNAR (burst version), TSB-AD version.

Port of run_missing_mnar_burst.py to the TSB-AD benchmark, built on the same skeleton as
run_missing_true_impact_tsbad.py.

The question is about the MECHANISM of data loss, not its volume. Every condition removes the
same `fraction` of points in the same number of contiguous blocks; only the RULE that picks
where the blocks go changes:

    mcar_burst          blocks at uniformly random positions            (control)
    mnar_extreme_burst  blocks where mean |z-score| is highest          (loss follows outliers)
    mnar_high_burst     blocks where mean positive z-score is highest   (loss follows peaks)

    burst_length = max(1, int(fraction * n / num_bursts))

MNAR is the realistic failure: a sensor that saturates, a logger that drops packets exactly
when the signal spikes, a transmitter that browns out under load. Anomalies live in precisely
those regions, so MNAR loss destroys anomalies far more often than MCAR loss of equal volume.

> Key question: at IDENTICAL loss volume, how much worse is it when the loss is correlated with
> the signal than when it is random?

That question needs the control arm to mean anything, which is the main change here (below).

Evaluation — "true impact", NO imputation (identical policy to the original):
  1. Drop the NaN points; the detector sees a SHORTER series.
  2. Map the scores back onto the original timeline.
  3. Lost ANOMALY points get the minimum score -> they count as false negatives.
  4. Lost NORMAL points are EXCLUDED from the evaluation.

Detection and metrics use the OFFICIAL TSB-AD pipeline (run_Unsupervise_AD /
run_Semisupervise_AD + find_length_rank + TSB_AD get_metrics) on the full 350-series
TSB-AD-U eval list.

DELIBERATE DEVIATIONS from the TSB-UAD original — all documented, none accidental:

  1. THE MCAR CONTROL ARM IS RESTORED. The original implements `mcar_burst`
     (run_missing_mnar_burst.py line 81) but leaves it out of `MECHANISMS` (line 58), so the
     experiment named "MCAR vs MNAR" only ever ran the two MNAR arms. Without the matched
     random-placement arm there is nothing to attribute the damage to: any drop could be the
     volume of loss rather than its correlation with the signal. All three mechanisms run here
     by default, so every MNAR cell has a same-fraction, same-num_bursts MCAR twin.
  2. A `clean` anchor condition (no loss at all) runs the identical pipeline, giving the x=0 of
     every degradation curve inside this run rather than importing it from elsewhere. On the
     clean condition nan_mask is empty, so it reduces exactly to the official TSB-AD baseline.
  3. Lost anomalies get score.min(), not a literal 0.0. TSB-AD returns UNNORMALIZED scores
     (IForest sits around [-0.06, +0.03] with most points below zero), so a hardcoded 0.0 would
     rank a destroyed anomaly in the top ~10% and INFLATE the metrics as more anomalies are
     lost. In the original 0.0 WAS the minimum, because it MinMax-scaled the scores first;
     score.min() carries exactly that semantics into a raw-score pipeline.
  4. Burst placement is vectorised (see _select_burst_starts). Selection order and the resulting
     starts are identical to the original; only the occupancy test changed, because the original
     builds a Python set of `burst_length` integers per candidate AND the MNAR candidates
     cluster together (extreme |z| sits around anomalies), so it rescans many overlapping starts
     before finding a free one. On a 900k-point series with burst_length = 180k that is
     pathological rather than merely slow.
  5. No MinMax on the scores and no max(window, 10) floor on the metric window — the same two
     deviations as every other TSB-AD port here, so the clean anchor reproduces the official
     baseline bit for bit. Rank-based metrics are invariant to the dropped MinMax.
  6. Extra seeds are spent only on the MCAR arm. MNAR placement is a deterministic greedy pick
     over z-scores: same series, same starts, every seed. Re-running it would burn compute to
     reproduce identical rows.

INTERPRETATION NOTE — read `pct_anomalies_lost` alongside every metric. MNAR destroys anomalies
by construction, and under true-impact evaluation a destroyed anomaly is a guaranteed false
negative, so part of the MNAR damage is arithmetic rather than a statement about the detector.
The finding is whatever damage EXCEEDS what the anomaly loss alone explains; the MCAR twin at
matched volume is what makes that separation possible.

Usage:
    python src/experiments/corrupt_tsb-ad/run_missing_mnar_burst_tsbad.py --test
    python src/experiments/corrupt_tsb-ad/run_missing_mnar_burst_tsbad.py --models IForest --workers 4
    # the contrast at one fraction, if the full grid is too big:
    python src/experiments/corrupt_tsb-ad/run_missing_mnar_burst_tsbad.py \
        --models IForest --fractions 0.10 --workers 4
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
import glob
import random
import hashlib
import argparse
import warnings
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

# ==========================================
# CONFIG
# ==========================================
_ROOT = Path(PROJECT_ROOT)
FILE_LIST_CSV = _ROOT / "TSB-AD" / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
DATA_DIR = _ROOT / "TSB-AD" / "Datasets" / "TSB-AD-U"
RESULTS_DIR = _ROOT / "results" / "experiments" / "missing_mnar_burst_tsbad"

# Same grid as the original run_missing_mnar_burst.py
FRACTIONS = [0.01, 0.05, 0.10, 0.20]
NUM_BURSTS = [1, 3, 5, 10]

# mcar_burst restored — see deviation 1 in the module docstring.
MECHANISMS = ['mcar_burst', 'mnar_extreme_burst', 'mnar_high_burst']

# Only mcar_burst is stochastic; the MNAR arms are a deterministic greedy pick over z-scores.
STOCHASTIC_MECHANISMS = {'mcar_burst'}

N_SEEDS = 1

METRIC_COLS = ['AUC_PR', 'AUC_ROC', 'VUS_PR', 'VUS_ROC', 'Standard_F1',
               'PA_F1', 'Event_based_F1', 'R_based_F1', 'Affiliation_F']

SEMISUP = {'AutoEncoder', 'AutoEncoder_2', 'StreamVAE', 'CNN', 'LSTMAD'}

# Models whose window is derived from the data via find_length_rank(periodicity). Under
# corruption the official wrapper would re-estimate it on the CORRUPTED (here: shortened)
# signal; the original experiment instead fixed the window from the CLEAN signal. In
# window_mode='clean' we replicate each wrapper body verbatim but inject the clean window.
# (IForest is absent on purpose: it uses a fixed slidingWindow=100, so corruption cannot move it.)
WINDOW_MODELS = {'MatrixProfile', 'POLY', 'Sub_PCA', 'KShapeAD', 'KMeansAD_U'}

LEGACY_CKPT = None  # set in main()


def _cond_name(mechanism, fraction, num_bursts):
    if mechanism is None:
        return "clean"
    return f"{mechanism}_frac_{fraction}_nb_{num_bursts}"


# ==========================================
# BURST PLACEMENT
# ==========================================
def _select_burst_starts(values, n, burst_length, num_bursts, mechanism, rng):
    """Start indices of `num_bursts` NON-OVERLAPPING blocks of length `burst_length`.

    Greedy, identical in semantics and in output to the original select_burst_starts():
      * mcar_burst -> walk a shuffled list of every legal start;
      * mnar_*     -> walk the starts ordered by descending mean window score;
    take a candidate whenever it does not touch an already placed block.

    Only the occupancy test differs. The original builds `set(range(c, c + burst_length))` for
    every candidate it examines; the MNAR candidates are clustered (extreme |z| sits around
    anomalies), so it examines many overlapping starts before finding a free one, and each of
    those checks allocates a set of burst_length integers. With burst_length = 180 000 on a
    900k-point series that is pathological. A boolean occupancy array makes the same test an
    O(burst_length) numpy slice with no allocation.

    Returns fewer than `num_bursts` starts if the series cannot hold them; the caller records
    the realised count.
    """
    if burst_length >= n:
        return [0]
    max_start = n - burst_length
    if max_start <= 0:
        return [0]

    if mechanism == 'mcar_burst':
        order = np.arange(max_start)
        rng.shuffle(order)
    else:
        # Mean window score via a prefix sum: score[i] = mean(w[i : i + burst_length]).
        z = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-10)
        if mechanism == 'mnar_extreme_burst':
            w = np.abs(z)                 # both tails: outliers of either sign
        elif mechanism == 'mnar_high_burst':
            w = np.maximum(z, 0)          # upper tail only: peaks
        else:
            raise ValueError(f'unknown mechanism: {mechanism}')
        c = np.concatenate([[0.0], np.cumsum(w)])
        position_scores = (c[burst_length:max_start + burst_length] - c[:max_start]) / burst_length
        order = np.argsort(-position_scores)      # highest first, same kind as the original

    used = np.zeros(n, dtype=bool)
    starts = []
    for idx in order:
        if len(starts) >= num_bursts:
            break
        i = int(idx)
        if not used[i:i + burst_length].any():
            starts.append(i)
            used[i:i + burst_length] = True
    return starts


# ==========================================
# MODELS (identical policy to run_missing_true_impact_tsbad.py)
# ==========================================
def _get_hp(model_name):
    from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
    key = 'AutoEncoder' if model_name in ('AutoEncoder', 'AutoEncoder_2') else model_name
    return dict(Optimal_Uni_algo_HP_dict.get(key, {}))


def _model_window(model_name, clean_data, hp, n_kept):
    """Window for periodicity-derived models, estimated on the CLEAN signal.

    Clamped to the length actually reaching the detector: this experiment DELETES points, so a
    window valid for the full series can exceed the survivor count. Same clamp as the original
    (`min(w, n_kept // 4)`, floor 10). Reported per row so a firing clamp is visible.
    """
    from TSB_AD.utils.slidingWindows import find_length_rank
    w = int(find_length_rank(clean_data, rank=hp.get('periodicity', 1)))
    w_clamped = max(min(w, n_kept // 4), 10)
    return w_clamped, (w_clamped != w)


def _run_model(model_name, data, clean_data, hp, window_mode, file_name, nan_mask):
    """Run a TSB-AD detector on the SURVIVING points.

    `data` is already the shortened array (NaN rows removed). `nan_mask` is over the ORIGINAL
    timeline and is needed to remap the semi-supervised train split.

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

    w, clamped = _model_window(model_name, clean_data, hp, n_kept)

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
    (file_path, mechanism, fraction, num_bursts, seed, model_names, window_mode) = job_args
    file_name = os.path.basename(file_path)
    condition_name = _cond_name(mechanism, fraction, num_bursts)

    try:
        from TSB_AD.evaluation.metrics import get_metrics
        from TSB_AD.utils.slidingWindows import find_length_rank

        # 1. Load (official TSB-AD convention)
        df = pd.read_csv(file_path).dropna().reset_index(drop=True)
        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]

        clean_data = df.iloc[:, 0:-1].values.astype(float)          # (N, feats)
        labels = df[label_col].astype(int).to_numpy()
        n = len(df)

        # Metric window from the CLEAN signal, rank=1 — the official formula, but fixed so the
        # measuring stick stays identical across every mechanism and loss level.
        sliding_window = int(find_length_rank(clean_data[:, 0].reshape(-1, 1), rank=1))

        # 2. Place the bursts (skip for the clean anchor). Whole ROWS are dropped, so every
        # channel stays aligned; the mechanism scores positions on channel 0.
        burst_length, n_bursts_actual = 0, 0
        if mechanism is None:
            nan_mask = np.zeros(n, dtype=bool)
        else:
            burst_length = max(1, int(fraction * n / num_bursts))
            rng = np.random.default_rng(seed)
            starts = _select_burst_starts(clean_data[:, 0], n, burst_length,
                                          num_bursts, mechanism, rng)
            n_bursts_actual = len(starts)
            nan_mask = np.zeros(n, dtype=bool)
            for s in starts:
                nan_mask[s:s + burst_length] = True

        masked_anomaly = nan_mask & (labels == 1)
        masked_normal = nan_mask & (labels == 0)
        n_lost_anomalies = int(masked_anomaly.sum())
        n_anomalies = int(labels.sum())
        pct_anomalies_lost = n_lost_anomalies / max(n_anomalies, 1)
        actual_missing_rate = float(nan_mask.sum()) / n

        # 3. The detector sees only the survivors — no imputation
        model_data = clean_data[~nan_mask]
        n_kept = len(model_data)

        if n_kept < sliding_window + 10:
            return {'status': 'skipped', 'file': file_name,
                    'reason': f'too few points after data loss: {n_kept}'}
        if n_anomalies == 0:
            return {'status': 'skipped', 'file': file_name,
                    'reason': 'file has no anomaly to evaluate'}

        # 4. Model + eval via the OFFICIAL pipeline
        results = []
        for model_name in model_names:
            base = {'file': file_name, 'mechanism': mechanism,
                    'fraction': fraction, 'num_bursts': num_bursts,
                    'burst_length': burst_length, 'num_bursts_actual': n_bursts_actual,
                    'actual_missing_rate': round(actual_missing_rate, 4),
                    'n_lost_anomalies': n_lost_anomalies,
                    'n_anomalies': n_anomalies,
                    'pct_anomalies_lost': round(pct_anomalies_lost, 4),
                    'n_original': n, 'n_kept': n_kept,
                    'metric_window': sliding_window,
                    'seed': seed, 'model': model_name, 'condition': condition_name}
            try:
                hp = _get_hp(model_name)
                # Deterministic per-job seeding: some TSB-AD models are stochastic (e.g. KMeansAD
                # builds sklearn KMeans without random_state). Stable md5, not Python's salted hash.
                _key = f"{file_name}|{condition_name}|{seed}|{model_name}".encode()
                _js = int(hashlib.md5(_key).hexdigest()[:8], 16) % (2**31 - 1)
                np.random.seed(_js)
                random.seed(_js)

                output, model_window, clamped = _run_model(
                    model_name, model_data, clean_data, hp, window_mode, file_name, nan_mask)

                if not isinstance(output, np.ndarray):
                    results.append({**base, 'model_window': model_window,
                                    'window_clamped': clamped,
                                    'error': f'wrapper returned: {str(output)[:120]}'})
                    continue

                score = _fit_length(output, n_kept)

                # --- TRUE IMPACT remapping ---
                # The original wrote a literal 0.0 for lost anomalies, valid there because it
                # MinMax-scaled scores to [0, 1] first, making 0.0 the true minimum.
                # TSB-AD wrappers return RAW scores: IForest lives in roughly [-0.06, +0.03] with
                # ~90% of points BELOW zero, so a literal 0.0 would rank a destroyed anomaly in
                # the top ~10% — inverting the intended meaning and inflating the metrics as more
                # anomalies are lost. Use the series minimum instead: same semantics as the
                # original (tied last with the least anomalous point), scale-free, and rank
                # metrics are invariant to the difference.
                lost_anomaly_score = float(score.min())
                full_score = np.full(n, np.nan)
                full_score[~nan_mask] = score
                full_score[masked_anomaly] = lost_anomaly_score   # destroyed anomaly = missed
                eval_mask = ~masked_normal            # normals destroyed by the loss = excluded

                eval_scores = full_score[eval_mask]
                eval_labels = labels[eval_mask]
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
        return {'status': 'error', 'file': file_name, 'error': traceback.format_exc()}


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
        # key on `condition` (a string): the clean anchor stores mechanism=None, which pandas
        # reads back as NaN, so a mechanism/fraction-based key would never match on resume
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


def compute_summary(df_results, output_path):
    cols = [c for c in METRIC_COLS if c in df_results.columns]
    ok = df_results[df_results['error'].isnull()]
    if ok.empty:
        print("No successful runs to summarize.")
        return
    keys = [k for k in ['condition', 'mechanism', 'fraction', 'num_bursts', 'model']
            if k in ok.columns]
    rows = []
    for name, g in ok.groupby(keys, dropna=False):
        row = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        row['n_runs'] = len(g)
        for extra in ('burst_length', 'num_bursts_actual', 'actual_missing_rate',
                      'n_lost_anomalies', 'pct_anomalies_lost', 'n_kept'):
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
# MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser(description='Missing values MCAR vs MNAR (burst) — TSB-AD (350)')
    ap.add_argument('--test', action='store_true', help='Smoke test: 3 files, small grid')
    ap.add_argument('--models', nargs='+', default=['IForest'],
                    help='TSB-AD models (IForest, MatrixProfile, Sub_PCA, POLY, KShapeAD, KMeansAD_U, ...)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--fractions', nargs='+', type=float, default=None,
                    help=f'Override fractions (default {FRACTIONS})')
    ap.add_argument('--num-bursts', nargs='+', type=int, default=None,
                    help=f'Override burst counts (default {NUM_BURSTS})')
    ap.add_argument('--mechanisms', nargs='+', default=None, choices=MECHANISMS,
                    help=f'Override mechanisms (default: all {len(MECHANISMS)}). Dropping '
                         'mcar_burst removes the control arm — see the module docstring.')
    ap.add_argument('--no-clean', action='store_true', help='Skip the clean (no-loss) anchor')
    ap.add_argument('--window-mode', choices=['clean', 'native'], default='clean',
                    help="'clean' (default): periodicity models use the window from the clean "
                         "signal, as in the original experiment. 'native': pure TSB-AD wrapper.")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    global LEGACY_CKPT
    LEGACY_CKPT = os.path.join(RESULTS_DIR, "checkpoint.csv")
    if not FILE_LIST_CSV.exists():
        print(f"[ERROR] File list not found: {FILE_LIST_CSV}")
        return

    files = pd.read_csv(FILE_LIST_CSV)['file_name'].tolist()
    file_paths = [str(DATA_DIR / f) for f in files]
    fractions = args.fractions if args.fractions else FRACTIONS
    num_bursts_list = args.num_bursts if args.num_bursts else NUM_BURSTS
    mechanisms = args.mechanisms if args.mechanisms else MECHANISMS
    if args.test:
        file_paths = file_paths[:3]
        fractions, num_bursts_list = [0.05, 0.20], [3]
        print("!!! TEST MODE !!!")

    if 'mcar_burst' not in mechanisms:
        print("[WARN] mcar_burst is not in the run: the MNAR arms will have no matched "
              "control, so damage cannot be attributed to the mechanism rather than the volume.")

    # (mechanism, fraction, num_bursts); None mechanism = the clean anchor
    conditions = [] if args.no_clean else [(None, None, 0)]
    conditions += [(mech, f, nb)
                   for mech in mechanisms for f in fractions for nb in num_bursts_list]

    print(f"\n{'='*66}\n  Missing values — MCAR vs MNAR (burst) — TSB-AD (full 350)\n{'='*66}")
    print("  Same volume, same block count — only the PLACEMENT RULE changes")
    print("  Lost anomalies -> minimum score (false negatives)")
    print("  Lost normal points -> excluded from evaluation")
    print(f"{'='*66}")
    print(f"Files: {len(file_paths)} | fractions: {fractions} | num_bursts: {num_bursts_list}")
    print(f"Mechanisms: {mechanisms}")
    print(f"Clean anchor: {not args.no_clean} | conditions: {len(conditions)}")
    print(f"Models: {args.models} | workers: {args.workers} | window_mode: {args.window_mode}")
    print(f"Seeds: {N_SEEDS} (MNAR placement is deterministic — extra seeds go to mcar_burst only)")
    print(f"{'='*66}\n")

    all_results, completed = _load_checkpoints(args.models)
    if all_results:
        print(f"Loaded checkpoints: {len(all_results)} successful results for {args.models}.")

    jobs = []
    for fp in file_paths:
        fn = os.path.basename(fp)
        for (mech, frac, nb) in conditions:
            cond = _cond_name(mech, frac, nb)
            # The clean anchor and the MNAR arms are deterministic: one seed reproduces them.
            n_seeds = N_SEEDS if mech in STOCHASTIC_MECHANISMS else 1
            for seed in range(n_seeds):
                need = [m for m in args.models if f"{fn}|{cond}|{seed}|{m}" not in completed]
                if need:
                    jobs.append((fp, mech, frac, nb, seed, need, args.window_mode))

    # Longest-first scheduling: MatrixProfile costs ~O(n^2) and series span 1k-900k points,
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
            for fut in tqdm(as_completed(futs), total=len(jobs), desc="missing(mnar-burst)"):
                res = fut.result()
                if res['status'] == 'success':
                    all_results.extend(res['results'])
                    new += len(res['results'])
                elif res['status'] == 'skipped':
                    skipped += 1
                else:
                    print(f"Error in {res.get('file','?')}: {str(res.get('error',''))[:200]}")
                if new >= 200:   # frequent autosave: a crash costs minutes, not hours
                    _save_checkpoints(all_results, args.models)
                    new = 0
        _save_checkpoints(all_results, args.models)
        if skipped:
            print(f"Skipped {skipped} job(s): too few points left / file has no anomaly.")
        print("\nFinal results saved to: "
              + ", ".join(os.path.basename(_ckpt_path(m)) for m in args.models))

    # Summary covers every model on disk, so it stays complete when models run one at a time.
    df_all = _load_all_results_for_summary()
    if not df_all.empty:
        print(f"Summarizing {len(df_all)} results across models: "
              f"{sorted(df_all['model'].dropna().unique())}")
        compute_summary(df_all, os.path.join(RESULTS_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
