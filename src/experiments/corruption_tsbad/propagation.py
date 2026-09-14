"""
Corruption Propagation — how far do localized corruption effects travel?, TSB-AD version.

Port of run_propagation.py to the TSB-AD benchmark.

Unlike every other corruption experiment here, this one does NOT measure AUC. It corrupts a
single centred segment (10% of the series) and asks WHERE the anomaly scores moved:

    score_diff[t] = |score_corrupted[t] - score_clean[t]|

Zones are defined by distance from the corrupted segment (corruption / near / mid / far), and

    propagation_ratio = #{t : normalised score_diff[t] > 0.05} / corruption_zone_size

A ratio of 1.0 means the damage stayed exactly inside the corrupted segment; 5.0 means the
scores moved over an area five times larger than the corruption itself.

Hypothesis (unchanged from the original): the radius depends on whether a model builds LOCAL
or GLOBAL features.
    global      IForest, Sub_PCA        split thresholds / eigenvectors depend on all the data
    neighbour   Sub_LOF                 k-NN reshuffling propagates in chains
    local       MatrixProfile           z-normalised subsequence distance is local by construction

Model mapping from the TSB-UAD original:
    IForest -> IForest        PCA -> Sub_PCA        LOF -> Sub_LOF
    MP      -> MatrixProfile  AE  -> AutoEncoder (semi-supervised in TSB-AD, see caveat below)

=== THE NORMALISATION PROBLEM — read this before using the numbers ===

The original MinMax-scales the clean and corrupted score vectors SEPARATELY to [0, 1] and then
subtracts them. Those are two DIFFERENT affine maps: corruption raises the maximum score, so
every other point is divided by a larger number and appears to move even when its raw score did
not. Measured here on real TSB-AD-U files:

    file        model           propagation_ratio (raw)   (separate MinMax)
    644_YAHOO   Sub_PCA                   9.965                 1.566      <- 6.4x apart
    644_YAHOO   IForest                   8.168                 7.888
    657_YAHOO   MatrixProfile             1.562                 1.562      <- unaffected

Under MinMax, Sub_PCA reads as a LOCAL model; under raw scores it reads as strongly GLOBAL.
The normalisation choice IS the finding, so this port does not pick one silently. Every row
carries the full zone analysis under all three:

    *_minmax : the original's per-run MinMax. Reproduces the TSB-UAD numbers.
    *_raw    : raw TSB-AD scores, differenced directly. Sensitive to a global shift or rescale
               of the score distribution — which for this research question is signal, not noise.
    *_rank   : both vectors converted to percentile ranks first. Invariant to ANY monotone
               rescaling of either run, so it isolates the one thing that changes a detector's
               verdict: did a point move relative to the others? The most defensible single
               number, and the one to prefer in the paper unless there is a reason not to.

DELIBERATE DEVIATIONS from the TSB-UAD original:
  1. The clean run happens ONCE per (file, model) instead of once per condition. The original
     re-ran it inside all 9 condition jobs — 18 model fits per (file, model) where 10 suffice.
     Same scores (the clean run does not depend on the condition), ~45% less compute.
  2. Missing values are INTERPOLATED, never dropped — same as the original, and required: the
     score vectors must stay on a common timeline to be subtracted. This is the one experiment
     where the true-impact policy of the other ports would be wrong.

WINDOW POLICY — the one thing this port must not get wrong:
ONE window per file, computed once on the clean signal as max(find_length_rank(clean, rank=1), 10),
and FORCED on every model, exactly as the original did. Two separate reasons, both fatal if missed:
  * clean vs corrupted are subtracted point by point, so a window that moved with the corruption
    would make the difference measure the window change instead of propagation;
  * a subsequence model with window w mechanically touches everything within w of the corruption,
    so models with different windows carry different mechanical floors and the cross-model table
    would rank window sizes rather than locality. TSB-AD hands each model its own window
    (IForest 100, Sub_PCA 24, Sub_LOF 49); the original handed all of them the same one.

Caveat on semi-supervised models (AutoEncoder, CNN, LSTMAD, StreamVAE): TSB-AD trains them on
the first `train_index` points taken from the filename. The corrupted segment is centred at 50%,
so for most files training data is untouched and only inference sees the corruption. That is a
different experimental setup from the unsupervised models and the ratios are not directly
comparable; the `train_overlaps_corruption` column flags the files where they do overlap.

Usage:
    python src/experiments/corruption_tsbad/propagation.py --test
    python src/experiments/corruption_tsbad/propagation.py --models IForest --workers 4
    # the hypothesis needs the contrast, so the interesting run is the spread:
    python src/experiments/corruption_tsbad/propagation.py \
        --models IForest Sub_PCA Sub_LOF MatrixProfile --workers 4
"""

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from ts_corruptor.localized import (  # noqa: E402
    inject_localized_missing, inject_localized_noise, inject_localized_spikes)
from tsbad import paths  # noqa: E402
from tsbad.checkpoint import (  # noqa: E402
    ckpt_path, load_all_for_summary, load_checkpoints, save_checkpoints)
from tsbad.env import silence_warnings  # noqa: E402
from tsbad.evaluate import fit_length  # noqa: E402
from tsbad.models import SEMISUP, get_hp, seed_job  # noqa: E402

# This runner does not use tsbad.harness.run_sweep: it produces no metrics row. It corrupts one
# centred segment and measures how far the anomaly scores move, per zone and per normalisation.

RESULTS_NAME = "propagation_tsbad"

# ==========================================
# CONFIG
# ==========================================
CORRUPTION_RATIO = 0.10       # corrupt 10% of the series, centred on the midpoint
SEED = 42                     # the original's fixed seed, reset per condition
SCORE_DIFF_THRESHOLD = 0.05   # a point counts as "impacted" above 5% of the max diff

# Severity grid — byte-identical to the original
CORRUPTIONS = {
    'noise':   {'low': {'snr_db': 20}, 'med': {'snr_db': 10}, 'high': {'snr_db': 5}},
    'spikes':  {'low': {'magnitude': 3}, 'med': {'magnitude': 6}, 'high': {'magnitude': 10}},
    'missing': {'low': {'fraction': 0.3},   # fraction OF THE CORRUPTION ZONE set to NaN
                'med': {'fraction': 0.5},
                'high': {'fraction': 0.8}},
}

# The three score normalisations compared on every row. See the module docstring.
NORMS = ('minmax', 'raw', 'rank')

# NOTE: there is deliberately no WINDOW_MODELS list here. Which models derive their window
# from the data is read from each wrapper's signature at call time (see _window_kwargs), because
# a hardcoded list silently misses ten of the fifteen periodicity-derived TSB-AD models.

ZONE_COLS = ['corruption_zone', 'near', 'mid', 'far']

# ==========================================
# LOCALIZED CORRUPTION (injectors in ts_corruptor.localized, verbatim from the original)
# ==========================================

def build_corrupted(kind, params, raw1d, start, end, sliding_window):
    """Apply one localized corruption and return a model-ready 1-D array (no NaNs left).

    Each condition gets a FRESH RandomState(SEED), exactly as the original did by constructing
    one per job. Consequence, preserved on purpose: severities of the same corruption type share
    the same random draws, so low/med/high differ only in amplitude — a controlled contrast
    rather than three unrelated realisations.
    """
    rng = np.random.RandomState(SEED)

    if kind == 'noise':
        return inject_localized_noise(raw1d, start, end, params['snr_db'], rng), 0

    if kind == 'spikes':
        return inject_localized_spikes(raw1d, start, end, params['magnitude'], rng), 0

    if kind == 'missing':
        corrupted = inject_localized_missing(raw1d, start, end, params['fraction'], rng)
        valid = ~np.isnan(corrupted)
        n_nan = int((~valid).sum())
        if valid.sum() < sliding_window + 10:
            raise ValueError(f'too few valid points after missing injection: {int(valid.sum())}')
        # Linear interpolation, NOT deletion: the two score vectors are subtracted point by
        # point, so the corrupted run must stay on the original timeline.
        filled = corrupted.copy()
        filled[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], corrupted[valid])
        return filled, n_nan

    raise ValueError(f'unknown corruption type: {kind}')

# ==========================================
# MODELS (official TSB-AD pipeline, one shared window per file)
# ==========================================

def _shared_window(clean_data):
    """ONE window per file, shared by every model — exactly what the original did.

    run_propagation.py computed `max(int(find_length(clean)), 10)` once per file and handed the
    same value to IForest, PCA, LOF and MatrixProfile alike. That is what makes the cross-model
    table mean anything: every detector then has the SAME mechanical footprint, so a difference
    in propagation is a difference in how the model builds features, not in how wide its window
    happens to be. TSB-AD instead gives each model its own window (IForest 100, Sub_PCA 24,
    Sub_LOF 49), which would silently turn a window comparison into the headline result.

    find_length_rank(rank=1) was measured to agree with the original's find_length on all 350
    files, and the max(., 10) floor is the original's.
    """
    from TSB_AD.utils.slidingWindows import find_length_rank
    return max(int(find_length_rank(clean_data, rank=1)), 10)


def _window_kwargs(model_name, hp, w):
    """Force window `w` on a TSB-AD detector, whichever way its wrapper takes one.

    Two families, discovered from the wrapper signature rather than a hardcoded list:
      * `slidingWindow=` accepted directly     -> IForest, LOF, PCA, HBOS, KNN
      * `periodicity=`, window derived inside  -> Sub_*, MatrixProfile, POLY, KShapeAD,
                                                  KMeansAD_U, NORMA, SAND, Series2Graph, SR
    The second family calls find_length_rank(data, ...) on whatever array it is given, so on the
    corrupted run it would re-estimate the window on the CORRUPTED series. Returns
    (hp, patch_needed); the caller pins find_length_rank when patch_needed is True.
    """
    import inspect
    from TSB_AD import model_wrapper as mw
    hp = dict(hp)
    fn = getattr(mw, f'run_{model_name}', None)
    if fn is None:
        return hp, False
    params = inspect.signature(fn).parameters
    if 'slidingWindow' in params:
        hp['slidingWindow'] = w
        return hp, False
    return hp, 'periodicity' in params


def _run_model(model_name, data, hp, window, file_name):
    """Run a TSB-AD detector with the shared per-file window forced on it.

    Every model — including IForest, whose TSB-AD default is a fixed slidingWindow=100 — is
    pinned to `window`, so clean and corrupted runs are the same detector and all models on a
    file share one mechanical footprint. That is the property the original had and the property
    the cross-model table depends on.
    """
    from TSB_AD import model_wrapper as mw
    from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD

    if model_name in SEMISUP:
        # Semi-supervised models carry a fixed window_size/win_size in their HP; there is no
        # data-derived window to pin, and the original used a per-file pretrained AE instead.
        train_index = int(file_name.split('.')[0].split('_')[-3])
        return run_Semisupervise_AD(model_name, data[:train_index, :], data, **hp)

    hp2, needs_patch = _window_kwargs(model_name, hp, window)
    if not needs_patch:
        return run_Unsupervise_AD(model_name, data, **hp2)

    # Pin find_length_rank for the duration of this one fit. Restored in `finally`, and each
    # worker is its own process, so nothing leaks between jobs.
    orig = mw.find_length_rank
    mw.find_length_rank = lambda _data, rank=1: window
    try:
        return run_Unsupervise_AD(model_name, data, **hp2)
    finally:
        mw.find_length_rank = orig


# ==========================================
# PROPAGATION ANALYSIS
# ==========================================

def _normalise(score, how):
    """Put a score vector on the scale the comparison is made in."""
    if how == 'raw':
        return np.asarray(score, dtype=float)
    if how == 'minmax':
        s = np.asarray(score, dtype=float)
        lo, hi = s.min(), s.max()
        return np.zeros_like(s) if hi - lo < 1e-12 else (s - lo) / (hi - lo)
    if how == 'rank':
        # Percentile position. Invariant to any monotone rescaling of either run, so a global
        # shift or squeeze of the score distribution cannot masquerade as propagation.
        from scipy.stats import rankdata
        return rankdata(score, method='average') / float(len(score))
    raise ValueError(f'unknown normalisation: {how}')


def compute_propagation(scores_clean, scores_corrupted, corrupt_start, corrupt_end, n_total):
    """Zone means and propagation ratio from two aligned score vectors.

    Zone geometry is the original's: near = within 10% of the series length from the corruption
    boundary, mid = 10-30%, far = beyond 30%.
    """
    score_diff = np.abs(scores_corrupted - scores_clean)
    max_diff = float(np.max(score_diff))
    corrupt_size = corrupt_end - corrupt_start

    if max_diff < 1e-10:
        return {'mean_diff_corruption_zone': 0.0, 'mean_diff_near': 0.0,
                'mean_diff_mid': 0.0, 'mean_diff_far': 0.0,
                'propagation_ratio': 1.0, 'max_diff': 0.0, 'n_impacted': 0}

    dn = score_diff / max_diff
    near_dist = int(0.10 * n_total)
    mid_dist = int(0.30 * n_total)

    corruption_zone = dn[corrupt_start:corrupt_end]

    near_left_start = max(0, corrupt_start - near_dist)
    near_right_end = min(n_total, corrupt_end + near_dist)
    near_zone = np.concatenate([dn[near_left_start:corrupt_start],
                                dn[corrupt_end:near_right_end]])

    mid_left_start = max(0, corrupt_start - mid_dist)
    mid_right_end = min(n_total, corrupt_end + mid_dist)
    mid_zone = np.concatenate([dn[mid_left_start:near_left_start],
                               dn[near_right_end:mid_right_end]])

    far_zone = np.concatenate([dn[:mid_left_start], dn[mid_right_end:]])

    n_impacted = int(np.sum(dn > SCORE_DIFF_THRESHOLD))

    return {
        'mean_diff_corruption_zone': float(np.mean(corruption_zone)) if len(corruption_zone) else 0.0,
        'mean_diff_near': float(np.mean(near_zone)) if len(near_zone) else 0.0,
        'mean_diff_mid': float(np.mean(mid_zone)) if len(mid_zone) else 0.0,
        'mean_diff_far': float(np.mean(far_zone)) if len(far_zone) else 0.0,
        'propagation_ratio': float(n_impacted / corrupt_size) if corrupt_size > 0 else 0.0,
        'max_diff': max_diff,
        'n_impacted': n_impacted,
    }


def propagation_all_norms(scores_clean, scores_corrupted, s, e, n):
    """Run the zone analysis under every normalisation, flattened into suffixed columns."""
    out = {}
    for how in NORMS:
        p = compute_propagation(_normalise(scores_clean, how),
                                _normalise(scores_corrupted, how), s, e, n)
        for k, v in p.items():
            out[f'{k}_{how}'] = v
    return out


# ==========================================
# CORE WORKER — one job = one (file, model), clean run reused across all 9 conditions
# ==========================================

def process_single_job(job_args):
    silence_warnings()
    (file_path, model_name, conditions) = job_args
    file_name = os.path.basename(file_path)

    try:
        df = pd.read_csv(file_path).dropna().reset_index(drop=True)
        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
        clean_data = df.iloc[:, 0:-1].values.astype(float)     # (N, feats)
        labels = df[label_col].astype(int).to_numpy()
        n = len(df)
        raw1d = clean_data[:, 0]

        # Corrupted segment: 10% of the series, centred (identical to the original)
        corrupt_size = int(CORRUPTION_RATIO * n)
        corrupt_start = (n - corrupt_size) // 2
        corrupt_end = corrupt_start + corrupt_size

        hp = get_hp(model_name)
        window = _shared_window(clean_data)   # one window per FILE, same for every model

        # Deterministic per-(file, model) seeding for stochastic detectors. The SAME seed is set
        # before the clean and every corrupted fit, so a score difference can only come from the
        # data, never from the detector's own randomness.
        seed_key = f"{file_name}|{model_name}"

        def _seeded_run(arr2d):
            seed_job(seed_key)
            return fit_length(_run_model(model_name, arr2d, hp, window, file_name), n)

        # --- the clean run happens ONCE and is reused by all nine conditions ---
        scores_clean = _seeded_run(clean_data)
        if not np.all(np.isfinite(scores_clean)):
            return {'status': 'error', 'file': file_name, 'model': model_name,
                    'error': 'clean scores contain NaN/inf'}

        train_index = None
        if model_name in SEMISUP:
            try:
                train_index = int(file_name.split('.')[0].split('_')[-3])
            except (ValueError, IndexError):
                train_index = None

        results = []
        for (ctype, sev, params) in conditions:
            base = {'file': file_name, 'length': n, 'corruption_type': ctype,
                    'severity': sev, 'model': model_name,
                    'model_window': window, 'hp_window': hp.get('window_size') or hp.get('win_size'),
                    'corrupt_start': corrupt_start, 'corrupt_end': corrupt_end,
                    'corrupt_size': corrupt_size,
                    'anomalies_in_zone': int(labels[corrupt_start:corrupt_end].sum()),
                    'train_overlaps_corruption': (None if train_index is None
                                                  else bool(train_index > corrupt_start))}
            for k, v in params.items():
                base[f'param_{k}'] = v
            try:
                corrupted1d, n_nan = build_corrupted(ctype, params, raw1d,
                                                     corrupt_start, corrupt_end, window or 100)
                corrupted_data = clean_data.copy()
                corrupted_data[:, 0] = corrupted1d

                scores_corrupted = _seeded_run(corrupted_data)
                if not np.all(np.isfinite(scores_corrupted)):
                    results.append({**base, 'error': 'corrupted scores contain NaN/inf'})
                    continue

                row = {**base, 'n_nan_injected': n_nan, 'error': None}
                row.update(propagation_all_norms(scores_clean, scores_corrupted,
                                                 corrupt_start, corrupt_end, n))
                results.append(row)
            except Exception as e:
                results.append({**base, 'error': str(e)[:300]})

        return {'status': 'success', 'results': results}

    except Exception:
        return {'status': 'error', 'file': file_name, 'model': model_name,
                'error': traceback.format_exc()}

# ==========================================
# CHECKPOINTS (one file per model)
# ==========================================
KEY_COLS = ('file', 'corruption_type', 'severity', 'model')


def _job_key(file_name, model):
    """Resume granularity is the job: one (file, model) produces all nine condition rows."""
    return f"{file_name}|{model}"


def _row_key(r):
    return f"{r['file']}|{r['corruption_type']}|{r['severity']}|{r['model']}"


def _load_checkpoints(results_dir, models):
    uniq, _ = load_checkpoints(results_dir, models, key=_row_key)
    # A (file, model) job counts as done only when all its condition rows are present.
    expected = sum(len(v) for v in CORRUPTIONS.values())
    counts = {}
    for r in uniq:
        counts[_job_key(r['file'], r['model'])] = counts.get(_job_key(r['file'], r['model']), 0) + 1
    done = {k for k, c in counts.items() if c >= expected}
    return uniq, done


# ==========================================
# SUMMARIES
# ==========================================

def write_summaries(df, output_dir):
    ok = df[df['error'].isnull()] if 'error' in df.columns else df
    if ok.empty:
        print("No successful runs to summarize.")
        return

    ok.to_csv(os.path.join(output_dir, "propagation_results.csv"), index=False)

    ratio_cols = [f'propagation_ratio_{h}' for h in NORMS if f'propagation_ratio_{h}' in ok.columns]
    far_cols = [f'mean_diff_far_{h}' for h in NORMS if f'mean_diff_far_{h}' in ok.columns]

    agg = {}
    for c in ratio_cols:
        agg[f'mean_{c}'] = (c, 'mean')
        agg[f'std_{c}'] = (c, 'std')
    for c in far_cols:
        agg[f'mean_{c}'] = (c, 'mean')
    agg['n'] = ('file', 'count')

    summary = ok.groupby(['corruption_type', 'severity', 'model']).agg(**agg).round(4)
    summary.to_csv(os.path.join(output_dir, "propagation_summary.csv"))

    # Compact: model x corruption_type, one table per normalisation. This is the table the
    # hypothesis is read off, so all three are printed side by side on purpose.
    print(f"\n{'='*70}\n  PROPAGATION RATIO — mean over files\n{'='*70}")
    for c in ratio_cols:
        piv = ok.groupby(['corruption_type', 'model'])[c].mean().unstack('model').round(2)
        print(f"\n--- {c} ---")
        print(piv.to_string())
        piv.to_csv(os.path.join(output_dir, f"propagation_compact_{c.split('_')[-1]}.csv"))

    # Zone decay: the shape of the falloff, per normalisation
    print(f"\n{'='*70}\n  ZONE DECAY — mean normalised score diff\n{'='*70}")
    for h in NORMS:
        cols = [f'mean_diff_{z}_{h}' for z in ZONE_COLS if f'mean_diff_{z}_{h}' in ok.columns]
        if not cols:
            continue
        z = ok.groupby(['corruption_type', 'model'])[cols].mean().round(4)
        print(f"\n--- {h} ---")
        print(z.to_string())
        z.to_csv(os.path.join(output_dir, f"propagation_zones_{h}.csv"))

    print(f"\nSaved: propagation_results.csv, propagation_summary.csv, "
          f"propagation_compact_*.csv, propagation_zones_*.csv")

# ==========================================
# MAIN
# ==========================================

def main():
    ap = argparse.ArgumentParser(description='Corruption propagation — TSB-AD (350)')
    ap.add_argument('--test', action='store_true', help='Smoke test: 3 files')
    ap.add_argument('--models', nargs='+', default=['IForest'],
                    help='TSB-AD models. The hypothesis needs the contrast: '
                         'IForest / Sub_PCA (global) vs MatrixProfile (local) vs Sub_LOF.')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--corruptions', nargs='+', default=None, choices=list(CORRUPTIONS),
                    help='Run only these corruption types (default: all three)')
    ap.add_argument('--files-csv', default=None,
                    help='Alternative file list (reads a file_name/file column)')
    ap.add_argument('--max-files', type=int, default=None, help='Cap the number of files')
    ap.add_argument('--results-dir', default=None,
                    help=f'Write checkpoints and summaries here instead of '
                         f'results/experiments/{RESULTS_NAME}/ (e.g. for a verification run)')
    ap.add_argument('--summary-only', action='store_true',
                    help='Rebuild the summary tables from existing checkpoints')
    args = ap.parse_args()

    results_dir = str(paths.resolve(args.results_dir) if args.results_dir
                      else paths.results_dir(RESULTS_NAME))
    os.makedirs(results_dir, exist_ok=True)

    if args.summary_only:
        df = load_all_for_summary(results_dir, KEY_COLS)
        if df.empty:
            print("No checkpoints found.")
            return
        write_summaries(df, results_dir)
        return

    list_path = paths.resolve(args.files_csv) if args.files_csv else paths.FILE_LIST_CSV
    if not list_path.exists():
        print(f"[ERROR] File list not found: {list_path}")
        return

    file_paths = [str(paths.DATA_DIR / f) for f in paths.read_file_list(list_path)]
    if args.test:
        idx = np.linspace(0, len(file_paths) - 1, 3, dtype=int)
        file_paths = [file_paths[i] for i in idx]
        print("!!! TEST MODE !!!")
    elif args.max_files:
        file_paths = file_paths[:args.max_files]

    types = args.corruptions if args.corruptions else list(CORRUPTIONS)
    conditions = [(ct, sev, params)
                  for ct in types for sev, params in CORRUPTIONS[ct].items()]

    print(f"\n{'='*70}\n  Corruption propagation — TSB-AD ({len(file_paths)} files)\n{'='*70}")
    print(f"  Corrupted segment: {CORRUPTION_RATIO:.0%} of the series, centred")
    print(f"  Zones: near <10% of length, mid 10-30%, far >30% from the boundary")
    print(f"  Impacted: normalised |score diff| > {SCORE_DIFF_THRESHOLD}")
    print(f"  Normalisations compared per row: {', '.join(NORMS)}")
    print(f"{'='*70}")
    print(f"File list: {list_path.name}")
    print(f"Corruption types: {types}  ->  {len(conditions)} conditions")
    print(f"Models: {args.models} | Workers: {args.workers}")
    print(f"Results: {results_dir}")
    print(f"Jobs (file x model): {len(file_paths) * len(args.models)}   "
          f"rows: {len(file_paths) * len(args.models) * len(conditions)}")
    print(f"Model fits: {len(file_paths) * len(args.models) * (1 + len(conditions))} "
          f"(the clean run is shared across conditions)")
    print(f"{'='*70}\n")

    all_results, done = _load_checkpoints(results_dir, args.models)
    if all_results:
        print(f"Loaded checkpoints: {len(all_results)} rows, {len(done)} complete (file, model) jobs.")

    jobs = [(fp, m, conditions)
            for fp in file_paths for m in args.models
            if _job_key(os.path.basename(fp), m) not in done]

    # Longest-first scheduling: MatrixProfile costs ~O(n^2) and series span 18k-900k points.
    try:
        sizes = {fp: os.path.getsize(fp) for fp in set(j[0] for j in jobs)}
        jobs.sort(key=lambda j: sizes.get(j[0], 0), reverse=True)
    except OSError:
        pass

    print(f"Jobs scheduled: {len(jobs)}")
    if jobs:
        new = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_single_job, j): j for j in jobs}
            for fut in tqdm(as_completed(futs), total=len(jobs), desc="propagation"):
                res = fut.result()
                if res['status'] == 'success':
                    all_results.extend(res['results'])
                    new += len(res['results'])
                else:
                    print(f"Error in {res.get('file','?')} / {res.get('model','?')}: "
                          f"{str(res.get('error',''))[:200]}")
                if new >= 200:
                    save_checkpoints(results_dir, all_results, args.models)
                    new = 0
        save_checkpoints(results_dir, all_results, args.models)
        print("\nFinal results saved to: "
              + ", ".join(os.path.basename(ckpt_path(results_dir, m)) for m in args.models))

    df = load_all_for_summary(results_dir, KEY_COLS)
    if not df.empty:
        print(f"\nSummarizing {len(df)} rows across models: "
              f"{sorted(df['model'].dropna().unique())}")
        write_summaries(df, results_dir)

    print(f"\n[Done] Results in {results_dir}")


if __name__ == "__main__":
    main()
