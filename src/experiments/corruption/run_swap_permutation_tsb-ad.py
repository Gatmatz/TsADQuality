"""
Swap Corruption — Segment Permutation — TSB-AD version.

Port of run_swap_permutation.py to the TSB-AD benchmark, built to mirror
run_spikes_tsbad.py step for step. Same corruption (split into N equal segments and
shuffle their order) and same parameter grid, but evaluated with the OFFICIAL TSB-AD
pipeline (run_Unsupervise_AD / run_Semisupervise_AD + find_length_rank + TSB_AD
get_metrics) on the full 350-series TSB-AD-U eval list.

Based on:
  - Um et al. (2017) — Permutation augmentation for wearable sensor data
  - Grover et al. (NeurIPS 2024) — Segment, Shuffle, and Stitch (S3)

The entire series is permuted — values are rearranged but preserved (no new values
created, no values removed). Labels stay in their ORIGINAL positions, which is what
makes this a corruption rather than a relabelling: the detector sees reordered data
but is scored against the original ground truth.

A `clean` anchor condition runs the identical pipeline with no permutation, so it
reproduces the TSB-AD baseline exactly and the degradation curve starts from the
right point.

Key question: "How much temporal reordering can anomaly detectors tolerate before
              performance breaks down?"

Usage:
    python src/experiments/corruption/run_swap_permutation_tsb-ad.py --test
    python src/experiments/corruption/run_swap_permutation_tsb-ad.py --models IForest --workers 4
    python src/experiments/corruption/run_swap_permutation_tsb-ad.py --models MatrixProfile --workers 4
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
RESULTS_DIR = _ROOT / "results" / "experiments" / "swap_permutation_tsbad"

# Same grid as the original TSB-UAD run_swap_permutation.py
N_SEGMENTS_LIST = [2, 4, 8, 16, 24]
N_SEEDS = 3  # Multiple seeds since the permutation is random

METRIC_COLS = ['AUC_PR', 'AUC_ROC', 'VUS_PR', 'VUS_ROC', 'Standard_F1',
               'PA_F1', 'Event_based_F1', 'R_based_F1', 'Affiliation_F']

SEMISUP = {'AutoEncoder', 'AutoEncoder_2', 'StreamVAE', 'CNN', 'LSTMAD'}

# Models whose window is derived from the data via find_length_rank(periodicity). Under
# corruption the official wrapper would re-estimate it on the CORRUPTED signal; the original
# experiment instead fixed the window from the CLEAN signal. In window_mode='clean' we
# replicate each wrapper body verbatim but inject the clean window. (IForest is absent on
# purpose: it uses a fixed slidingWindow=100, so corruption cannot move it.)
#
# Permutation makes this matter more than any other corruption in the suite: reordering
# segments destroys the periodicity that find_length_rank measures, so a native-window run
# would change the window AND the data at once and the two effects could not be separated.
WINDOW_MODELS = {'MatrixProfile', 'POLY', 'Sub_PCA', 'KShapeAD', 'KMeansAD_U'}

LEGACY_CKPT = None  # set in main()


def _cond_name(n_segments):
    return "clean" if n_segments is None else f"nseg_{n_segments}"


def permute_segments(values, n_segments, rng):
    """Split into n_segments near-equal parts along axis 0 and shuffle their order.

    Works on row indices rather than on the values themselves, so a multivariate series
    keeps its channels aligned: every column is reordered by the same permutation.

    Returns (permuted_values, perm_order).
    """
    n = len(values)
    seg_len, remainder = divmod(n, n_segments)

    # Segment boundaries; the remainder is spread over the first segments so the
    # concatenation is exactly the original length.
    bounds, start = [], 0
    for i in range(n_segments):
        end = start + seg_len + (1 if i < remainder else 0)
        bounds.append((start, end))
        start = end

    # Shuffle until the order actually differs from the identity, otherwise the
    # "corrupted" condition would silently be a clean one.
    original_order = list(range(n_segments))
    perm = original_order.copy()
    for _ in range(100):
        rng.shuffle(perm)
        if perm != original_order:
            break

    order = np.concatenate([np.arange(*bounds[i]) for i in perm])
    return values[order], perm


def _get_hp(model_name):
    from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
    key = 'AutoEncoder' if model_name in ('AutoEncoder', 'AutoEncoder_2') else model_name
    return dict(Optimal_Uni_algo_HP_dict.get(key, {}))


def _run_model(model_name, data, clean_data, hp, window_mode, file_name):
    """Run a TSB-AD detector.

    window_mode='native' -> pure official wrapper (re-estimates the window on the given data).
    window_mode='clean'  -> same model/HP, but periodicity-derived models use the window
                            estimated from the clean signal (matches the original experiment).
    On the clean condition both modes coincide exactly.
    """
    from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD
    from TSB_AD.utils.slidingWindows import find_length_rank

    if model_name in SEMISUP:
        train_index = int(file_name.split('.')[0].split('_')[-3])
        return run_Semisupervise_AD(model_name, data[:train_index, :], data, **hp)

    if window_mode == 'native' or model_name not in WINDOW_MODELS:
        return run_Unsupervise_AD(model_name, data, **hp)

    # Each model uses ITS OWN official periodicity (e.g. KMeansAD_U uses rank=2),
    # estimated on the CLEAN signal so corruption cannot move it.
    w = int(find_length_rank(clean_data, rank=hp.get('periodicity', 1)))
    if model_name == 'MatrixProfile':
        from TSB_AD.models.MatrixProfile import MatrixProfile
        clf = MatrixProfile(window=w); clf.fit(data)
        return clf.decision_scores_.ravel()
    if model_name == 'POLY':
        from TSB_AD.models.POLY import POLY
        clf = POLY(power=hp.get('power', 3), window=w); clf.fit(data)
        return clf.decision_scores_.ravel()
    if model_name == 'Sub_PCA':
        from TSB_AD.models.PCA import PCA
        clf = PCA(slidingWindow=w, n_components=hp.get('n_components')); clf.fit(data)
        return clf.decision_scores_.ravel()
    if model_name == 'KShapeAD':
        from TSB_AD.models.SAND import SAND
        clf = SAND(pattern_length=w, subsequence_length=4 * w)
        clf.fit(data.squeeze(), overlaping_rate=int(1.5 * w))
        return clf.decision_scores_.ravel()
    if model_name == 'KMeansAD_U':
        from TSB_AD.models.KMeansAD import KMeansAD
        clf = KMeansAD(k=hp.get('n_clusters', 20), window_size=w, stride=1, n_jobs=1)
        return clf.fit_predict(data).ravel()
    raise ValueError(f'unhandled window model: {model_name}')


def process_single_job(job_args):
    _silence_warnings()   # defensive: ensure filters are active in this worker process
    (file_path, n_segments, seed, model_names, window_mode) = job_args
    file_name = os.path.basename(file_path)
    condition_name = _cond_name(n_segments)

    try:
        from TSB_AD.evaluation.metrics import get_metrics
        from TSB_AD.utils.slidingWindows import find_length_rank

        # 1. Load (official TSB-AD convention)
        df = pd.read_csv(file_path).dropna()
        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]

        clean_data = df.iloc[:, 0:-1].values.astype(float)          # (N, feats)
        label = df[label_col].astype(int).to_numpy()
        n = len(df)

        # metric window from the CLEAN signal, rank=1 — the official formula, but fixed so the
        # measuring stick is identical across conditions
        sliding_window = find_length_rank(clean_data[:, 0].reshape(-1, 1), rank=1)

        if n_segments is not None and n < n_segments * 2:
            return {'status': 'skipped', 'file': file_name,
                    'reason': f'Series too short ({n}) for {n_segments} segments'}

        # 2. Permute (skip for the clean anchor). Labels are NOT permuted: they stay in
        # their original positions, which is the point of the experiment.
        if n_segments is None:
            data, perm_order, seg_len = clean_data, None, None
        else:
            rng = np.random.RandomState(seed)
            data, perm_order = permute_segments(clean_data, n_segments, rng)
            seg_len = n // n_segments

        if np.isnan(data).all():
            return {'status': 'error', 'file': file_name, 'error': 'All NaN after permutation'}

        # 3. Model + eval via the OFFICIAL pipeline
        results = []
        for model_name in model_names:
            base = {'file': file_name, 'n_segments': n_segments, 'segment_length': seg_len,
                    'permutation': str(perm_order), 'seed': seed,
                    'model': model_name, 'condition': condition_name}
            try:
                hp = _get_hp(model_name)
                # Deterministic per-job seeding: some TSB-AD models are stochastic (e.g. KMeansAD
                # builds sklearn KMeans without random_state). Stable md5, not Python's salted hash.
                _key = f"{file_name}|{condition_name}|{seed}|{model_name}".encode()
                _js = int(hashlib.md5(_key).hexdigest()[:8], 16) % (2**31 - 1)
                np.random.seed(_js)
                random.seed(_js)

                output = _run_model(model_name, data, clean_data, hp, window_mode, file_name)

                if not isinstance(output, np.ndarray):
                    results.append({**base, 'error': f'wrapper returned: {str(output)[:120]}'})
                    continue

                m = get_metrics(output, label, slidingWindow=sliding_window)
                row = {**base, 'error': None}
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


def _job_key(r):
    return f"{r['file']}|{r['condition']}|{r['seed']}|{r['model']}"


def _load_checkpoints(models):
    """Previous successful results for `models` only, plus the legacy combined file."""
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
        k = _job_key(r)
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


def compute_summary(df_results, output_path):
    cols = [c for c in METRIC_COLS if c in df_results.columns]
    ok = df_results[df_results['error'].isnull()]
    if ok.empty:
        print("No successful runs to summarize.")
        return
    keys = ['condition', 'n_segments', 'model']
    keys = [k for k in keys if k in ok.columns]
    rows = []
    for name, g in ok.groupby(keys, dropna=False):
        row = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        row['n_runs'] = len(g)
        if 'segment_length' in g.columns:
            row['mean_segment_length'] = round(g['segment_length'].mean(), 1)
        for c in cols:
            row[f'mean_{c}'] = round(g[c].mean(), 4)
            row[f'std_{c}'] = round(g[c].std(), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(description='Swap — segment permutation — TSB-AD (350)')
    ap.add_argument('--test', action='store_true', help='Smoke test: 3 files, small grid')
    ap.add_argument('--models', nargs='+', default=['IForest'],
                    help='TSB-AD models (IForest, MatrixProfile, Sub_PCA, POLY, KShapeAD, KMeansAD_U, ...)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-clean', action='store_true', help='Skip the clean (unpermuted) anchor condition')
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
    n_segments_list, n_seeds = N_SEGMENTS_LIST, N_SEEDS
    if args.test:
        file_paths = file_paths[:3]
        n_segments_list, n_seeds = [2, 8], 1
        print("!!! TEST MODE !!!")

    # None = the clean anchor
    conditions = [] if args.no_clean else [None]
    conditions += list(n_segments_list)

    print(f"\n{'='*60}\n  Swap — Segment Permutation — TSB-AD (full 350)\n{'='*60}")
    print(f"  Um et al. (2017), S3 (Grover et al., NeurIPS 2024)")
    print(f"  Split into N segments, shuffle order; labels stay in place")
    print(f"{'='*60}")
    print(f"Files: {len(file_paths)} | n_segments: {n_segments_list} | seeds: {n_seeds}")
    print(f"Clean anchor: {not args.no_clean} | conditions: {len(conditions)}")
    print(f"Models: {args.models} | Workers: {args.workers} | window_mode: {args.window_mode}")
    print(f"Total jobs: {len(file_paths) * len(conditions) * n_seeds}\n{'='*60}\n")

    all_results, completed = _load_checkpoints(args.models)
    if all_results:
        print(f"Loaded checkpoints: {len(all_results)} successful results for {args.models}.")

    jobs = []
    for fp in file_paths:
        fn = os.path.basename(fp)
        for nseg in conditions:
            cond = _cond_name(nseg)
            # The clean anchor is deterministic: one seed is enough, more would only
            # repeat identical work.
            seeds = [0] if nseg is None else range(n_seeds)
            for seed in seeds:
                need = [m for m in args.models if f"{fn}|{cond}|{seed}|{m}" not in completed]
                if need:
                    jobs.append((fp, nseg, seed, need, args.window_mode))

    # Longest-first scheduling: MatrixProfile costs ~O(n^2) and series span 18k-900k points,
    # so a few files carry most of the work. Ordering affects scheduling only, never results.
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
            for fut in tqdm(as_completed(futs), total=len(jobs), desc="permutation"):
                res = fut.result()
                if res['status'] == 'success':
                    all_results.extend(res['results'])
                    new += len(res['results'])
                elif res['status'] == 'error':
                    print(f"Error in {res.get('file','?')}: {str(res.get('error',''))[:200]}")
                if new >= 200:   # frequent autosave: a crash costs minutes, not hours
                    _save_checkpoints(all_results, args.models)
                    new = 0
        _save_checkpoints(all_results, args.models)
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
