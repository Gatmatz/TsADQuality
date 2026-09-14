"""
Swap Corruption — Segment Swap — TSB-AD version.

Port of run_swap_segment.py to the TSB-AD benchmark, built to mirror
run_swap_point_tsb-ad.py step for step. Same corruption and same parameter grid, but
evaluated with the OFFICIAL TSB-AD pipeline (run_Unsupervise_AD / run_Semisupervise_AD
+ find_length_rank + TSB_AD get_metrics) on the full 350-series TSB-AD-U eval list.

The experiment holds the TOTAL swapped fraction constant and varies the GRANULARITY:

    swap_length = max(1, int(fraction * n / (2 * num_swaps)))

so  num_swaps large -> many short segments swapped
    num_swaps small -> few long segments swapped
with the same number of corrupted points either way. swap_length is therefore derived
per series, not fixed.

Key question: "For the same amount of swap corruption, is it worse to have a few large
              segment swaps or many small ones?"

Labels stay in place — only values move. This disrupts temporal dependencies while
preserving the marginal distribution of values exactly (it is a permutation of the
original samples).

A `clean` anchor condition runs the identical pipeline with no swapping, so it
reproduces the TSB-AD baseline exactly and the degradation curve starts from the
right point.

--- Why the swap is reimplemented here (correctness, not just speed) ---
ts_corruptor.injectors.inject_swap does NOT reliably deliver the requested fraction once
swap_length > 1, which is fatal for an experiment whose whole premise is "same fraction,
different granularity". Two mechanisms:

  1. Bounds skip: if idx1 + L > n the swap is dropped silently (the RNG is consumed and
     the candidate array is not pruned), so the pair is simply lost.
  2. Cross-pair overlap: after a swap it removes only the exact used indices from the
     candidate array, so a later segment may still START just before an earlier one and
     overlap it. Overlapping points are counted once, so the delivered fraction falls.

Measured on n=20,000 (delivered points, 5 seeds; every cell should be 6000):

    frac=0.30, num_swaps=1  -> 6000 6000 6000 6000     0   <- one seed corrupted NOTHING
    frac=0.30, num_swaps=3  -> 6000 6000 5061 4000  4000
    frac=0.30, num_swaps=5  -> 3600 4291 6000 6000  5486
    frac=0.30, num_swaps=20 -> 5726 5512 5631 5254  5459

The shortfall is large, seed-dependent AND correlated with num_swaps — i.e. it moves with
the very variable under study, confounding the comparison.

_swap_segments_fast instead samples 2*num_swaps STRICTLY DISJOINT segments of length L,
uniformly, using the standard bijection: choose k starts without replacement from
n - k*(L-1) slots, then expand by i*(L-1). Every condition then delivers exactly
2*num_swaps*L swapped points, so "fraction held constant" is actually true. It is also
O(n) rather than the injector's per-swap O(n) pruning pass.

The shared injector is deliberately left untouched so previously produced TSB-UAD results
stay reproducible. `--injector reference` runs the original path for cross-checking.

Usage:
    python src/experiments/corruption/run_swap_segment_tsbad.py --test
    python src/experiments/corruption/run_swap_segment_tsbad.py --models IForest --workers 4
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
RESULTS_DIR = _ROOT / "results" / "experiments" / "swap_segment_tsbad"

# Same grid as the original TSB-UAD run_swap_segment.py
FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30]
NUM_SWAPS = [1, 3, 5, 10, 20]   # swap_length = fraction * n / (2 * num_swaps), per series
N_SEEDS = 1

METRIC_COLS = ['AUC_PR', 'AUC_ROC', 'VUS_PR', 'VUS_ROC', 'Standard_F1',
               'PA_F1', 'Event_based_F1', 'R_based_F1', 'Affiliation_F']

SEMISUP = {'AutoEncoder', 'AutoEncoder_2', 'StreamVAE', 'CNN', 'LSTMAD'}

# Models whose window is derived from the data via find_length_rank(periodicity). Under
# corruption the official wrapper would re-estimate it on the CORRUPTED signal; the original
# experiment instead fixed the window from the CLEAN signal. In window_mode='clean' we
# replicate each wrapper body verbatim but inject the clean window. (IForest is absent on
# purpose: it uses a fixed slidingWindow=100, so corruption cannot move it.)
WINDOW_MODELS = {'MatrixProfile', 'POLY', 'Sub_PCA', 'KShapeAD', 'KMeansAD_U'}

LEGACY_CKPT = None  # set in main()


def _cond_name(fraction, num_swaps):
    return "clean" if fraction is None else f"frac_{fraction}_ns_{num_swaps}"


def _derive_swap_length(n, fraction, num_swaps):
    """swap_length as defined by the experiment: total fraction split over num_swaps pairs."""
    return max(1, int(fraction * n / (2 * num_swaps)))


def _swap_segments_fast(values, fraction, num_swaps, rng):
    """Swap `num_swaps` pairs of contiguous, strictly disjoint segments of length L.

    L = max(1, int(fraction*n / (2*num_swaps))), the definition used by the experiment.

    Disjoint starts are drawn uniformly via the standard bijection between "k disjoint
    blocks of length L in n slots" and "k distinct starts in n-k*(L-1) slots": draw the
    starts without replacement, sort them, then push start i right by i*(L-1). The largest
    start is then n-L at most, so no segment can run off the end and none can overlap.
    That is what guarantees exactly 2*num_swaps*L swapped points in every condition —
    the property the whole experiment rests on.

    Rows are swapped whole, so a multivariate series keeps its channels aligned.

    Returns (new_values, n_points_swapped, swap_length, num_swaps_actual).
    """
    n = len(values)
    L = _derive_swap_length(n, fraction, num_swaps)

    # Fit as many pairs as the series can actually hold disjointly. Only bites on series
    # shorter than 2*num_swaps*L, i.e. tiny ones where L has been clamped to 1.
    pairs = min(num_swaps, n // (2 * L))
    if pairs < 1:
        return values, 0, L, 0

    k = 2 * pairs                       # segments to place
    free = n - k * (L - 1)              # slots in the compressed coordinate system
    starts = np.sort(rng.choice(free, size=k, replace=False)) + np.arange(k) * (L - 1)

    order = rng.permutation(k)          # random pairing of the placed segments
    a, b = starts[order[:pairs]], starts[order[pairs:]]

    out = values.copy()
    # <= 20 iterations; the segments are disjoint, so the order of the swaps is irrelevant
    for s1, s2 in zip(a, b):
        tmp = out[s1:s1 + L].copy()
        out[s1:s1 + L] = out[s2:s2 + L]
        out[s2:s2 + L] = tmp
    return out, k * L, L, pairs


def _swap_segments_reference(df, value_col, label_col, fraction, num_swaps, seed):
    """Original ts_corruptor path — kept for cross-checking. See the module docstring for
    why it under-delivers the requested fraction once swap_length > 1."""
    from ts_corruptor.core import TSCorruptor
    import ts_corruptor.injectors
    n = len(df)
    L = _derive_swap_length(n, fraction, num_swaps)
    corruptor = TSCorruptor(df.copy(), value_col=value_col, label_col=label_col, seed=seed)
    ts_corruptor.injectors.inject_swap(
        corruptor, fraction=fraction, swap_length=L, max_distance=None)
    n_swapped = int(corruptor.corruption_mask.sum())
    return corruptor.get_corrupted_df(), n_swapped, L, (n_swapped // (2 * L) if L else 0)


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
    (file_path, fraction, num_swaps, seed, model_names, window_mode, injector) = job_args
    file_name = os.path.basename(file_path)
    condition_name = _cond_name(fraction, num_swaps)

    try:
        from TSB_AD.evaluation.metrics import get_metrics
        from TSB_AD.utils.slidingWindows import find_length_rank

        # 1. Load (official TSB-AD convention). reset_index guards the corruptor path:
        # get_target_indices() returns label indices while the swap slices positionally,
        # so a gapped index would silently misplace swaps. No file in TSB-AD-U currently
        # loses a row to dropna(), which makes this free insurance rather than a fix.
        df = pd.read_csv(file_path).dropna().reset_index(drop=True)
        value_col = df.columns[0]           # 'Data'
        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]

        clean_data = df.iloc[:, 0:-1].values.astype(float)          # (N, feats)
        label = df[label_col].astype(int).to_numpy()
        n = len(df)

        # metric window from the CLEAN signal, rank=1 — the official formula, but fixed so the
        # measuring stick is identical across conditions
        sliding_window = find_length_rank(clean_data[:, 0].reshape(-1, 1), rank=1)

        # 2. Swap (skip for the clean anchor). Labels are NOT touched: they stay in their
        # original positions, so a swapped segment is judged against the labels it landed on.
        if fraction is None:
            data, n_swapped, swap_length, ns_actual = clean_data, 0, 0, 0
        elif injector == 'reference':
            df_work, n_swapped, swap_length, ns_actual = _swap_segments_reference(
                df, value_col, label_col, fraction, num_swaps, seed)
            data = df_work.iloc[:, 0:-1].values.astype(float)
        else:
            rng = np.random.RandomState(seed)
            data, n_swapped, swap_length, ns_actual = _swap_segments_fast(
                clean_data, fraction, num_swaps, rng)

        if np.isnan(data).all():
            return {'status': 'error', 'file': file_name, 'error': 'All NaN after corruption'}

        # 3. Model + eval via the OFFICIAL pipeline
        results = []
        for model_name in model_names:
            base = {'file': file_name, 'fraction': fraction, 'num_swaps': num_swaps,
                    'swap_length': swap_length, 'num_swaps_actual': ns_actual,
                    'n_swapped_points': n_swapped,
                    'fraction_actual': round(n_swapped / n, 6) if n else 0.0,
                    'seed': seed, 'model': model_name, 'condition': condition_name}
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
    keys = [k for k in ['condition', 'fraction', 'num_swaps', 'model'] if k in ok.columns]
    rows = []
    for name, g in ok.groupby(keys, dropna=False):
        row = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        row['n_runs'] = len(g)
        # swap_length is derived per series, so only its mean is meaningful across files
        if 'swap_length' in g.columns:
            row['mean_swap_length'] = round(g['swap_length'].mean(), 1)
        if 'fraction_actual' in g.columns:
            row['mean_fraction_actual'] = round(g['fraction_actual'].mean(), 6)
        for c in cols:
            row[f'mean_{c}'] = round(g[c].mean(), 4)
            row[f'std_{c}'] = round(g[c].std(), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(description='Swap — segment swap — TSB-AD (350)')
    ap.add_argument('--test', action='store_true', help='Smoke test: 3 files, small grid')
    ap.add_argument('--models', nargs='+', default=['IForest'],
                    help='TSB-AD models (IForest, MatrixProfile, Sub_PCA, POLY, KShapeAD, KMeansAD_U, ...)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-clean', action='store_true', help='Skip the clean (unswapped) anchor condition')
    ap.add_argument('--window-mode', choices=['clean', 'native'], default='clean',
                    help="'clean' (default): periodicity models use the window from the clean "
                         "signal, as in the original experiment. 'native': pure TSB-AD wrapper.")
    ap.add_argument('--injector', choices=['fast', 'reference'], default='fast',
                    help="'fast' (default): strictly disjoint segments, exact fraction. "
                         "'reference': the original ts_corruptor.inject_swap, which "
                         "under-delivers the fraction for swap_length > 1 (see docstring).")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    global LEGACY_CKPT
    LEGACY_CKPT = os.path.join(RESULTS_DIR, "checkpoint.csv")
    if not FILE_LIST_CSV.exists():
        print(f"[ERROR] File list not found: {FILE_LIST_CSV}")
        return

    files = pd.read_csv(FILE_LIST_CSV)['file_name'].tolist()
    file_paths = [str(DATA_DIR / f) for f in files]
    fractions, num_swaps_list = FRACTIONS, NUM_SWAPS
    if args.test:
        file_paths = file_paths[:3]
        fractions = [0.05, 0.20]
        num_swaps_list = [1, 10]
        print("!!! TEST MODE !!!")

    # (None, None) = the clean anchor; (fraction, num_swaps) otherwise
    conditions = [] if args.no_clean else [(None, None)]
    conditions += [(f, ns) for f in fractions for ns in num_swaps_list]

    print(f"\n{'='*60}\n  Swap — Segment Swap — TSB-AD (full 350)\n{'='*60}")
    print("  swap_length = fraction * n / (2 * num_swaps), derived per series")
    print("  total swapped fraction held constant; granularity varies")
    print(f"{'='*60}")
    print(f"Files: {len(file_paths)} | fractions: {fractions} | num_swaps: {num_swaps_list}")
    print(f"Seeds: {N_SEEDS} | clean anchor: {not args.no_clean} | conditions: {len(conditions)}")
    print(f"Models: {args.models} | Workers: {args.workers}")
    print(f"window_mode: {args.window_mode} | injector: {args.injector}")
    print(f"Total jobs: {len(file_paths) * len(conditions) * N_SEEDS}\n{'='*60}\n")

    if args.injector == 'reference':
        print("NOTE: --injector reference does not deliver the requested fraction for "
              "swap_length > 1 (bounds skips + cross-pair overlap). Use it for "
              "cross-checking only, not for the reported run.\n")

    all_results, completed = _load_checkpoints(args.models)
    if all_results:
        print(f"Loaded checkpoints: {len(all_results)} successful results for {args.models}.")

    jobs = []
    for fp in file_paths:
        fn = os.path.basename(fp)
        for frac, ns in conditions:
            cond = _cond_name(frac, ns)
            # The clean anchor is deterministic: one seed is enough, more would only
            # repeat identical work.
            seeds = [0] if frac is None else range(N_SEEDS)
            for seed in seeds:
                need = [m for m in args.models if f"{fn}|{cond}|{seed}|{m}" not in completed]
                if need:
                    jobs.append((fp, frac, ns, seed, need, args.window_mode, args.injector))

    # Longest-first scheduling: MatrixProfile costs ~O(n^2) and series span 1k-900k points,
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
            for fut in tqdm(as_completed(futs), total=len(jobs), desc="swap (segment)"):
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
