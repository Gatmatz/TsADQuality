"""
Spike Corruption on NORMAL points only — TSB-AD version.

Port of run_spikes_normal_only.py (the corrected methodology) to the TSB-AD benchmark,
mirroring white_noise_snr.py step for step.

Why normal-only, quoting the original script:
  - Spikes landing on anomaly points artificially boost AUC — the model flags them
    "correctly" but for the wrong reason (the spike, not the anomaly pattern).
  - Spikes on normal points are the real threat: false anomalies that confuse the model.

So corruption_target='only_normal', which is what makes this the methodologically sound
variant of run_spikes.py (that one used the default 'global' target).

Evaluation uses the OFFICIAL TSB-AD pipeline (run_Unsupervise_AD / run_Semisupervise_AD +
find_length_rank + TSB_AD get_metrics) on the full 350-series TSB-AD-U eval list. A `clean`
anchor condition runs the identical pipeline with no corruption, so it reproduces the TSB-AD
baseline exactly and the degradation curve starts from the right point.

Key question: "How much do false spikes in normal data degrade anomaly detection?"

Usage:
    python src/experiments/corruption_tsbad/spikes_normal_only.py --test
    python src/experiments/corruption_tsbad/spikes_normal_only.py --models IForest --workers 4
    python src/experiments/corruption_tsbad/spikes_normal_only.py --models MatrixProfile --workers 4
    # optional: also sweep burst spikes (still normal-only)
    python src/experiments/corruption_tsbad/spikes_normal_only.py --models IForest --burst 3 10
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

from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors  # registers injectors

# ==========================================
# CONFIG
# ==========================================
_ROOT = Path(PROJECT_ROOT)
FILE_LIST_CSV = _ROOT / "TSB-AD" / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
DATA_DIR = _ROOT / "TSB-AD" / "Datasets" / "TSB-AD-U"
RESULTS_DIR = _ROOT / "results" / "experiments" / "spikes_normal_only_tsbad"

# Same grid as the original run_spikes_normal_only.py
FRACTIONS = [0.01, 0.05, 0.10, 0.20]
MULTIPLIERS = [3.0, 5.0, 10.0]
N_SEEDS = 1

# Spikes go only on non-anomaly points — the whole point of this variant.
CORRUPTION_TARGET = 'only_normal'

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


def _cond_name(fraction, multiplier, seq_len):
    """seq_len 1 = point spikes (the original's only mode); >1 = burst."""
    if fraction is None:
        return "clean"
    base = f"frac_{fraction}_mult_{multiplier}"
    return base if seq_len <= 1 else f"{base}_burst_{seq_len}"


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
    (file_path, fraction, multiplier, seq_len, seed, model_names, window_mode) = job_args
    file_name = os.path.basename(file_path)
    condition_name = _cond_name(fraction, multiplier, seq_len)

    try:
        from TSB_AD.evaluation.metrics import get_metrics
        from TSB_AD.utils.slidingWindows import find_length_rank

        # 1. Load (official TSB-AD convention)
        df = pd.read_csv(file_path).dropna()
        value_col = df.columns[0]           # 'Data'
        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]

        clean_data = df.iloc[:, 0:-1].values.astype(float)          # (N, feats)
        # metric window from the CLEAN signal, rank=1 — the official formula, but fixed so the
        # measuring stick stays identical across conditions
        sliding_window = find_length_rank(clean_data[:, 0].reshape(-1, 1), rank=1)

        n_normal = int((df[label_col].astype(int).to_numpy() == 0).sum())
        n_spikes = 0

        # 2. Corrupt — spikes on NORMAL points only (skip for the clean anchor)
        if fraction is None:
            df_work = df
        else:
            corruptor = TSCorruptor(df.copy(), value_col=value_col, label_col=label_col,
                                    seed=seed, corruption_target=CORRUPTION_TARGET)
            ts_corruptor.injectors.inject_spikes(
                corruptor, fraction=fraction, multiplier=multiplier,
                sequential=(seq_len > 1), sequence_length=max(1, seq_len))
            df_work = corruptor.get_corrupted_df()
            # record how many points were actually spiked (provenance, as in the original)
            for action in corruptor.get_corruption_report()['action_details']:
                if action['type'] == 'spikes':
                    n_spikes = action['count']

        data = df_work.iloc[:, 0:-1].values.astype(float)
        label = df_work[label_col].astype(int).to_numpy()

        if np.isnan(data).all():
            return {'status': 'error', 'file': file_name, 'error': 'All NaN after corruption'}

        # 3. Model + eval via the OFFICIAL pipeline
        results = []
        for model_name in model_names:
            base = {'file': file_name, 'fraction': fraction, 'multiplier': multiplier,
                    'sequence_length': seq_len, 'n_spikes': n_spikes, 'n_normal': n_normal,
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
        # key on `condition` (a string): the clean anchor stores fraction=None, which pandas
        # reads back as NaN, so a fraction-based key would never match on resume
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


def compute_summary(df_results, output_path):
    cols = [c for c in METRIC_COLS if c in df_results.columns]
    ok = df_results[df_results['error'].isnull()]
    if ok.empty:
        print("No successful runs to summarize.")
        return
    keys = [k for k in ['condition', 'fraction', 'multiplier', 'sequence_length', 'model']
            if k in ok.columns]
    rows = []
    for name, g in ok.groupby(keys, dropna=False):
        row = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        row['n_runs'] = len(g)
        if 'n_spikes' in g.columns:
            row['mean_n_spikes'] = round(g['n_spikes'].mean(), 1)
        for c in cols:
            row[f'mean_{c}'] = round(g[c].mean(), 4)
            row[f'std_{c}'] = round(g[c].std(), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(description='Spikes on normal points only — TSB-AD (350)')
    ap.add_argument('--test', action='store_true', help='Smoke test: 3 files, small grid')
    ap.add_argument('--models', nargs='+', default=['IForest'],
                    help='TSB-AD models (IForest, MatrixProfile, Sub_PCA, POLY, KShapeAD, KMeansAD_U, ...)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--fractions', nargs='+', type=float, default=None,
                    help=f'Override fractions (default {FRACTIONS})')
    ap.add_argument('--multipliers', nargs='+', type=float, default=None,
                    help=f'Override multipliers (default {MULTIPLIERS})')
    ap.add_argument('--burst', nargs='+', type=int, default=None,
                    help='Optional burst lengths to sweep in ADDITION to point spikes '
                         '(e.g. --burst 3 10). Still normal-only. Off by default, matching '
                         'the original run_spikes_normal_only.py.')
    ap.add_argument('--no-clean', action='store_true', help='Skip the clean (no-spike) anchor')
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
    multipliers = args.multipliers if args.multipliers else MULTIPLIERS
    seq_lens = [1] + (args.burst if args.burst else [])   # 1 = point spikes
    if args.test:
        file_paths = file_paths[:3]
        fractions, multipliers, seq_lens = [0.05, 0.20], [3.0], [1]
        print("!!! TEST MODE !!!")

    # (fraction, multiplier, seq_len); None fraction = the clean anchor
    conditions = [] if args.no_clean else [(None, None, 1)]
    conditions += [(f, m, s) for f in fractions for m in multipliers for s in seq_lens]

    print(f"\n{'='*60}\n  Spikes on NORMAL points only — TSB-AD (full 350)\n{'='*60}")
    print(f"Files: {len(file_paths)} | fractions: {fractions} | multipliers: {multipliers}")
    print(f"Spike shape: {'point' if seq_lens == [1] else f'point + bursts {seq_lens[1:]}'}")
    print(f"corruption_target: {CORRUPTION_TARGET} | clean anchor: {not args.no_clean}")
    print(f"Conditions: {len(conditions)} | Models: {args.models} | Workers: {args.workers} "
          f"| window_mode: {args.window_mode}")
    print(f"Total jobs: {len(file_paths) * len(conditions) * N_SEEDS}\n{'='*60}\n")

    all_results, completed = _load_checkpoints(args.models)
    if all_results:
        print(f"Loaded checkpoints: {len(all_results)} successful results for {args.models}.")

    jobs = []
    for fp in file_paths:
        fn = os.path.basename(fp)
        for (frac, mult, sl) in conditions:
            cond = _cond_name(frac, mult, sl)
            for seed in range(N_SEEDS):
                need = [m for m in args.models if f"{fn}|{cond}|{seed}|{m}" not in completed]
                if need:
                    jobs.append((fp, frac, mult, sl, seed, need, args.window_mode))

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
            for fut in tqdm(as_completed(futs), total=len(jobs), desc="spikes(normal-only)"):
                res = fut.result()
                if res['status'] == 'success':
                    all_results.extend(res['results'])
                    new += len(res['results'])
                else:
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
