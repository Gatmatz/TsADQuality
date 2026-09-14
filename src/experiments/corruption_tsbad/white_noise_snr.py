"""
White Gaussian Noise (SNR) robustness experiment — TSB-AD version.

Port of run_whitenoise_snr.py to the TSB-AD benchmark. Keeps the SAME corruption
(ts_corruptor.inject_white_noise_snr) and SAME SNR grid, but evaluates with the
OFFICIAL TSB-AD pipeline (run_Unsupervise_AD / run_Semisupervise_AD + find_length_rank
+ TSB_AD get_metrics) on the FULL 350-series TSB-AD-U eval set — identical to the
baseline setup, so clean-vs-corrupted is directly comparable.

Usage:
    python src/experiments/corruption_tsbad/white_noise_snr.py --test
    python src/experiments/corruption_tsbad/white_noise_snr.py --models IForest
    python src/experiments/corruption_tsbad/white_noise_snr.py --models IForest Sub_PCA POLY KShapeAD KMeansAD_U --workers 8
"""
import os

# Pin every numeric backend to one thread BEFORE numpy/numba/stumpy are imported.
# We already parallelise across files with ProcessPoolExecutor; without this each worker
# also grabs every core (stumpy.stump, used by MatrixProfile, is numba-parallel) and the
# resulting oversubscription slows the run badly. The project's other batch scripts all
# set these. Must stay above the numpy import to take effect.
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

    sklearn raises UndefinedMetricWarning whenever a threshold yields no predicted
    positives (precision = 0/0) — routine at low SNR where scores flatten, and it only
    touches the threshold-dependent F1s, never AUC/VUS. Called at import time so the
    spawned worker processes (which re-import this module) inherit the filters too.
    """
    from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # numpy moved RankWarning in 2.0; POLY's polyfit raises it on flat/noisy segments
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
# Full 350-series official eval list (the leaderboard set)
FILE_LIST_CSV = _ROOT / "TSB-AD" / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
DATA_DIR = _ROOT / "TSB-AD" / "Datasets" / "TSB-AD-U"
RESULTS_DIR = _ROOT / "results" / "experiments" / "white_noise_snr_tsbad"

SNRS_DB = [40, 30, 20, 10, 5, 0, -5, -10, -20]
N_SEEDS = 1

# Which TSB-AD wrapper each model uses (semi-supervised models need a train split)
SEMISUP = {'AutoEncoder', 'AutoEncoder_2', 'StreamVAE', 'CNN', 'LSTMAD'}

# Models whose window is derived from the data via find_length_rank(periodicity).
# Under corruption the official wrapper would re-estimate it on the NOISY signal; the
# original TSB-UAD experiment instead fixed the window from the CLEAN signal. In
# window_mode='clean' we replicate each wrapper body verbatim but inject the clean window.
# (IForest is absent on purpose: it uses a fixed slidingWindow=100, so corruption cannot
# change it, and it stays on the pure wrapper path.)
WINDOW_MODELS = {'MatrixProfile', 'POLY', 'Sub_PCA', 'KShapeAD', 'KMeansAD_U'}


def _run_model(model_name, data, clean_data, hp, window_mode, file_name):
    """Run a TSB-AD detector.

    window_mode='native' -> pure official wrapper (re-estimates window on the given data).
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

    # Each model's window uses ITS OWN official periodicity (e.g. KMeansAD_U uses rank=2),
    # estimated on the CLEAN signal so noise cannot move it.
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


def _get_hp(model_name):
    from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
    # AutoEncoder_2 shares AutoEncoder's HP
    key = 'AutoEncoder' if model_name in ('AutoEncoder', 'AutoEncoder_2') else model_name
    return dict(Optimal_Uni_algo_HP_dict.get(key, {}))


def process_single_job(job_args):
    _silence_warnings()   # defensive: ensure filters are active in this worker process
    (file_path, snr_db, seed, model_names, window_mode) = job_args
    file_name = os.path.basename(file_path)
    condition_name = "clean" if snr_db is None else f"snr_{snr_db}dB"

    try:
        from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD
        from TSB_AD.evaluation.metrics import get_metrics
        from TSB_AD.utils.slidingWindows import find_length_rank

        # 1. Load (official TSB-AD convention)
        df = pd.read_csv(file_path).dropna()
        value_col = df.columns[0]           # 'Data'
        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]

        clean_data = df.iloc[:, 0:-1].values.astype(float)          # (N, feats)
        # sliding window from the CLEAN signal (consistent with baseline, not confounded by noise)
        sliding_window = find_length_rank(clean_data[:, 0].reshape(-1, 1), rank=1)

        # 2. Corrupt (skip for the clean anchor)
        if snr_db is None:
            df_work = df
        else:
            corruptor = TSCorruptor(df.copy(), value_col=value_col, label_col=label_col, seed=seed)
            ts_corruptor.injectors.inject_white_noise_snr(corruptor, snr_db=snr_db)
            df_work = corruptor.get_corrupted_df()

        data = df_work.iloc[:, 0:-1].values.astype(float)
        label = df_work[label_col].astype(int).to_numpy()

        if np.isnan(data).all():
            return {'status': 'error', 'file': file_name, 'error': 'All NaN after corruption'}

        # 3. Model + eval via OFFICIAL pipeline
        results = []
        for model_name in model_names:
            try:
                hp = _get_hp(model_name)
                # Deterministic per-job seeding: some TSB-AD models are stochastic
                # (e.g. KMeansAD builds sklearn KMeans without random_state), so without
                # this the same job would give different scores on every run.
                # stable across processes/runs (Python's hash() is salted per process)
                _key = f"{file_name}|{snr_db}|{seed}|{model_name}".encode()
                _js = int(hashlib.md5(_key).hexdigest()[:8], 16) % (2**31 - 1)
                np.random.seed(_js)
                random.seed(_js)
                output = _run_model(model_name, data, clean_data, hp, window_mode, file_name)

                if not isinstance(output, np.ndarray):
                    results.append({'file': file_name, 'snr_db': snr_db, 'seed': seed,
                                    'model': model_name, 'condition': condition_name,
                                    'error': f'wrapper returned: {str(output)[:120]}'})
                    continue

                m = get_metrics(output, label, slidingWindow=sliding_window)
                row = {'file': file_name, 'snr_db': snr_db, 'seed': seed,
                       'model': model_name, 'condition': condition_name, 'error': None}
                # store ALL metrics returned by TSB-AD get_metrics (keys -> underscores)
                for k, v in m.items():
                    row[k.replace('-', '_')] = v
                results.append(row)
            except Exception as e:
                results.append({'file': file_name, 'snr_db': snr_db, 'seed': seed,
                                'model': model_name, 'condition': condition_name,
                                'error': str(e)})
        return {'status': 'success', 'results': results}

    except Exception:
        return {'status': 'error', 'file': file_name, 'error': traceback.format_exc()}


def _ckpt_path(model):
    """One checkpoint per model, so runs of different models never clobber each other."""
    return os.path.join(RESULTS_DIR, f"checkpoint_{model}.csv")


LEGACY_CKPT = None  # set in main(); the old single combined checkpoint.csv


def _load_checkpoints(models):
    """Load previous successful results for `models` only.

    Reads each model's own checkpoint plus, for backward compatibility, the legacy
    combined checkpoint.csv (filtered to these models). Returns (records, completed_keys).
    """
    records = []
    paths = [_ckpt_path(m) for m in models]
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

    # de-duplicate (legacy file may overlap a per-model file)
    seen, uniq, keys = set(), [], set()
    for r in records:
        k = f"{r['file']}|{r['condition']}|{r['seed']}|{r['model']}"
        if k in seen:
            continue
        seen.add(k); keys.add(k); uniq.append(r)
    return uniq, keys


def _save_checkpoints(all_results, models):
    """Write results back, one file per model. Only touches this run's models."""
    if not all_results:
        return
    df = pd.DataFrame(all_results)
    for m in models:
        sub = df[df['model'] == m]
        if not sub.empty:
            sub.to_csv(_ckpt_path(m), index=False)


def _load_all_results_for_summary():
    """Every checkpoint on disk (all models, plus legacy), de-duplicated."""
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
    metric_cols = ['AUC_PR', 'AUC_ROC', 'VUS_PR', 'VUS_ROC', 'Standard_F1',
                   'PA_F1', 'Event_based_F1', 'R_based_F1', 'Affiliation_F']
    metric_cols = [c for c in metric_cols if c in df_results.columns]
    ok = df_results[df_results['error'].isnull()]
    if ok.empty:
        print("No successful runs to summarize.")
        return
    rows = []
    for (snr_db, condition, model), g in ok.groupby(['snr_db', 'condition', 'model'], dropna=False):
        row = {'snr_db': snr_db, 'condition': condition, 'model': model, 'n_runs': len(g)}
        for c in metric_cols:
            if c in g.columns:
                row[f'mean_{c}'] = round(g[c].mean(), 4)
                row[f'std_{c}'] = round(g[c].std(), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(description='White Noise SNR robustness — TSB-AD (350)')
    ap.add_argument('--test', action='store_true', help='Smoke test: 3 files, 2 SNRs')
    ap.add_argument('--models', nargs='+', default=['IForest'],
                    help='TSB-AD models (IForest, Sub_PCA, POLY, KShapeAD, KMeansAD_U, AutoEncoder_2, StreamVAE, ...)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-clean', action='store_true', help='Skip the clean (no-noise) anchor condition')
    ap.add_argument('--window-mode', choices=['clean', 'native'], default='clean',
                    help="'clean' (default): periodicity models use the window from the clean "
                         "signal, as in the original experiment. 'native': pure TSB-AD wrapper "
                         "(re-estimates the window on the corrupted signal).")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    global LEGACY_CKPT
    LEGACY_CKPT = os.path.join(RESULTS_DIR, "checkpoint.csv")
    if not FILE_LIST_CSV.exists():
        print(f"[ERROR] File list not found: {FILE_LIST_CSV}")
        return

    files = pd.read_csv(FILE_LIST_CSV)['file_name'].tolist()
    file_paths = [str(DATA_DIR / f) for f in files]
    snrs = list(SNRS_DB)
    n_seeds = N_SEEDS
    if args.test:
        file_paths = file_paths[:3]
        snrs = [40, 10]
        print("!!! TEST MODE !!!")
    conditions = ([None] if not args.no_clean else []) + snrs

    print(f"\n{'='*60}\n  White Noise (SNR) Robustness — TSB-AD (full 350)\n{'='*60}")
    print(f"Files: {len(file_paths)} | SNRs: {snrs} | clean anchor: {not args.no_clean}")
    print(f"Models: {args.models} | Workers: {args.workers} | window_mode: {args.window_mode}")
    print(f"Total jobs: {len(file_paths) * len(conditions) * n_seeds}\n{'='*60}\n")

    # Resume keys use `condition` (a string), not snr_db: the clean anchor stores
    # snr_db=None, which pandas reads back as NaN, so an snr_db-based key would never
    # match on resume and clean rows would be re-run and duplicated.
    all_results, completed = _load_checkpoints(args.models)
    if all_results:
        print(f"Loaded checkpoints: {len(all_results)} successful results for "
              f"{args.models} (failures will retry).")

    jobs = []
    for fp in file_paths:
        fn = os.path.basename(fp)
        for snr_db in conditions:
            cond = "clean" if snr_db is None else f"snr_{snr_db}dB"
            for seed in range(n_seeds):
                need = [m for m in args.models
                        if f"{fn}|{cond}|{seed}|{m}" not in completed]
                if need:
                    jobs.append((fp, snr_db, seed, need, args.window_mode))

    # Longest-first scheduling. MatrixProfile costs ~O(n^2) and series length spans
    # 18k to 900k points, so ~10 files carry ~75% of the work. Starting those first
    # keeps every worker busy instead of leaving one grinding a 900k series at the end.
    # Ordering only changes scheduling, never results (each job is independent).
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
            for fut in tqdm(as_completed(futs), total=len(jobs), desc="Evaluating"):
                res = fut.result()
                if res['status'] == 'success':
                    all_results.extend(res['results'])
                    new += len(res['results'])
                else:
                    print(f"Error in {res.get('file','?')}: {str(res.get('error',''))[:200]}")
                if new >= 200:   # frequent autosave: a crash should cost minutes, not hours
                    _save_checkpoints(all_results, args.models)
                    new = 0
        _save_checkpoints(all_results, args.models)
        print("\nFinal results saved to: "
              + ", ".join(os.path.basename(_ckpt_path(m)) for m in args.models))

    # Summary covers EVERY model on disk, not just the ones this run touched,
    # so summary.csv stays complete when models are run one at a time.
    df_all = _load_all_results_for_summary()
    if not df_all.empty:
        print(f"Summarizing {len(df_all)} results across models: "
              f"{sorted(df_all['model'].dropna().unique())}")
        compute_summary(df_all, os.path.join(RESULTS_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
