"""
Gilbert-Elliott Channel Model — True Impact Experiment, TSB-AD version.

Port of run_gilbert_elliott_true_impact.py to the TSB-AD benchmark, mirroring
run_whitenoise_snr_tsbad.py / run_spikes_normal_only_tsbad.py step for step.

Gilbert-Elliott model (2-state Markov chain over the series):
  - Good state: no corruption.  Bad state: the point is dropped (set to NaN).
  - alpha = p(Good -> Bad): burst frequency.
  - beta  = p(Bad -> Good):  burst duration, expected length 1/beta.
  - Expected corruption rate = alpha / (alpha + beta).

Evaluation — "true impact", i.e. NO imputation (identical policy to the original):
  1. Drop the NaN points; the detector sees a SHORTER series.
  2. Map the scores back onto the original timeline.
  3. Lost ANOMALY points get score = 0  -> they count as false negatives.
  4. Lost NORMAL points are EXCLUDED from the evaluation.
This is what makes the number honest: burst loss cannot be rewarded for hiding
the anomalies it destroyed.

Detection and metrics use the OFFICIAL TSB-AD pipeline (run_Unsupervise_AD /
run_Semisupervise_AD + find_length_rank + TSB_AD get_metrics) on the full
350-series TSB-AD-U eval list. A `clean` anchor condition runs the identical
pipeline with no corruption; there nan_mask is empty, so it reduces exactly to
the official TSB-AD baseline and the degradation curve starts from the right point.

Key question: "How do bursty missing patterns (vs random missing) affect
               anomaly detection?"

Usage:
    python src/experiments/corruption/run_gilbert_elliott_true_impact_tsbad.py --test
    python src/experiments/corruption/run_gilbert_elliott_true_impact_tsbad.py --models IForest --workers 4
    python src/experiments/corruption/run_gilbert_elliott_true_impact_tsbad.py --models MatrixProfile --workers 4
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
RESULTS_DIR = _ROOT / "results" / "experiments" / "gilbert_elliott_true_impact_tsbad"

# Same grid as the original run_gilbert_elliott_true_impact.py
ALPHAS = [0.01, 0.05, 0.10]    # p(Good -> Bad) — burst frequency
BETAS = [0.10, 0.25, 0.50]     # p(Bad -> Good) — burst duration (1/beta)
N_SEEDS = 1

# The original passes no corruption_target, i.e. the default. Bursts may therefore land on
# anomaly points — deliberately: losing anomalies is the effect this experiment measures.
CORRUPTION_TARGET = 'global'

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


def _cond_name(alpha, beta):
    return "clean" if alpha is None else f"a_{alpha}_b_{beta}"


def _get_hp(model_name):
    from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
    key = 'AutoEncoder' if model_name in ('AutoEncoder', 'AutoEncoder_2') else model_name
    return dict(Optimal_Uni_algo_HP_dict.get(key, {}))


def _model_window(model_name, clean_data, hp, n_kept):
    """Window for periodicity-derived models, estimated on the CLEAN signal.

    Clamped to the length actually reaching the detector: this experiment DELETES points, so a
    window valid for the full series can exceed the survivor count. Same clamp as the original
    (`min(w, n_kept // 4)`, floor 10). It is reported per row so a firing clamp is visible.
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
            raise ValueError(f'train split collapsed after burst loss: {train_kept} points')
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


def process_single_job(job_args):
    _silence_warnings()   # defensive: ensure filters are active in this worker process
    (file_path, alpha, beta, seed, model_names, window_mode) = job_args
    file_name = os.path.basename(file_path)
    condition_name = _cond_name(alpha, beta)
    expected_rate = None if alpha is None else alpha / (alpha + beta)
    expected_burst_len = None if alpha is None else 1.0 / beta

    try:
        from TSB_AD.evaluation.metrics import get_metrics
        from TSB_AD.utils.slidingWindows import find_length_rank

        # 1. Load (official TSB-AD convention). reset_index because the injectors address rows
        # by LABEL (df.loc[...]) while generating POSITIONS — a dropna() gap would misalign them.
        df = pd.read_csv(file_path).dropna().reset_index(drop=True)
        value_col = df.columns[0]           # 'Data'
        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]

        clean_data = df.iloc[:, 0:-1].values.astype(float)          # (N, feats)
        labels = df[label_col].astype(int).to_numpy()
        n = len(df)

        # Metric window from the CLEAN signal, rank=1 — the official formula, but fixed so the
        # measuring stick stays identical across every burst level.
        sliding_window = int(find_length_rank(clean_data[:, 0].reshape(-1, 1), rank=1))

        # 2. Corrupt — Gilbert-Elliott bursts of MISSING points (skip for the clean anchor)
        if alpha is None:
            data_full = clean_data
            nan_mask = np.zeros(n, dtype=bool)
        else:
            corruptor = TSCorruptor(df.copy(), value_col=value_col, label_col=label_col,
                                    seed=seed, corruption_target=CORRUPTION_TARGET)
            ts_corruptor.injectors.inject_gilbert_elliott(
                corruptor, p_good_to_bad=alpha, p_bad_to_good=beta, noise_type='missing')
            df_work = corruptor.get_corrupted_df()
            data_full = df_work.iloc[:, 0:-1].values.astype(float)
            nan_mask = np.isnan(data_full).any(axis=1)

        masked_anomaly = nan_mask & (labels == 1)
        masked_normal = nan_mask & (labels == 0)
        n_lost_anomalies = int(masked_anomaly.sum())
        actual_missing_rate = float(nan_mask.sum()) / n

        # 3. The detector sees only the survivors — no imputation
        model_data = data_full[~nan_mask]
        n_kept = len(model_data)

        if n_kept < sliding_window + 10:
            return {'status': 'skipped', 'file': file_name,
                    'reason': f'too few points after burst loss: {n_kept}'}
        if labels[~masked_normal].sum() == 0:
            return {'status': 'skipped', 'file': file_name,
                    'reason': 'no anomaly survives in the evaluation set'}

        # 4. Model + eval via the OFFICIAL pipeline
        results = []
        for model_name in model_names:
            base = {'file': file_name, 'alpha': alpha, 'beta': beta,
                    'expected_rate': None if expected_rate is None else round(expected_rate, 4),
                    'expected_burst_len': None if expected_burst_len is None else round(expected_burst_len, 1),
                    'actual_missing_rate': round(actual_missing_rate, 4),
                    'n_lost_anomalies': n_lost_anomalies,
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
                eval_mask = ~masked_normal            # normals destroyed by the burst = excluded

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
        # key on `condition` (a string): the clean anchor stores alpha=None, which pandas
        # reads back as NaN, so an alpha-based key would never match on resume
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
    keys = [k for k in ['condition', 'alpha', 'beta', 'model'] if k in ok.columns]
    rows = []
    for name, g in ok.groupby(keys, dropna=False):
        row = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        row['n_runs'] = len(g)
        for extra in ('expected_rate', 'expected_burst_len'):
            if extra in g.columns:
                row[extra] = g[extra].iloc[0]
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


def main():
    ap = argparse.ArgumentParser(description='Gilbert-Elliott true impact — TSB-AD (350)')
    ap.add_argument('--test', action='store_true', help='Smoke test: 3 files, small grid')
    ap.add_argument('--models', nargs='+', default=['IForest'],
                    help='TSB-AD models (IForest, MatrixProfile, Sub_PCA, POLY, KShapeAD, KMeansAD_U, ...)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--alphas', nargs='+', type=float, default=None,
                    help=f'Override alphas, p(Good->Bad) (default {ALPHAS})')
    ap.add_argument('--betas', nargs='+', type=float, default=None,
                    help=f'Override betas, p(Bad->Good) (default {BETAS})')
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
    alphas = args.alphas if args.alphas else ALPHAS
    betas = args.betas if args.betas else BETAS
    if args.test:
        file_paths = file_paths[:3]
        alphas, betas = [0.05], [0.10, 0.50]
        print("!!! TEST MODE !!!")

    # (alpha, beta); None alpha = the clean anchor
    conditions = [] if args.no_clean else [(None, None)]
    conditions += [(a, b) for a in alphas for b in betas]

    print(f"\n{'='*60}\n  Gilbert-Elliott — True Impact (no imputation) — TSB-AD (full 350)\n{'='*60}")
    print("  Lost anomalies -> score=0 (false negatives)")
    print("  Lost normal points -> excluded from evaluation")
    print(f"{'='*60}")
    print(f"Files: {len(file_paths)} | alphas (p G->B): {alphas} | betas (p B->G): {betas}")
    print(f"corruption_target: {CORRUPTION_TARGET} | clean anchor: {not args.no_clean}")
    print(f"Conditions: {len(conditions)} | Models: {args.models} | Workers: {args.workers} "
          f"| window_mode: {args.window_mode}")
    print()
    print(f"{'alpha':>8} {'beta':>8} {'exp_rate':>10} {'exp_burst':>10}")
    print("-" * 40)
    for a in alphas:
        for b in betas:
            print(f"{a:>8.2f} {b:>8.2f} {a/(a+b):>10.2%} {1/b:>10.1f}")
    print(f"\nTotal jobs: {len(file_paths) * len(conditions) * N_SEEDS}\n{'='*60}\n")

    all_results, completed = _load_checkpoints(args.models)
    if all_results:
        print(f"Loaded checkpoints: {len(all_results)} successful results for {args.models}.")

    jobs = []
    for fp in file_paths:
        fn = os.path.basename(fp)
        for (a, b) in conditions:
            cond = _cond_name(a, b)
            for seed in range(N_SEEDS):
                need = [m for m in args.models if f"{fn}|{cond}|{seed}|{m}" not in completed]
                if need:
                    jobs.append((fp, a, b, seed, need, args.window_mode))

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
            for fut in tqdm(as_completed(futs), total=len(jobs), desc="gilbert-elliott(true-impact)"):
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
            print(f"Skipped {skipped} job(s): too few points / no anomaly left after burst loss.")
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
