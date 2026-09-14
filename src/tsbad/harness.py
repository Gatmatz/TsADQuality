"""The shared main() of every TSB-AD corruption runner.

A runner describes its experiment in a `Spec` and calls `run_sweep(SPEC)`. The harness
does everything else: CLI, file list, resume from per-model checkpoints, longest-first
scheduling on a process pool, autosave, and the summary.

What a runner provides
----------------------
build_conditions(args) -> [condition(...), ...]
    The grid, including the clean anchor unless args.no_clean, reduced when args.test.

corrupt(ctx, params) -> (data, fields)
    Called once per (file, condition, seed). `ctx` carries the loaded series (see
    `_context`). Returns the (N, features) float array the detector will see — NaN rows
    for deleted points in the survivors family — and extra per-row columns. May raise
    `Skip(reason)`.

Evaluation families
-------------------
family='full'       The detector sees the whole corrupted series; metrics on all points.
family='survivors'  Points may be deleted (NaN rows). The detector sees only the
                    survivors, lost anomalies count as missed, lost normals are excluded
                    (`tsbad.evaluate.true_impact`), and the model window is clamped to
                    the survivor count.
    survivor_rules='always'      survivor path on every condition, clean included
                                 (missing / MNAR / Gilbert-Elliott runners)
    survivor_rules='if_dropped'  survivor path only when the condition actually deleted
                                 points; otherwise the plain full-series evaluation
                                 (compound runner)

Hooks must be module-level functions, not lambdas: jobs are pickled to worker processes.
"""
import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import paths
from .checkpoint import ckpt_path, load_all_for_summary, load_checkpoints, save_checkpoints
from .env import silence_warnings
from .evaluate import METRIC_COLS, fit_length, metric_fields, true_impact
from .models import get_hp, run_model_full, run_model_survivors, seed_job


class Skip(Exception):
    """Raised by a corrupt hook when a (file, condition) cannot be evaluated."""


def condition(name, row, params=None, n_seeds=1):
    """One grid point.

    name     the `condition` string — part of the resume key, so it must never change
    row      columns written into every result row of this condition
    params   what `corrupt` receives (defaults to `row`)
    n_seeds  corruption seeds to run for this condition
    """
    return {'name': name, 'row': dict(row), 'params': dict(row) if params is None else params,
            'n_seeds': n_seeds}


def default_seed_key(file_name, cond, seed, model_name):
    return f"{file_name}|{cond['name']}|{seed}|{model_name}"


@dataclass(frozen=True)
class Spec:
    name: str                                   # results/experiments/<name>/
    title: str
    family: str                                 # 'full' | 'survivors'
    build_conditions: Callable
    corrupt: Callable
    summary_keys: Sequence[str]                 # groupby columns of summary.csv
    survivor_rules: str = 'always'              # survivors family only: 'always' | 'if_dropped'
    summary_first: Sequence[str] = ()           # copied from the first row of each group
    summary_means: Sequence[Tuple[str, int]] = ()   # (column, decimals) -> mean_<column>
    seed_key: Callable = default_seed_key
    add_args: Optional[Callable] = None         # (ArgumentParser) -> None
    job_options: Optional[Callable] = None      # (args) -> dict, available as ctx.options
    before_run: Optional[Callable] = None       # (args, conditions) -> None, informational
    post_summary: Optional[Callable] = None     # (df_all, results_dir, args) -> None
    test_help: str = 'Smoke test: 3 files, reduced grid'
    test_files: int = 3
    progress_desc: str = 'Evaluating'

    def __post_init__(self):
        if self.family not in ('full', 'survivors'):
            raise ValueError(f"family must be 'full' or 'survivors', got {self.family!r}")
        if self.survivor_rules not in ('always', 'if_dropped'):
            raise ValueError(f"survivor_rules must be 'always' or 'if_dropped', "
                             f"got {self.survivor_rules!r}")


# ==========================================
# WORKER
# ==========================================
def _context(file_path, seed, options, family):
    # Official TSB-AD loading convention. reset_index because the injectors address rows by
    # LABEL (df.loc[...]) while generating POSITIONS — a dropna() gap would misalign them.
    df = pd.read_csv(file_path).dropna().reset_index(drop=True)
    value_col = df.columns[0]           # 'Data'
    label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
    clean_data = df.iloc[:, 0:-1].values.astype(float)          # (N, feats)

    from TSB_AD.utils.slidingWindows import find_length_rank
    # Metric window from the CLEAN signal, rank=1 — the official formula, but fixed so the
    # measuring stick stays identical across every condition of the grid.
    sliding_window = find_length_rank(clean_data[:, 0].reshape(-1, 1), rank=1)
    if family == 'survivors':
        sliding_window = int(sliding_window)

    return SimpleNamespace(
        df=df, value_col=value_col, label_col=label_col, clean_data=clean_data,
        labels=df[label_col].astype(int).to_numpy(), n=len(df),
        sliding_window=sliding_window, seed=seed,
        file_name=os.path.basename(file_path), options=options)


def _job(spec, file_path, cond, seed, model_names, window_mode, options):
    silence_warnings()   # defensive: ensure filters are active in this worker process
    file_name = os.path.basename(file_path)
    condition_name = cond['name']
    try:
        ctx = _context(file_path, seed, options, spec.family)
        try:
            data, fields = spec.corrupt(ctx, cond['params'])
            evaluate = _evaluate_full if spec.family == 'full' else _evaluate_survivors
            results = evaluate(spec, ctx, cond, data, fields, model_names, window_mode)
        except Skip as s:
            return {'status': 'skipped', 'file': file_name, 'condition': condition_name,
                    'reason': str(s)}
        return {'status': 'success', 'results': results}
    except Exception:
        return {'status': 'error', 'file': file_name, 'condition': condition_name,
                'error': traceback.format_exc()}


def _evaluate_full(spec, ctx, cond, data, fields, model_names, window_mode):
    from TSB_AD.evaluation.metrics import get_metrics

    if np.isnan(data).all():
        raise RuntimeError('All NaN after corruption')

    results = []
    for model_name in model_names:
        base = {'file': ctx.file_name, **cond['row'], **fields,
                'seed': ctx.seed, 'model': model_name, 'condition': cond['name']}
        try:
            hp = get_hp(model_name)
            seed_job(spec.seed_key(ctx.file_name, cond, ctx.seed, model_name))
            output = run_model_full(model_name, data, ctx.clean_data, hp, window_mode,
                                    ctx.file_name)

            if not isinstance(output, np.ndarray):
                results.append({**base, 'error': f'wrapper returned: {str(output)[:120]}'})
                continue

            m = get_metrics(output, ctx.labels, slidingWindow=ctx.sliding_window)
            results.append({**base, 'error': None, **metric_fields(m)})
        except Exception as e:
            results.append({**base, 'error': str(e)})
    return results


def _evaluate_survivors(spec, ctx, cond, data_full, fields, model_names, window_mode):
    from TSB_AD.evaluation.metrics import get_metrics

    n, labels = ctx.n, ctx.labels
    # The evaluation follows the REALISED NaNs, not the declared condition: a missing level
    # that happened to drop nothing is still evaluated on what actually happened.
    nan_mask = np.isnan(data_full).any(axis=1)
    dropped = True if spec.survivor_rules == 'always' else bool(nan_mask.any())

    masked_anomaly = nan_mask & (labels == 1)
    masked_normal = nan_mask & (labels == 0)
    n_lost_anomalies = int(masked_anomaly.sum())
    actual_missing_rate = float(nan_mask.sum()) / n

    # The detector sees only the survivors — no imputation
    model_data = data_full[~nan_mask] if dropped else data_full
    n_kept = len(model_data)

    if n_kept < ctx.sliding_window + 10:
        raise Skip(f'too few points after data loss: {n_kept}')
    # ~masked_normal keeps every anomaly position (lost ones are scored as missed), so this is
    # the same test as "the file has no anomaly at all"
    if labels[~masked_normal].sum() == 0:
        raise Skip('no anomaly survives in the evaluation set')

    results = []
    for model_name in model_names:
        base = {'file': ctx.file_name, **cond['row'], **fields,
                'actual_missing_rate': round(actual_missing_rate, 4),
                'n_lost_anomalies': n_lost_anomalies,
                'n_original': n, 'n_kept': n_kept,
                'metric_window': ctx.sliding_window,
                'seed': ctx.seed, 'model': model_name, 'condition': cond['name']}
        try:
            hp = get_hp(model_name)
            seed_job(spec.seed_key(ctx.file_name, cond, ctx.seed, model_name))
            output, model_window, clamped = run_model_survivors(
                model_name, model_data, ctx.clean_data, hp, window_mode,
                ctx.file_name, nan_mask, clamp=dropped)

            if not isinstance(output, np.ndarray):
                results.append({**base, 'model_window': model_window,
                                'window_clamped': clamped,
                                'error': f'wrapper returned: {str(output)[:120]}'})
                continue

            score = fit_length(output, n_kept)
            if dropped:
                eval_scores, eval_labels, eval_mask = true_impact(score, n, nan_mask, labels)
            else:
                eval_scores, eval_labels, eval_mask = score, labels, np.ones(n, dtype=bool)

            if np.isnan(eval_scores).any():
                results.append({**base, 'model_window': model_window,
                                'window_clamped': clamped,
                                'error': 'NaN left in evaluation scores'})
                continue

            m = get_metrics(eval_scores, eval_labels, slidingWindow=ctx.sliding_window)
            results.append({**base, 'model_window': model_window, 'window_clamped': clamped,
                            'n_evaluated': int(eval_mask.sum()), 'error': None,
                            **metric_fields(m)})
        except Exception as e:
            results.append({**base, 'error': str(e)})
    return results


# ==========================================
# SUMMARY
# ==========================================
def compute_summary(df_results, output_path, keys, first=(), means=()):
    cols = [c for c in METRIC_COLS if c in df_results.columns]
    ok = df_results[df_results['error'].isnull()]
    if ok.empty:
        print("No successful runs to summarize.")
        return
    keys = [k for k in keys if k in ok.columns]
    rows = []
    for name, g in ok.groupby(keys, dropna=False):
        row = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        row['n_runs'] = len(g)
        for c in first:
            if c in g.columns:
                row[c] = g[c].iloc[0]
        for c, decimals in means:
            if c in g.columns:
                row[f'mean_{c}'] = round(g[c].mean(), decimals)
        if 'window_clamped' in g.columns:
            row['n_window_clamped'] = int(g['window_clamped'].fillna(False).astype(bool).sum())
        for c in cols:
            row[f'mean_{c}'] = round(g[c].mean(), 4)
            row[f'std_{c}'] = round(g[c].std(), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


def _summarize(spec, results_dir, args):
    # Covers EVERY model on disk, not just the ones this run touched, so summary.csv stays
    # complete when models are run one at a time.
    df_all = load_all_for_summary(results_dir)
    if df_all.empty:
        print("No checkpoints found.")
        return
    print(f"Summarizing {len(df_all)} results across models: "
          f"{sorted(df_all['model'].dropna().unique())}")
    compute_summary(df_all, os.path.join(results_dir, "summary.csv"),
                    spec.summary_keys, spec.summary_first, spec.summary_means)
    if spec.post_summary:
        spec.post_summary(df_all, results_dir, args)


# ==========================================
# MAIN
# ==========================================
def _parser(spec):
    ap = argparse.ArgumentParser(description=spec.title)
    ap.add_argument('--test', action='store_true', help=spec.test_help)
    ap.add_argument('--models', nargs='+', default=['IForest'],
                    help='TSB-AD models (IForest, MatrixProfile, Sub_PCA, POLY, KShapeAD, '
                         'KMeansAD_U, AutoEncoder_2, StreamVAE, ...)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-clean', action='store_true',
                    help='Skip the clean (no-corruption) anchor condition')
    ap.add_argument('--window-mode', choices=['clean', 'native'], default='clean',
                    help="'clean' (default): periodicity models use the window from the clean "
                         "signal, as in the original experiment. 'native': pure TSB-AD wrapper "
                         "(re-estimates the window on the corrupted signal).")
    ap.add_argument('--files-csv', default=None,
                    help='Alternative file list (e.g. a validated subset table). '
                         'Reads a file_name/file column. Relative paths start at the project root.')
    ap.add_argument('--max-files', type=int, default=None, help='Cap the number of files')
    ap.add_argument('--results-dir', default=None,
                    help=f'Write checkpoints and summaries here instead of '
                         f'results/experiments/{spec.name}/ (e.g. for a verification run)')
    ap.add_argument('--summary-only', action='store_true',
                    help='Recompute the summary from existing checkpoints, run nothing')
    if spec.add_args:
        spec.add_args(ap)
    return ap


def run_sweep(spec, argv=None):
    args = _parser(spec).parse_args(argv)

    results_dir = str(paths.resolve(args.results_dir) if args.results_dir
                      else paths.results_dir(spec.name))
    os.makedirs(results_dir, exist_ok=True)

    if args.summary_only:
        _summarize(spec, results_dir, args)
        return

    list_path = paths.resolve(args.files_csv) if args.files_csv else paths.FILE_LIST_CSV
    if not list_path.exists():
        print(f"[ERROR] File list not found: {list_path}")
        return
    file_paths = [str(paths.DATA_DIR / f) for f in paths.read_file_list(list_path)]

    conditions = spec.build_conditions(args)
    if args.test:
        file_paths = file_paths[:spec.test_files]
        print("!!! TEST MODE !!!")
    elif args.max_files:
        file_paths = file_paths[:args.max_files]

    n_jobs_total = sum(len(file_paths) * c['n_seeds'] for c in conditions)
    print(f"\n{'='*60}\n  {spec.title}\n{'='*60}")
    print(f"File list: {list_path.name} | files: {len(file_paths)} | "
          f"conditions: {len(conditions)}")
    print(f"Models: {args.models} | Workers: {args.workers} | window_mode: {args.window_mode}")
    print(f"Results: {results_dir}")
    print(f"Total jobs: {n_jobs_total}\n{'='*60}\n")
    if spec.before_run:
        spec.before_run(args, conditions)

    all_results, completed = load_checkpoints(results_dir, args.models)
    if all_results:
        print(f"Loaded checkpoints: {len(all_results)} successful results for "
              f"{args.models} (failures will retry).")

    options = spec.job_options(args) if spec.job_options else {}
    jobs = []
    for fp in file_paths:
        fn = os.path.basename(fp)
        for cond in conditions:
            for seed in range(cond['n_seeds']):
                need = [m for m in args.models
                        if f"{fn}|{cond['name']}|{seed}|{m}" not in completed]
                if need:
                    jobs.append((fp, cond, seed, need, args.window_mode, options))

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
        new, skipped = 0, 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_job, spec, *j): j for j in jobs}
            for fut in tqdm(as_completed(futs), total=len(jobs), desc=spec.progress_desc):
                res = fut.result()
                if res['status'] == 'success':
                    all_results.extend(res['results'])
                    new += len(res['results'])
                elif res['status'] == 'skipped':
                    skipped += 1
                else:
                    print(f"Error in {res.get('file', '?')} / {res.get('condition', '?')}: "
                          f"{str(res.get('error', ''))[:200]}")
                if new >= 200:   # frequent autosave: a crash should cost minutes, not hours
                    save_checkpoints(results_dir, all_results, args.models)
                    new = 0
        save_checkpoints(results_dir, all_results, args.models)
        if skipped:
            print(f"Skipped {skipped} job(s) (too short / nothing left to evaluate).")
        print("\nFinal results saved to: "
              + ", ".join(os.path.basename(ckpt_path(results_dir, m)) for m in args.models))

    _summarize(spec, results_dir, args)
