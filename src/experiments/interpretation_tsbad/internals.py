"""
Internal algorithmic analysis on TSB-AD — WHY each detector is robust or vulnerable.

The TSB-AD counterpart of src/experiments/analysis/run_internal_analysis.py (thesis, TSB-UAD).
Same measurements (see tsbad.internals), same 74 corruption conditions, taken from the TSB-AD
detectors fitted exactly as the corruption experiments fit them.

Conditions are not redefined here: they come from the TSB-AD corruption runners themselves
(build_conditions + corrupt), so the corruption is identical to the one the performance
numbers were measured on:

    white_noise_snr       9 SNR levels
    spikes_normal_only    4 fractions x 3 multipliers
    missing_mnar          mcar and mnar_extreme x 4 fractions (the thesis arms)
    swap_segment          5 fractions x 5 num_swaps
    freeze                4 fractions x 5 num_stucks

One job = one (file, model): the clean internals are measured once per window policy (the clean
anchor of the full-family experiments, and the clamped clean anchor of the missing experiment),
then every condition is compared against the clean reference of its own experiment.
Rows are keyed on file|experiment|condition|seed|model (condition names repeat across
experiments, e.g. frac_0.01_ns_1 exists in both freeze and swap_segment).

Missing values follow the thesis: internals on the surviving points with their labels, the model
window clamped as in the experiment, no nearest-neighbour index comparison for MatrixProfile.
Files the experiment itself skips (too few points, no anomaly left) are recorded as skipped.

Usage:
    python src/experiments/interpretation_tsbad/internals.py --test
    python src/experiments/interpretation_tsbad/internals.py --models IForest MatrixProfile Sub_PCA --workers 4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/experiments/ (corruption_tsbad.*)
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import argparse  # noqa: E402
import importlib  # noqa: E402
import os  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from tsbad import paths  # noqa: E402
from tsbad.env import silence_warnings  # noqa: E402
from tsbad.harness import Skip, _context, _parser  # noqa: E402
from tsbad.internals import PHASE1_MODELS, compare, measure  # noqa: E402
from tsbad.models import get_hp, run_model_full, run_model_survivors, seed_job  # noqa: E402

RESULTS_NAME = 'internal_analysis_tsbad'
DEFAULT_FILES_CSV = 'results/tables/representative_subset_tsb_ad_vuspr_big_n200.csv'

# (corruption_tsbad runner, argument overrides) — together the thesis' 74 conditions
EXPERIMENTS = [
    ('white_noise_snr', {}),
    ('spikes_normal_only', {}),
    ('missing_mnar', {'mechanisms': ['mcar', 'mnar_extreme']}),
    ('swap_segment', {}),
    ('freeze', {}),
]

KEY_COLS = ['file', 'experiment', 'condition', 'seed', 'model']


def experiment_grid(name, overrides, test):
    """(spec, job options, conditions) of one corruption runner, clean anchor excluded."""
    spec = importlib.import_module(f'corruption_tsbad.{name}').SPEC
    args = _parser(spec).parse_args(['--test'] if test else [])
    for key, value in overrides.items():
        setattr(args, key, value)
    options = spec.job_options(args) if spec.job_options else {}
    conditions = [c for c in spec.build_conditions(args) if c['name'] != 'clean']
    return spec, options, conditions


def expected_keys(test):
    return {(spec.name, c['name'], seed)
            for name, overrides in EXPERIMENTS
            for spec, _, conds in [experiment_grid(name, overrides, test)]
            for c in conds for seed in range(c['n_seeds'])}


def _same_scores(a, b):
    return isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)


# ==========================================
# WORKER
# ==========================================
def process_file_model(job):
    """One (file, model): clean internals once, then every condition of every experiment."""
    silence_warnings()
    file_path, model_name, test, check_scores = job
    file_name = os.path.basename(file_path)
    try:
        base = _context(file_path, seed=0, options={}, family='full')
        clean, labels, n = base.clean_data, base.labels, base.n
        hp = get_hp(model_name)
        clean_refs = {}

        def clean_reference(clamp):
            """Clean internals under the window policy of the experiment's own clean anchor.

            Full-family experiments use the raw clean-signal window; the missing experiments clamp
            it (floor 10) even on the clean anchor. Comparing a corrupted run against a clean run
            with a different window would measure the window change, not the corruption.
            """
            if clamp not in clean_refs:
                key = f"{file_name}|clean|0|{model_name}"
                seed_job(key)
                scores, internals = measure(model_name, clean, clean, labels, hp, clamp=clamp)
                if check_scores:
                    seed_job(key)
                    if clamp:
                        ref = run_model_survivors(model_name, clean, clean, hp, 'clean', file_name,
                                                  np.zeros(n, dtype=bool), clamp=True)[0]
                    else:
                        ref = run_model_full(model_name, clean, clean, hp, 'clean', file_name)
                    if not _same_scores(scores, ref):
                        raise RuntimeError('score check failed on the clean series')
                clean_refs[clamp] = internals
            return clean_refs[clamp]

        rows = []
        for exp_name, overrides in EXPERIMENTS:
            spec, options, conditions = experiment_grid(exp_name, overrides, test)
            for cond in conditions:
                for seed in range(cond['n_seeds']):
                    row = {'file': file_name, 'experiment': spec.name, 'condition': cond['name'],
                           'seed': seed, 'model': model_name}
                    try:
                        ctx = SimpleNamespace(**vars(base))
                        ctx.seed, ctx.options = seed, options
                        if spec.family == 'survivors':
                            ctx.sliding_window = int(ctx.sliding_window)
                        data, _ = spec.corrupt(ctx, cond['params'])
                        key = spec.seed_key(file_name, cond, seed, model_name)

                        if spec.family == 'full':
                            clean_int = clean_reference(clamp=False)
                            seed_job(key)
                            scores, corr_int = measure(model_name, data, clean, labels, hp, clamp=False)
                            can_compare_nn = True
                            if check_scores:
                                seed_job(key)
                                ref = run_model_full(model_name, data, clean, hp, 'clean', file_name)
                        else:
                            nan_mask = np.isnan(data).any(axis=1)
                            dropped = True if spec.survivor_rules == 'always' else bool(nan_mask.any())
                            model_data = data[~nan_mask] if dropped else data
                            kept_labels = labels[~nan_mask] if dropped else labels
                            n_kept = len(model_data)
                            # the experiment skips these files, so there is no performance to relate to
                            if n_kept < ctx.sliding_window + 10:
                                raise Skip(f'too few points after data loss: {n_kept}')
                            if labels[~(nan_mask & (labels == 0))].sum() == 0:
                                raise Skip('no anomaly survives in the evaluation set')
                            clean_int = clean_reference(clamp=spec.survivor_rules == 'always')
                            seed_job(key)
                            scores, corr_int = measure(model_name, model_data, clean, kept_labels, hp,
                                                       clamp=dropped)
                            can_compare_nn = not dropped   # indices refer to different timelines
                            row['n_kept'] = n_kept
                            if check_scores:
                                seed_job(key)
                                ref = run_model_survivors(model_name, model_data, clean, hp, 'clean',
                                                          file_name, nan_mask, clamp=dropped)[0]

                        if check_scores and not _same_scores(scores, ref):
                            row['error'] = 'score check failed'
                        else:
                            row['clean_window'] = clean_int['window']
                            row['model_window'] = corr_int['window']
                            row.update(compare(model_name, clean_int, corr_int, can_compare_nn))
                            row['error'] = None
                    except Skip as s:
                        row['error'] = f'skipped: {s}'
                    except Exception as e:
                        row['error'] = str(e)[:200]
                    rows.append(row)

        return {'status': 'success', 'rows': rows}
    except Exception:
        return {'status': 'error', 'file': file_name, 'model': model_name,
                'error': traceback.format_exc()[-400:]}


# ==========================================
# CHECKPOINTS (one file per model)
# ==========================================
def _ckpt(results_dir, model):
    return os.path.join(results_dir, f"checkpoint_{model}.csv")


def load_rows(results_dir, models):
    frames = [pd.read_csv(_ckpt(results_dir, m), low_memory=False)
              for m in models if os.path.exists(_ckpt(results_dir, m))]
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    # failures are retried; skips are final (the experiment skips the same files)
    keep = df['error'].isna() | df['error'].astype(str).str.startswith('skipped')
    return df[keep].drop_duplicates(KEY_COLS, keep='last').to_dict('records')


def save_rows(results_dir, rows, models):
    if not rows:
        return
    df = pd.DataFrame(rows).drop_duplicates(KEY_COLS, keep='last')
    for m in models:
        sub = df[df['model'] == m]
        if not sub.empty:
            sub.to_csv(_ckpt(results_dir, m), index=False)


# ==========================================
# SUMMARY & CASE STUDIES (as in the thesis, grouped per experiment and condition)
# ==========================================
def compute_aggregate_summary(df_all, output_path):
    """Mean ± std per (experiment, condition, model) for each metric."""
    df_ok = df_all[df_all['error'].isna()] if 'error' in df_all.columns else df_all
    if df_ok.empty:
        print("No successful results to summarize.")
        return
    metric_cols = [c for c in df_ok.columns if c not in KEY_COLS + ['error']]
    summary_rows = []
    for (experiment, condition, model), group in df_ok.groupby(['experiment', 'condition', 'model']):
        row = {'experiment': experiment, 'condition': condition, 'model': model, 'n_files': len(group)}
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors='coerce').dropna()
            if len(vals) > 0:
                row[f'{col}_mean'] = round(vals.mean(), 4)
                row[f'{col}_std'] = round(vals.std(), 4)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)
    print(f"Aggregate summary saved: {output_path} ({len(summary_rows)} rows)")


def save_case_studies(df_all, output_dir):
    """Save detailed results for 5 representative files."""
    df_ok = df_all[df_all['error'].isna()] if 'error' in df_all.columns else df_all
    if df_ok.empty:
        return
    files = df_ok['file'].unique()
    indices = np.linspace(0, len(files) - 1, min(5, len(files)), dtype=int)
    case_dir = os.path.join(output_dir, "case_studies")
    os.makedirs(case_dir, exist_ok=True)
    for cf in (files[i] for i in indices):
        safe_name = cf.replace('.', '_').replace('/', '_').replace('\\', '_')
        df_ok[df_ok['file'] == cf].to_csv(os.path.join(case_dir, f"{safe_name}_internals.csv"), index=False)
    print(f"Case studies saved for {len(indices)} files in {case_dir}")


def summarize(results_dir, models):
    frames = [pd.read_csv(_ckpt(results_dir, m), low_memory=False)
              for m in models if os.path.exists(_ckpt(results_dir, m))]
    if not frames:
        print("No checkpoints found.")
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(results_dir, "raw_internals.csv"), index=False)
    compute_aggregate_summary(df, os.path.join(results_dir, "aggregate_summary.csv"))
    save_case_studies(df, results_dir)


# ==========================================
# MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser(description='Internal algorithmic analysis — TSB-AD')
    ap.add_argument('--test', action='store_true',
                    help='3 files and the reduced --test grid of each corruption runner')
    ap.add_argument('--models', nargs='+', default=list(PHASE1_MODELS), choices=list(PHASE1_MODELS))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--files-csv', default=DEFAULT_FILES_CSV,
                    help='File list (file_name/file column); default: the validated 200-series subset')
    ap.add_argument('--max-files', type=int, default=None)
    ap.add_argument('--results-dir', default=None,
                    help=f'Write here instead of results/experiments/{RESULTS_NAME}/')
    ap.add_argument('--check-scores', dest='check_scores', action='store_true', default=None,
                    help='Refit through the harness and require identical scores (default: on with --test)')
    ap.add_argument('--no-check-scores', dest='check_scores', action='store_false')
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    check_scores = args.test if args.check_scores is None else args.check_scores

    results_dir = str(paths.resolve(args.results_dir) if args.results_dir
                      else paths.results_dir(RESULTS_NAME))
    os.makedirs(results_dir, exist_ok=True)
    if args.summary_only:
        summarize(results_dir, args.models)
        return

    list_path = paths.resolve(args.files_csv)
    if not list_path.exists():
        print(f"[ERROR] File list not found: {list_path}")
        return
    file_paths = [str(paths.DATA_DIR / f) for f in paths.read_file_list(list_path)]
    if args.test:
        file_paths = file_paths[:3]
        print("!!! TEST MODE !!!")
    elif args.max_files:
        file_paths = file_paths[:args.max_files]

    wanted = expected_keys(args.test)
    print(f"\n{'=' * 60}\n  Internal Algorithmic Analysis — TSB-AD\n{'=' * 60}")
    print(f"File list: {list_path.name} | files: {len(file_paths)}")
    print(f"Models: {args.models} | Workers: {args.workers} | score check: {check_scores}")
    print(f"Conditions per (file, model): {len(wanted)}  "
          f"({', '.join(name for name, _ in EXPERIMENTS)})")
    print(f"Results: {results_dir}\n{'=' * 60}\n")

    all_rows = load_rows(results_dir, args.models)
    have = {}
    for r in all_rows:
        have.setdefault((r['file'], r['model']), set()).add((r['experiment'], r['condition'], int(r['seed'])))
    jobs = [(fp, m, args.test, check_scores) for fp in file_paths for m in args.models
            if not wanted.issubset(have.get((os.path.basename(fp), m), set()))]
    done_pairs = {pair for pair, keys in have.items() if wanted.issubset(keys)}
    if all_rows:
        print(f"Loaded checkpoints: {len(all_rows)} rows, {len(done_pairs)} complete (file, model) jobs.")

    # longest-first: MatrixProfile is ~O(n^2) and series span 18k-900k points
    try:
        sizes = {fp: os.path.getsize(fp) for fp in set(j[0] for j in jobs)}
        jobs.sort(key=lambda j: sizes.get(j[0], 0), reverse=True)
    except OSError:
        pass

    print(f"Jobs scheduled: {len(jobs)}")
    if jobs:
        todo = {(os.path.basename(j[0]), j[1]) for j in jobs}
        all_rows = [r for r in all_rows if (r['file'], r['model']) not in todo]   # redo partial jobs whole
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_file_model, j): j for j in jobs}
            for fut in tqdm(as_completed(futs), total=len(jobs), desc="internals"):
                res = fut.result()
                if res['status'] == 'success':
                    all_rows.extend(res['rows'])
                    save_rows(results_dir, all_rows, args.models)   # one (file, model) job is minutes of work
                else:
                    print(f"Error in {res.get('file', '?')} / {res.get('model', '?')}: "
                          f"{str(res.get('error', ''))[:300]}")
        save_rows(results_dir, all_rows, args.models)

    summarize(results_dir, args.models)
    print(f"\n[Done] Results in {results_dir}")


if __name__ == "__main__":
    main()
