"""
Missing Values — MCAR vs MNAR (point version), TSB-AD version.

Port of run_missing_mnar.py to the TSB-AD benchmark, built on the same skeleton as
missing_mnar_burst.py / missing_true_impact.py.

The question is about the MECHANISM of data loss, not its volume. Every condition removes
exactly `int(fraction * n)` SCATTERED points; only the rule that picks which points changes:

    mcar          every point equally likely                          (control)
    mnar_extreme  probability proportional to |z|                     (sensor overload)
    mnar_high     probability proportional to max(z, 0) + 0.01        (sensor caps out)

This is the scattered counterpart of missing_mnar_burst.py: same three mechanisms,
same fractions, but the loss is spread point-by-point instead of concentrated in blocks.
Together the two scripts separate WHERE loss lands (mechanism) from HOW it clusters (burstiness).

> Key question: given the same % of missing data, does the mechanism matter?

Real-world motivation: sensors fail when readings are extreme (overload, saturation), which
means anomalies — usually extreme — are lost disproportionately. MCAR is the control that
makes that attributable.

Evaluation — "true impact", NO imputation (identical policy to the original):
  1. Drop the NaN points; the detector sees a SHORTER series.
  2. Map the scores back onto the original timeline.
  3. Lost ANOMALY points get the minimum score -> they count as false negatives.
  4. Lost NORMAL points are EXCLUDED from the evaluation.

Detection and metrics use the OFFICIAL TSB-AD pipeline (run_Unsupervise_AD /
run_Semisupervise_AD + find_length_rank + TSB_AD get_metrics) on the full 350-series
TSB-AD-U eval list.

DELIBERATE DEVIATIONS from the TSB-UAD original — the same set as every other TSB-AD port here:
  1. A `clean` anchor condition (no loss at all) runs the identical pipeline, giving the x=0 of
     every degradation curve inside this run rather than importing it. On the clean condition
     nan_mask is empty, so it reduces exactly to the official TSB-AD baseline.
  2. Lost anomalies get score.min(), not a literal 0.0. TSB-AD returns UNNORMALIZED scores
     (IForest sits around [-0.06, +0.03] with most points below zero), so a hardcoded 0.0 would
     rank a destroyed anomaly in the top ~10% and INFLATE the metrics as more anomalies are
     lost. In the original 0.0 WAS the minimum, because it MinMax-scaled the scores first;
     score.min() carries exactly that semantics into a raw-score pipeline.
  3. No MinMax on the scores and no max(window, 10) floor on the metric window, so the clean
     anchor reproduces the official baseline bit for bit. Rank-based metrics are invariant to
     the dropped MinMax.
  4. Checkpoints are per model, jobs are seeded deterministically per (file, condition, seed,
     model), and files are scheduled longest-first.

The selection rule itself (`ts_corruptor.missing.select_missing_indices`) is a verbatim transcription of the
original, weights and floors included. Two properties of it were left ALONE on purpose — they
are the original's behaviour, not bugs introduced here, but they are worth knowing:

  * ASYMMETRIC FLOOR. `mnar_high` adds `+ 0.01` to its weights so every point keeps some
    chance; `mnar_extreme` does not. A point sitting exactly at the mean therefore has
    probability 0 under mnar_extreme, and if fewer than `int(fraction*n)` points have non-zero
    |z| numpy raises "Fewer non-zero entries in p than size". That surfaces as an error row
    rather than a crash, and on real series it should be vanishingly rare.
  * WEIGHTED SAMPLING WITHOUT REPLACEMENT. `rng.choice(n, size=n_missing, replace=False,
    p=weights)` is materially more expensive than the unweighted path. At fraction=0.20 on the
    900k-point series that is 180 000 weighted draws without replacement. Worth timing on the
    longest file before committing to a full run.

INTERPRETATION NOTE — read `pct_anomalies_lost` alongside every metric. MNAR destroys anomalies
by construction, and under true-impact evaluation a destroyed anomaly is a guaranteed false
negative, so part of the MNAR damage is arithmetic rather than a statement about the detector.
The finding is whatever damage EXCEEDS what the anomaly loss alone explains; the MCAR arm at
matched volume is what makes that separation possible.

Usage:
    python src/experiments/corruption_tsbad/missing_mnar.py --test
    python src/experiments/corruption_tsbad/missing_mnar.py --models IForest --workers 4
    # one fraction, if the full grid is too big:
    python src/experiments/corruption_tsbad/missing_mnar.py \
        --models IForest --fractions 0.10 --workers 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import numpy as np  # noqa: E402

from ts_corruptor.missing import select_missing_indices  # noqa: E402
from tsbad.harness import Spec, condition, run_sweep  # noqa: E402

FRACTIONS = [0.01, 0.05, 0.10, 0.20]
MECHANISMS = ['mcar', 'mnar_extreme', 'mnar_high']
N_SEEDS = 1


def _cond_name(mechanism, fraction):
    if mechanism is None:
        return "clean"
    return f"frac_{fraction}_{mechanism}"


def add_args(ap):
    ap.add_argument('--fractions', nargs='+', type=float, default=None,
                    help=f'Override fractions (default {FRACTIONS})')
    ap.add_argument('--mechanisms', nargs='+', default=None, choices=MECHANISMS,
                    help=f'Override mechanisms (default: all {len(MECHANISMS)}). Dropping mcar '
                         'removes the control arm that makes the MNAR damage attributable.')


def _grid(args):
    fractions = args.fractions if args.fractions else FRACTIONS
    mechanisms = args.mechanisms if args.mechanisms else MECHANISMS
    if args.test:
        fractions = [0.05, 0.20]
        mechanisms = ['mcar', 'mnar_extreme']
    return fractions, mechanisms


def build_conditions(args):
    fractions, mechanisms = _grid(args)
    # (mechanism, fraction); None mechanism = the clean anchor
    grid = [] if args.no_clean else [(None, None)]
    grid += [(mech, f) for mech in mechanisms for f in fractions]
    return [condition(_cond_name(mech, f), {'mechanism': mech, 'fraction': f},
                      n_seeds=1 if mech is None else N_SEEDS)
            for (mech, f) in grid]


def before_run(args, conditions):
    _, mechanisms = _grid(args)
    if 'mcar' not in mechanisms:
        print("[WARN] mcar is not in the run: the MNAR arms will have no matched control, "
              "so damage cannot be attributed to the mechanism rather than the volume.")
    print("  Same number of missing points in every arm — only the SELECTION RULE changes")
    print("  Lost anomalies -> minimum score (false negatives)")
    print("  Lost normal points -> excluded from evaluation")
    print(f"Mechanisms: {mechanisms} | seeds: {N_SEEDS}\n")


def corrupt(ctx, p):
    """Choose the missing points (none for the clean anchor).

    Whole ROWS are dropped, so every channel stays aligned; the mechanism weights positions
    by channel 0.
    """
    n, labels = ctx.n, ctx.labels
    if p['mechanism'] is None:
        nan_mask = np.zeros(n, dtype=bool)
        data_full = ctx.clean_data
    else:
        rng = np.random.default_rng(ctx.seed)
        missing_idx = select_missing_indices(ctx.clean_data[:, 0], p['fraction'], p['mechanism'], rng)
        nan_mask = np.zeros(n, dtype=bool)
        nan_mask[missing_idx] = True
        data_full = ctx.clean_data.copy()
        data_full[nan_mask] = np.nan

    n_lost_anomalies = int((nan_mask & (labels == 1)).sum())
    n_anomalies = int(labels.sum())
    return data_full, {'n_missing': int(nan_mask.sum()),
                       'n_lost_normal': int((nan_mask & (labels == 0)).sum()),
                       'n_anomalies': n_anomalies,
                       'pct_anomalies_lost': round(n_lost_anomalies / max(n_anomalies, 1), 4)}


SPEC = Spec(
    name='missing_mnar_tsbad',
    title='Missing values — MCAR vs MNAR (point) — TSB-AD',
    family='survivors',
    survivor_rules='always',
    build_conditions=build_conditions,
    corrupt=corrupt,
    add_args=add_args,
    before_run=before_run,
    summary_keys=('condition', 'mechanism', 'fraction', 'model'),
    summary_means=(('actual_missing_rate', 4), ('n_lost_anomalies', 4), ('n_lost_normal', 4),
                   ('pct_anomalies_lost', 4), ('n_kept', 4)),
    test_help='Smoke test: 3 files, small grid',
)

if __name__ == "__main__":
    run_sweep(SPEC)
