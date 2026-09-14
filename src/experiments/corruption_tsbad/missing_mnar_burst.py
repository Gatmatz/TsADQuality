"""
Missing Values — MCAR vs MNAR (burst version), TSB-AD version.

Port of run_missing_mnar_burst.py to the TSB-AD benchmark, built on the same skeleton as
missing_true_impact.py.

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
  4. Burst placement is vectorised (see ts_corruptor.missing.select_burst_starts). Selection order and the resulting
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
    python src/experiments/corruption_tsbad/missing_mnar_burst.py --test
    python src/experiments/corruption_tsbad/missing_mnar_burst.py --models IForest --workers 4
    # the contrast at one fraction, if the full grid is too big:
    python src/experiments/corruption_tsbad/missing_mnar_burst.py \
        --models IForest --fractions 0.10 --workers 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import numpy as np  # noqa: E402

from ts_corruptor.missing import select_burst_starts  # noqa: E402
from tsbad.harness import Spec, condition, run_sweep  # noqa: E402

FRACTIONS = [0.01, 0.05, 0.10, 0.20]
NUM_BURSTS = [1, 3, 5, 10]
MECHANISMS = ['mcar_burst', 'mnar_extreme_burst', 'mnar_high_burst']
# MNAR placement is a deterministic ranking; only mcar_burst draws random positions
STOCHASTIC_MECHANISMS = {'mcar_burst'}
N_SEEDS = 1


def _cond_name(mechanism, fraction, num_bursts):
    if mechanism is None:
        return "clean"
    return f"{mechanism}_frac_{fraction}_nb_{num_bursts}"


def add_args(ap):
    ap.add_argument('--fractions', nargs='+', type=float, default=None,
                    help=f'Override fractions (default {FRACTIONS})')
    ap.add_argument('--num-bursts', nargs='+', type=int, default=None,
                    help=f'Override burst counts (default {NUM_BURSTS})')
    ap.add_argument('--mechanisms', nargs='+', default=None, choices=MECHANISMS,
                    help=f'Override mechanisms (default: all {len(MECHANISMS)}). Dropping '
                         'mcar_burst removes the control arm — see the module docstring.')


def _grid(args):
    fractions = args.fractions if args.fractions else FRACTIONS
    num_bursts_list = args.num_bursts if args.num_bursts else NUM_BURSTS
    mechanisms = args.mechanisms if args.mechanisms else MECHANISMS
    if args.test:
        fractions, num_bursts_list = [0.05, 0.20], [3]
    return fractions, num_bursts_list, mechanisms


def build_conditions(args):
    fractions, num_bursts_list, mechanisms = _grid(args)
    # (mechanism, fraction, num_bursts); None mechanism = the clean anchor
    grid = [] if args.no_clean else [(None, None, 0)]
    grid += [(mech, f, nb) for mech in mechanisms for f in fractions for nb in num_bursts_list]
    return [condition(_cond_name(mech, f, nb),
                      {'mechanism': mech, 'fraction': f, 'num_bursts': nb},
                      n_seeds=N_SEEDS if mech in STOCHASTIC_MECHANISMS else 1)
            for (mech, f, nb) in grid]


def before_run(args, conditions):
    _, _, mechanisms = _grid(args)
    if 'mcar_burst' not in mechanisms:
        print("[WARN] mcar_burst is not in the run: the MNAR arms will have no matched "
              "control, so damage cannot be attributed to the mechanism rather than the volume.")
    print("  Same volume, same block count — only the PLACEMENT RULE changes")
    print("  Lost anomalies -> minimum score (false negatives)")
    print("  Lost normal points -> excluded from evaluation")
    print(f"Mechanisms: {mechanisms}")
    print(f"Seeds: {N_SEEDS} (MNAR placement is deterministic — extra seeds go to mcar_burst only)\n")


def corrupt(ctx, p):
    """Place the bursts (none for the clean anchor).

    Whole ROWS are dropped, so every channel stays aligned; the mechanism scores positions
    on channel 0.
    """
    n, labels = ctx.n, ctx.labels
    burst_length, n_bursts_actual = 0, 0
    if p['mechanism'] is None:
        nan_mask = np.zeros(n, dtype=bool)
        data_full = ctx.clean_data
    else:
        burst_length = max(1, int(p['fraction'] * n / p['num_bursts']))
        rng = np.random.default_rng(ctx.seed)
        starts = select_burst_starts(ctx.clean_data[:, 0], n, burst_length,
                                     p['num_bursts'], p['mechanism'], rng)
        n_bursts_actual = len(starts)
        nan_mask = np.zeros(n, dtype=bool)
        for s in starts:
            nan_mask[s:s + burst_length] = True
        data_full = ctx.clean_data.copy()
        data_full[nan_mask] = np.nan

    n_lost_anomalies = int((nan_mask & (labels == 1)).sum())
    n_anomalies = int(labels.sum())
    return data_full, {'burst_length': burst_length, 'num_bursts_actual': n_bursts_actual,
                       'n_anomalies': n_anomalies,
                       'pct_anomalies_lost': round(n_lost_anomalies / max(n_anomalies, 1), 4)}


SPEC = Spec(
    name='missing_mnar_burst_tsbad',
    title='Missing values — MCAR vs MNAR (burst) — TSB-AD',
    family='survivors',
    survivor_rules='always',
    build_conditions=build_conditions,
    corrupt=corrupt,
    add_args=add_args,
    before_run=before_run,
    summary_keys=('condition', 'mechanism', 'fraction', 'num_bursts', 'model'),
    summary_means=(('burst_length', 4), ('num_bursts_actual', 4), ('actual_missing_rate', 4),
                   ('n_lost_anomalies', 4), ('pct_anomalies_lost', 4), ('n_kept', 4)),
    test_help='Smoke test: 3 files, small grid',
)

if __name__ == "__main__":
    run_sweep(SPEC)
