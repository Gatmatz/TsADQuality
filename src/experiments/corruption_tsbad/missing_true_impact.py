"""
Missing Values — True Impact Experiment, TSB-AD version.

Port of run_missing_true_impact.py to the TSB-AD benchmark, mirroring
gilbert_elliott_true_impact.py step for step.

Two missing patterns at MATCHED volume — this is the controlled contrast:
  - point : `fraction` of points dropped at random, scattered (MCAR).
  - burst : the same `fraction` of points dropped, but concentrated in
            `num_bursts` contiguous blocks of length fraction*n/num_bursts.
Holding `fraction` fixed and varying `num_bursts` isolates the effect of HOW the
loss is distributed from HOW MUCH is lost. num_bursts=1 is the most concentrated
case, and the point condition is the fully dispersed limit.

This is the non-bursty reference arm for gilbert_elliott_true_impact.py,
which cannot separate volume from burstiness on its own (alpha and beta move both
at once).

Evaluation — "true impact", i.e. NO imputation (identical policy to the original):
  1. Drop the NaN points; the detector sees a SHORTER series.
  2. Map the scores back onto the original timeline.
  3. Lost ANOMALY points get the minimum score -> they count as false negatives.
  4. Lost NORMAL points are EXCLUDED from the evaluation.

Detection and metrics use the OFFICIAL TSB-AD pipeline (run_Unsupervise_AD /
run_Semisupervise_AD + find_length_rank + TSB_AD get_metrics) on the full
350-series TSB-AD-U eval list. A `clean` anchor condition runs the identical
pipeline with no corruption; there nan_mask is empty, so it reduces exactly to
the official TSB-AD baseline and the degradation curve starts from the right point.

Key question: "Does it matter HOW missing data is distributed, or only HOW MUCH?"

Usage:
    python src/experiments/corruption_tsbad/missing_true_impact.py --test
    python src/experiments/corruption_tsbad/missing_true_impact.py --models IForest --workers 4
    python src/experiments/corruption_tsbad/missing_true_impact.py --models MatrixProfile --workers 4
    # just the point/burst contrast at one fraction, if the full grid is too big:
    python src/experiments/corruption_tsbad/missing_true_impact.py --models IForest --fractions 0.10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import ts_corruptor.injectors  # noqa: E402
from ts_corruptor.core import TSCorruptor  # noqa: E402
from tsbad.harness import Spec, condition, run_sweep  # noqa: E402

FRACTIONS = [0.01, 0.05, 0.10, 0.20]
NUM_BURSTS = [1, 3, 5, 10]
N_SEEDS = 1
CORRUPTION_TARGET = 'global'


def _cond_name(missing_type, fraction, num_bursts):
    if missing_type is None:
        return "clean"
    if missing_type == 'point':
        return f"point_frac_{fraction}"
    return f"burst_frac_{fraction}_nb_{num_bursts}"


def add_args(ap):
    ap.add_argument('--fractions', nargs='+', type=float, default=None,
                    help=f'Override fractions (default {FRACTIONS})')
    ap.add_argument('--num-bursts', nargs='+', type=int, default=None,
                    help=f'Override burst counts (default {NUM_BURSTS})')
    ap.add_argument('--no-point', action='store_true', help='Skip the scattered (MCAR) arm')
    ap.add_argument('--no-burst', action='store_true', help='Skip the burst arm')


def _grid(args):
    fractions = args.fractions if args.fractions else FRACTIONS
    num_bursts_list = args.num_bursts if args.num_bursts else NUM_BURSTS
    if args.test:
        fractions, num_bursts_list = [0.05, 0.20], [3]
    return fractions, num_bursts_list


def build_conditions(args):
    fractions, num_bursts_list = _grid(args)
    # (missing_type, fraction, num_bursts); None type = the clean anchor
    grid = [] if args.no_clean else [(None, None, 0)]
    if not args.no_point:
        grid += [('point', f, 0) for f in fractions]
    if not args.no_burst:
        grid += [('burst', f, nb) for f in fractions for nb in num_bursts_list]
    return [condition(_cond_name(mt, f, nb),
                      {'missing_type': mt, 'fraction': f, 'num_bursts': nb}, n_seeds=N_SEEDS)
            for (mt, f, nb) in grid]


def before_run(args, conditions):
    fractions, num_bursts_list = _grid(args)
    print("  Lost anomalies -> minimum score (false negatives)")
    print("  Lost normal points -> excluded from evaluation")
    print(f"corruption_target: {CORRUPTION_TARGET}")
    print("  Matched-volume contrast — same fraction, different concentration:")
    print(f"  {'fraction':>10}{'point':>10}" + "".join(f"{'nb=' + str(nb):>10}" for nb in num_bursts_list))
    for f in fractions:
        print(f"  {f:>10.2f}{'scattered':>10}" + "".join(f"{nb:>10}" for nb in num_bursts_list))
    print()


def corrupt(ctx, p):
    """Drop points, scattered or in bursts (nothing for the clean anchor)."""
    burst_length = 0
    if p['missing_type'] is None:
        return ctx.clean_data, {'burst_length': burst_length}
    corruptor = TSCorruptor(ctx.df.copy(), value_col=ctx.value_col, label_col=ctx.label_col,
                            seed=ctx.seed, corruption_target=CORRUPTION_TARGET)
    if p['missing_type'] == 'point':
        ts_corruptor.injectors.inject_point_missing(corruptor, fraction=p['fraction'])
    else:
        # same total volume as the point condition, concentrated in num_bursts blocks
        burst_length = max(1, int(p['fraction'] * ctx.n / p['num_bursts']))
        ts_corruptor.injectors.inject_burst_missing(
            corruptor, num_bursts=p['num_bursts'], burst_length=burst_length)
    data_full = corruptor.get_corrupted_df().iloc[:, 0:-1].values.astype(float)
    return data_full, {'burst_length': burst_length}


SPEC = Spec(
    name='missing_true_impact_tsbad',
    title='Missing values — True Impact (no imputation) — TSB-AD',
    family='survivors',
    survivor_rules='always',
    build_conditions=build_conditions,
    corrupt=corrupt,
    add_args=add_args,
    before_run=before_run,
    summary_keys=('condition', 'missing_type', 'fraction', 'num_bursts', 'model'),
    summary_means=(('burst_length', 4), ('actual_missing_rate', 4), ('n_lost_anomalies', 4),
                   ('n_kept', 4)),
    test_help='Smoke test: 3 files, small grid',
)

if __name__ == "__main__":
    run_sweep(SPEC)
