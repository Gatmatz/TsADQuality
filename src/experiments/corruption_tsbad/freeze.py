"""
Freeze (sensor-stuck) robustness experiment — TSB-AD version.

Port of run_freeze.py to the TSB-AD benchmark, built to mirror spikes.py /
white_noise_snr.py step for step. Same corruption
(ts_corruptor.inject_sensor_stuck) and the same parameter grid, but evaluated with the
OFFICIAL TSB-AD pipeline (run_Unsupervise_AD / run_Semisupervise_AD + find_length_rank +
TSB_AD get_metrics) on the full 350-series TSB-AD-U eval list.

A `clean` anchor condition runs the identical pipeline with no corruption, so it reproduces
the TSB-AD baseline exactly and the degradation curve starts from the right point.

Deliberate deviations from the TSB-UAD original (documented, not accidents):
  * no `max(window, 10)` floor — the official TSB-AD runner uses find_length_rank raw;
  * scores are used UNNORMALIZED, exactly as Run_Detector_U.py does (TSB-AD metrics are
    rank-based, so a MinMax rescale would be a no-op here anyway);
  * a `clean` anchor condition is added (pure addition — the other conditions are untouched);
  * `n_corrupted` / `corrupted_frac` / `n_value_changed` are recorded. These matter: with
    num_stucks=1 a block starting near the end is truncated, so the REALISED freeze fraction
    can fall well short of the nominal one (measured: 14.2% realised for a nominal 20% on
    331_UCR, n=900k). And on quantized signals many frozen points already equalled the stuck
    value, so `n_value_changed` < `n_corrupted`. Both are diagnostics only — they never enter
    a metric.

Usage:
    python src/experiments/corruption_tsbad/freeze.py --test
    python src/experiments/corruption_tsbad/freeze.py --models IForest --workers 4
    python src/experiments/corruption_tsbad/freeze.py --models MatrixProfile --workers 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import ts_corruptor.injectors  # noqa: E402
from ts_corruptor.core import TSCorruptor  # noqa: E402
from tsbad.harness import Spec, condition, run_sweep  # noqa: E402

FRACTIONS = [0.01, 0.05, 0.10, 0.20]
NUM_STUCKS = [1, 3, 5, 10, 20]     # dynamic: stuck_length = fraction * n / num_stucks
N_SEEDS = 1


def _cond_name(fraction, num_stucks):
    """Byte-identical to the TSB-UAD original, so old and new tables group on `condition`."""
    if fraction is None:
        return "clean"
    return f"frac_{fraction}_ns_{num_stucks}"


def add_args(ap):
    ap.add_argument('--fractions', nargs='+', type=float, default=None,
                    help=f'Override fractions (default {FRACTIONS})')
    ap.add_argument('--num-stucks', nargs='+', type=int, default=None,
                    help=f'Override num_stucks (default {NUM_STUCKS})')


def build_conditions(args):
    fractions = args.fractions if args.fractions else FRACTIONS
    num_stucks_list = args.num_stucks if args.num_stucks else NUM_STUCKS
    if args.test:
        fractions = [0.05, 0.20]
        num_stucks_list = [3]
    grid = [] if args.no_clean else [(None, None)]
    grid += [(f, ns) for f in fractions for ns in num_stucks_list]
    return [condition(_cond_name(f, ns), {'fraction': f, 'num_stucks': ns}, n_seeds=N_SEEDS)
            for (f, ns) in grid]


def corrupt(ctx, p):
    n = ctx.n
    fraction, num_stucks = p['fraction'], p['num_stucks']
    # Same dynamic rule as the original: the nominal frozen fraction is split into
    # num_stucks blocks. Total nominal frozen points = fraction * n, independent of
    # num_stucks; only the block geometry changes.
    stuck_length = 0 if fraction is None else max(1, int(fraction * n / num_stucks))

    if fraction is None:
        data, n_corrupted, n_value_changed = ctx.clean_data, 0, 0
    else:
        corruptor = TSCorruptor(ctx.df.copy(), value_col=ctx.value_col, label_col=ctx.label_col,
                                seed=ctx.seed)
        ts_corruptor.injectors.inject_sensor_stuck(
            corruptor, num_stucks=num_stucks, stuck_length=stuck_length)
        df_work = corruptor.get_corrupted_df()
        data = df_work.iloc[:, 0:-1].values.astype(float)
        # Realised (not nominal) corruption: blocks are clipped at the series end and may
        # overlap each other, so this can be well below fraction * n.
        n_corrupted = int(corruptor.corruption_mask.sum())
        n_value_changed = int(
            (df_work[ctx.value_col].to_numpy() != ctx.df[ctx.value_col].to_numpy()).sum())

    return data, {'stuck_length': stuck_length, 'n_points': n,
                  'n_corrupted': n_corrupted,
                  'corrupted_frac': round(n_corrupted / n, 6) if n else 0.0,
                  'n_value_changed': n_value_changed,
                  'metric_window': int(ctx.sliding_window)}


SPEC = Spec(
    name='freeze_tsbad',
    title='Freeze (sensor stuck) Robustness — TSB-AD',
    family='full',
    build_conditions=build_conditions,
    corrupt=corrupt,
    add_args=add_args,
    summary_keys=('condition', 'fraction', 'num_stucks', 'model'),
    summary_means=(('stuck_length', 1), ('corrupted_frac', 4)),
    test_help='Smoke test: 3 files, small grid',
)

if __name__ == "__main__":
    run_sweep(SPEC)
