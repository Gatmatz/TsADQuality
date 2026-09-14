"""
Swap Corruption — Segment Swap — TSB-AD version.

Port of run_swap_segment.py to the TSB-AD benchmark, built to mirror
swap_point.py step for step. Same corruption and same parameter grid, but
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

--- Why not ts_corruptor.injectors.inject_swap (correctness, not just speed) ---
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

ts_corruptor.swap.swap_segments_fast instead samples 2*num_swaps STRICTLY DISJOINT segments of length L,
uniformly, using the standard bijection: choose k starts without replacement from
n - k*(L-1) slots, then expand by i*(L-1). Every condition then delivers exactly
2*num_swaps*L swapped points, so "fraction held constant" is actually true. It is also
O(n) rather than the injector's per-swap O(n) pruning pass.

The shared injector is deliberately left untouched so previously produced TSB-UAD results
stay reproducible. `--injector reference` runs the original path for cross-checking.

Usage:
    python src/experiments/corruption_tsbad/swap_segment.py --test
    python src/experiments/corruption_tsbad/swap_segment.py --models IForest --workers 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import numpy as np  # noqa: E402

from ts_corruptor.swap import swap_segments_fast, swap_segments_reference  # noqa: E402
from tsbad.harness import Spec, condition, run_sweep  # noqa: E402

FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30]
NUM_SWAPS = [1, 3, 5, 10, 20]   # dynamic: swap_length = fraction * n / (2 * num_swaps)
N_SEEDS = 1


def _cond_name(fraction, num_swaps):
    return "clean" if fraction is None else f"frac_{fraction}_ns_{num_swaps}"


def add_args(ap):
    ap.add_argument('--injector', choices=['fast', 'reference'], default='fast',
                    help="'fast' (default): strictly disjoint segments, exact fraction. "
                         "'reference': the original ts_corruptor.inject_swap, which "
                         "under-delivers the fraction for swap_length > 1 (see docstring).")


def build_conditions(args):
    fractions, num_swaps_list = FRACTIONS, NUM_SWAPS
    if args.test:
        fractions = [0.05, 0.20]
        num_swaps_list = [1, 10]
    grid = [] if args.no_clean else [(None, None)]
    grid += [(f, ns) for f in fractions for ns in num_swaps_list]
    return [condition(_cond_name(f, ns), {'fraction': f, 'num_swaps': ns},
                      n_seeds=1 if f is None else N_SEEDS)
            for (f, ns) in grid]


def job_options(args):
    return {'injector': args.injector}


def before_run(args, conditions):
    print(f"window_mode: {args.window_mode} | injector: {args.injector}")
    if args.injector == 'reference':
        print("NOTE: --injector reference does not deliver the requested fraction for "
              "swap_length > 1 (bounds skips + cross-pair overlap). Use it for "
              "cross-checking only, not for the reported run.\n")


def corrupt(ctx, p):
    n, fraction, num_swaps = ctx.n, p['fraction'], p['num_swaps']
    # Labels are NOT touched: they stay in their original positions, so a swapped segment is
    # judged against the labels it landed on.
    if fraction is None:
        data, n_swapped, swap_length, ns_actual = ctx.clean_data, 0, 0, 0
    elif ctx.options['injector'] == 'reference':
        df_work, n_swapped, swap_length, ns_actual = swap_segments_reference(
            ctx.df, ctx.value_col, ctx.label_col, fraction, num_swaps, ctx.seed)
        data = df_work.iloc[:, 0:-1].values.astype(float)
    else:
        rng = np.random.RandomState(ctx.seed)
        data, n_swapped, swap_length, ns_actual = swap_segments_fast(
            ctx.clean_data, fraction, num_swaps, rng)
    return data, {'swap_length': swap_length, 'num_swaps_actual': ns_actual,
                  'n_swapped_points': n_swapped,
                  'fraction_actual': round(n_swapped / n, 6) if n else 0.0}


SPEC = Spec(
    name='swap_segment_tsbad',
    title='Segment Swap Robustness — TSB-AD',
    family='full',
    build_conditions=build_conditions,
    corrupt=corrupt,
    add_args=add_args,
    job_options=job_options,
    before_run=before_run,
    summary_keys=('condition', 'fraction', 'num_swaps', 'model'),
    summary_means=(('swap_length', 1), ('fraction_actual', 6)),
    test_help='Smoke test: 3 files, small grid',
)

if __name__ == "__main__":
    run_sweep(SPEC)
