"""
Swap Corruption — Point Swap — TSB-AD version.

Port of run_swap_point.py to the TSB-AD benchmark, built to mirror
swap_permutation.py step for step. Same corruption (randomly pick pairs of
points and swap their values) and same parameter grid, but evaluated with the OFFICIAL
TSB-AD pipeline (run_Unsupervise_AD / run_Semisupervise_AD + find_length_rank + TSB_AD
get_metrics) on the full 350-series TSB-AD-U eval list.

Labels stay in place — only values move. This disrupts temporal dependencies while
preserving the marginal distribution of values exactly (it is a permutation of the
original samples).

A `clean` anchor condition runs the identical pipeline with no swapping, so it
reproduces the TSB-AD baseline exactly and the degradation curve starts from the
right point.

Key question: "How much random point swapping is needed to degrade AD?"

--- Why not ts_corruptor.injectors.inject_swap ---
ts_corruptor.injectors.inject_swap prunes its candidate-index array inside the swap
loop (`target_indices[~np.isin(target_indices, used)]`). That is O(n) per swap with
O(n*fraction) swaps, i.e. O(n^2) overall. TSB-UAD series are short enough for this not
to matter; TSB-AD-U runs to 900k points, where the full grid would cost ~8 hours of
pure corruption before a single detector runs (measured: 3.3s at n=40k, fraction=0.40,
scaling quadratically).

For swap_length=1 with max_distance=None the loop is equivalent to "draw 2*num_swaps
distinct indices uniformly and pair them up", which vectorises to a single O(n) pass.
ts_corruptor.swap.swap_points_fast does exactly that. The shared injector is deliberately left untouched
so previously produced TSB-UAD results stay reproducible.

`--injector reference` runs the original ts_corruptor path instead, for cross-checking
on small series. Note the two draw different *specific* pairs (they consume the RNG
differently); what matches is the number of swapped points and the distribution.

Usage:
    python src/experiments/corruption_tsbad/swap_point.py --test
    python src/experiments/corruption_tsbad/swap_point.py --models IForest --workers 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import numpy as np  # noqa: E402

from ts_corruptor.swap import swap_points_fast, swap_points_reference  # noqa: E402
from tsbad.harness import Spec, condition, run_sweep  # noqa: E402

FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]
SWAP_LENGTH = 1          # point swap
N_SEEDS = 1


def _cond_name(fraction):
    return "clean" if fraction is None else f"frac_{fraction}"


def add_args(ap):
    ap.add_argument('--injector', choices=['fast', 'reference'], default='fast',
                    help="'fast' (default): vectorised O(n) point swap. 'reference': the original "
                         "ts_corruptor.inject_swap, O(n^2) — usable only on short series.")


def build_conditions(args):
    fractions = [0.05, 0.20] if args.test else FRACTIONS
    grid = [] if args.no_clean else [None]   # None = the clean anchor
    grid += list(fractions)
    return [condition(_cond_name(f), {'fraction': f}, n_seeds=1 if f is None else N_SEEDS)
            for f in grid]


def job_options(args):
    return {'injector': args.injector}


def before_run(args, conditions):
    print(f"window_mode: {args.window_mode} | injector: {args.injector}")
    if args.injector == 'reference':
        print("NOTE: --injector reference is O(n^2); on the 900k-point series the corruption "
              "alone costs over an hour per condition.\n")


def corrupt(ctx, p):
    n, fraction = ctx.n, p['fraction']
    # Labels are NOT touched: they stay in their original positions, so a swapped point is
    # judged against the label it landed on.
    if fraction is None:
        data, n_swapped = ctx.clean_data, 0
    elif ctx.options['injector'] == 'reference':
        df_work, n_swapped = swap_points_reference(
            ctx.df, ctx.value_col, ctx.label_col, fraction, ctx.seed, SWAP_LENGTH)
        data = df_work.iloc[:, 0:-1].values.astype(float)
    else:
        rng = np.random.RandomState(ctx.seed)
        data, n_swapped = swap_points_fast(ctx.clean_data, fraction, rng, SWAP_LENGTH)
    return data, {'swap_length': SWAP_LENGTH, 'n_swapped_points': n_swapped,
                  'fraction_actual': round(n_swapped / n, 6) if n else 0.0}


SPEC = Spec(
    name='swap_point_tsbad',
    title='Point Swap Robustness — TSB-AD',
    family='full',
    build_conditions=build_conditions,
    corrupt=corrupt,
    add_args=add_args,
    job_options=job_options,
    before_run=before_run,
    summary_keys=('condition', 'fraction', 'model'),
    summary_means=(('fraction_actual', 6),),
    test_help='Smoke test: 3 files, small grid',
)

if __name__ == "__main__":
    run_sweep(SPEC)
