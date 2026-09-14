"""
Gilbert-Elliott Channel Model — True Impact Experiment, TSB-AD version.

Port of run_gilbert_elliott_true_impact.py to the TSB-AD benchmark, mirroring
white_noise_snr.py / spikes_normal_only.py step for step.

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
    python src/experiments/corruption_tsbad/gilbert_elliott_true_impact.py --test
    python src/experiments/corruption_tsbad/gilbert_elliott_true_impact.py --models IForest --workers 4
    python src/experiments/corruption_tsbad/gilbert_elliott_true_impact.py --models MatrixProfile --workers 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import ts_corruptor.injectors  # noqa: E402
from ts_corruptor.core import TSCorruptor  # noqa: E402
from tsbad.harness import Spec, condition, run_sweep  # noqa: E402

ALPHAS = [0.01, 0.05, 0.10]    # p(Good -> Bad) — burst frequency
BETAS = [0.10, 0.25, 0.50]     # p(Bad -> Good) — burst duration (1/beta)
N_SEEDS = 1
CORRUPTION_TARGET = 'global'


def _cond_name(alpha, beta):
    return "clean" if alpha is None else f"a_{alpha}_b_{beta}"


def add_args(ap):
    ap.add_argument('--alphas', nargs='+', type=float, default=None,
                    help=f'Override alphas, p(Good->Bad) (default {ALPHAS})')
    ap.add_argument('--betas', nargs='+', type=float, default=None,
                    help=f'Override betas, p(Bad->Good) (default {BETAS})')


def _grid(args):
    alphas = args.alphas if args.alphas else ALPHAS
    betas = args.betas if args.betas else BETAS
    if args.test:
        alphas, betas = [0.05], [0.10, 0.50]
    return alphas, betas


def build_conditions(args):
    alphas, betas = _grid(args)
    # (alpha, beta); None alpha = the clean anchor
    grid = [] if args.no_clean else [(None, None)]
    grid += [(a, b) for a in alphas for b in betas]
    return [condition(_cond_name(a, b), {'alpha': a, 'beta': b}, n_seeds=N_SEEDS)
            for (a, b) in grid]


def before_run(args, conditions):
    alphas, betas = _grid(args)
    print("  Lost anomalies -> minimum score (false negatives)")
    print("  Lost normal points -> excluded from evaluation")
    print(f"corruption_target: {CORRUPTION_TARGET}")
    print(f"{'alpha':>8} {'beta':>8} {'exp_rate':>10} {'exp_burst':>10}")
    print("-" * 40)
    for a in alphas:
        for b in betas:
            print(f"{a:>8.2f} {b:>8.2f} {a/(a+b):>10.2%} {1/b:>10.1f}")
    print()


def corrupt(ctx, p):
    """Gilbert-Elliott bursts of MISSING points (nothing for the clean anchor)."""
    alpha, beta = p['alpha'], p['beta']
    expected_rate = None if alpha is None else alpha / (alpha + beta)
    expected_burst_len = None if alpha is None else 1.0 / beta
    fields = {'expected_rate': None if expected_rate is None else round(expected_rate, 4),
              'expected_burst_len': None if expected_burst_len is None else round(expected_burst_len, 1)}
    if alpha is None:
        return ctx.clean_data, fields
    corruptor = TSCorruptor(ctx.df.copy(), value_col=ctx.value_col, label_col=ctx.label_col,
                            seed=ctx.seed, corruption_target=CORRUPTION_TARGET)
    ts_corruptor.injectors.inject_gilbert_elliott(
        corruptor, p_good_to_bad=alpha, p_bad_to_good=beta, noise_type='missing')
    return corruptor.get_corrupted_df().iloc[:, 0:-1].values.astype(float), fields


SPEC = Spec(
    name='gilbert_elliott_true_impact_tsbad',
    title='Gilbert-Elliott — True Impact (no imputation) — TSB-AD',
    family='survivors',
    survivor_rules='always',
    build_conditions=build_conditions,
    corrupt=corrupt,
    add_args=add_args,
    before_run=before_run,
    summary_keys=('condition', 'alpha', 'beta', 'model'),
    summary_first=('expected_rate', 'expected_burst_len'),
    summary_means=(('actual_missing_rate', 4), ('n_lost_anomalies', 4), ('n_kept', 4)),
    test_help='Smoke test: 3 files, small grid',
)

if __name__ == "__main__":
    run_sweep(SPEC)
