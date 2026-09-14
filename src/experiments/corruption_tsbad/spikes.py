"""
Spike-error robustness experiment — TSB-AD version.

Port of run_spikes.py to the TSB-AD benchmark, built to mirror white_noise_snr.py
step for step. Same corruption (ts_corruptor.inject_spikes) and same parameter grid, but
evaluated with the OFFICIAL TSB-AD pipeline (run_Unsupervise_AD / run_Semisupervise_AD +
find_length_rank + TSB_AD get_metrics) on the full 350-series TSB-AD-U eval list.

A `clean` anchor condition runs the identical pipeline with no corruption, so it reproduces
the TSB-AD baseline exactly and the degradation curve starts from the right point.

Usage:
    python src/experiments/corruption_tsbad/spikes.py --test
    python src/experiments/corruption_tsbad/spikes.py --models IForest --workers 4
    python src/experiments/corruption_tsbad/spikes.py --models MatrixProfile --workers 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import ts_corruptor.injectors  # noqa: E402
from ts_corruptor.core import TSCorruptor  # noqa: E402
from tsbad.harness import Spec, condition, run_sweep  # noqa: E402

# Same grid as the original TSB-UAD run_spikes.py
FRACTIONS = [0.01, 0.05, 0.10, 0.20]
MULTIPLIERS = [3.0, 10.0]
# (sequential, sequence_length): False = point spike, True = burst spike
MODES = [(False, 1), (True, 3), (True, 10)]
N_SEEDS = 1


def _cond_name(fraction, multiplier, sequential, seq_len):
    if fraction is None:
        return "clean"
    mode = "point" if not sequential else f"burst_{seq_len}"
    return f"frac_{fraction}_mult_{multiplier}_{mode}"


def build_conditions(args):
    fractions, multipliers, modes = FRACTIONS, MULTIPLIERS, MODES
    if args.test:
        fractions, multipliers, modes = [0.05], [3.0], [(False, 1), (True, 3)]
    grid = [] if args.no_clean else [(None, None, False, 0)]
    grid += [(f, m, s, sl) for f in fractions for m in multipliers for (s, sl) in modes]
    return [condition(_cond_name(f, m, s, sl),
                      {'fraction': f, 'multiplier': m, 'sequential': s, 'sequence_length': sl},
                      n_seeds=N_SEEDS)
            for (f, m, s, sl) in grid]


def corrupt(ctx, p):
    if p['fraction'] is None:
        return ctx.clean_data, {}
    corruptor = TSCorruptor(ctx.df.copy(), value_col=ctx.value_col, label_col=ctx.label_col,
                            seed=ctx.seed)
    ts_corruptor.injectors.inject_spikes(
        corruptor, fraction=p['fraction'], multiplier=p['multiplier'],
        sequential=p['sequential'], sequence_length=p['sequence_length'])
    return corruptor.get_corrupted_df().iloc[:, 0:-1].values.astype(float), {}


SPEC = Spec(
    name='spikes_tsbad',
    title='Spike Robustness — TSB-AD',
    family='full',
    build_conditions=build_conditions,
    corrupt=corrupt,
    summary_keys=('condition', 'fraction', 'multiplier', 'sequential', 'sequence_length', 'model'),
    test_help='Smoke test: 3 files, small grid',
)

if __name__ == "__main__":
    run_sweep(SPEC)
