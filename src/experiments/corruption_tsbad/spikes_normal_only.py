"""
Spike Corruption on NORMAL points only — TSB-AD version.

Port of run_spikes_normal_only.py (the corrected methodology) to the TSB-AD benchmark,
mirroring white_noise_snr.py step for step.

Why normal-only, quoting the original script:
  - Spikes landing on anomaly points artificially boost AUC — the model flags them
    "correctly" but for the wrong reason (the spike, not the anomaly pattern).
  - Spikes on normal points are the real threat: false anomalies that confuse the model.

So corruption_target='only_normal', which is what makes this the methodologically sound
variant of run_spikes.py (that one used the default 'global' target).

Evaluation uses the OFFICIAL TSB-AD pipeline (run_Unsupervise_AD / run_Semisupervise_AD +
find_length_rank + TSB_AD get_metrics) on the full 350-series TSB-AD-U eval list. A `clean`
anchor condition runs the identical pipeline with no corruption, so it reproduces the TSB-AD
baseline exactly and the degradation curve starts from the right point.

Key question: "How much do false spikes in normal data degrade anomaly detection?"

Usage:
    python src/experiments/corruption_tsbad/spikes_normal_only.py --test
    python src/experiments/corruption_tsbad/spikes_normal_only.py --models IForest --workers 4
    python src/experiments/corruption_tsbad/spikes_normal_only.py --models MatrixProfile --workers 4
    # optional: also sweep burst spikes (still normal-only)
    python src/experiments/corruption_tsbad/spikes_normal_only.py --models IForest --burst 3 10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import ts_corruptor.injectors  # noqa: E402
from ts_corruptor.core import TSCorruptor  # noqa: E402
from tsbad.harness import Spec, condition, run_sweep  # noqa: E402

FRACTIONS = [0.01, 0.05, 0.10, 0.20]
MULTIPLIERS = [3.0, 5.0, 10.0]
N_SEEDS = 1
CORRUPTION_TARGET = 'only_normal'


def _cond_name(fraction, multiplier, seq_len):
    """seq_len 1 = point spikes (the original's only mode); >1 = burst."""
    if fraction is None:
        return "clean"
    base = f"frac_{fraction}_mult_{multiplier}"
    return base if seq_len <= 1 else f"{base}_burst_{seq_len}"


def add_args(ap):
    ap.add_argument('--fractions', nargs='+', type=float, default=None,
                    help=f'Override fractions (default {FRACTIONS})')
    ap.add_argument('--multipliers', nargs='+', type=float, default=None,
                    help=f'Override multipliers (default {MULTIPLIERS})')
    ap.add_argument('--burst', nargs='+', type=int, default=None,
                    help='Optional burst lengths to sweep in ADDITION to point spikes '
                         '(e.g. --burst 3 10). Still normal-only. Off by default, matching '
                         'the original run_spikes_normal_only.py.')


def build_conditions(args):
    fractions = args.fractions if args.fractions else FRACTIONS
    multipliers = args.multipliers if args.multipliers else MULTIPLIERS
    seq_lens = [1] + (args.burst if args.burst else [])   # 1 = point spikes
    if args.test:
        fractions, multipliers, seq_lens = [0.05, 0.20], [3.0], [1]
    grid = [] if args.no_clean else [(None, None, 1)]
    grid += [(f, m, s) for f in fractions for m in multipliers for s in seq_lens]
    return [condition(_cond_name(f, m, sl),
                      {'fraction': f, 'multiplier': m, 'sequence_length': sl},
                      n_seeds=N_SEEDS)
            for (f, m, sl) in grid]


def corrupt(ctx, p):
    n_normal = int((ctx.labels == 0).sum())
    if p['fraction'] is None:
        return ctx.clean_data, {'n_spikes': 0, 'n_normal': n_normal}
    # spikes on NORMAL points only
    corruptor = TSCorruptor(ctx.df.copy(), value_col=ctx.value_col, label_col=ctx.label_col,
                            seed=ctx.seed, corruption_target=CORRUPTION_TARGET)
    seq_len = p['sequence_length']
    ts_corruptor.injectors.inject_spikes(
        corruptor, fraction=p['fraction'], multiplier=p['multiplier'],
        sequential=(seq_len > 1), sequence_length=max(1, seq_len))
    # record how many points were actually spiked (provenance, as in the original)
    n_spikes = 0
    for action in corruptor.get_corruption_report()['action_details']:
        if action['type'] == 'spikes':
            n_spikes = action['count']
    data = corruptor.get_corrupted_df().iloc[:, 0:-1].values.astype(float)
    return data, {'n_spikes': n_spikes, 'n_normal': n_normal}


SPEC = Spec(
    name='spikes_normal_only_tsbad',
    title='Spikes on NORMAL points only — TSB-AD',
    family='full',
    build_conditions=build_conditions,
    corrupt=corrupt,
    add_args=add_args,
    summary_keys=('condition', 'fraction', 'multiplier', 'sequence_length', 'model'),
    summary_means=(('n_spikes', 1),),
    test_help='Smoke test: 3 files, small grid',
)

if __name__ == "__main__":
    run_sweep(SPEC)
