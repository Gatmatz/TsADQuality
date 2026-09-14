"""
White Gaussian Noise (SNR) robustness experiment — TSB-AD version.

Port of run_whitenoise_snr.py to the TSB-AD benchmark. Keeps the SAME corruption
(ts_corruptor.inject_white_noise_snr) and SAME SNR grid, but evaluates with the
OFFICIAL TSB-AD pipeline (run_Unsupervise_AD / run_Semisupervise_AD + find_length_rank
+ TSB_AD get_metrics) on the FULL 350-series TSB-AD-U eval set — identical to the
baseline setup, so clean-vs-corrupted is directly comparable.

Usage:
    python src/experiments/corruption_tsbad/white_noise_snr.py --test
    python src/experiments/corruption_tsbad/white_noise_snr.py --models IForest
    python src/experiments/corruption_tsbad/white_noise_snr.py --models IForest Sub_PCA POLY KShapeAD KMeansAD_U --workers 8
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import ts_corruptor.injectors  # noqa: E402
from ts_corruptor.core import TSCorruptor  # noqa: E402
from tsbad.harness import Spec, condition, run_sweep  # noqa: E402

SNRS_DB = [40, 30, 20, 10, 5, 0, -5, -10, -20]
N_SEEDS = 1


def build_conditions(args):
    snrs = [40, 10] if args.test else list(SNRS_DB)
    grid = ([None] if not args.no_clean else []) + snrs
    return [condition("clean" if snr_db is None else f"snr_{snr_db}dB", {'snr_db': snr_db},
                      n_seeds=N_SEEDS)
            for snr_db in grid]


def corrupt(ctx, params):
    if params['snr_db'] is None:
        return ctx.clean_data, {}
    corruptor = TSCorruptor(ctx.df.copy(), value_col=ctx.value_col, label_col=ctx.label_col,
                            seed=ctx.seed)
    ts_corruptor.injectors.inject_white_noise_snr(corruptor, snr_db=params['snr_db'])
    return corruptor.get_corrupted_df().iloc[:, 0:-1].values.astype(float), {}


def seed_key(file_name, cond, seed, model_name):
    # This runner predates the shared key and seeds on snr_db (None for the clean anchor)
    # instead of the condition name. Kept: changing it would change KMeansAD_U scores.
    return f"{file_name}|{cond['params']['snr_db']}|{seed}|{model_name}"


SPEC = Spec(
    name='white_noise_snr_tsbad',
    title='White Noise (SNR) Robustness — TSB-AD',
    family='full',
    build_conditions=build_conditions,
    corrupt=corrupt,
    seed_key=seed_key,
    summary_keys=('snr_db', 'condition', 'model'),
    test_help='Smoke test: 3 files, 2 SNRs',
)

if __name__ == "__main__":
    run_sweep(SPEC)
