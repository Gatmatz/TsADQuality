"""
Swap Corruption — Segment Permutation — TSB-AD version.

Port of run_swap_permutation.py to the TSB-AD benchmark, built to mirror
spikes.py step for step. Same corruption (split into N equal segments and
shuffle their order) and same parameter grid, but evaluated with the OFFICIAL TSB-AD
pipeline (run_Unsupervise_AD / run_Semisupervise_AD + find_length_rank + TSB_AD
get_metrics) on the full 350-series TSB-AD-U eval list.

Based on:
  - Um et al. (2017) — Permutation augmentation for wearable sensor data
  - Grover et al. (NeurIPS 2024) — Segment, Shuffle, and Stitch (S3)

The entire series is permuted — values are rearranged but preserved (no new values
created, no values removed). Labels stay in their ORIGINAL positions, which is what
makes this a corruption rather than a relabelling: the detector sees reordered data
but is scored against the original ground truth.

A `clean` anchor condition runs the identical pipeline with no permutation, so it
reproduces the TSB-AD baseline exactly and the degradation curve starts from the
right point.

Key question: "How much temporal reordering can anomaly detectors tolerate before
              performance breaks down?"

Usage:
    python src/experiments/corruption_tsbad/swap_permutation.py --test
    python src/experiments/corruption_tsbad/swap_permutation.py --models IForest --workers 4
    python src/experiments/corruption_tsbad/swap_permutation.py --models MatrixProfile --workers 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import numpy as np  # noqa: E402

from ts_corruptor.swap import permute_segments  # noqa: E402
from tsbad.harness import Skip, Spec, condition, run_sweep  # noqa: E402

N_SEGMENTS_LIST = [2, 4, 8, 16, 24]
N_SEEDS = 3


def _cond_name(n_segments):
    return "clean" if n_segments is None else f"nseg_{n_segments}"


def build_conditions(args):
    n_segments_list, n_seeds = N_SEGMENTS_LIST, N_SEEDS
    if args.test:
        n_segments_list, n_seeds = [2, 8], 1
    grid = [] if args.no_clean else [None]
    grid += list(n_segments_list)
    # the clean anchor is deterministic: one seed is enough
    return [condition(_cond_name(ns), {'n_segments': ns}, n_seeds=1 if ns is None else n_seeds)
            for ns in grid]


def corrupt(ctx, p):
    n, n_segments = ctx.n, p['n_segments']
    if n_segments is not None and n < n_segments * 2:
        raise Skip(f'Series too short ({n}) for {n_segments} segments')

    # Labels are NOT permuted: they stay in their original positions, which is the point of
    # the experiment.
    if n_segments is None:
        data, perm_order, seg_len = ctx.clean_data, None, None
    else:
        rng = np.random.RandomState(ctx.seed)
        data, perm_order = permute_segments(ctx.clean_data, n_segments, rng)
        seg_len = n // n_segments
    return data, {'segment_length': seg_len, 'permutation': str(perm_order)}


SPEC = Spec(
    name='swap_permutation_tsbad',
    title='Segment Permutation Robustness — TSB-AD',
    family='full',
    build_conditions=build_conditions,
    corrupt=corrupt,
    summary_keys=('condition', 'n_segments', 'model'),
    summary_means=(('segment_length', 1),),
    test_help='Smoke test: 3 files, small grid',
)

if __name__ == "__main__":
    run_sweep(SPEC)
