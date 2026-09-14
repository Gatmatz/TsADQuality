"""Swap corruptions on numpy arrays, used by the TSB-AD swap experiments.

Rows are moved whole, so a multivariate series keeps its channels aligned, and labels
never move (only values do).

Why not injectors.inject_swap
-----------------------------
`injectors.inject_swap` does NOT reliably deliver the requested fraction once
swap_length > 1. Two mechanisms:

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

`swap_segments_fast` instead samples 2*num_swaps STRICTLY DISJOINT segments, so every
condition delivers exactly 2*num_swaps*L swapped points. `inject_swap` itself is left
untouched so previously produced TSB-UAD results stay reproducible; the `*_reference`
functions run it for cross-checking.
"""
import numpy as np

from . import injectors
from .core import TSCorruptor


def swap_points_fast(values, fraction, rng, swap_length=1):
    """Randomly swap disjoint pairs of rows. Vectorised O(n) equivalent of inject_swap.

    Matches ts_corruptor.injectors.inject_swap for swap_length=1 / max_distance=None:
    num_swaps = int(n*fraction) // (2*swap_length) disjoint pairs, drawn uniformly.
    Rows are swapped whole, so a multivariate series keeps its channels aligned.

    Returns (new_values, n_points_swapped).
    """
    n = len(values)
    num_swaps = int(n * fraction) // (2 * swap_length)
    if num_swaps == 0 or n < 2 * swap_length:
        return values, 0

    # Draw 2*num_swaps distinct indices in one shot, then pair the halves. Sampling
    # without replacement is what makes the pairs disjoint, which is exactly what the
    # original loop achieved by pruning its candidate array.
    k = min(2 * num_swaps, n)
    k -= k % 2
    chosen = rng.choice(n, size=k, replace=False)
    i1, i2 = chosen[:k // 2], chosen[k // 2:]

    out = values.copy()
    out[i1], out[i2] = values[i2], values[i1]
    return out, k


def swap_points_reference(df, value_col, label_col, fraction, seed, swap_length=1):
    """Original ts_corruptor path — kept for cross-checking on small series."""
    corruptor = TSCorruptor(df.copy(), value_col=value_col, label_col=label_col, seed=seed)
    injectors.inject_swap(
        corruptor, fraction=fraction, swap_length=swap_length, max_distance=None)
    n_swapped = int(corruptor.corruption_mask.sum())
    return corruptor.get_corrupted_df(), n_swapped


def derive_swap_length(n, fraction, num_swaps):
    """swap_length as defined by the experiment: total fraction split over num_swaps pairs."""
    return max(1, int(fraction * n / (2 * num_swaps)))


def swap_segments_fast(values, fraction, num_swaps, rng):
    """Swap `num_swaps` pairs of contiguous, strictly disjoint segments of length L.

    L = max(1, int(fraction*n / (2*num_swaps))), the definition used by the experiment.

    Disjoint starts are drawn uniformly via the standard bijection between "k disjoint
    blocks of length L in n slots" and "k distinct starts in n-k*(L-1) slots": draw the
    starts without replacement, sort them, then push start i right by i*(L-1). The largest
    start is then n-L at most, so no segment can run off the end and none can overlap.
    That is what guarantees exactly 2*num_swaps*L swapped points in every condition —
    the property the whole experiment rests on.

    Rows are swapped whole, so a multivariate series keeps its channels aligned.

    Returns (new_values, n_points_swapped, swap_length, num_swaps_actual).
    """
    n = len(values)
    L = derive_swap_length(n, fraction, num_swaps)

    # Fit as many pairs as the series can actually hold disjointly. Only bites on series
    # shorter than 2*num_swaps*L, i.e. tiny ones where L has been clamped to 1.
    pairs = min(num_swaps, n // (2 * L))
    if pairs < 1:
        return values, 0, L, 0

    k = 2 * pairs                       # segments to place
    free = n - k * (L - 1)              # slots in the compressed coordinate system
    starts = np.sort(rng.choice(free, size=k, replace=False)) + np.arange(k) * (L - 1)

    order = rng.permutation(k)          # random pairing of the placed segments
    a, b = starts[order[:pairs]], starts[order[pairs:]]

    out = values.copy()
    # <= 20 iterations; the segments are disjoint, so the order of the swaps is irrelevant
    for s1, s2 in zip(a, b):
        tmp = out[s1:s1 + L].copy()
        out[s1:s1 + L] = out[s2:s2 + L]
        out[s2:s2 + L] = tmp
    return out, k * L, L, pairs


def swap_segments_reference(df, value_col, label_col, fraction, num_swaps, seed):
    """Original ts_corruptor path — kept for cross-checking. See this module's docstring for
    why it under-delivers the requested fraction once swap_length > 1."""
    n = len(df)
    L = derive_swap_length(n, fraction, num_swaps)
    corruptor = TSCorruptor(df.copy(), value_col=value_col, label_col=label_col, seed=seed)
    injectors.inject_swap(
        corruptor, fraction=fraction, swap_length=L, max_distance=None)
    n_swapped = int(corruptor.corruption_mask.sum())
    return corruptor.get_corrupted_df(), n_swapped, L, (n_swapped // (2 * L) if L else 0)


def permute_segments(values, n_segments, rng):
    """Split into n_segments near-equal parts along axis 0 and shuffle their order.

    Works on row indices rather than on the values themselves, so a multivariate series
    keeps its channels aligned: every column is reordered by the same permutation.

    Returns (permuted_values, perm_order).
    """
    n = len(values)
    seg_len, remainder = divmod(n, n_segments)

    # Segment boundaries; the remainder is spread over the first segments so the
    # concatenation is exactly the original length.
    bounds, start = [], 0
    for i in range(n_segments):
        end = start + seg_len + (1 if i < remainder else 0)
        bounds.append((start, end))
        start = end

    # Shuffle until the order actually differs from the identity, otherwise the
    # "corrupted" condition would silently be a clean one.
    original_order = list(range(n_segments))
    perm = original_order.copy()
    for _ in range(100):
        rng.shuffle(perm)
        if perm != original_order:
            break

    order = np.concatenate([np.arange(*bounds[i]) for i in perm])
    return values[order], perm
