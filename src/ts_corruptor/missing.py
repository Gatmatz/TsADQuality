"""Which points go missing — MCAR vs MNAR selection, scattered or in bursts.

Both functions only choose positions; the caller turns them into NaNs. Used by the
TSB-AD MNAR experiments.
"""
import numpy as np


def select_missing_indices(values, fraction, mechanism, rng):
    """Which indices go missing, per mechanism.

    Verbatim transcription of the original select_missing_indices(), weights and floors
    included (its unused `labels` parameter is dropped). All mechanisms produce exactly the
    same number of missing points, which is what makes the comparison fair.
    """
    n = len(values)
    n_missing = max(1, int(fraction * n))

    if mechanism == 'mcar':
        # Equal probability for all points
        return rng.choice(n, size=n_missing, replace=False)

    if mechanism == 'mnar_extreme':
        # Probability proportional to |z-score| — extreme values in EITHER tail
        z = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-10)
        weights = np.abs(z)
        weights = weights / weights.sum()
        return rng.choice(n, size=n_missing, replace=False, p=weights)

    if mechanism == 'mnar_high':
        # Probability proportional to the POSITIVE part of z — sensor caps out at the top
        z = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-10)
        weights = np.maximum(z, 0) + 0.01     # small floor so all points have some chance
        weights = weights / weights.sum()
        return rng.choice(n, size=n_missing, replace=False, p=weights)

    raise ValueError(f"Unknown mechanism: {mechanism}")


def select_burst_starts(values, n, burst_length, num_bursts, mechanism, rng):
    """Start indices of `num_bursts` NON-OVERLAPPING blocks of length `burst_length`.

    Greedy, identical in semantics and in output to the original select_burst_starts():
      * mcar_burst -> walk a shuffled list of every legal start;
      * mnar_*     -> walk the starts ordered by descending mean window score;
    take a candidate whenever it does not touch an already placed block.

    Only the occupancy test differs. The original builds `set(range(c, c + burst_length))` for
    every candidate it examines; the MNAR candidates are clustered (extreme |z| sits around
    anomalies), so it examines many overlapping starts before finding a free one, and each of
    those checks allocates a set of burst_length integers. With burst_length = 180 000 on a
    900k-point series that is pathological. A boolean occupancy array makes the same test an
    O(burst_length) numpy slice with no allocation.

    Returns fewer than `num_bursts` starts if the series cannot hold them; the caller records
    the realised count.
    """
    if burst_length >= n:
        return [0]
    max_start = n - burst_length
    if max_start <= 0:
        return [0]

    if mechanism == 'mcar_burst':
        order = np.arange(max_start)
        rng.shuffle(order)
    else:
        # Mean window score via a prefix sum: score[i] = mean(w[i : i + burst_length]).
        z = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-10)
        if mechanism == 'mnar_extreme_burst':
            w = np.abs(z)                 # both tails: outliers of either sign
        elif mechanism == 'mnar_high_burst':
            w = np.maximum(z, 0)          # upper tail only: peaks
        else:
            raise ValueError(f'unknown mechanism: {mechanism}')
        c = np.concatenate([[0.0], np.cumsum(w)])
        position_scores = (c[burst_length:max_start + burst_length] - c[:max_start]) / burst_length
        order = np.argsort(-position_scores)      # highest first, same kind as the original

    used = np.zeros(n, dtype=bool)
    starts = []
    for idx in order:
        if len(starts) >= num_bursts:
            break
        i = int(idx)
        if not used[i:i + burst_length].any():
            starts.append(i)
            used[i:i + burst_length] = True
    return starts
