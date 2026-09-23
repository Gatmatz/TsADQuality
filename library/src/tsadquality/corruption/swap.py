import numpy as np

from .base import BaseCorruptor


class SwapInjector(BaseCorruptor):
    """
    Swaps `num_swaps` pairs of equal-length segments of the time series.

    The segment length is derived from the fraction and num_swaps: the requested
    fraction is split evenly over num_swaps segment pairs, i.e. swap_length =
    max(1, floor(fraction * N / (2 * num_swaps))). When num_swaps is omitted (point
    swap), swap_length is 1 and num_swaps is derived from the fraction instead, so
    every swap exchanges two single points.

    Segments are only placed where a full segment fits among the targetable points and
    never overlap an earlier swap, so the requested fraction is delivered whenever the
    series has room for it.
    """

    def inject(self, fraction, num_swaps=None):
        n = len(self.df)
        target_indices = np.array(self.get_target_indices(), dtype=int)

        if num_swaps is None:
            swap_length = 1
            num_swaps = max(1, int(fraction * n / 2))
        else:
            swap_length = max(1, int(fraction * n / (2 * num_swaps)))

        if len(target_indices) < 2 * swap_length or num_swaps == 0:
            return self.df

        values = self.df[self.value_col].to_numpy().copy()

        if swap_length == 1:
            # Point swap: draw all pairs at once, so every point is used at most once.
            k = min(num_swaps, len(target_indices) // 2)
            chosen = self.rng.choice(target_indices, size=2 * k, replace=False)
            first, second = chosen[:k], chosen[k:]
            values[first], values[second] = values[second].copy(), values[first].copy()
            swapped_indices = chosen
        else:
            available = np.zeros(n, dtype=bool)
            available[target_indices] = True
            swapped = []
            for _ in range(num_swaps):
                # Starts whose whole segment is still available (targetable and unused).
                counts = np.concatenate([[0], np.cumsum(available)])
                starts = np.flatnonzero(counts[swap_length:] - counts[:-swap_length] == swap_length)
                if len(starts) == 0:
                    break
                start1 = self.rng.choice(starts)
                starts2 = starts[np.abs(starts - start1) >= swap_length]
                if len(starts2) == 0:
                    break
                start2 = self.rng.choice(starts2)

                seg1 = slice(start1, start1 + swap_length)
                seg2 = slice(start2, start2 + swap_length)
                values[seg1], values[seg2] = values[seg2].copy(), values[seg1].copy()
                available[seg1] = False
                available[seg2] = False
                swapped.extend(range(start1, start1 + swap_length))
                swapped.extend(range(start2, start2 + swap_length))
            swapped_indices = np.array(swapped, dtype=int)

        self.df[self.value_col] = values

        if len(swapped_indices) > 0:
            self.record_corruption('swap', np.unique(swapped_indices), {
                'fraction': fraction,
                'num_swaps': num_swaps,
                'swap_length': swap_length,
            })
        return self.df
