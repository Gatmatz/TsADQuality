import numpy as np

from tsadquality.reproducibility import ReproducibleOperations

from .BaseError import BaseCorruptor


class SwapInjector(BaseCorruptor):
    """
    Swaps `num_swaps` segments of the time series using NumPy arrays.

    The segment length is derived from the fraction and num_swaps, i.e. the requested
    fraction is split evenly over num_swaps segment pairs: swap_length = max(1,
    floor(fraction * N / (2 * num_swaps))). When num_swaps is omitted (point swap),
    swap_length is forced to 1 and num_swaps is derived from the fraction instead, so
    every swap is a single-point swap.

    Known issue, kept as is so the TSB-UAD results stay reproducible: once swap_length > 1
    this under-delivers the requested fraction (out-of-bounds pairs are dropped silently and
    later segments can overlap earlier ones), sometimes swapping nothing at all. For an exact
    fraction use tsadquality.corruption_utils.swap.swap_segments_fast (ported alongside
    src/ts_corruptor/swap.py); see that module's docstring.
    """

    def inject(self, fraction, num_swaps=None):
        n = len(self.df)
        target_indices = np.array(self.get_target_indices())

        if num_swaps is None:
            swap_length = 1
            num_swaps = max(1, int(fraction * n / 2))
        else:
            swap_length = max(1, int(fraction * n / (2 * num_swaps)))

        if len(target_indices) < 2 * swap_length or num_swaps == 0:
            return self.df

        # Extract data as a fast, writable NumPy array
        values = self.df[self.value_col].to_numpy().copy()
        swapped_indices = []

        # We still need a loop, but we do it on raw NumPy arrays which is 100x faster than Pandas .loc
        for _ in range(num_swaps):
            # Pick first random start index
            idx1 = ReproducibleOperations.choice(target_indices)

            mask = np.abs(target_indices - idx1) >= swap_length
            valid_targets_for_idx2 = target_indices[mask]

            if len(valid_targets_for_idx2) == 0:
                continue

            idx2 = ReproducibleOperations.choice(valid_targets_for_idx2)

            # Ensure segments are within bounds
            if (idx1 + swap_length <= n) and (idx2 + swap_length <= n):
                # Perform the swap on the NumPy array
                val1 = values[idx1: idx1 + swap_length].copy()
                val2 = values[idx2: idx2 + swap_length].copy()

                values[idx1: idx1 + swap_length] = val2
                values[idx2: idx2 + swap_length] = val1

                # Record affected indices
                used1 = set(range(idx1, idx1 + swap_length))
                used2 = set(range(idx2, idx2 + swap_length))
                swapped_indices.extend(used1)
                swapped_indices.extend(used2)

                # Remove used indices to prevent duplicate swaps
                used_all = used1 | used2
                target_indices = target_indices[~np.isin(target_indices, list(used_all))]

        # Write back to pandas only ONCE at the end
        self.df[self.value_col] = values

        if swapped_indices:
            unique_swapped = np.unique(swapped_indices)
            self.record_corruption('swap', unique_swapped, {
                'fraction': fraction,
                'num_swaps': num_swaps,
                'swap_length': swap_length,
            })

        return self.df
