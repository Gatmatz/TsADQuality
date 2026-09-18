import numpy as np

from tsadquality.reproducibility import ReproducibleOperations

from .BaseError import BaseCorruptor


class PermutationInjector(BaseCorruptor):
    """
    Splits the series into `num_permutations` near-equal segments and shuffles their
    order. Values are rearranged but preserved (no new values created, no values
    removed); labels stay in their ORIGINAL positions, which is what makes this a
    corruption rather than a relabelling — the detector sees reordered data but is
    scored against the original ground truth.

    Based on:
      - Um et al. (2017) — Permutation augmentation for wearable sensor data
      - Grover et al. (NeurIPS 2024) — Segment, Shuffle, and Stitch (S3)
    """

    def inject(self, num_permutations):
        n = len(self.df)
        if num_permutations < 2 or n < num_permutations * 2:
            return self.df

        # Segment boundaries; the remainder is spread over the first segments so the
        # concatenation is exactly the original length.
        seg_len, remainder = divmod(n, num_permutations)
        bounds, start = [], 0
        for i in range(num_permutations):
            end = start + seg_len + (1 if i < remainder else 0)
            bounds.append((start, end))
            start = end

        # A derangement guarantees the segment order actually changes, otherwise the
        # "corrupted" condition would silently be a clean one.
        perm = ReproducibleOperations.derangement(list(range(num_permutations)))
        order = np.concatenate([np.arange(*bounds[i]) for i in perm])

        values = self.df[self.value_col].to_numpy().copy()
        self.df[self.value_col] = values[order]

        all_indices = np.arange(n)
        self.record_corruption('permutation', all_indices, {
            'num_permutations': num_permutations,
            'segment_length': seg_len,
        })
        return self.df
