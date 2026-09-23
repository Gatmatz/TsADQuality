import numpy as np

from .base import BaseCorruptor


class SensorStuckInjector(BaseCorruptor):
    """
    Simulates a frozen sensor hardware failure. Captures the value at each block's
    start index and freezes the signal to that constant value for the block length.
    The `fraction` of the series is split evenly over `stuck_blocks` blocks.
    """

    def inject(self, fraction, stuck_blocks):
        total_points = int(len(self.df) * fraction)
        stuck_length = max(1, total_points // stuck_blocks)

        target_indices = self.get_injectable_indices(stuck_blocks)
        if len(target_indices) == 0:
            return self.df

        expanded_indices = set()
        for start_idx in target_indices:
            stuck_value = self.df.loc[start_idx, self.value_col]
            for i in range(stuck_length):
                idx = start_idx + i
                if idx < len(self.df):
                    self.df.loc[idx, self.value_col] = stuck_value
                    expanded_indices.add(idx)

        stuck_array = np.array(sorted(expanded_indices))
        params = {'fraction': fraction, 'stuck_blocks': stuck_blocks, 'stuck_length': stuck_length}
        self.record_corruption('sensor_stuck', stuck_array, params)
        return self.df
