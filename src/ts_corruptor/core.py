import numpy as np
import pandas as pd

class TSCorruptor:
    """
    The Core engine handling DataFrame tracking, history logging, and targeting logic.
    
    corruption_target controls WHERE corruption is applied relative to anomalies:
    
        'global'              — All points (no position filtering)
        'only_normal'         — Only normal (non-anomaly) points
        'overlapping_anomaly' — Only the anomaly points themselves
        'near_anomaly'        — Normal points within ±window_size of any anomaly
        'before_anomaly'      — Normal points in [anomaly - window_size, anomaly - 1]
        'after_anomaly'       — Normal points in [anomaly + 1, anomaly + window_size]
        'far_from_anomaly'    — Normal points outside ±window_size of all anomalies
    
    Example with window_size=3 and an anomaly at index 10:
    
        ... [7 8 9] [10] [11 12 13] ...
             before  anom   after
             |----- near ---------|
        far                          far
    """
    
    def __init__(self, df, value_col='value', label_col="is_anomaly", seed=None, corruption_target='global', window_size=50):
        self.df = df.copy()
        self.df_original = df.copy()
        self.value_col = value_col
        self.label_col = label_col
        self.corruption_target = corruption_target
        self.window_size = window_size
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        self.history = []
        self.corruption_mask = pd.Series(False, index=self.df.index)
        self.std = self.df[self.value_col].std()
        if self.std == 0 or np.isnan(self.std):
            self.std = 1.0

    def get_target_indices(self):
        """Returns indices where corruption can be applied based on corruption_target."""
        all_indices = self.df.index.tolist()
        anomaly_indices = self.df[self.df[self.label_col] == 1].index.tolist()
        normal_indices = self.df[self.df[self.label_col] == 0].index.tolist()
        n = len(self.df)

        if self.corruption_target == 'global':
            return all_indices
        elif self.corruption_target == 'only_normal':
            return normal_indices
        elif self.corruption_target == 'overlapping_anomaly':
            return anomaly_indices
        elif self.corruption_target == 'before_anomaly':
            # Normal points in [anomaly - window_size, anomaly - 1]
            before_indices = set()
            for idx in anomaly_indices:
                for i in range(max(0, idx - self.window_size), idx):
                    before_indices.add(i)
            return list(before_indices.intersection(normal_indices))
        elif self.corruption_target == 'after_anomaly':
            # Normal points in [anomaly + 1, anomaly + window_size]
            after_indices = set()
            for idx in anomaly_indices:
                for i in range(idx + 1, min(n, idx + self.window_size + 1)):
                    after_indices.add(i)
            return list(after_indices.intersection(normal_indices))
        elif self.corruption_target == 'near_anomaly':
            near_indices = set()
            for idx in anomaly_indices:
                for i in range(max(0, idx - self.window_size), min(n, idx + self.window_size + 1)):
                    near_indices.add(i)
            # Only corrupt normal points near anomalies
            return list(near_indices.intersection(normal_indices))
        elif self.corruption_target == 'far_from_anomaly':
            near_indices = set()
            for idx in anomaly_indices:
                for i in range(max(0, idx - self.window_size), min(n, idx + self.window_size + 1)):
                    near_indices.add(i)
            far_indices = set(all_indices) - near_indices
            return list(far_indices.intersection(normal_indices))
        
        return all_indices

    def get_injectable_indices(self, required_amount):
        target_indices = self.get_target_indices()
        if required_amount > len(target_indices):
            required_amount = len(target_indices)
        return self.rng.choice(target_indices, size=required_amount, replace=False)

    def record_corruption(self, corruption_type, indices, params):
        if len(indices) > 0:
            self.history.append({
                'type': corruption_type,
                'count': len(indices),
                'first_idx': int(min(indices)),
                'last_idx': int(max(indices)),
                'params': params
            })
            self.corruption_mask.loc[indices] = True

    def get_corruption_report(self):
        total_len = len(self.df)
        total_corrupted = self.corruption_mask.sum()
        
        anomaly_indices = np.where(self.df[self.label_col] == 1)[0]
        corrupted_indices = np.where(self.corruption_mask == True)[0]
        avg_dist = -1
        if len(anomaly_indices) > 0 and len(corrupted_indices) > 0:
            distances = [np.min(np.abs(anomaly_indices - c_idx)) for c_idx in corrupted_indices]
            avg_dist = np.mean(distances)

        return {
            'summary': {
                'total_points': total_len,
                'corrupted_points': int(total_corrupted),
                'corruption_percentage': (total_corrupted / total_len) * 100,
                'avg_distance_to_anomaly': float(avg_dist),
                'corruption_target_mode': self.corruption_target
            },
            'action_details': self.history
        }

    def compare_statistics(self):
        orig = self.df_original[self.value_col]
        corr = self.df[self.value_col]
        mask = ~(orig.isna() | corr.isna())
        s_orig, s_corr = orig[mask], corr[mask]
        
        if len(s_orig) == 0: return {}
        
        noise = s_corr - s_orig
        signal_power = np.mean(s_orig.to_numpy() ** 2)
        noise_power  = np.mean(noise.to_numpy() ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 50.0

        return {
            'snr_db': float(snr_db),
            'mean_shift_std': float(abs(s_corr.mean() - s_orig.mean()) / s_orig.std()),
            'variance_ratio': float(s_corr.var() / s_orig.var()),
            'missing_count': int(self.df[self.value_col].isna().sum())
        }

    def get_corrupted_df(self):
        return self.df

    def get_corruption_mask(self):
        return self.corruption_mask
