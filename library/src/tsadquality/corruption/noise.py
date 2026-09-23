import numpy as np

from .base import BaseCorruptor


class WhiteNoiseSNRInjector(BaseCorruptor):
    """
    Adds Gaussian white noise to achieve a specific Signal-to-Noise Ratio (SNR)
    in decibels (dB). Respects the corruptor's corruption_target setting.

    SNR is calculated from the full signal power, but noise is only applied
    to the indices selected by the corruption_target mode.

    Formula: SNR_dB = 10 * log10( P_signal / P_noise )
    """

    def inject(self, snr_db):
        signal = self.df[self.value_col].to_numpy('float')

        # Ignore NaNs for power calculation if any exist
        clean_signal = signal[~np.isnan(signal)]
        if len(clean_signal) == 0:
            return self.df

        # Signal power is always taken from the full signal for a consistent SNR
        signal_power = np.mean(clean_signal ** 2)
        if signal_power == 0:
            signal_power = 1e-6

        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(loc=0, scale=np.sqrt(noise_power), size=len(signal))

        target_indices = np.array(self.get_target_indices())
        valid_mask = ~np.isnan(signal[target_indices]) if len(target_indices) > 0 else np.array([], dtype=bool)
        target_indices = target_indices[valid_mask]
        if len(target_indices) == 0:
            return self.df

        # Ensure column is float to avoid dtype incompatibility (int64 += float64)
        if not np.issubdtype(self.df[self.value_col].dtype, np.floating):
            self.df[self.value_col] = self.df[self.value_col].astype(float)

        self.df.loc[target_indices, self.value_col] += noise[target_indices]
        self.record_corruption('white_noise_snr', target_indices, {'snr_db': snr_db, 'noise_power': noise_power})
        return self.df
