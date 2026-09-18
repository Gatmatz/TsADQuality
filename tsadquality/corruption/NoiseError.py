import numpy as np

from tsadquality.reproducibility import ReproducibleOperations

from .BaseError import BaseCorruptor


class WhiteNoiseSNRInjector(BaseCorruptor):
    """
    Adds Gaussian white noise to achieve a specific Signal-to-Noise Ratio (SNR)
    in decibels (dB). Respects the corruptor's corruption_target setting.

    SNR is calculated from the full signal power, but noise is only applied
    to the indices selected by the corruption_target mode.

    Formula: SNR_dB = 10 * log10( P_signal / P_noise )
    """

    def inject(self, snr_db):
        # 1. Get the original signal
        signal = self.df[self.value_col].to_numpy('float')

        # Ignore NaNs for power calculation if any exist
        clean_signal = signal[~np.isnan(signal)]
        if len(clean_signal) == 0:
            return self.df

        # 2. Calculate Signal Power (always from full signal for consistent SNR)
        signal_power = np.mean(clean_signal ** 2)
        if signal_power == 0:
            signal_power = 1e-6

        # 3. Calculate Target Noise Power based on desired SNR dB
        noise_power = signal_power / (10 ** (snr_db / 10))

        # 4. Generate Gaussian Noise
        noise_std = np.sqrt(noise_power)
        noise = ReproducibleOperations.normal(loc=0, scale=noise_std, size=len(signal))

        # 5. Get target indices based on corruption_target mode
        target_indices = np.array(self.get_target_indices())

        # Filter out indices with NaN values
        valid_mask = ~np.isnan(signal[target_indices]) if len(target_indices) > 0 else np.array([], dtype=bool)
        target_indices = target_indices[valid_mask]

        if len(target_indices) == 0:
            return self.df

        # Ensure column is float to avoid dtype incompatibility (int64 += float64)
        if not np.issubdtype(self.df[self.value_col].dtype, np.floating):
            self.df[self.value_col] = self.df[self.value_col].astype(float)

        # 6. Add noise only to target indices
        self.df.loc[target_indices, self.value_col] += noise[target_indices]

        self.record_corruption('white_noise_snr', target_indices, {'snr_db': snr_db, 'noise_power': noise_power})

        return self.df
