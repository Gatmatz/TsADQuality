"""Corruptions confined to one segment data[start:end], used by the propagation experiment."""
import numpy as np


def inject_localized_noise(data, start, end, snr_db, rng):
    """Inject white noise into data[start:end] at the given SNR."""
    corrupted = data.copy()
    segment = data[start:end]
    signal_power = np.mean(segment ** 2)
    if signal_power < 1e-10:
        signal_power = 1e-10
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), size=end - start)
    corrupted[start:end] = segment + noise
    return corrupted


def inject_localized_spikes(data, start, end, magnitude, rng):
    """Inject random spikes into data[start:end]. Magnitude is in units of the GLOBAL std."""
    corrupted = data.copy()
    n_spikes = max(1, (end - start) // 10)   # ~10% of the segment becomes a spike
    spike_indices = rng.choice(range(start, end), size=n_spikes, replace=False)
    std = np.std(data)
    if std < 1e-10:
        std = 1.0
    for idx in spike_indices:
        sign = rng.choice([-1, 1])
        corrupted[idx] = data[idx] + sign * magnitude * std
    return corrupted


def inject_localized_missing(data, start, end, fraction, rng):
    """Set a fraction of data[start:end] to NaN."""
    corrupted = data.copy()
    segment_len = end - start
    n_missing = max(1, int(fraction * segment_len))
    missing_indices = rng.choice(range(start, end), size=n_missing, replace=False)
    corrupted[missing_indices] = np.nan
    return corrupted
