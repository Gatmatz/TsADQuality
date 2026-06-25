"""
Simple, interpretable data cleaning methods for time series.

Each method takes a 1-D numpy array and returns a cleaned copy.
These are intentionally simple — the goal is to show that even basic
cleaning can recover anomaly detection performance.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter


# ======================================================================
# Spike / Outlier removal
# ======================================================================

def clean_zscore_clip(data, threshold=3.0):
    """Replace points beyond ±threshold std with the clipped boundary."""
    cleaned = data.copy()
    mu, sigma = np.nanmean(cleaned), np.nanstd(cleaned)
    if sigma < 1e-12:
        return cleaned
    upper = mu + threshold * sigma
    lower = mu - threshold * sigma
    cleaned = np.clip(cleaned, lower, upper)
    return cleaned


def clean_median_filter(data, kernel_size=5):
    """Replace each point with the median of its local window."""
    return median_filter(data.astype(float), size=kernel_size)


def clean_iqr_replace(data, k=1.5):
    """Replace IQR outliers with the nearest fence value."""
    cleaned = data.copy()
    q1, q3 = np.nanpercentile(cleaned, 25), np.nanpercentile(cleaned, 75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    cleaned = np.clip(cleaned, lower, upper)
    return cleaned


# ======================================================================
# Noise smoothing
# ======================================================================

def clean_moving_average(data, window=5):
    """Simple moving average smoothing."""
    kernel = np.ones(window) / window
    # Pad to preserve length
    padded = np.pad(data, (window // 2, window // 2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(data)]


def clean_savgol(data, window_length=11, polyorder=2):
    """Savitzky-Golay filter (preserves sharp features better than MA)."""
    wl = min(window_length, len(data))
    if wl % 2 == 0:
        wl -= 1
    if wl < polyorder + 2:
        return data.copy()
    return savgol_filter(data, wl, polyorder)


# ======================================================================
# Missing value imputation
# ======================================================================

def clean_interpolate_linear(data):
    """Linear interpolation for NaN values."""
    cleaned = data.copy()
    nans = np.isnan(cleaned)
    if not nans.any():
        return cleaned
    x = np.arange(len(cleaned))
    cleaned[nans] = np.interp(x[nans], x[~nans], cleaned[~nans])
    return cleaned


def clean_forward_fill(data):
    """Forward fill NaN values."""
    cleaned = data.copy()
    mask = np.isnan(cleaned)
    if not mask.any():
        return cleaned
    idx = np.where(~mask, np.arange(len(cleaned)), 0)
    np.maximum.accumulate(idx, out=idx)
    cleaned = cleaned[idx]
    # Handle leading NaNs with backward fill
    if np.isnan(cleaned[0]):
        first_valid = np.where(~np.isnan(data))[0]
        if len(first_valid) > 0:
            cleaned[:first_valid[0]] = data[first_valid[0]]
    return cleaned


# ======================================================================
# Freeze / stuck sensor
# ======================================================================

def clean_replace_constant_runs(data, min_run_length=10):
    """Detect constant-value runs and replace with linear interpolation."""
    cleaned = data.copy()
    n = len(cleaned)
    i = 0
    while i < n:
        j = i + 1
        while j < n and cleaned[j] == cleaned[i]:
            j += 1
        run_len = j - i
        if run_len >= min_run_length:
            # Replace with linear interpolation between boundaries
            left_val = cleaned[max(0, i - 1)]
            right_val = cleaned[min(n - 1, j)]
            cleaned[i:j] = np.linspace(left_val, right_val, run_len)
        i = j
    return cleaned


# ======================================================================
# Convenience: cleaning pipelines per corruption type
# ======================================================================

# Named wrapper functions (picklable, unlike lambdas)
def _ma5(d): return clean_moving_average(d, window=5)
def _ma11(d): return clean_moving_average(d, window=11)
def _savgol(d): return clean_savgol(d, window_length=11, polyorder=2)
def _zscore3(d): return clean_zscore_clip(d, threshold=3.0)
def _zscore2(d): return clean_zscore_clip(d, threshold=2.0)
def _median5(d): return clean_median_filter(d, kernel_size=5)
def _median11(d): return clean_median_filter(d, kernel_size=11)
def _iqr15(d): return clean_iqr_replace(d, k=1.5)
def _const_runs10(d): return clean_replace_constant_runs(d, min_run_length=10)
def _const_runs5(d): return clean_replace_constant_runs(d, min_run_length=5)


CLEANING_METHODS = {
    'white_noise': [
        ('Moving Average (w=5)', _ma5),
        ('Moving Average (w=11)', _ma11),
        ('Savitzky-Golay', _savgol),
    ],
    'spikes': [
        ('Z-score clip (3s)', _zscore3),
        ('Z-score clip (2s)', _zscore2),
        ('Median filter (k=5)', _median5),
        ('IQR replace (1.5)', _iqr15),
    ],
    'missing': [
        ('Linear interpolation', clean_interpolate_linear),
        ('Forward fill', clean_forward_fill),
    ],
    'freeze': [
        ('Replace constant runs (10)', _const_runs10),
        ('Replace constant runs (5)', _const_runs5),
    ],
    'swap': [
        ('Median filter (k=5)', _median5),
        ('Median filter (k=11)', _median11),
        ('Moving Average (w=5)', _ma5),
    ],
}
