import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Registry for all available corruptors
INJECTORS = {}

def register_injector(name):
    """Decorator to register a new injector function."""
    def decorator(func):
        INJECTORS[name] = func
        return func
    return decorator

@register_injector('missing')
def inject_point_missing(corruptor, fraction=0.05):
    """
    Injects missing values (NaN) into the dataframe based on the corruptor's target criteria.
    """
    count = int(len(corruptor.df) * fraction)
    indices = corruptor.get_injectable_indices(count)
    
    if len(indices) > 0:
        corruptor.df.loc[indices, corruptor.value_col] = np.nan
        corruptor.record_corruption('point_missing', indices, {'fraction': fraction})
    return corruptor.df

@register_injector('noise')
def inject_white_noise(corruptor, level=0.1):
    """
    Adds simple Gaussian white noise to the targeted segments of the dataframe based on the standard deviation.
    (Legacy version - prefer 'white_noise_snr' for scientific experiments).
    """
    target_indices = corruptor.get_target_indices()
    if len(target_indices) > 0:
        noise = corruptor.rng.normal(0, level * corruptor.std, size=len(target_indices))
        corruptor.df.loc[target_indices, corruptor.value_col] += noise
        corruptor.record_corruption('white_noise', target_indices, {'level': level})
    return corruptor.df


@register_injector('white_noise_snr')
def inject_white_noise_snr(corruptor, snr_db=20.0):
    """
    Adds Gaussian white noise to achieve a specific Signal-to-Noise Ratio (SNR)
    in decibels (dB). Respects the corruptor's corruption_target setting.
    
    SNR is calculated from the full signal power, but noise is only applied
    to the indices selected by the corruption_target mode.
    
    Formula: SNR_dB = 10 * log10( P_signal / P_noise )
    """
    # 1. Get the original signal
    signal = corruptor.df[corruptor.value_col].to_numpy('float')
    
    # Ignore NaNs for power calculation if any exist
    clean_signal = signal[~np.isnan(signal)]
    if len(clean_signal) == 0:
        return corruptor.df
        
    # 2. Calculate Signal Power (always from full signal for consistent SNR)
    signal_power = np.mean(clean_signal ** 2)
    if signal_power == 0:
        signal_power = 1e-6
        
    # 3. Calculate Target Noise Power based on desired SNR dB
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # 4. Generate Gaussian Noise
    noise_std = np.sqrt(noise_power)
    noise = corruptor.rng.normal(0, noise_std, size=len(signal))
    
    # 5. Get target indices based on corruption_target mode
    target_indices = np.array(corruptor.get_target_indices())
    
    # Filter out indices with NaN values
    valid_mask = ~np.isnan(signal[target_indices]) if len(target_indices) > 0 else np.array([], dtype=bool)
    target_indices = target_indices[valid_mask]
    
    if len(target_indices) == 0:
        return corruptor.df
    
    # Ensure column is float to avoid dtype incompatibility (int64 += float64)
    if not np.issubdtype(corruptor.df[corruptor.value_col].dtype, np.floating):
        corruptor.df[corruptor.value_col] = corruptor.df[corruptor.value_col].astype(float)
    
    # 6. Add noise only to target indices
    corruptor.df.loc[target_indices, corruptor.value_col] += noise[target_indices]
    
    corruptor.record_corruption('white_noise_snr', target_indices, {'snr_db': snr_db, 'noise_power': noise_power})
    
    return corruptor.df

@register_injector('spikes')
def inject_spikes(corruptor, fraction=0.01, multiplier=5.0, sequential=False, sequence_length=3):
    """
    Injects sudden extreme spikes. 
    If sequential=True, it will create back-to-back spikes simulating sensor hardware failures.
    """
    total_points = int(len(corruptor.df) * fraction)
    if sequential:
        # Divide by sequence_length so total corrupted points match fraction
        count = max(1, total_points // sequence_length)
    else:
        count = total_points
    target_indices = corruptor.get_injectable_indices(count)

    if len(target_indices) == 0:
        return corruptor.df

    if sequential:
        expanded_indices = set()
        for idx in target_indices:
            for i in range(sequence_length):
                if idx + i < len(corruptor.df):
                    expanded_indices.add(idx + i)
        target_indices = np.array(list(expanded_indices))
        
    signs = corruptor.rng.choice([-1, 1], size=len(target_indices))
    corruptor.df.loc[target_indices, corruptor.value_col] += signs * multiplier * corruptor.std
    
    params = {'fraction': fraction, 'multiplier': multiplier, 'sequential': sequential, 'seq_len': sequence_length}
    corruptor.record_corruption('spikes', target_indices, params)
    return corruptor.df

@register_injector('drift')
def inject_drift(corruptor, drift_factor=0.5):
    """
    Injects a steady linear concept drift across the requested target areas.
    """
    n = len(corruptor.df)
    # The drift increases linearly across the entire dataset timeline
    drift_line = np.linspace(0, drift_factor * corruptor.std, n)
    target_indices = np.array(corruptor.get_target_indices())
    
    if len(target_indices) > 0:
        # Patch to avoid pd.Index list lookup issues
        corruptor.df.loc[target_indices, corruptor.value_col] += drift_line[target_indices]
        corruptor.record_corruption('drift', target_indices, {'drift_factor': drift_factor})
        
    return corruptor.df

@register_injector('burst_missing')
def inject_burst_missing(corruptor, num_bursts=3, burst_length=100):
    """
    Simulates packet-loss by dropping chunk arrays of continuous data.
    """
    target_indices = corruptor.get_injectable_indices(num_bursts)
    if len(target_indices) == 0:
        return corruptor.df
        
    expanded_indices = set()
    for start_idx in target_indices:
        for i in range(burst_length):
            if start_idx + i < len(corruptor.df):
                expanded_indices.add(start_idx + i)
                
    burst_array = np.array(list(expanded_indices))
    corruptor.df.loc[burst_array, corruptor.value_col] = np.nan
    corruptor.record_corruption('burst_missing', burst_array, {'num_bursts': num_bursts, 'burst_length': burst_length})
    return corruptor.df

@register_injector('sensor_stuck')
def inject_sensor_stuck(corruptor, num_stucks=2, stuck_length=200):
    """
    Simulates a frozen sensor hardware failure. Captures the value at the start index
    and freezes the signal to that constant value for block length.
    """
    target_indices = corruptor.get_injectable_indices(num_stucks)
    if len(target_indices) == 0:
        return corruptor.df
        
    expanded_indices = set()
    for start_idx in target_indices:
        stuck_value = corruptor.df.loc[start_idx, corruptor.value_col]
        
        for i in range(stuck_length):
            idx = start_idx + i
            if idx < len(corruptor.df):
                corruptor.df.loc[idx, corruptor.value_col] = stuck_value
                expanded_indices.add(idx)
                
    stuck_array = np.array(list(expanded_indices))
    corruptor.record_corruption('sensor_stuck', stuck_array, {'num_stucks': num_stucks, 'stuck_length': stuck_length})
    return corruptor.df

@register_injector('swap')
def inject_swap(corruptor, fraction=0.05, swap_length=1, max_distance=None):
    """
    Swaps segments of the time series efficiently using NumPy arrays.

    Known issue, kept as is so the TSB-UAD results stay reproducible: once swap_length > 1
    this under-delivers the requested fraction (out-of-bounds pairs are dropped silently and
    later segments can overlap earlier ones), sometimes swapping nothing at all. For an exact
    fraction use ts_corruptor.swap.swap_segments_fast; see that module's docstring.
    """
    n = len(corruptor.df)
    target_indices = np.array(corruptor.get_target_indices())
    
    total_points_to_swap = int(n * fraction)
    num_swaps = total_points_to_swap // (2 * swap_length)
    
    if len(target_indices) < 2 * swap_length or num_swaps == 0:
        return corruptor.df
        
    # Extract data as a fast NumPy array
    values = corruptor.df[corruptor.value_col].to_numpy()
    swapped_indices = []
    
    # We still need a loop, but we do it on raw NumPy arrays which is 100x faster than Pandas .loc
    for _ in range(num_swaps):
        # Pick first random start index
        idx1 = corruptor.rng.choice(target_indices)
        
        # Determine valid range for idx2 based on max_distance
        if max_distance is not None:
            min_idx2 = max(0, idx1 - max_distance)
            max_idx2 = min(n - 1, idx1 + max_distance)
            # Masking in numpy is fast
            mask = (target_indices >= min_idx2) & (target_indices <= max_idx2) & (np.abs(target_indices - idx1) >= swap_length)
            valid_targets_for_idx2 = target_indices[mask]
        else:
            mask = np.abs(target_indices - idx1) >= swap_length
            valid_targets_for_idx2 = target_indices[mask]
            
        if len(valid_targets_for_idx2) == 0:
            continue
            
        idx2 = corruptor.rng.choice(valid_targets_for_idx2)
        
        # Ensure segments are within bounds
        if (idx1 + swap_length <= n) and (idx2 + swap_length <= n):
            # Perform the swap on the NumPy array
            val1 = values[idx1 : idx1 + swap_length].copy()
            val2 = values[idx2 : idx2 + swap_length].copy()
            
            values[idx1 : idx1 + swap_length] = val2
            values[idx2 : idx2 + swap_length] = val1
            
            # Record affected indices
            used1 = set(range(idx1, idx1 + swap_length))
            used2 = set(range(idx2, idx2 + swap_length))
            swapped_indices.extend(used1)
            swapped_indices.extend(used2)
            
            # Remove used indices to prevent duplicate swaps
            used_all = used1 | used2
            target_indices = target_indices[~np.isin(target_indices, list(used_all))]
            
    # Write back to pandas only ONCE at the end
    corruptor.df[corruptor.value_col] = values
            
    if swapped_indices:
        unique_swapped = np.unique(swapped_indices)
        actual_fraction = len(unique_swapped) / n
        actual_num_swaps = len(unique_swapped) // (2 * swap_length)
        corruptor.record_corruption('swap', unique_swapped, {
            'fraction_requested': fraction,
            'fraction_actual': round(actual_fraction, 6),
            'num_swaps_requested': num_swaps,
            'num_swaps_actual': actual_num_swaps,
            'swap_length': swap_length,
            'max_distance': max_distance
        })
    
    return corruptor.df

# ---------------------------------------------------------
# Gilbert-Elliott (Markovian Bursty) Injector
# ---------------------------------------------------------

@register_injector('gilbert_elliott')
def inject_gilbert_elliott(corruptor, p_good_to_bad=0.05, p_bad_to_good=0.2, 
                           noise_type='missing', noise_intensity=3.0):
    """
    Injects realistic bursty noise using a 2-state Markov chain (Gilbert-Elliott model).
    
    The model alternates between Good (no corruption) and Bad (corruption applied) states.
    Transition probabilities control burst frequency and duration:
      - p_good_to_bad (α): probability of entering a burst. Higher = more frequent bursts.
      - p_bad_to_good (β): probability of leaving a burst. Lower = longer bursts.
    
    Expected corruption rate ≈ α / (α + β)
    Expected avg burst length ≈ 1 / β
    
    Args:
        corruptor: TSCorruptor instance
        p_good_to_bad: Transition probability Good → Bad (α)
        p_bad_to_good: Transition probability Bad → Good (β)
        noise_type: Type of corruption in Bad state ('missing', 'gaussian', 'stuck')
        noise_intensity: Multiplier for gaussian noise (× std of signal)
    """
    n = len(corruptor.df)
    target_indices = set(corruptor.get_target_indices())
    
    # Simulate Markov chain
    state = 0  # 0 = Good, 1 = Bad
    bad_indices = []
    
    for i in range(n):
        if i not in target_indices:
            continue
            
        if state == 0:  # Good state
            if corruptor.rng.random() < p_good_to_bad:
                state = 1
        else:  # Bad state
            if corruptor.rng.random() < p_bad_to_good:
                state = 0
        
        if state == 1:
            bad_indices.append(i)
    
    if len(bad_indices) == 0:
        return corruptor.df
    
    bad_indices = np.array(bad_indices)
    
    # Apply corruption based on noise_type
    if noise_type == 'missing':
        corruptor.df.loc[bad_indices, corruptor.value_col] = np.nan
        
    elif noise_type == 'gaussian':
        noise = corruptor.rng.normal(0, noise_intensity * corruptor.std, size=len(bad_indices))
        corruptor.df.loc[bad_indices, corruptor.value_col] += noise
        
    elif noise_type == 'stuck':
        # Freeze at the last good value before each burst
        i = 0
        while i < len(bad_indices):
            burst_start = bad_indices[i]
            # Find the value just before the burst
            if burst_start > 0:
                stuck_value = corruptor.df_original.loc[burst_start - 1, corruptor.value_col]
            else:
                stuck_value = corruptor.df_original.loc[0, corruptor.value_col]
            
            # Apply stuck value to the entire burst
            j = i
            while j < len(bad_indices) and (j == i or bad_indices[j] == bad_indices[j-1] + 1):
                corruptor.df.loc[bad_indices[j], corruptor.value_col] = stuck_value
                j += 1
            i = j
    
    corruptor.record_corruption('gilbert_elliott', bad_indices, {
        'p_good_to_bad': p_good_to_bad,
        'p_bad_to_good': p_bad_to_good,
        'noise_type': noise_type,
        'noise_intensity': noise_intensity if noise_type == 'gaussian' else None,
        'actual_corruption_rate': len(bad_indices) / n,
        'expected_corruption_rate': p_good_to_bad / (p_good_to_bad + p_bad_to_good),
        'expected_avg_burst_length': 1.0 / p_bad_to_good
    })
    
    return corruptor.df

# ---------------------------------------------------------
# Adversarial Injectors (Black-Box Surrogate Approach)
# ---------------------------------------------------------

class SurrogateAE(nn.Module):
    def __init__(self, window_size):
        super(SurrogateAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, window_size // 2),
            nn.ReLU(),
            nn.Linear(window_size // 2, max(2, window_size // 4)),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(max(2, window_size // 4), window_size // 2),
            nn.ReLU(),
            nn.Linear(window_size // 2, window_size)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def _train_surrogate(T_normal, window_size, epochs=20):
    model = SurrogateAE(window_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    # Ensure window_size isn't larger than total length
    if len(T_normal) < window_size:
        return model # Too short to train
    windows = T_normal.unfold(0, window_size, 1)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(windows), windows)
        loss.backward()
        optimizer.step()
    return model

def _apply_smoothing(perturbation, smooth_window):
    if smooth_window <= 1: return perturbation
    kernel = torch.ones(1, 1, smooth_window) / smooth_window
    p_reshaped = perturbation.view(1, 1, -1)
    pad_left = smooth_window // 2
    pad_right = smooth_window - 1 - pad_left
    p_padded = torch.nn.functional.pad(p_reshaped, (pad_left, pad_right), mode='replicate')
    return torch.nn.functional.conv1d(p_padded, kernel).view(-1)

def _generate_adversarial_noise(corruptor, epsilon_factor=0.2, alpha_factor=0.02, iterations=20, smooth_window=1):
    original_data = corruptor.df_original[corruptor.value_col].to_numpy()
    original_data = np.nan_to_num(original_data) # Clean NaNs safely
    
    target_indices = corruptor.get_injectable_indices(len(corruptor.df))
    
    T_tensor = torch.tensor(original_data, dtype=torch.float32)
    window_size = corruptor.window_size if hasattr(corruptor, 'window_size') else 20
    
    if len(T_tensor) < window_size:
        return target_indices, np.zeros_like(original_data)
        
    # Fast train surrogate on original data
    model = _train_surrogate(T_tensor, window_size, epochs=20)
    model.eval()
    
    std = corruptor.std
    epsilon = epsilon_factor * std
    alpha = alpha_factor * std
    
    T_adv = T_tensor.clone().detach().requires_grad_(True)
    
    for _ in range(iterations):
        windows = T_adv.unfold(0, window_size, 1)
        reconstruction = model(windows)
        loss_per_window = torch.mean((reconstruction - windows) ** 2, dim=1)
        
        total_loss = 0
        if len(target_indices) > 0:
            # Vectorized loss computation instead of a slow python loop
            # Create a mask of the same size as loss_per_window
            num_windows = len(loss_per_window)
            window_mask = torch.zeros(num_windows, dtype=torch.bool, device=T_adv.device)
            
            # Find which windows contain the target indices
            # A window starting at 'w' contains indices 'w' to 'w + window_size - 1'
            # So window 'w' contains target 't' if: w <= t < w + window_size
            # => w <= t  AND  w > t - window_size
            # For each target t, we set the mask to true for windows in [t - window_size + 1, t]
            
            for t in target_indices:
                start_w = max(0, t - window_size + 1)
                end_w = min(num_windows, t + 1)
                if start_w < end_w:
                    window_mask[start_w:end_w] = True
                    
            total_loss = torch.sum(loss_per_window[window_mask])
        else:
            total_loss = torch.sum(loss_per_window)
            
        if isinstance(total_loss, int) and total_loss == 0 or total_loss.item() == 0:
            break
            
        total_loss.backward()
        gradients = T_adv.grad
        
        with torch.no_grad():
            perturbation = gradients.sign()
            if smooth_window > 1:
                perturbation = _apply_smoothing(perturbation, smooth_window)

            # Gradient descent: subtract to MINIMIZE anomaly score (hide anomalies)
            T_adv_update = T_adv - alpha * perturbation
            eta = torch.clamp(T_adv_update - T_tensor, min=-epsilon, max=epsilon)
            T_adv = (T_tensor + eta).detach().requires_grad_(True)
            
    final_noise = (T_adv.detach() - T_tensor).numpy()
    return target_indices, final_noise

@register_injector('adversarial_bim')
def inject_adversarial_bim(corruptor, epsilon_factor=0.2, iterations=20):
    """
    Injects visible, spiky adversarial perturbations (Basic Iterative Method)
    by attacking a dynamically trained surrogate autoencoder.
    """
    alpha_factor = epsilon_factor / max(1.0, (iterations / 2.0))
    target_indices, final_noise = _generate_adversarial_noise(
        corruptor, epsilon_factor, alpha_factor, iterations, smooth_window=1
    )
    
    if len(target_indices) > 0:
        corruptor.df.loc[target_indices, corruptor.value_col] += final_noise[target_indices]
        corruptor.record_corruption('adversarial_bim', target_indices, 
                                    {'epsilon_factor': epsilon_factor, 'iterations': iterations})
    return corruptor.df

@register_injector('adversarial_smooth')
def inject_adversarial_smooth(corruptor, epsilon_factor=0.2, iterations=20, smooth_window=10):
    """
    Injects stealthy, smooth adversarial perturbations (Pialla et al.)
    by attacking a dynamically trained surrogate autoencoder.
    """
    alpha_factor = epsilon_factor / max(1.0, (iterations / 2.0))
    target_indices, final_noise = _generate_adversarial_noise(
        corruptor, epsilon_factor, alpha_factor, iterations, smooth_window=smooth_window
    )
    
    if len(target_indices) > 0:
        corruptor.df.loc[target_indices, corruptor.value_col] += final_noise[target_indices]
        corruptor.record_corruption('adversarial_smooth', target_indices, 
                                    {'epsilon_factor': epsilon_factor, 'iterations': iterations, 'smooth_window': smooth_window})
    return corruptor.df
