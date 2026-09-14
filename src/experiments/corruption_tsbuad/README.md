# Corruption experiments — TSB-UAD generation (thesis)

The corruption runners used for the BSc thesis, evaluated on the 141-series TSB-UAD
subset (`results/tables/final_subset.csv`) with the TSB-UAD metrics.

**Frozen.** These scripts are kept so the thesis numbers stay reproducible. They are not
maintained and are not being moved onto the shared TSB-AD harness. New work goes in
[`../corruption_tsbad/`](../corruption_tsbad/).

Run them exactly as before, from the repository root:

```
python src/experiments/corruption_tsbuad/run_whitenoise_snr.py --test
```

Results go to `results/experiments/<name>/`, unchanged.

## Known issues when reading these numbers

| Script | Issue | Fixed in |
|---|---|---|
| `run_swap.py`, `run_swap_ratio.py`, `run_swap_segment.py` | `ts_corruptor.injectors.inject_swap` under-delivers once `swap_length > 1`: for some series it swaps far fewer points than requested, sometimes none. The segment-length axis is confounded. | `corruption_tsbad/swap_segment.py` (own swap implementation) |
| `run_missing_mnar_burst.py` | `MECHANISMS` has only the two MNAR arms; the `mcar_burst` control is implemented but never run, so there is no random-burst baseline to compare against. | `corruption_tsbad/missing_mnar_burst.py` |

## TSB-AD counterpart of each script

| TSB-UAD script | TSB-AD version |
|---|---|
| `run_whitenoise_snr.py` | `white_noise_snr.py` |
| `run_spikes.py` | `spikes.py` |
| `run_spikes_normal_only.py` | `spikes_normal_only.py` |
| `run_freeze.py` | `freeze.py` |
| `run_swap_point.py` | `swap_point.py` |
| `run_swap_segment.py` | `swap_segment.py` |
| `run_swap_permutation.py` | `swap_permutation.py` |
| `run_missing_true_impact.py` | `missing_true_impact.py` |
| `run_missing_mnar.py` | `missing_mnar.py` |
| `run_missing_mnar_burst.py` | `missing_mnar_burst.py` |
| `run_gilbert_elliott_true_impact.py` | `gilbert_elliott_true_impact.py` |
| `run_compound_corruptions.py` | `compound_corruptions.py` |
| `run_propagation.py` | `propagation.py` |

Covered by a TSB-AD experiment without a separate port:

| TSB-UAD script | Why no port |
|---|---|
| `run_swap.py`, `run_swap_ratio.py` | superseded by `swap_point.py` + `swap_segment.py` (and affected by the swap bug above) |
| `run_freeze_stuck_length.py` | same question as the `NUM_STUCKS` axis of `freeze.py` (length at fixed fraction), parametrised by count instead of ratio |
| `run_missing_burst_length.py` | same, via the `NUM_BURSTS` axis of `missing_true_impact.py` |
| `run_spikes_no_zscore.py` | the official TSB-AD pipeline feeds raw data (no input z-score), so `spikes.py` already is the no-z-score setting |
| `run_propagation_fixed_scaler_multiseed.py` | the scaler confound came from the TSB-UAD `StandardScaler`, absent in TSB-AD; the multi-seed part is not ported |
| `run_combined_noise_missing.py` | largely covered by `noise_missing` in `compound_corruptions.py` (SNR 20/10/5/0 x missing 5/10/20%; SNR 30 and 1% missing are not) |

No TSB-AD version yet:

- `run_noise_position.py` — noise position relative to anomalies
- `run_anomaly_aware_corruption.py` — fixed budget on normal / anomalous / mixed points
- `run_gradual_drift.py` — gradual vs uniform mean / variance drift
- `run_propagation_multiscale.py` — propagation at 5 / 10 / 20 % (the port runs 10 % only)
- `run_missing_masking.py` — evaluation on surviving points only (lost anomalies dropped, not penalised)

`run_cleaning_recovery.py` has no port: recovery/cleaning experiments are not being carried over.
