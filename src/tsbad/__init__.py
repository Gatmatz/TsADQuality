"""Shared evaluation layer for the TSB-AD corruption experiments.

`ts_corruptor` decides WHAT is done to a series; `tsbad` decides HOW the corrupted series
is evaluated: official TSB-AD models and metrics, per-model checkpoints, the process pool
and the summary. Every runner in src/experiments/corruption_tsbad/ is a config on top of
`tsbad.harness.run_sweep` (propagation uses the building blocks directly).

Import `tsbad.env` before numpy — see that module.
"""
