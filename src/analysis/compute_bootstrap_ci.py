"""
Compute bootstrap confidence intervals and significance for corruption impact.

This script is post-processing only: it does NOT rerun anomaly detection models.
It reads existing experiment checkpoints, builds per-file delta AUC:

    delta_auc = auc_corrupted - auc_baseline

Then for each condition/model it computes:
  - mean delta AUC
  - bootstrap 95% CI (mean delta)
  - one-sample Wilcoxon test against 0 delta
  - Holm-corrected p-values per model

Outputs:
  - results\\analysis\\ci_table.csv
  - results\\analysis\\pvalues.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import zlib

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis_common import PROJECT_ROOT, RESULTS_DIR, build_delta_dataset, holm_adjust


DEFAULT_EXPERIMENTS = [
    "white_noise_snr",
    "missing_true_impact",
    "freeze",
    "spikes",
    "compound_corruptions",
]


@dataclass
class BootstrapConfig:
    n_bootstrap: int = 2000
    random_seed: int = 42
    ci_alpha: float = 0.05


def bootstrap_mean_ci(values: np.ndarray, cfg: BootstrapConfig) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        v = float(values[0])
        return v, v, v

    rng = np.random.default_rng(cfg.random_seed)
    indices = rng.integers(0, n, size=(cfg.n_bootstrap, n))
    samples = values[indices]
    means = samples.mean(axis=1)

    alpha = cfg.ci_alpha
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    mean = float(values.mean())
    return mean, low, high


def wilcoxon_p(delta_values: np.ndarray) -> float:
    x = np.asarray(delta_values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    if np.allclose(x, 0.0):
        return 1.0
    try:
        return float(wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def compute_condition_stats(df_delta: pd.DataFrame, cfg: BootstrapConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "experiment",
        "corruption_family",
        "model",
        "condition",
        "frontier_group",
        "axis_name",
        "axis_value",
        "severity_score",
    ]

    rows = []
    for keys, group in df_delta.groupby(group_cols, dropna=False):
        (experiment, family, model, condition, frontier_group, axis_name, axis_value, severity_score) = keys
        delta = group["delta_auc"].to_numpy(dtype=float)
        # Deterministic but condition-specific resampling stream.
        seed_text = f"{experiment}|{model}|{condition}"
        condition_seed = int(zlib.crc32(seed_text.encode("utf-8")) % (2**31 - 1))
        local_cfg = BootstrapConfig(
            n_bootstrap=cfg.n_bootstrap,
            random_seed=cfg.random_seed + condition_seed,
            ci_alpha=cfg.ci_alpha,
        )
        mean_delta, ci_low, ci_high = bootstrap_mean_ci(delta, local_cfg)
        p_raw = wilcoxon_p(delta)

        rows.append(
            {
                "experiment": experiment,
                "corruption_family": family,
                "model": model,
                "condition": condition,
                "frontier_group": frontier_group,
                "axis_name": axis_name,
                "axis_value": axis_value,
                "severity_score": severity_score,
                "n_files": int(group["file"].nunique()),
                "mean_auc_corrupted": float(group["auc_corrupted"].mean()),
                "mean_auc_baseline": float(group["baseline_auc"].mean()),
                "mean_delta_auc": mean_delta,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "wilcoxon_p": p_raw,
            }
        )

    ci_table = pd.DataFrame(rows)
    if ci_table.empty:
        return ci_table, ci_table

    pvalue_rows = []
    for model, sub in ci_table.groupby("model", dropna=False):
        p = sub["wilcoxon_p"].to_numpy(dtype=float)
        mask = np.isfinite(p)
        adjusted = np.full_like(p, np.nan, dtype=float)
        if mask.any():
            adjusted_vals = holm_adjust(p[mask])
            adjusted[mask] = adjusted_vals

        local = sub.copy()
        local["holm_p"] = adjusted
        local["significant_0_05"] = local["holm_p"] < 0.05
        local["direction"] = np.where(local["mean_delta_auc"] < 0, "degradation", "improvement")
        pvalue_rows.append(local)

    pvalues = pd.concat(pvalue_rows, ignore_index=True)
    pvalues = pvalues.sort_values(
        ["model", "experiment", "corruption_family", "condition"],
        ascending=[True, True, True, True],
    )

    merged = ci_table.merge(
        pvalues[["model", "experiment", "condition", "holm_p", "significant_0_05", "direction"]],
        on=["model", "experiment", "condition"],
        how="left",
    )
    merged = merged.sort_values(
        ["model", "experiment", "corruption_family", "condition"],
        ascending=[True, True, True, True],
    )

    return merged, pvalues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap CI and significance for delta AUC conditions.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help=f"Experiments to include. Default: {' '.join(DEFAULT_EXPERIMENTS)}",
    )
    parser.add_argument("--bootstrap", type=int, default=2000, help="Number of bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed.")
    parser.add_argument(
        "--quick-files",
        type=int,
        default=0,
        help="If >0, keep only first N files for quick smoke runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR / "analysis"),
        help="Output directory for ci_table.csv and pvalues.csv.",
    )
    return parser.parse_args()


def _normalize_experiment_names(experiments: Iterable[str]) -> list[str]:
    normalized = []
    alias = {
        "noise": "white_noise_snr",
        "white_noise": "white_noise_snr",
        "missing": "missing_true_impact",
        "compound": "compound_corruptions",
    }
    for exp in experiments:
        normalized.append(alias.get(exp, exp))
    # preserve order and uniqueness
    seen = set()
    out = []
    for item in normalized:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def main() -> None:
    args = parse_args()
    experiments = _normalize_experiment_names(args.experiments)
    cfg = BootstrapConfig(n_bootstrap=args.bootstrap, random_seed=args.seed)

    print("=" * 70)
    print("BOOTSTRAP CI + SIGNIFICANCE (delta AUC)")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Experiments:  {experiments}")
    print(f"Bootstrap B: {cfg.n_bootstrap}")
    print(f"Quick files: {args.quick_files if args.quick_files > 0 else 'disabled'}")
    print("=" * 70)

    df_delta = build_delta_dataset(include_experiments=experiments, quick_files=args.quick_files)
    if df_delta.empty:
        print("[ERROR] No usable rows loaded from selected experiments.")
        return

    print(f"[INFO] Delta dataset rows: {len(df_delta)}")
    print(f"[INFO] Files: {df_delta['file'].nunique()} | Models: {df_delta['model'].nunique()}")

    ci_table, pvalues = compute_condition_stats(df_delta, cfg)
    if ci_table.empty:
        print("[ERROR] No grouped condition rows produced.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ci_path = output_dir / "ci_table.csv"
    p_path = output_dir / "pvalues.csv"

    ci_table.to_csv(ci_path, index=False)
    pvalues.to_csv(p_path, index=False)

    print(f"[DONE] Saved CI table: {ci_path}")
    print(f"[DONE] Saved p-values: {p_path}")

    # Short CLI summary: most degraded and most improved conditions by model.
    summary_cols = ["model", "condition", "mean_delta_auc", "ci95_low", "ci95_high", "holm_p"]
    for model, sub in ci_table.groupby("model", dropna=False):
        s = sub.sort_values("mean_delta_auc")
        worst = s.head(1)[summary_cols]
        best = s.tail(1)[summary_cols]
        print(f"\n[MODEL] {model}")
        print("  Worst condition:")
        print("   ", worst.to_dict("records")[0])
        print("  Best condition:")
        print("   ", best.to_dict("records")[0])


if __name__ == "__main__":
    main()
