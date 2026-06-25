"""
Error Analysis for corruption experiments (white noise, spikes, missing).

Keeps the same experiment style as existing analysis scripts:
  - reads `checkpoint.csv`
  - pairs each corrupted run with per-file baseline
  - writes CSV outputs under `<experiment>/error_analysis/`

Outputs per experiment:
  - paired_file_metrics.csv
  - condition_summary.csv
  - critical_severity.csv
  - worst_cases.csv

Usage:
  python error_analysis_corruptions.py
  python error_analysis_corruptions.py --experiments white_noise spikes missing
  python error_analysis_corruptions.py --auc-threshold 0.70 --top-k 30
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_CSV = PROJECT_ROOT / "results" / "tables" / "baseline_final_subset.csv"

METRICS = ["AUC_ROC", "AUC_PR", "Precision", "Recall", "F"]
MODEL_RENAME = {"Autoencoder": "AE"}

EXPERIMENTS = {
    "white_noise": {
        "label": "White Noise (SNR)",
        "results_dir": PROJECT_ROOT / "results" / "experiments" / "white_noise_snr",
        "checkpoint": "checkpoint.csv",
        "condition_keys": ["snr_db", "condition"],
        "critical_group_keys": ["model"],
    },
    "spikes": {
        "label": "Spikes (normal-only)",
        "results_dir": PROJECT_ROOT / "results" / "experiments" / "spikes_normal_only",
        "checkpoint": "checkpoint.csv",
        "condition_keys": ["fraction", "multiplier", "condition"],
        "critical_group_keys": ["model"],
    },
    "missing": {
        "label": "Missing (true impact)",
        "results_dir": PROJECT_ROOT / "results" / "experiments" / "missing_true_impact",
        "checkpoint": "checkpoint.csv",
        "condition_keys": ["missing_type", "fraction", "num_bursts", "condition"],
        "critical_group_keys": ["model", "missing_type"],
    },
}


def _normalize_model_names(df: pd.DataFrame) -> pd.DataFrame:
    if "model" in df.columns:
        df["model"] = df["model"].replace(MODEL_RENAME)
    return df


def _load_baseline() -> pd.DataFrame:
    bl = pd.read_csv(BASELINE_CSV)
    bl = _normalize_model_names(bl)
    if "F1" in bl.columns and "F" not in bl.columns:
        bl = bl.rename(columns={"F1": "F"})

    keep_cols = ["file", "model"] + [m for m in METRICS if m in bl.columns]
    bl = bl[keep_cols].copy()
    for m in METRICS:
        if m in bl.columns:
            bl[m] = pd.to_numeric(bl[m], errors="coerce")
    bl = bl.dropna(subset=["file", "model"])
    bl = bl.groupby(["file", "model"], dropna=False)[[m for m in METRICS if m in bl.columns]].mean().reset_index()
    return bl


def _format_condition(row: pd.Series, condition_keys: list[str]) -> str:
    parts = []
    for k in condition_keys:
        if k not in row.index:
            continue
        v = row[k]
        if pd.isna(v):
            continue
        if isinstance(v, float) and float(v).is_integer():
            v = int(v)
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def _add_severity_score(df: pd.DataFrame, experiment_key: str) -> pd.DataFrame:
    df = df.copy()
    if experiment_key == "white_noise":
        df["severity_score"] = -pd.to_numeric(df["snr_db"], errors="coerce")
    elif experiment_key == "spikes":
        frac = pd.to_numeric(df["fraction"], errors="coerce")
        mult = pd.to_numeric(df["multiplier"], errors="coerce")
        df["severity_score"] = frac * mult
    elif experiment_key == "missing":
        frac = pd.to_numeric(df["fraction"], errors="coerce")
        nb = pd.to_numeric(df.get("num_bursts", 0), errors="coerce").fillna(0)
        # Primary order by fraction, tiny tie-break by num_bursts
        df["severity_score"] = frac + (nb / 1000.0)
    else:
        df["severity_score"] = np.nan
    return df


def _load_experiment_rows(exp_cfg: dict) -> pd.DataFrame:
    checkpoint_path = exp_cfg["results_dir"] / exp_cfg["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    df = pd.read_csv(checkpoint_path)
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    df = _normalize_model_names(df)

    for col in ["snr_db", "fraction", "multiplier", "num_bursts", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for m in METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    condition_keys = [k for k in exp_cfg["condition_keys"] if k in df.columns]
    group_keys = condition_keys + ["file", "model"]
    metric_cols = [m for m in METRICS if m in df.columns]
    if not metric_cols:
        raise ValueError("No supported metrics found in checkpoint.")

    # Average over seeds/repeats to keep one paired value per file/condition/model
    df_cond = df.groupby(group_keys, dropna=False)[metric_cols].mean().reset_index()
    return df_cond


def _build_paired(df_cond: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    paired = df_cond.merge(
        baseline,
        on=["file", "model"],
        how="inner",
        suffixes=("_corr", "_base"),
    )
    for m in METRICS:
        corr_col = f"{m}_corr"
        base_col = f"{m}_base"
        if corr_col in paired.columns and base_col in paired.columns:
            paired[f"drop_{m}"] = paired[base_col] - paired[corr_col]
    return paired


def _condition_summary(paired: pd.DataFrame, condition_keys: list[str]) -> pd.DataFrame:
    group_keys = [k for k in condition_keys if k in paired.columns] + ["model"]
    aggs = {"n_files": ("file", "nunique")}

    for m in METRICS:
        corr_col = f"{m}_corr"
        base_col = f"{m}_base"
        drop_col = f"drop_{m}"
        if corr_col in paired.columns:
            aggs[f"mean_corrupted_{m}"] = (corr_col, "mean")
        if base_col in paired.columns:
            aggs[f"mean_baseline_{m}"] = (base_col, "mean")
        if drop_col in paired.columns:
            aggs[f"mean_drop_{m}"] = (drop_col, "mean")
            aggs[f"median_drop_{m}"] = (drop_col, "median")

    summary = paired.groupby(group_keys, dropna=False).agg(**aggs).reset_index()
    return summary


def _critical_severity(
    summary: pd.DataFrame,
    condition_keys: list[str],
    critical_group_keys: list[str],
    auc_threshold: float,
) -> pd.DataFrame:
    if "mean_corrupted_AUC_ROC" not in summary.columns:
        return pd.DataFrame()

    rows = []
    for group_vals, sub in summary.groupby(critical_group_keys, dropna=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        group_dict = dict(zip(critical_group_keys, group_vals))
        sub = sub.sort_values("severity_score")
        valid = sub.dropna(subset=["mean_corrupted_AUC_ROC"])
        if valid.empty:
            continue

        best_row = valid.loc[valid["mean_corrupted_AUC_ROC"].idxmax()]
        worst_row = valid.loc[valid["mean_corrupted_AUC_ROC"].idxmin()]

        crossed = valid[valid["mean_corrupted_AUC_ROC"] <= auc_threshold]
        if crossed.empty:
            critical_row = None
        else:
            critical_row = crossed.iloc[0]

        row = {
            **group_dict,
            "auc_threshold": auc_threshold,
            "crosses_threshold": critical_row is not None,
            "best_condition": _format_condition(best_row, condition_keys),
            "best_mean_auc": round(float(best_row["mean_corrupted_AUC_ROC"]), 4),
            "worst_condition": _format_condition(worst_row, condition_keys),
            "worst_mean_auc": round(float(worst_row["mean_corrupted_AUC_ROC"]), 4),
        }

        if critical_row is not None:
            row["critical_condition"] = _format_condition(critical_row, condition_keys)
            row["critical_mean_auc"] = round(float(critical_row["mean_corrupted_AUC_ROC"]), 4)
            row["critical_severity_score"] = round(float(critical_row["severity_score"]), 6)
        else:
            row["critical_condition"] = None
            row["critical_mean_auc"] = None
            row["critical_severity_score"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def _worst_cases(paired: pd.DataFrame, condition_keys: list[str], top_k: int) -> pd.DataFrame:
    if "drop_AUC_ROC" not in paired.columns:
        return pd.DataFrame()

    cols = [k for k in condition_keys if k in paired.columns] + ["file", "model"]
    for m in METRICS:
        for suffix in ["_base", "_corr"]:
            c = f"{m}{suffix}"
            if c in paired.columns:
                cols.append(c)
        d = f"drop_{m}"
        if d in paired.columns:
            cols.append(d)

    worst = paired.sort_values("drop_AUC_ROC", ascending=False)[cols].head(top_k).copy()
    return worst


def _run_single_experiment(
    experiment_key: str,
    baseline: pd.DataFrame,
    auc_threshold: float,
    top_k: int,
) -> None:
    cfg = EXPERIMENTS[experiment_key]
    print("\n" + "=" * 72)
    print(f"  Error Analysis: {cfg['label']}")
    print("=" * 72)

    df_cond = _load_experiment_rows(cfg)
    paired = _build_paired(df_cond, baseline)
    if paired.empty:
        print("No paired rows after baseline alignment. Skipping.")
        return

    summary = _condition_summary(paired, cfg["condition_keys"])
    summary = _add_severity_score(summary, experiment_key)
    summary = summary.sort_values(["model", "severity_score"]).reset_index(drop=True)

    critical = _critical_severity(
        summary,
        cfg["condition_keys"],
        cfg["critical_group_keys"],
        auc_threshold,
    )
    worst = _worst_cases(paired, cfg["condition_keys"], top_k)

    out_dir = cfg["results_dir"] / "error_analysis"
    os.makedirs(out_dir, exist_ok=True)

    paired.to_csv(out_dir / "paired_file_metrics.csv", index=False)
    summary.to_csv(out_dir / "condition_summary.csv", index=False)
    critical.to_csv(out_dir / "critical_severity.csv", index=False)
    worst.to_csv(out_dir / "worst_cases.csv", index=False)

    print(f"Paired rows: {len(paired)}")
    print(f"Summary rows: {len(summary)}")
    print(f"Critical rows: {len(critical)}")
    print(f"Worst cases: {len(worst)}")
    print(f"Saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis for corruption experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        choices=list(EXPERIMENTS.keys()),
        help="Experiments to analyze",
    )
    parser.add_argument(
        "--auc-threshold",
        type=float,
        default=0.70,
        help="AUC threshold for critical severity",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-K worst file-level cases by AUC drop",
    )
    args = parser.parse_args()

    baseline = _load_baseline()
    print(f"Loaded baseline rows: {len(baseline)}")

    for exp_key in args.experiments:
        _run_single_experiment(exp_key, baseline, args.auc_threshold, args.top_k)

    print("\nDone.")


if __name__ == "__main__":
    main()

