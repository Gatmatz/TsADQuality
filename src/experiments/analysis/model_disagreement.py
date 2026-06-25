"""
Model Disagreement as an Unsupervised Corruption Indicator
==========================================================
Hypothesis: As data corruption increases, anomaly detection models
disagree MORE with each other (measured via metric-level divergence).

This analysis uses existing checkpoint CSVs — no re-running needed.

Metrics computed per (file × condition):
  - std(AUC_ROC) across models
  - range(AUC_ROC) = max - min across models
  - coefficient of variation = std / mean
  - Kendall's tau of model ranking vs clean baseline
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import kendalltau, spearmanr
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = PROJECT_ROOT / "results" / "experiments"
BASELINE_CSV = PROJECT_ROOT / "results" / "tables" / "baseline_final_subset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "model_disagreement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Models present in both baseline and experiments
MODELS = ["IForest", "LOF", "MP", "AE"]
METRIC = "AUC_ROC"

# ── experiment definitions ─────────────────────────────────────────
# Each entry: (dir_name, severity_column, condition_column, description)
EXPERIMENTS = [
    ("white_noise_snr", "snr_db", "condition",
     "White Noise (SNR)"),
    ("freeze", "fraction", "condition",
     "Sensor Freeze"),
    ("spikes_normal_only", "fraction", "condition",
     "Spikes (normal region)"),
    ("swap_segment", "fraction", "condition",
     "Segment Swap"),
    ("missing_true_impact", "fraction", "condition",
     "Missing Data (true impact)"),
]


def load_baseline():
    """Load baseline results, harmonize model names."""
    df = pd.read_csv(BASELINE_CSV)
    # Harmonize: baseline uses 'Autoencoder', experiments use 'AE'
    df["model"] = df["model"].replace({"Autoencoder": "AE"})
    # Keep only our 4 models
    df = df[df["model"].isin(MODELS)].copy()
    # Rename F1 to F if needed
    if "F1" in df.columns and "F" not in df.columns:
        df = df.rename(columns={"F1": "F"})
    return df


def load_experiment(exp_name):
    """Load experiment checkpoint, keep valid rows with our 4 models."""
    path = EXPERIMENTS_DIR / exp_name / "checkpoint.csv"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None
    df = pd.read_csv(path)
    # Keep only our models and non-error rows
    df = df[df["model"].isin(MODELS)].copy()
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")].copy()
    # Ensure metric column is numeric
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
    df = df.dropna(subset=[METRIC])
    return df


def compute_disagreement(group):
    """Compute disagreement metrics for a group of models on one file×condition."""
    values = group[METRIC].values
    if len(values) < 2:
        return pd.Series({
            "n_models": len(values),
            "auc_std": np.nan,
            "auc_range": np.nan,
            "auc_cv": np.nan,
            "auc_mean": np.nan,
        })
    return pd.Series({
        "n_models": len(values),
        "auc_std": np.std(values, ddof=1),
        "auc_range": np.max(values) - np.min(values),
        "auc_cv": np.std(values, ddof=1) / np.mean(values) if np.mean(values) > 0 else np.nan,
        "auc_mean": np.mean(values),
    })


def compute_rank_correlation(baseline_df, experiment_df, condition_col):
    """
    For each condition, compute Kendall's tau between the model ranking
    on clean data vs corrupted data (averaged across files).
    """
    # Baseline ranking per file: pivot to file × model
    bl_pivot = baseline_df.pivot_table(
        index="file", columns="model", values=METRIC, aggfunc="first"
    )

    results = []
    for cond, grp in experiment_df.groupby(condition_col):
        exp_pivot = grp.pivot_table(
            index="file", columns="model", values=METRIC, aggfunc="first"
        )
        # Keep only common files and models
        common_files = bl_pivot.index.intersection(exp_pivot.index)
        common_models = bl_pivot.columns.intersection(exp_pivot.columns)
        if len(common_files) < 5 or len(common_models) < 3:
            continue

        taus = []
        for f in common_files:
            bl_ranks = bl_pivot.loc[f, common_models].rank()
            exp_ranks = exp_pivot.loc[f, common_models].rank()
            if bl_ranks.isna().any() or exp_ranks.isna().any():
                continue
            tau, _ = kendalltau(bl_ranks.values, exp_ranks.values)
            taus.append(tau)

        if taus:
            results.append({
                "condition": cond,
                "kendall_tau_mean": np.mean(taus),
                "kendall_tau_std": np.std(taus),
                "kendall_tau_median": np.median(taus),
                "n_files": len(taus),
                "pct_inversions": np.mean([1 for t in taus if t < 0]) * 100,
            })

    return pd.DataFrame(results)


def analyze_experiment(exp_name, severity_col, condition_col, description, baseline_df):
    """Full disagreement analysis for one experiment."""
    print(f"\n{'='*60}")
    print(f"  {description} ({exp_name})")
    print(f"{'='*60}")

    df = load_experiment(exp_name)
    if df is None or df.empty:
        return None

    print(f"  Loaded {len(df)} rows, {df['file'].nunique()} files, "
          f"models: {sorted(df['model'].unique())}")

    # ── 1. Per-file disagreement ──
    # Only keep files that have all 4 models
    file_cond_counts = df.groupby(["file", condition_col])["model"].nunique()
    valid = file_cond_counts[file_cond_counts >= 3].reset_index()[["file", condition_col]]
    df_valid = df.merge(valid, on=["file", condition_col])

    disagreement = df_valid.groupby(["file", condition_col]).apply(
        compute_disagreement, include_groups=False
    ).reset_index()

    # ── 2. Baseline disagreement ──
    bl_files = baseline_df[baseline_df["file"].isin(df_valid["file"].unique())]
    bl_disagree = bl_files.groupby("file").apply(
        compute_disagreement, include_groups=False
    ).reset_index()
    bl_disagree[condition_col] = "baseline_clean"

    # ── 3. Aggregate: mean disagreement per condition ──
    agg = disagreement.groupby(condition_col).agg(
        auc_std_mean=("auc_std", "mean"),
        auc_std_median=("auc_std", "median"),
        auc_range_mean=("auc_range", "mean"),
        auc_cv_mean=("auc_cv", "mean"),
        auc_mean_mean=("auc_mean", "mean"),
        n_files=("auc_std", "count"),
    ).reset_index()

    # Add baseline row
    bl_agg = pd.DataFrame([{
        condition_col: "baseline_clean",
        "auc_std_mean": bl_disagree["auc_std"].mean(),
        "auc_std_median": bl_disagree["auc_std"].median(),
        "auc_range_mean": bl_disagree["auc_range"].mean(),
        "auc_cv_mean": bl_disagree["auc_cv"].mean(),
        "auc_mean_mean": bl_disagree["auc_mean"].mean(),
        "n_files": len(bl_disagree),
    }])
    agg = pd.concat([bl_agg, agg], ignore_index=True)

    print(f"\n  Disagreement summary (std of {METRIC} across models):")
    print(agg.to_string(index=False))

    # ── 4. Rank correlation ──
    rank_df = compute_rank_correlation(baseline_df, df_valid, condition_col)
    if not rank_df.empty:
        print(f"\n  Rank stability (Kendall's tau vs clean baseline):")
        print(rank_df.to_string(index=False))

    # ── 5. Save ──
    exp_out = OUTPUT_DIR / exp_name
    exp_out.mkdir(parents=True, exist_ok=True)
    agg.to_csv(exp_out / "disagreement_summary.csv", index=False)
    disagreement.to_csv(exp_out / "disagreement_per_file.csv", index=False)
    if not rank_df.empty:
        rank_df.to_csv(exp_out / "rank_stability.csv", index=False)

    return {
        "experiment": description,
        "baseline_std": bl_disagree["auc_std"].mean(),
        "max_corrupted_std": agg[agg[condition_col] != "baseline_clean"]["auc_std_mean"].max()
        if len(agg) > 1 else np.nan,
        "disagreement_ratio": (
            agg[agg[condition_col] != "baseline_clean"]["auc_std_mean"].max()
            / bl_disagree["auc_std"].mean()
        ) if bl_disagree["auc_std"].mean() > 0 else np.nan,
    }


def create_summary_table(all_results):
    """Cross-experiment summary: which corruptions cause most disagreement?"""
    summary = pd.DataFrame(all_results)
    summary = summary.sort_values("disagreement_ratio", ascending=False)

    print(f"\n{'='*60}")
    print("  CROSS-EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Disagreement ratio = max_corrupted_std / baseline_std")
    print(f"  >1 means corruption INCREASES disagreement\n")
    print(summary.to_string(index=False))

    summary.to_csv(OUTPUT_DIR / "cross_experiment_summary.csv", index=False)
    return summary


def main():
    print("Model Disagreement Analysis")
    print(f"Output: {OUTPUT_DIR}\n")

    # Load baseline
    baseline_df = load_baseline()
    print(f"Baseline: {len(baseline_df)} rows, {baseline_df['file'].nunique()} files, "
          f"models: {sorted(baseline_df['model'].unique())}")

    # Analyze each experiment
    all_results = []
    for exp_name, severity_col, condition_col, description in EXPERIMENTS:
        result = analyze_experiment(
            exp_name, severity_col, condition_col, description, baseline_df
        )
        if result:
            all_results.append(result)

    # Cross-experiment summary
    if all_results:
        create_summary_table(all_results)

    print(f"\nDone. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
