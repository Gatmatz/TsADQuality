"""
Failure Transferability under Corruption Severity
===================================================
For each corruption condition, identifies which FILES each model fails on,
then computes pairwise Jaccard similarity between models' failure sets.

High Jaccard = shared structural weakness (data is inherently vulnerable)
Low Jaccard  = model-specific weakness (different models break differently)

Uses existing checkpoint CSVs — no new experiments needed.
"""

import re
import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = PROJECT_ROOT / "results" / "experiments"
BASELINE_CSV = PROJECT_ROOT / "results" / "tables" / "baseline_final_subset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "failure_transferability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────
MODELS = ["IForest", "LOF", "MP", "AE", "PCA"]
FN_THRESHOLD = 0.05   # Recall < this => FN-failure
FP_THRESHOLD = 0.10   # Precision < this => FP-failure

EXPERIMENTS = [
    ("white_noise_snr",     "White Noise",
     lambda c: float(re.search(r"snr_([-\d]+)dB", c).group(1)),
     "SNR (dB)", True),
    ("freeze",              "Sensor Freeze",
     lambda c: float(re.search(r"frac_([\d.]+)_ns_(\d+)", c).group(1)),
     "Corruption Fraction", False),
    ("spikes_normal_only",  "Spikes",
     lambda c: float(re.search(r"frac_([\d.]+)_mult", c).group(1)),
     "Corruption Fraction", False),
    ("swap_segment",        "Segment Swap",
     lambda c: float(re.search(r"frac_([\d.]+)_ns", c).group(1)),
     "Corruption Fraction", False),
    ("missing_true_impact", "Missing Data",
     lambda c: float(re.search(r"frac_([\d.]+)", c).group(1)),
     "Corruption Fraction", False),
]

MODEL_PAIRS = list(combinations(MODELS, 2))


def load_baseline():
    df = pd.read_csv(BASELINE_CSV)
    df["model"] = df["model"].replace({"Autoencoder": "AE"})
    df = df[df["model"].isin(MODELS)].copy()
    if "F1" in df.columns and "F" not in df.columns:
        df = df.rename(columns={"F1": "F"})
    return df


def load_experiment(exp_name):
    path = EXPERIMENTS_DIR / exp_name / "checkpoint.csv"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None
    df = pd.read_csv(path)
    df = df[df["model"].isin(MODELS)].copy()
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")].copy()
    for col in ["Precision", "Recall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Precision", "Recall"])
    return df


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return np.nan
    union = set_a | set_b
    if len(union) == 0:
        return np.nan
    return len(set_a & set_b) / len(union)


def build_failure_sets(df, condition_col, condition_val, failure_type="fn"):
    """Build {model: set(failed_files)} for one condition."""
    sub = df[df[condition_col] == condition_val]
    failure_sets = {}
    for model in MODELS:
        model_df = sub[sub["model"] == model]
        if failure_type == "fn":
            failed = set(model_df[model_df["Recall"] < FN_THRESHOLD]["file"])
        else:  # fp
            failed = set(model_df[model_df["Precision"] < FP_THRESHOLD]["file"])
        failure_sets[model] = failed
    return failure_sets


def compute_pairwise_jaccard(failure_sets):
    """Compute Jaccard for all 6 model pairs."""
    rows = []
    for m_a, m_b in MODEL_PAIRS:
        j = jaccard(failure_sets[m_a], failure_sets[m_b])
        rows.append({
            "model_a": m_a, "model_b": m_b,
            "jaccard": j,
            "intersection": len(failure_sets[m_a] & failure_sets[m_b]),
            "union": len(failure_sets[m_a] | failure_sets[m_b]),
            "size_a": len(failure_sets[m_a]),
            "size_b": len(failure_sets[m_b]),
        })
    return rows


def analyze_experiment(exp_name, display_name, severity_extractor,
                       severity_label, invert_x, baseline_df):
    print(f"\n{'='*60}")
    print(f"  {display_name} ({exp_name})")
    print(f"{'='*60}")

    df = load_experiment(exp_name)
    if df is None or df.empty:
        return None

    conditions = sorted(df["condition"].unique())
    print(f"  {len(df)} rows, {df['file'].nunique()} files, "
          f"{len(conditions)} conditions")

    all_jaccard = []
    shared_failures = []

    for cond in conditions:
        try:
            severity = severity_extractor(cond)
        except Exception:
            continue

        for ftype in ["fn", "fp"]:
            fsets = build_failure_sets(df, "condition", cond, ftype)

            # Pairwise Jaccard
            pairs = compute_pairwise_jaccard(fsets)
            for p in pairs:
                p.update({
                    "experiment": display_name,
                    "condition": cond,
                    "severity": severity,
                    "failure_type": ftype,
                })
            all_jaccard.extend(pairs)

            # Shared failures (all 4 models fail)
            all_failed = set.intersection(*fsets.values()) if all(isinstance(s, set) for s in fsets.values()) else set()
            if all_failed:
                for f in all_failed:
                    shared_failures.append({
                        "experiment": display_name,
                        "condition": cond,
                        "severity": severity,
                        "failure_type": ftype,
                        "file": f,
                    })

    # ── Baseline Jaccard ──
    baseline_jaccard = []
    exp_files = set(df["file"].unique())
    bl = baseline_df[baseline_df["file"].isin(exp_files)]
    for ftype in ["fn", "fp"]:
        bl_fsets = {}
        for model in MODELS:
            model_df = bl[bl["model"] == model]
            if ftype == "fn":
                bl_fsets[model] = set(model_df[model_df["Recall"] < FN_THRESHOLD]["file"])
            else:
                bl_fsets[model] = set(model_df[model_df["Precision"] < FP_THRESHOLD]["file"])
        pairs = compute_pairwise_jaccard(bl_fsets)
        for p in pairs:
            p.update({
                "experiment": display_name,
                "condition": "baseline_clean",
                "severity": None,
                "failure_type": ftype,
            })
        baseline_jaccard.extend(pairs)

    # ── Summary ──
    df_j = pd.DataFrame(all_jaccard)
    if not df_j.empty:
        summary = df_j.groupby(["severity", "failure_type"])["jaccard"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        summary = summary.sort_values("severity", ascending=not invert_x)

        print(f"\n  Mean Jaccard by severity (FN-failure):")
        fn_sum = summary[summary["failure_type"] == "fn"]
        if not fn_sum.empty:
            print(fn_sum[["severity", "mean", "count"]].to_string(index=False))

        print(f"\n  Mean Jaccard by severity (FP-failure):")
        fp_sum = summary[summary["failure_type"] == "fp"]
        if not fp_sum.empty:
            print(fp_sum[["severity", "mean", "count"]].to_string(index=False))

    # Baseline summary
    bl_j = pd.DataFrame(baseline_jaccard)
    if not bl_j.empty:
        bl_mean = bl_j.groupby("failure_type")["jaccard"].mean()
        print(f"\n  Baseline Jaccard: FN={bl_mean.get('fn', np.nan):.3f}, "
              f"FP={bl_mean.get('fp', np.nan):.3f}")

    return {
        "jaccard": all_jaccard,
        "baseline_jaccard": baseline_jaccard,
        "shared_failures": shared_failures,
        "display_name": display_name,
    }


def main():
    print("Failure Transferability Analysis")
    print(f"Thresholds: FN-fail = Recall < {FN_THRESHOLD}, "
          f"FP-fail = Precision < {FP_THRESHOLD}")
    print(f"Output: {OUTPUT_DIR}\n")

    baseline_df = load_baseline()
    print(f"Baseline: {len(baseline_df)} rows, {baseline_df['file'].nunique()} files")

    all_jaccard = []
    all_baseline = []
    all_shared = []

    for exp_name, display, extractor, label, invert in EXPERIMENTS:
        result = analyze_experiment(
            exp_name, display, extractor, label, invert, baseline_df
        )
        if result:
            all_jaccard.extend(result["jaccard"])
            all_baseline.extend(result["baseline_jaccard"])
            all_shared.extend(result["shared_failures"])

    # ── Save CSVs ──
    df_jaccard = pd.DataFrame(all_jaccard)
    df_jaccard.to_csv(OUTPUT_DIR / "pairwise_jaccard_all.csv", index=False)

    df_baseline = pd.DataFrame(all_baseline)
    df_baseline.to_csv(OUTPUT_DIR / "baseline_jaccard.csv", index=False)

    if all_shared:
        df_shared = pd.DataFrame(all_shared)
        df_shared.to_csv(OUTPUT_DIR / "shared_failures.csv", index=False)
        print(f"\nShared failure files: {len(df_shared)}")

    # ── Mean Jaccard by severity ──
    if not df_jaccard.empty:
        mean_j = df_jaccard.groupby(
            ["experiment", "severity", "failure_type"]
        )["jaccard"].agg(["mean", "std", "count"]).reset_index()
        mean_j.to_csv(OUTPUT_DIR / "mean_jaccard_by_severity.csv", index=False)

        # Shared failure count by severity
        if all_shared:
            shared_count = df_shared.groupby(
                ["experiment", "severity", "failure_type"]
            )["file"].nunique().reset_index()
            shared_count.columns = ["experiment", "severity", "failure_type", "n_shared_files"]
            shared_count.to_csv(OUTPUT_DIR / "shared_failure_count.csv", index=False)

    # ── Cross-experiment summary ──
    if not df_jaccard.empty:
        cross = df_jaccard.groupby(["experiment", "failure_type"]).agg(
            jaccard_mean=("jaccard", "mean"),
            jaccard_max=("jaccard", "max"),
            jaccard_min=("jaccard", "min"),
        ).reset_index()
        cross = cross.sort_values(["failure_type", "jaccard_mean"], ascending=[True, False])
        cross.to_csv(OUTPUT_DIR / "cross_experiment_summary.csv", index=False)

        print(f"\n{'='*60}")
        print("  CROSS-EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(cross.to_string(index=False))

    print(f"\nDone. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
