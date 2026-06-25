"""
Deep Interpretation of Single Corruption Experiments
=====================================================
1. Distribution of AUC drops per file (not just mean)
2. Dataset characteristics → vulnerability correlation
3. Critical threshold ("breaking point") per model × corruption
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = PROJECT_ROOT / "results" / "experiments"
BASELINE_CSV = PROJECT_ROOT / "results" / "tables" / "baseline_final_subset.csv"
SUBSET_CSV = PROJECT_ROOT / "results" / "tables" / "final_subset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "deep_interpretation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["IForest", "LOF", "MP", "AE"]

EXPERIMENTS = [
    ("white_noise_snr", "White Noise",
     lambda c: float(re.search(r"snr_([-\d]+)dB", c).group(1)),
     True),  # invert: lower SNR = more corruption
    ("freeze", "Sensor Freeze",
     lambda c: float(re.search(r"frac_([\d.]+)", c).group(1)),
     False),
    ("spikes_normal_only", "Spikes",
     lambda c: float(re.search(r"frac_([\d.]+)_mult", c).group(1)),
     False),
    ("swap_segment", "Segment Swap",
     lambda c: float(re.search(r"frac_([\d.]+)_ns", c).group(1)),
     False),
    ("missing_true_impact", "Missing Data",
     lambda c: float(re.search(r"frac_([\d.]+)", c).group(1)),
     False),
]


def load_baseline():
    df = pd.read_csv(BASELINE_CSV)
    df["model"] = df["model"].replace({"Autoencoder": "AE"})
    df = df[df["model"].isin(MODELS)].copy()
    if "F1" in df.columns:
        df = df.rename(columns={"F1": "F"})
    return df


def load_subset_meta():
    df = pd.read_csv(SUBSET_CSV)
    # Normalize filename for merging
    df["file"] = df["filename_actual"]
    return df[["file", "folder", "data_len", "ratio", "difficulty", "type_an"]].copy()


def load_experiment(exp_name):
    path = EXPERIMENTS_DIR / exp_name / "checkpoint.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["model"].isin(MODELS)].copy()
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")].copy()
    df["AUC_ROC"] = pd.to_numeric(df["AUC_ROC"], errors="coerce")
    df = df.dropna(subset=["AUC_ROC"])
    return df


# ═══════════════════════════════════════════════════════════════
# 1. DISTRIBUTION OF AUC DROPS
# ═══════════════════════════════════════════════════════════════

def compute_auc_drops(exp_df, baseline_df, severity_extractor):
    """Compute per-file AUC drop = baseline_AUC - corrupted_AUC."""
    bl_pivot = baseline_df.set_index(["file", "model"])["AUC_ROC"]

    rows = []
    for _, r in exp_df.iterrows():
        key = (r["file"], r["model"])
        if key in bl_pivot.index:
            bl_auc = bl_pivot[key]
            drop = bl_auc - r["AUC_ROC"]
            try:
                severity = severity_extractor(r["condition"])
            except Exception:
                continue
            rows.append({
                "file": r["file"], "model": r["model"],
                "condition": r["condition"], "severity": severity,
                "baseline_auc": bl_auc, "corrupted_auc": r["AUC_ROC"],
                "auc_drop": drop, "relative_drop": drop / bl_auc if bl_auc > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def analyze_distributions(drops_df, exp_name, display_name):
    """Analyze distribution of AUC drops."""
    print(f"\n  Distribution analysis:")

    results = []
    for model in MODELS:
        for sev in sorted(drops_df["severity"].unique()):
            sub = drops_df[(drops_df["model"] == model) & (drops_df["severity"] == sev)]
            if sub.empty:
                continue
            d = sub["auc_drop"]
            results.append({
                "experiment": display_name, "model": model, "severity": sev,
                "mean_drop": d.mean(), "median_drop": d.median(),
                "std_drop": d.std(), "min_drop": d.min(), "max_drop": d.max(),
                "pct_improved": (d < 0).mean() * 100,  # files that IMPROVED
                "pct_catastrophic": (d > 0.2).mean() * 100,  # >20% AUC loss
                "skewness": d.skew(),
            })

    df_dist = pd.DataFrame(results)

    # Print key findings
    for model in MODELS:
        m_df = df_dist[df_dist["model"] == model]
        if m_df.empty:
            continue
        worst = m_df.loc[m_df["mean_drop"].idxmax()]
        imp = m_df["pct_improved"].mean()
        cat = m_df.loc[m_df["mean_drop"].idxmax(), "pct_catastrophic"]
        print(f"    {model}: worst severity={worst['severity']}, "
              f"mean_drop={worst['mean_drop']:.3f}, "
              f"catastrophic={cat:.0f}%, improved={imp:.1f}%")

    return df_dist


# ═══════════════════════════════════════════════════════════════
# 2. DATASET CHARACTERISTICS → VULNERABILITY
# ═══════════════════════════════════════════════════════════════

def analyze_vulnerability_factors(drops_df, meta_df, display_name):
    """Correlate dataset properties with AUC drop."""
    # Merge drops with metadata
    merged = drops_df.merge(meta_df, on="file", how="left")

    print(f"\n  Vulnerability factors (Spearman correlation with AUC drop):")

    numeric_features = ["data_len", "ratio", "baseline_auc"]
    categorical_features = ["difficulty", "folder", "type_an"]

    corr_rows = []
    for model in MODELS:
        for sev in sorted(drops_df["severity"].unique()):
            sub = merged[(merged["model"] == model) & (merged["severity"] == sev)]
            if len(sub) < 10:
                continue

            for feat in numeric_features:
                if feat in sub.columns and sub[feat].notna().sum() > 10:
                    rho, pval = spearmanr(sub[feat], sub["auc_drop"], nan_policy="omit")
                    corr_rows.append({
                        "experiment": display_name, "model": model,
                        "severity": sev, "feature": feat,
                        "spearman_rho": rho, "p_value": pval,
                        "significant": pval < 0.05,
                    })

    df_corr = pd.DataFrame(corr_rows)

    # Print significant correlations
    if not df_corr.empty:
        sig = df_corr[df_corr["significant"]]
        if not sig.empty:
            # Average across severities
            avg_sig = sig.groupby(["model", "feature"])["spearman_rho"].mean().reset_index()
            avg_sig = avg_sig.sort_values("spearman_rho", key=abs, ascending=False)
            for _, r in avg_sig.head(8).iterrows():
                direction = "more vulnerable" if r["spearman_rho"] > 0 else "more robust"
                print(f"    {r['model']}: higher {r['feature']} -> {direction} "
                      f"(rho={r['spearman_rho']:.3f})")

    # Categorical: difficulty breakdown
    cat_rows = []
    for model in MODELS:
        for sev in sorted(drops_df["severity"].unique()):
            sub = merged[(merged["model"] == model) & (merged["severity"] == sev)]
            if sub.empty:
                continue
            for cat in categorical_features:
                if cat not in sub.columns:
                    continue
                for val in sub[cat].dropna().unique():
                    grp = sub[sub[cat] == val]["auc_drop"]
                    if len(grp) < 3:
                        continue
                    cat_rows.append({
                        "experiment": display_name, "model": model,
                        "severity": sev, "feature": cat, "value": val,
                        "mean_drop": grp.mean(), "n_files": len(grp),
                    })

    df_cat = pd.DataFrame(cat_rows)

    # Print difficulty breakdown at max severity
    if not df_cat.empty:
        max_sev = drops_df["severity"].max() if not drops_df.empty else None
        if max_sev is not None:
            diff = df_cat[(df_cat["feature"] == "difficulty") & (df_cat["severity"] == max_sev)]
            if not diff.empty:
                print(f"\n  AUC drop by difficulty (severity={max_sev}):")
                pivot = diff.pivot_table(index="value", columns="model",
                                         values="mean_drop", aggfunc="mean")
                print(pivot.to_string())

    return df_corr, df_cat


# ═══════════════════════════════════════════════════════════════
# 3. CRITICAL THRESHOLD ("BREAKING POINT")
# ═══════════════════════════════════════════════════════════════

def find_breaking_points(drops_df, exp_df, display_name, invert_severity):
    """Find severity at which AUC drops below key thresholds."""
    thresholds = {
        "below_random": 0.5,
        "below_0.6": 0.6,
        "10pct_drop": None,  # relative
    }

    print(f"\n  Breaking points:")

    bp_rows = []
    for model in MODELS:
        model_df = drops_df[drops_df["model"] == model]
        if model_df.empty:
            continue

        # Mean corrupted AUC per severity
        agg = model_df.groupby("severity").agg(
            mean_auc=("corrupted_auc", "mean"),
            mean_drop=("auc_drop", "mean"),
            mean_baseline=("baseline_auc", "mean"),
        ).reset_index().sort_values("severity", ascending=not invert_severity)

        baseline_mean = agg["mean_baseline"].iloc[0]

        for thresh_name, thresh_val in thresholds.items():
            if thresh_name == "10pct_drop":
                # Find where mean drop > 10% of baseline
                target = baseline_mean * 0.10
                breach = agg[agg["mean_drop"] >= target]
            else:
                breach = agg[agg["mean_auc"] <= thresh_val]

            if not breach.empty:
                if invert_severity:
                    bp_severity = breach["severity"].max()  # highest SNR where it breaks
                else:
                    bp_severity = breach["severity"].min()  # lowest fraction where it breaks
                bp_rows.append({
                    "experiment": display_name, "model": model,
                    "threshold": thresh_name, "breaking_severity": bp_severity,
                    "auc_at_break": breach.iloc[0]["mean_auc"] if not breach.empty else np.nan,
                })
                print(f"    {model}: {thresh_name} at severity={bp_severity}")
            else:
                bp_rows.append({
                    "experiment": display_name, "model": model,
                    "threshold": thresh_name, "breaking_severity": np.nan,
                    "auc_at_break": np.nan,
                })

    return pd.DataFrame(bp_rows)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("Deep Interpretation Analysis")
    print(f"Output: {OUTPUT_DIR}\n")

    baseline_df = load_baseline()
    meta_df = load_subset_meta()

    all_distributions = []
    all_correlations = []
    all_categorical = []
    all_breaking = []
    all_drops = []

    for exp_name, display, extractor, invert in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  {display} ({exp_name})")
        print(f"{'='*60}")

        exp_df = load_experiment(exp_name)
        if exp_df is None or exp_df.empty:
            continue

        # Compute per-file drops
        drops = compute_auc_drops(exp_df, baseline_df, extractor)
        if drops.empty:
            continue
        drops["experiment"] = display
        all_drops.append(drops)

        # 1. Distributions
        dist = analyze_distributions(drops, exp_name, display)
        all_distributions.append(dist)

        # 2. Vulnerability factors
        corr, cat = analyze_vulnerability_factors(drops, meta_df, display)
        all_correlations.append(corr)
        all_categorical.append(cat)

        # 3. Breaking points
        bp = find_breaking_points(drops, exp_df, display, invert)
        all_breaking.append(bp)

    # ── Save ──
    if all_drops:
        pd.concat(all_drops).to_csv(OUTPUT_DIR / "per_file_drops.csv", index=False)
    if all_distributions:
        pd.concat(all_distributions).to_csv(OUTPUT_DIR / "drop_distributions.csv", index=False)
    if all_correlations:
        pd.concat(all_correlations).to_csv(OUTPUT_DIR / "vulnerability_correlations.csv", index=False)
    if all_categorical:
        pd.concat(all_categorical).to_csv(OUTPUT_DIR / "vulnerability_by_category.csv", index=False)
    if all_breaking:
        pd.concat(all_breaking).to_csv(OUTPUT_DIR / "breaking_points.csv", index=False)

    # ── Cross-experiment breaking point summary ──
    if all_breaking:
        bp_all = pd.concat(all_breaking)
        bp_10 = bp_all[bp_all["threshold"] == "10pct_drop"].dropna(subset=["breaking_severity"])
        if not bp_10.empty:
            print(f"\n{'='*60}")
            print("  BREAKING POINTS: 10% AUC drop threshold")
            print(f"{'='*60}")
            pivot = bp_10.pivot_table(index="experiment", columns="model",
                                      values="breaking_severity", aggfunc="first")
            print(pivot.to_string())

    print(f"\nDone. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
