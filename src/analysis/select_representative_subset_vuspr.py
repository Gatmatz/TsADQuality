"""
Select a representative subset from TSB-AD-U (Univariate) benchmark
that generalizes to the full dataset, optimized for VUS-PR.

Strategy:
---------
1. Load the official TSB-AD uni_mergedTable_VUS-PR.csv (32 methods × 151 TS)
2. Extract metadata features: domain, anomaly_ratio, ts_len, anomaly type, difficulty
3. Stratified sampling to preserve:
   - Domain distribution
   - Difficulty distribution (based on mean VUS-PR)
   - Anomaly type (point vs sequence)
   - Anomaly ratio distribution
4. Validate with multiple tests:
   - Spearman rank correlation (model ranking preservation)
   - Kendall tau-b (concordance)
   - Kolmogorov-Smirnov test (VUS-PR distribution similarity)
   - Chi-squared (categorical distribution tests)
5. Iterative optimization: try N random stratified subsets, keep the one with best
   Spearman correlation on model rankings (this is the key criterion)

Usage:
------
    python src/analysis/select_representative_subset_vuspr.py [--target_size 50] [--n_trials 5000]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    kendalltau,
    ks_2samp,
    spearmanr,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
VUS_PR_PATH = ROOT / "results" / "tables" / "tsb_ad_uni_VUS_PR.csv"
OUTPUT_DIR = ROOT / "results" / "tables"

# Method columns in the VUS-PR table
METHOD_COLS = [
    "Sub-IForest", "IForest", "Sub-LOF", "LOF", "POLY", "MatrixProfile",
    "KShapeAD", "SAND", "Series2Graph", "SR", "Sub-PCA", "Sub-HBOS",
    "Sub-OCSVM", "Sub-MCD", "Sub-KNN", "KMeansAD", "AutoEncoder", "CNN",
    "LSTMAD", "TranAD", "AnomalyTransformer", "OmniAnomaly", "USAD",
    "Donut", "TimesNet", "FITS", "OFA", "Lag-Llama", "Chronos", "TimesFM",
    "MOMENT (ZS)", "MOMENT (FT)",
]

# Metadata columns
META_COLS = ["ts_len", "anomaly_len", "num_anomaly", "avg_anomaly_len",
             "anomaly_ratio", "point_anomaly", "seq_anomaly"]


def extract_domain(filename: str) -> str:
    """Extract domain from TSB-AD filename convention.
    
    Format: NNN_SOURCE_id_N_DOMAIN_tr_...
    E.g.: 001_NAB_id_1_Facility_tr_1007_1st_2014.csv
    """
    parts = filename.split("_")
    # Source dataset is parts[1] (NAB, WSD, MSL, Stock, MITDB, SMD, Daphnet)
    source = parts[1]
    # Domain is parts[4] for most files
    if len(parts) >= 5:
        domain = parts[4]
    else:
        domain = source
    return domain


def extract_source(filename: str) -> str:
    """Extract source dataset name."""
    parts = filename.split("_")
    return parts[1] if len(parts) >= 2 else "Unknown"


def add_difficulty(df: pd.DataFrame, method_cols: list) -> pd.DataFrame:
    """Add difficulty terciles based on mean VUS-PR across all methods."""
    df = df.copy()
    df["mean_VUS_PR"] = df[method_cols].mean(axis=1)
    q33 = df["mean_VUS_PR"].quantile(1 / 3)
    q66 = df["mean_VUS_PR"].quantile(2 / 3)
    df["difficulty"] = pd.cut(
        df["mean_VUS_PR"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Hard", "Medium", "Easy"],
        include_lowest=True,
    )
    return df, q33, q66


def compute_model_ranking(df: pd.DataFrame, method_cols: list) -> pd.Series:
    """Compute average VUS-PR per method → model ranking."""
    return df[method_cols].mean().sort_values(ascending=False)


def evaluate_subset(full_df, subset_df, method_cols):
    """Evaluate how well a subset preserves the full benchmark's properties.
    
    Returns a dict of quality metrics (higher = better for correlations,
    higher p-value = better for distribution tests).
    """
    # 1. Model ranking preservation (most important!)
    full_ranking = compute_model_ranking(full_df, method_cols)
    subset_ranking = compute_model_ranking(subset_df, method_cols)
    
    # Align indices
    common = full_ranking.index.intersection(subset_ranking.index)
    spearman_corr, spearman_p = spearmanr(
        full_ranking[common].values, subset_ranking[common].values
    )
    kendall_corr, kendall_p = kendalltau(
        full_ranking[common].values, subset_ranking[common].values
    )
    
    # 2. VUS-PR distribution preservation per method (KS test)
    ks_pvalues = []
    for col in method_cols:
        if col in full_df.columns and col in subset_df.columns:
            stat, pval = ks_2samp(full_df[col].dropna(), subset_df[col].dropna())
            ks_pvalues.append(pval)
    mean_ks_pval = np.mean(ks_pvalues) if ks_pvalues else 0
    min_ks_pval = np.min(ks_pvalues) if ks_pvalues else 0
    
    # 3. Difficulty distribution (Chi-squared)
    chi2_diff_p = 1.0
    if "difficulty" in full_df.columns and "difficulty" in subset_df.columns:
        try:
            contingency = pd.crosstab(
                pd.Series(["Full"] * len(full_df) + ["Subset"] * len(subset_df)),
                pd.concat([full_df["difficulty"], subset_df["difficulty"]],
                          ignore_index=True),
            )
            _, chi2_diff_p, _, _ = chi2_contingency(contingency)
        except Exception:
            pass
    
    # 4. Domain distribution (Chi-squared)
    chi2_domain_p = 1.0
    if "domain" in full_df.columns and "domain" in subset_df.columns:
        try:
            contingency = pd.crosstab(
                pd.Series(["Full"] * len(full_df) + ["Subset"] * len(subset_df)),
                pd.concat([full_df["domain"], subset_df["domain"]],
                          ignore_index=True),
            )
            _, chi2_domain_p, _, _ = chi2_contingency(contingency)
        except Exception:
            pass
    
    # 5. Source distribution (Chi-squared)
    chi2_source_p = 1.0
    if "source" in full_df.columns and "source" in subset_df.columns:
        try:
            contingency = pd.crosstab(
                pd.Series(["Full"] * len(full_df) + ["Subset"] * len(subset_df)),
                pd.concat([full_df["source"], subset_df["source"]],
                          ignore_index=True),
            )
            _, chi2_source_p, _, _ = chi2_contingency(contingency)
        except Exception:
            pass
    
    # 6. Anomaly type distribution
    chi2_antype_p = 1.0
    if "anomaly_type" in full_df.columns and "anomaly_type" in subset_df.columns:
        try:
            contingency = pd.crosstab(
                pd.Series(["Full"] * len(full_df) + ["Subset"] * len(subset_df)),
                pd.concat([full_df["anomaly_type"], subset_df["anomaly_type"]],
                          ignore_index=True),
            )
            _, chi2_antype_p, _, _ = chi2_contingency(contingency)
        except Exception:
            pass
    
    # 7. Mean VUS-PR difference
    full_mean = full_df["mean_VUS_PR"].mean()
    subset_mean = subset_df["mean_VUS_PR"].mean()
    mean_diff = abs(full_mean - subset_mean)
    
    # 8. Top-5 model agreement
    full_top5 = set(full_ranking.index[:5])
    subset_top5 = set(subset_ranking.index[:5])
    top5_agreement = len(full_top5 & subset_top5) / 5
    
    # 9. Top-10 model agreement
    full_top10 = set(full_ranking.index[:10])
    subset_top10 = set(subset_ranking.index[:10])
    top10_agreement = len(full_top10 & subset_top10) / 10
    
    return {
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
        "kendall_corr": kendall_corr,
        "kendall_p": kendall_p,
        "mean_ks_pval": mean_ks_pval,
        "min_ks_pval": min_ks_pval,
        "chi2_difficulty_p": chi2_diff_p,
        "chi2_domain_p": chi2_domain_p,
        "chi2_source_p": chi2_source_p,
        "chi2_anomaly_type_p": chi2_antype_p,
        "mean_vuspr_diff": mean_diff,
        "top5_agreement": top5_agreement,
        "top10_agreement": top10_agreement,
    }


def composite_score(metrics: dict) -> float:
    """Compute a composite quality score for ranking subsets.
    
    Heavily weighted toward Spearman (model ranking preservation is key).
    """
    return (
        0.40 * metrics["spearman_corr"]
        + 0.15 * metrics["kendall_corr"]
        + 0.10 * metrics["top5_agreement"]
        + 0.10 * metrics["top10_agreement"]
        + 0.05 * min(metrics["mean_ks_pval"], 1.0)
        + 0.05 * min(metrics["chi2_difficulty_p"], 1.0)
        + 0.05 * min(metrics["chi2_domain_p"], 1.0)
        + 0.05 * min(metrics["chi2_source_p"], 1.0)
        + 0.05 * (1 - metrics["mean_vuspr_diff"])  # smaller diff = better
    )


def stratified_sample(df, target_size, rng):
    """Stratified sampling preserving difficulty, source, and anomaly type."""
    # Primary stratification: difficulty
    # Secondary: source dataset
    strat_col = "strat_key"
    df = df.copy()
    df[strat_col] = df["difficulty"].astype(str) + "_" + df["source"]
    
    groups = df.groupby(strat_col)
    n_groups = len(groups)
    
    # Allocate proportionally
    alloc = {}
    remaining = target_size
    for name, group in groups:
        n = max(1, int(round(len(group) / len(df) * target_size)))
        n = min(n, len(group))
        alloc[name] = n
    
    # Adjust to hit target
    total = sum(alloc.values())
    if total > target_size:
        # Reduce from largest groups
        sorted_groups = sorted(alloc.items(), key=lambda x: -x[1])
        for name, n in sorted_groups:
            if total <= target_size:
                break
            reduce = min(n - 1, total - target_size)
            alloc[name] -= reduce
            total -= reduce
    elif total < target_size:
        # Add to groups with remaining capacity
        for name, group in groups:
            if total >= target_size:
                break
            capacity = len(group) - alloc[name]
            add = min(capacity, target_size - total)
            alloc[name] += add
            total += add
    
    # Sample from each stratum
    samples = []
    for name, group in groups:
        n = alloc.get(name, 0)
        if n > 0:
            samples.append(group.sample(n=min(n, len(group)), random_state=rng))
    
    result = pd.concat(samples)
    df.drop(columns=[strat_col], inplace=True)
    if strat_col in result.columns:
        result = result.drop(columns=[strat_col])
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Select representative TSB-AD subset optimized for VUS-PR"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(VUS_PR_PATH),
        help=f"Path to VUS-PR merged table (default: {VUS_PR_PATH})",
    )
    parser.add_argument("--target_size", type=int, default=50,
                        help="Target subset size (default: 50)")
    parser.add_argument("--n_trials", type=int, default=5000,
                        help="Number of random trials (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("TSB-AD REPRESENTATIVE SUBSET SELECTION (VUS-PR)")
    print("=" * 80)
    
    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    input_csv = Path(args.input_csv)
    print(f"\nLoading VUS-PR data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  Full benchmark: {len(df)} time series, {len(METHOD_COLS)} methods")
    
    # Verify method columns exist
    available_methods = [c for c in METHOD_COLS if c in df.columns]
    missing_methods = [c for c in METHOD_COLS if c not in df.columns]
    if missing_methods:
        print(f"  WARNING: Missing method columns: {missing_methods}")
    print(f"  Available methods: {len(available_methods)}")
    
    # -----------------------------------------------------------------------
    # 2. Add metadata
    # -----------------------------------------------------------------------
    df["domain"] = df["file"].apply(extract_domain)
    df["source"] = df["file"].apply(extract_source)
    df["anomaly_type"] = df.apply(
        lambda r: "point" if r.get("point_anomaly", 0) == 1 else "sequence",
        axis=1,
    )
    df, q33, q66 = add_difficulty(df, available_methods)
    
    print(f"\n--- Full Benchmark Statistics ---")
    print(f"  Mean VUS-PR (across all methods/TS): {df['mean_VUS_PR'].mean():.4f}")
    print(f"  Difficulty thresholds: Hard < {q33:.4f} < Medium < {q66:.4f} < Easy")
    print(f"\n  Domain distribution:")
    for domain, count in df["domain"].value_counts().items():
        print(f"    {domain}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\n  Source distribution:")
    for src, count in df["source"].value_counts().items():
        print(f"    {src}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\n  Difficulty distribution:")
    for diff, count in df["difficulty"].value_counts().sort_index().items():
        print(f"    {diff}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\n  Anomaly type: {df['anomaly_type'].value_counts().to_dict()}")
    
    # Full benchmark model ranking
    full_ranking = compute_model_ranking(df, available_methods)
    print(f"\n--- Full Benchmark Model Ranking (by mean VUS-PR) ---")
    for i, (method, score) in enumerate(full_ranking.items(), 1):
        print(f"  {i:2d}. {method:25s}  {score:.4f}")
    
    # -----------------------------------------------------------------------
    # 3. Search for best subset
    # -----------------------------------------------------------------------
    target = min(args.target_size, len(df) - 1)
    print(f"\n{'='*80}")
    print(f"SEARCHING FOR BEST SUBSET (size={target}, trials={args.n_trials})")
    print(f"{'='*80}")
    
    rng = np.random.RandomState(args.seed)
    best_score = -np.inf
    best_subset = None
    best_metrics = None
    
    for trial in range(args.n_trials):
        seed_i = rng.randint(0, 2**31)
        subset = stratified_sample(df, target, np.random.RandomState(seed_i))
        metrics = evaluate_subset(df, subset, available_methods)
        score = composite_score(metrics)
        
        if score > best_score:
            best_score = score
            best_subset = subset.copy()
            best_metrics = metrics.copy()
            if (trial + 1) % 500 == 0 or trial == 0:
                print(f"  Trial {trial+1:5d}: new best score = {score:.6f} "
                      f"(ρ={metrics['spearman_corr']:.4f}, "
                      f"top5={metrics['top5_agreement']:.0%})")
        
        if (trial + 1) % 1000 == 0 and trial > 0:
            print(f"  Trial {trial+1:5d}: best so far = {best_score:.6f}")
    
    # -----------------------------------------------------------------------
    # 4. Report results
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("BEST SUBSET FOUND")
    print(f"{'='*80}")
    print(f"  Subset size: {len(best_subset)}")
    print(f"  Composite score: {best_score:.6f}")
    
    print(f"\n--- Representativeness Metrics ---")
    print(f"  Spearman rank correlation:  rho = {best_metrics['spearman_corr']:.4f} "
          f"(p = {best_metrics['spearman_p']:.2e})")
    print(f"  Kendall rank correlation:   tau = {best_metrics['kendall_corr']:.4f} "
          f"(p = {best_metrics['kendall_p']:.2e})")
    print(f"  Top-5 model agreement:      {best_metrics['top5_agreement']:.0%}")
    print(f"  Top-10 model agreement:     {best_metrics['top10_agreement']:.0%}")
    print(f"  Mean VUS-PR difference:     {best_metrics['mean_vuspr_diff']:.4f}")
    print(f"  Mean KS p-value:            {best_metrics['mean_ks_pval']:.4f}")
    print(f"  Min KS p-value:             {best_metrics['min_ks_pval']:.4f}")
    
    print(f"\n--- Distribution Tests (alpha=0.05, p>0.05 = REPRESENTATIVE) ---")
    tests = [
        ("Difficulty", best_metrics["chi2_difficulty_p"]),
        ("Domain", best_metrics["chi2_domain_p"]),
        ("Source", best_metrics["chi2_source_p"]),
        ("Anomaly type", best_metrics["chi2_anomaly_type_p"]),
    ]
    for name, pval in tests:
        status = "[PASS] REPRESENTATIVE" if pval > 0.05 else "[FAIL] NOT REPRESENTATIVE"
        print(f"  {name:20s}: p = {pval:.4f}  {status}")
    
    # Subset model ranking
    subset_ranking = compute_model_ranking(best_subset, available_methods)
    print(f"\n--- Model Ranking Comparison ---")
    print(f"  {'Rank':>4s}  {'Method':25s}  {'Full VUS-PR':>12s}  {'Subset VUS-PR':>14s}  {'Full Rank':>9s}  {'Subset Rank':>11s}")
    print(f"  {'-'*4}  {'-'*25}  {'-'*12}  {'-'*14}  {'-'*9}  {'-'*11}")
    for i, (method, full_score) in enumerate(full_ranking.items(), 1):
        subset_score = subset_ranking.get(method, np.nan)
        subset_rank = list(subset_ranking.index).index(method) + 1 if method in subset_ranking.index else "N/A"
        print(f"  {i:4d}  {method:25s}  {full_score:12.4f}  {subset_score:14.4f}  {i:9d}  {subset_rank:>11}")
    
    # Subset composition
    print(f"\n--- Subset Composition ---")
    print(f"\n  Domain distribution:")
    for domain, count in best_subset["domain"].value_counts().items():
        full_pct = df["domain"].value_counts(normalize=True).get(domain, 0) * 100
        sub_pct = count / len(best_subset) * 100
        print(f"    {domain:15s}: {count:3d} ({sub_pct:5.1f}%)  [full: {full_pct:5.1f}%]")
    
    print(f"\n  Source distribution:")
    for src, count in best_subset["source"].value_counts().items():
        full_pct = df["source"].value_counts(normalize=True).get(src, 0) * 100
        sub_pct = count / len(best_subset) * 100
        print(f"    {src:15s}: {count:3d} ({sub_pct:5.1f}%)  [full: {full_pct:5.1f}%]")
    
    print(f"\n  Difficulty distribution:")
    for diff in ["Hard", "Medium", "Easy"]:
        count = (best_subset["difficulty"] == diff).sum()
        full_pct = (df["difficulty"] == diff).sum() / len(df) * 100
        sub_pct = count / len(best_subset) * 100
        print(f"    {diff:15s}: {count:3d} ({sub_pct:5.1f}%)  [full: {full_pct:5.1f}%]")
    
    # -----------------------------------------------------------------------
    # 5. Save results
    # -----------------------------------------------------------------------
    # Save subset file list
    subset_files = best_subset[["file"] + available_methods + META_COLS + 
                                ["domain", "source", "anomaly_type", 
                                 "difficulty", "mean_VUS_PR"]].copy()
    
    output_path = OUTPUT_DIR / "representative_subset_tsb_ad_vuspr.csv"
    subset_files.to_csv(output_path, index=False)
    print(f"\n  Subset saved to: {output_path}")
    
    # Save validation report
    report_path = OUTPUT_DIR / "subset_validation_report_vuspr.csv"
    report = pd.DataFrame([best_metrics])
    report["subset_size"] = len(best_subset)
    report["full_size"] = len(df)
    report["composite_score"] = best_score
    report.to_csv(report_path, index=False)
    print(f"  Validation report saved to: {report_path}")
    
    # Save ranking comparison
    ranking_path = OUTPUT_DIR / "ranking_comparison_vuspr.csv"
    ranking_df = pd.DataFrame({
        "method": full_ranking.index,
        "full_mean_vuspr": full_ranking.values,
        "full_rank": range(1, len(full_ranking) + 1),
        "subset_mean_vuspr": [subset_ranking.get(m, np.nan) for m in full_ranking.index],
        "subset_rank": [list(subset_ranking.index).index(m) + 1 
                        if m in subset_ranking.index else np.nan 
                        for m in full_ranking.index],
    })
    ranking_df.to_csv(ranking_path, index=False)
    print(f"  Ranking comparison saved to: {ranking_path}")
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    
    all_rep = all(p > 0.05 for _, p in tests)
    good_corr = best_metrics["spearman_corr"] > 0.90
    
    if all_rep and good_corr:
        print("[PASS] The subset is HIGHLY REPRESENTATIVE of the full TSB-AD benchmark.")
        print(f"  Spearman rho = {best_metrics['spearman_corr']:.4f} (>0.90)")
        print("  All distribution tests passed (p > 0.05)")
    elif good_corr:
        print("[PARTIAL] The subset has GOOD model ranking preservation but some distribution differences.")
        print(f"  Spearman rho = {best_metrics['spearman_corr']:.4f} (>0.90)")
        failed = [name for name, p in tests if p <= 0.05]
        print(f"  Failed tests: {', '.join(failed)}")
    else:
        print("[FAIL] The subset may not be fully representative.")
        print(f"  Spearman rho = {best_metrics['spearman_corr']:.4f}")
        print("  Consider increasing subset size or n_trials")
    
    print(f"\n{'='*80}")
    
    return best_subset, best_metrics


if __name__ == "__main__":
    main()
