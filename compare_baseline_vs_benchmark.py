"""
Compare baseline results (from our subset) with TSB-UAD benchmark results.
"""
import pandas as pd
import numpy as np
import os

# Paths
BASELINE_PATH = r"results\tables\baseline_final_subset.csv"
BENCHMARK_DIR = r"TSB-UAD\result\accuracy_table"

# Load baseline
baseline = pd.read_csv(BASELINE_PATH)

# Load benchmark tables
bench_auc_roc = pd.read_csv(os.path.join(BENCHMARK_DIR, "mergedTable_AUC_ROC.csv"))
bench_auc_pr = pd.read_csv(os.path.join(BENCHMARK_DIR, "mergedTable_AUC_PR.csv"))
bench_f1 = pd.read_csv(os.path.join(BENCHMARK_DIR, "mergedTable_F.csv"))

# Model mapping: baseline name -> benchmark column name
MODEL_MAP = {
    "IForest": "IFOREST",
    "LOF": "LOF",
    "MP": "MP",
    "PCA": "PCA",
    "Autoencoder": "AE",
}

# Get unique files from baseline
baseline_files = baseline["file"].unique()
print(f"=== Baseline vs TSB-UAD Benchmark Comparison ===\n")
print(f"Baseline: {len(baseline)} rows, {len(baseline_files)} unique files, "
      f"{len(baseline['model'].unique())} models ({', '.join(baseline['model'].unique())})")
print(f"Benchmark tables: {len(bench_auc_roc)} files\n")

# Find matching files between baseline and benchmark
bench_filenames = set(bench_auc_roc["filename"].values)
matched_files = [f for f in baseline_files if f in bench_filenames]
unmatched_files = [f for f in baseline_files if f not in bench_filenames]
print(f"Matched files: {len(matched_files)} / {len(baseline_files)}")
if unmatched_files:
    print(f"Unmatched files ({len(unmatched_files)}): {unmatched_files[:10]}...")

# Build comparison dataframe
rows = []
for _, brow in baseline.iterrows():
    fname = brow["file"]
    model = brow["model"]
    bench_col = MODEL_MAP.get(model)
    if bench_col is None or fname not in bench_filenames:
        continue

    # Get benchmark values
    bench_auc_roc_row = bench_auc_roc[bench_auc_roc["filename"] == fname]
    bench_auc_pr_row = bench_auc_pr[bench_auc_pr["filename"] == fname]
    bench_f1_row = bench_f1[bench_f1["filename"] == fname]

    if bench_auc_roc_row.empty:
        continue

    bench_auc_roc_val = bench_auc_roc_row[bench_col].values[0]
    bench_auc_pr_val = bench_auc_pr_row[bench_col].values[0] if not bench_auc_pr_row.empty else np.nan
    bench_f1_val = bench_f1_row[bench_col].values[0] if not bench_f1_row.empty else np.nan

    rows.append({
        "file": fname,
        "model": model,
        "folder": brow["folder"],
        "baseline_AUC_ROC": brow["AUC_ROC"],
        "benchmark_AUC_ROC": bench_auc_roc_val,
        "diff_AUC_ROC": brow["AUC_ROC"] - bench_auc_roc_val,
        "baseline_AUC_PR": brow["AUC_PR"],
        "benchmark_AUC_PR": bench_auc_pr_val,
        "diff_AUC_PR": brow["AUC_PR"] - bench_auc_pr_val,
        "baseline_F1": brow["F1"],
        "benchmark_F1": bench_f1_val,
        "diff_F1": brow["F1"] - bench_f1_val,
    })

comp = pd.DataFrame(rows)
print(f"\nComparison rows: {len(comp)}")

# === SUMMARY STATISTICS ===
print("\n" + "="*70)
print("OVERALL COMPARISON: Mean metrics per model")
print("="*70)

for model in sorted(comp["model"].unique()):
    m = comp[comp["model"] == model]
    print(f"\n--- {model} ---")
    for metric in ["AUC_ROC", "AUC_PR", "F1"]:
        bl = m[f"baseline_{metric}"].mean()
        bm = m[f"benchmark_{metric}"].mean()
        diff = m[f"diff_{metric}"].mean()
        sign = "+" if diff > 0 else ""
        print(f"  {metric:10s}: Baseline={bl:.4f}  Benchmark={bm:.4f}  Diff={sign}{diff:.4f}")

print("\n" + "="*70)
print("OVERALL AVERAGES (across all models)")
print("="*70)
for metric in ["AUC_ROC", "AUC_PR", "F1"]:
    bl = comp[f"baseline_{metric}"].mean()
    bm = comp[f"benchmark_{metric}"].mean()
    diff = comp[f"diff_{metric}"].mean()
    sign = "+" if diff > 0 else ""
    print(f"  {metric:10s}: Baseline={bl:.4f}  Benchmark={bm:.4f}  Diff={sign}{diff:.4f}")

# Per-model aggregation
print("\n" + "="*70)
print("COMPARISON TABLE (Baseline avg vs Benchmark avg)")
print("="*70)
summary = comp.groupby("model").agg({
    "baseline_AUC_ROC": "mean",
    "benchmark_AUC_ROC": "mean",
    "diff_AUC_ROC": "mean",
    "baseline_AUC_PR": "mean",
    "benchmark_AUC_PR": "mean",
    "diff_AUC_PR": "mean",
    "baseline_F1": "mean",
    "benchmark_F1": "mean",
    "diff_F1": "mean",
}).round(4)
print(summary.to_string())

# Per-folder (dataset family) comparison
print("\n" + "="*70)
print("PER DATASET FAMILY: Mean AUC-ROC (Baseline vs Benchmark)")
print("="*70)
folder_summary = comp.groupby("folder").agg({
    "baseline_AUC_ROC": "mean",
    "benchmark_AUC_ROC": "mean",
    "diff_AUC_ROC": "mean",
}).round(4)
print(folder_summary.to_string())

# Percentage of cases where baseline > benchmark
print("\n" + "="*70)
print("WIN RATE: % of cases baseline >= benchmark")
print("="*70)
for metric in ["AUC_ROC", "AUC_PR", "F1"]:
    wins = (comp[f"diff_{metric}"] >= 0).sum()
    total = len(comp)
    pct = wins / total * 100
    print(f"  {metric:10s}: {wins}/{total} ({pct:.1f}%)")

# Per-model win rate
print("\n  Per model:")
for model in sorted(comp["model"].unique()):
    m = comp[comp["model"] == model]
    for metric in ["AUC_ROC", "AUC_PR", "F1"]:
        wins = (m[f"diff_{metric}"] >= 0).sum()
        total = len(m)
        pct = wins / total * 100
        print(f"    {model:12s} {metric:10s}: {wins}/{total} ({pct:.1f}%)")

# Show biggest differences
print("\n" + "="*70)
print("TOP 10 BIGGEST POSITIVE DIFFERENCES (Baseline > Benchmark) - AUC_ROC")
print("="*70)
top_pos = comp.nlargest(10, "diff_AUC_ROC")[["file", "model", "folder", "baseline_AUC_ROC", "benchmark_AUC_ROC", "diff_AUC_ROC"]]
print(top_pos.to_string(index=False))

print("\n" + "="*70)
print("TOP 10 BIGGEST NEGATIVE DIFFERENCES (Benchmark > Baseline) - AUC_ROC")
print("="*70)
top_neg = comp.nsmallest(10, "diff_AUC_ROC")[["file", "model", "folder", "baseline_AUC_ROC", "benchmark_AUC_ROC", "diff_AUC_ROC"]]
print(top_neg.to_string(index=False))

# Save comparison
output_path = r"results\tables\baseline_vs_benchmark_comparison.csv"
comp.to_csv(output_path, index=False)
print(f"\nFull comparison saved to: {output_path}")
