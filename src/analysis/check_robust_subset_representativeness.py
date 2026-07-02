"""
Analysis script to check if the robust subset (150 datasets) is representative 
of the full benchmark.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Paths
ROOT = Path(__file__).resolve().parents[2]
FULL_BENCHMARK_PATH = ROOT / 'TSB-UAD' / 'result' / 'accuracy_table' / 'mergedTable_AUC_ROC.csv'
ROBUST_SUBSET_PATH = ROOT / 'results' / 'tables' / 'robust_subset_TSB.csv'

# Parameters
SPECIAL_FOLDERS = ['YAHOO', 'ECG', 'GHL', 'MITDB', 'SVDB', 'Occupancy', 'NASA-SMAP', 'NASA-MSL']
AUC_COLUMNS = ['IFOREST', 'LOF', 'MP', 'NORMA', 'IFOREST1', 'HBOS', 'OCSVM', 'PCA', 'AE', 'CNN', 'LSTM', 'POLY']

print("=" * 80)
print("ROBUST SUBSET REPRESENTATIVENESS ANALYSIS")
print("=" * 80)
print()

# Load data
print("Loading data...")
benchmark_df = pd.read_csv(FULL_BENCHMARK_PATH)
robust_subset_df = pd.read_csv(ROBUST_SUBSET_PATH)

print(f"Full benchmark shape: {benchmark_df.shape}")
print(f"Robust subset shape: {robust_subset_df.shape}")
print()

# Process full benchmark
print("Processing full benchmark...")

# Compute mean AUC_ROC
benchmark_df['mean_AUC_ROC'] = benchmark_df[AUC_COLUMNS].mean(axis=1)

# Add folder column
benchmark_df['folder'] = benchmark_df['dataset'].str.replace('_', '-')

# Apply filters (same as create_robust_subset.py)
print(f"Applying filters...")
filtered_benchmark = benchmark_df.copy()

# Filter: keep SPECIAL_FOLDERS or (length between 1000-100000 AND ratio <= 0.20)
def apply_robust_filters(df):
    """Apply filters based on create_robust_subset.py logic"""
    mask = df['folder'].isin(SPECIAL_FOLDERS) | (
        (df['length'] >= 1000) & 
        (df['length'] <= 100000) & 
        (df['ratio'] <= 0.20)
    )
    return df[mask].copy()

filtered_benchmark = apply_robust_filters(benchmark_df)
print(f"Benchmark after filtering: {len(filtered_benchmark)} datasets (from {len(benchmark_df)})")

# Add difficulty terciles for filtered benchmark
q33 = filtered_benchmark['mean_AUC_ROC'].quantile(1/3)
q66 = filtered_benchmark['mean_AUC_ROC'].quantile(2/3)

filtered_benchmark['difficulty'] = pd.cut(
    filtered_benchmark['mean_AUC_ROC'],
    bins=[-np.inf, q33, q66, np.inf],
    labels=['Hard', 'Medium', 'Easy'],
    include_lowest=True
)

# Process robust subset
print("Processing robust subset...")
robust_subset_df['mean_AUC_ROC'] = robust_subset_df[AUC_COLUMNS].mean(axis=1)
robust_subset_df['folder'] = robust_subset_df['dataset'].str.replace('_', '-')

# Add difficulty terciles using same boundaries as filtered benchmark
robust_subset_df['difficulty'] = pd.cut(
    robust_subset_df['mean_AUC_ROC'],
    bins=[-np.inf, q33, q66, np.inf],
    labels=['Hard', 'Medium', 'Easy'],
    include_lowest=True
)

print(f"Robust subset size: {len(robust_subset_df)}")
print()

# Ensure robust subset is a subset of filtered benchmark
robust_datasets = set(robust_subset_df['dataset'].values)
filtered_datasets = set(filtered_benchmark['dataset'].values)
in_filtered = robust_datasets.issubset(filtered_datasets)
print(f"Robust subset is subset of filtered benchmark: {in_filtered}")
if not in_filtered:
    missing = robust_datasets - filtered_datasets
    print(f"  WARNING: {len(missing)} datasets in robust subset but not in filtered benchmark")
    print(f"  Examples: {list(missing)[:5]}")
print()

# COMPARISON ANALYSIS
print("=" * 80)
print("DISTRIBUTION COMPARISONS")
print("=" * 80)
print()

# 1. Difficulty Distribution
print("1. DIFFICULTY DISTRIBUTION")
print("-" * 40)
difficulty_bench = filtered_benchmark['difficulty'].value_counts(normalize=True).sort_index() * 100
difficulty_robust = robust_subset_df['difficulty'].value_counts(normalize=True).sort_index() * 100

comparison_difficulty = pd.DataFrame({
    'Full Benchmark %': difficulty_bench,
    'Robust Subset %': difficulty_robust,
    'Difference': difficulty_robust - difficulty_bench
})
print(comparison_difficulty.round(2))
print()

# Chi-squared test for difficulty
contingency_difficulty = pd.crosstab(
    pd.Series(['Benchmark']*len(filtered_benchmark) + ['Robust']*len(robust_subset_df)),
    pd.concat([filtered_benchmark['difficulty'], robust_subset_df['difficulty']], ignore_index=True)
)
chi2_diff, p_diff, dof_diff, expected_diff = chi2_contingency(contingency_difficulty)
print(f"Chi-squared test (Difficulty):")
print(f"  Chi2 = {chi2_diff:.4f}, p-value = {p_diff:.4f}, dof = {dof_diff}")
print(f"  Result: {'REPRESENTATIVE' if p_diff > 0.05 else 'NOT REPRESENTATIVE'} (α=0.05)")
print()

# 2. Anomaly Type Distribution
print("2. ANOMALY TYPE (type_an) DISTRIBUTION")
print("-" * 40)
if 'type_an' in filtered_benchmark.columns:
    type_an_bench = filtered_benchmark['type_an'].value_counts(normalize=True).sort_index() * 100
    type_an_robust = robust_subset_df['type_an'].value_counts(normalize=True).sort_index() * 100
    
    # Align indices
    all_types = sorted(set(type_an_bench.index) | set(type_an_robust.index))
    type_an_bench = type_an_bench.reindex(all_types, fill_value=0)
    type_an_robust = type_an_robust.reindex(all_types, fill_value=0)
    
    comparison_type = pd.DataFrame({
        'Full Benchmark %': type_an_bench,
        'Robust Subset %': type_an_robust,
        'Difference': type_an_robust - type_an_bench
    })
    print(comparison_type.round(2))
    print()
    
    # Chi-squared test for anomaly type
    contingency_type = pd.crosstab(
        pd.Series(['Benchmark']*len(filtered_benchmark) + ['Robust']*len(robust_subset_df)),
        pd.concat([filtered_benchmark['type_an'], robust_subset_df['type_an']], ignore_index=True)
    )
    chi2_type, p_type, dof_type, expected_type = chi2_contingency(contingency_type)
    print(f"Chi-squared test (Anomaly Type):")
    print(f"  Chi2 = {chi2_type:.4f}, p-value = {p_type:.4f}, dof = {dof_type}")
    print(f"  Result: {'REPRESENTATIVE' if p_type > 0.05 else 'NOT REPRESENTATIVE'} (α=0.05)")
else:
    print("Column 'type_an' not found in data")
print()

# 3. Folder Distribution
print("3. FOLDER DISTRIBUTION")
print("-" * 40)
folder_bench = filtered_benchmark['folder'].value_counts(normalize=True).sort_index() * 100
folder_robust = robust_subset_df['folder'].value_counts(normalize=True).sort_index() * 100

# Align indices
all_folders = sorted(set(folder_bench.index) | set(folder_robust.index))
folder_bench = folder_bench.reindex(all_folders, fill_value=0)
folder_robust = folder_robust.reindex(all_folders, fill_value=0)

comparison_folder = pd.DataFrame({
    'Full Benchmark %': folder_bench,
    'Robust Subset %': folder_robust,
    'Difference': folder_robust - folder_bench
})
print(comparison_folder.round(2))
print()

# Chi-squared test for folders
contingency_folder = pd.crosstab(
    pd.Series(['Benchmark']*len(filtered_benchmark) + ['Robust']*len(robust_subset_df)),
    pd.concat([filtered_benchmark['folder'], robust_subset_df['folder']], ignore_index=True)
)
chi2_folder, p_folder, dof_folder, expected_folder = chi2_contingency(contingency_folder)
print(f"Chi-squared test (Folder):")
print(f"  Chi2 = {chi2_folder:.4f}, p-value = {p_folder:.4f}, dof = {dof_folder}")
print(f"  Result: {'REPRESENTATIVE' if p_folder > 0.05 else 'NOT REPRESENTATIVE'} (α=0.05)")
print()

# 4. Mean AUC_ROC Statistics
print("4. MEAN AUC_ROC STATISTICS")
print("-" * 40)
auc_stats = pd.DataFrame({
    'Full Benchmark': [
        filtered_benchmark['mean_AUC_ROC'].mean(),
        filtered_benchmark['mean_AUC_ROC'].median(),
        filtered_benchmark['mean_AUC_ROC'].std(),
        filtered_benchmark['mean_AUC_ROC'].min(),
        filtered_benchmark['mean_AUC_ROC'].max(),
    ],
    'Robust Subset': [
        robust_subset_df['mean_AUC_ROC'].mean(),
        robust_subset_df['mean_AUC_ROC'].median(),
        robust_subset_df['mean_AUC_ROC'].std(),
        robust_subset_df['mean_AUC_ROC'].min(),
        robust_subset_df['mean_AUC_ROC'].max(),
    ]
}, index=['Mean', 'Median', 'Std Dev', 'Min', 'Max'])
print(auc_stats.round(4))
print()

# 5. Data Length Statistics
print("5. DATA LENGTH STATISTICS")
print("-" * 40)
length_stats = pd.DataFrame({
    'Full Benchmark': [
        filtered_benchmark['length'].mean(),
        filtered_benchmark['length'].median(),
        filtered_benchmark['length'].std(),
        filtered_benchmark['length'].min(),
        filtered_benchmark['length'].max(),
    ],
    'Robust Subset': [
        robust_subset_df['length'].mean(),
        robust_subset_df['length'].median(),
        robust_subset_df['length'].std(),
        robust_subset_df['length'].min(),
        robust_subset_df['length'].max(),
    ]
}, index=['Mean', 'Median', 'Std Dev', 'Min', 'Max'])
print(length_stats.round(0))
print()

# 6. Anomaly Ratio Statistics (if available)
print("6. ANOMALY RATIO (RATIO) STATISTICS")
print("-" * 40)
if 'ratio' in filtered_benchmark.columns:
    ratio_stats = pd.DataFrame({
        'Full Benchmark': [
            filtered_benchmark['ratio'].mean(),
            filtered_benchmark['ratio'].median(),
            filtered_benchmark['ratio'].std(),
            filtered_benchmark['ratio'].min(),
            filtered_benchmark['ratio'].max(),
        ],
        'Robust Subset': [
            robust_subset_df['ratio'].mean(),
            robust_subset_df['ratio'].median(),
            robust_subset_df['ratio'].std(),
            robust_subset_df['ratio'].min(),
            robust_subset_df['ratio'].max(),
        ]
    }, index=['Mean', 'Median', 'Std Dev', 'Min', 'Max'])
    print(ratio_stats.round(4))
else:
    print("Column 'ratio' not found in data")
print()

# SUMMARY
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"Full Benchmark Size: {len(filtered_benchmark)} datasets")
print(f"Robust Subset Size: {len(robust_subset_df)} datasets")
print(f"Subset Percentage: {len(robust_subset_df)/len(filtered_benchmark)*100:.1f}%")
print()
print("REPRESENTATIVENESS ASSESSMENT (Chi-squared tests, α=0.05):")
print(f"  Difficulty distribution: {'✓ REPRESENTATIVE' if p_diff > 0.05 else '✗ NOT REPRESENTATIVE'} (p={p_diff:.4f})")
print(f"  Anomaly type distribution: {'✓ REPRESENTATIVE' if p_type > 0.05 else '✗ NOT REPRESENTATIVE'} (p={p_type:.4f})")
print(f"  Folder distribution: {'✓ REPRESENTATIVE' if p_folder > 0.05 else '✗ NOT REPRESENTATIVE'} (p={p_folder:.4f})")
print()

all_representative = (p_diff > 0.05) and (p_type > 0.05) and (p_folder > 0.05)
if all_representative:
    print("CONCLUSION: The robust subset appears to be REPRESENTATIVE of the full benchmark.")
else:
    print("CONCLUSION: The robust subset shows SIGNIFICANT DIFFERENCES from the full benchmark.")
    if p_diff <= 0.05:
        print(f"  - Difficulty distribution differs (p={p_diff:.4f})")
    if p_type <= 0.05:
        print(f"  - Anomaly type distribution differs (p={p_type:.4f})")
    if p_folder <= 0.05:
        print(f"  - Folder distribution differs (p={p_folder:.4f})")
print()
print("=" * 80)
