import pandas as pd
import glob
import numpy as np

# --- 1. Load our results ---
our_file = glob.glob('**/run_IForest_Wrapper/results_final_tsb_ad.csv', recursive=True)[0]
our_df = pd.read_csv(our_file)
print(f"=== OUR RESULTS ===")
print(f"Total rows: {len(our_df)}")
print(f"Unique files: {our_df['file'].nunique()}")
print(f"Our mean VUS-PR: {our_df['VUS_PR'].mean():.6f}")

# --- 2. Load their published per-file results ---
their_df = pd.read_csv('TSB-AD/benchmark_exp/benchmark_eval_results/uni_mergedTable_VUS-PR.csv')
print(f"\n=== THEIR PUBLISHED RESULTS ===")
print(f"Total rows: {len(their_df)}")
print(f"Their mean VUS-PR (IForest): {their_df['IForest'].mean():.6f}")
print(f"Their IForest non-null count: {their_df['IForest'].notna().sum()}")

# --- 3. Per-file comparison ---
# Their file names are in first column
their_df = their_df.rename(columns={their_df.columns[0]: 'file'})
their_df['file_base'] = their_df['file'].apply(lambda x: x.split('/')[-1] if '/' in str(x) else x)

our_df['file_base'] = our_df['file']

# Merge
merged = our_df.merge(their_df[['file_base', 'IForest']], on='file_base', how='outer', suffixes=('_ours', '_theirs'))
merged = merged.rename(columns={'IForest': 'VUS_PR_theirs', 'VUS_PR': 'VUS_PR_ours'})

print(f"\n=== MERGE RESULTS ===")
print(f"Total merged rows: {len(merged)}")
print(f"Only in ours: {merged['VUS_PR_ours'].notna().sum() - merged[['VUS_PR_ours', 'VUS_PR_theirs']].dropna().shape[0]}")
print(f"Only in theirs: {merged['VUS_PR_theirs'].notna().sum() - merged[['VUS_PR_ours', 'VUS_PR_theirs']].dropna().shape[0]}")
print(f"In both: {merged[['VUS_PR_ours', 'VUS_PR_theirs']].dropna().shape[0]}")

# --- 4. Compute differences ---
both = merged.dropna(subset=['VUS_PR_ours', 'VUS_PR_theirs'])
both['diff'] = both['VUS_PR_ours'] - both['VUS_PR_theirs']
both['abs_diff'] = both['diff'].abs()

print(f"\n=== DIFFERENCES (ours - theirs) ===")
print(f"Mean diff: {both['diff'].mean():.6f}")
print(f"Median diff: {both['diff'].median():.6f}")
print(f"Max abs diff: {both['abs_diff'].max():.6f}")
print(f"Files with exact match (diff=0): {(both['abs_diff'] < 1e-10).sum()}")
print(f"Files with diff > 0.01: {(both['abs_diff'] > 0.01).sum()}")
print(f"Files with diff > 0.05: {(both['abs_diff'] > 0.05).sum()}")
print(f"Files with diff > 0.10: {(both['abs_diff'] > 0.10).sum()}")

# --- 5. Show top 20 largest differences ---
print(f"\n=== TOP 20 LARGEST DIFFERENCES ===")
top20 = both.nlargest(20, 'abs_diff')[['file_base', 'VUS_PR_ours', 'VUS_PR_theirs', 'diff']]
print(top20.to_string(index=False))

# --- 6. Files missing from one side ---
only_ours = merged[merged['VUS_PR_theirs'].isna()]
only_theirs = merged[merged['VUS_PR_ours'].isna()]
if len(only_ours) > 0:
    print(f"\n=== FILES ONLY IN OURS ({len(only_ours)}) ===")
    print(only_ours['file_base'].tolist())
if len(only_theirs) > 0:
    print(f"\n=== FILES ONLY IN THEIRS ({len(only_theirs)}) ===")
    print(only_theirs['file_base'].tolist())
