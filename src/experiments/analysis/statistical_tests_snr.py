"""
Statistical Tests for White Noise (SNR) Experiment

Tests:
  1. Wilcoxon signed-rank test — per (model, SNR): is the AUC drop
     from baseline statistically significant?
  2. Friedman test — per SNR: do models differ significantly?
     + Nemenyi post-hoc for pairwise comparisons
  3. Cliff's delta — effect size per (model, SNR): how large is the drop?
  4. Model ranking stability — does the best model change under noise?

Output:
  - wilcoxon_results.csv
  - friedman_results.csv
  - cliffs_delta_results.csv
  - ranking_stability.csv
  - Console summary

Usage:
    python statistical_tests_snr.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "white_noise_snr")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "statistical_tests")


def load_data():
    """Load experiment + baseline, return paired per-file data."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    df = df[df['error'].isna()]

    bl = pd.read_csv(BASELINE_CSV)

    return df, bl


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size.

    δ = (#{x_i > y_j} - #{x_i < y_j}) / (n_x * n_y)

    Interpretation (Romano et al. 2006):
      |δ| < 0.147  -> negligible
      |δ| < 0.33   -> small
      |δ| < 0.474  -> medium
      |δ| >= 0.474 -> large
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, 'negligible'

    # Vectorized computation
    more = np.sum(np.array(x)[:, None] > np.array(y)[None, :])
    less = np.sum(np.array(x)[:, None] < np.array(y)[None, :])
    delta = (more - less) / (n_x * n_y)

    abs_d = abs(delta)
    if abs_d < 0.147:
        magnitude = 'negligible'
    elif abs_d < 0.33:
        magnitude = 'small'
    elif abs_d < 0.474:
        magnitude = 'medium'
    else:
        magnitude = 'large'

    return delta, magnitude


# ================================================================
# 1. WILCOXON SIGNED-RANK TEST
# ================================================================
def run_wilcoxon(df, bl):
    """For each (model, SNR), test if AUC differs from baseline.

    Paired test: same file under baseline vs corruption.
    H0: median difference = 0
    H1: median difference ≠ 0
    """
    print("\n" + "=" * 70)
    print("  1. WILCOXON SIGNED-RANK TEST (corruption vs baseline)")
    print("=" * 70)

    rows = []
    models = sorted(df['model'].unique())
    snr_levels = sorted(df['snr_db'].unique(), reverse=True)

    for model in models:
        bl_model = bl[bl['model'] == model].set_index('file')['AUC_ROC']

        for snr in snr_levels:
            corrupted = df[(df['model'] == model) & (df['snr_db'] == snr)]
            corr_auc = corrupted.set_index('file')['AUC_ROC']

            # Align on common files
            common = bl_model.index.intersection(corr_auc.index)
            if len(common) < 10:
                continue

            baseline_vals = bl_model.loc[common].values
            corrupted_vals = corr_auc.loc[common].values
            diff = baseline_vals - corrupted_vals

            # Wilcoxon requires non-zero differences
            non_zero = diff[diff != 0]
            if len(non_zero) < 5:
                continue

            stat, p_value = stats.wilcoxon(non_zero, alternative='two-sided')

            mean_drop = np.mean(diff)
            median_drop = np.median(diff)
            mean_baseline = np.mean(baseline_vals)
            mean_corrupted = np.mean(corrupted_vals)
            pct_drop = (mean_drop / mean_baseline * 100) if mean_baseline > 0 else 0

            # Significance with Bonferroni correction
            n_tests = len(models) * len(snr_levels)
            significant = p_value < (0.05 / n_tests)

            rows.append({
                'model': model,
                'snr_db': snr,
                'n_paired': len(common),
                'mean_baseline_auc': round(mean_baseline, 4),
                'mean_corrupted_auc': round(mean_corrupted, 4),
                'mean_drop': round(mean_drop, 4),
                'median_drop': round(median_drop, 4),
                'pct_drop': round(pct_drop, 2),
                'wilcoxon_stat': round(stat, 1),
                'p_value': p_value,
                'p_value_bonferroni': min(p_value * n_tests, 1.0),
                'significant_bonferroni': significant,
            })

    df_results = pd.DataFrame(rows)

    # Print summary
    print(f"\n{'Model':<10} {'SNR':>5} {'Baseline':>9} {'Corrupt':>9} "
          f"{'Drop':>7} {'%Drop':>6} {'p-value':>10} {'Sig?':>5}")
    print("-" * 72)
    for _, r in df_results.iterrows():
        sig = "***" if r['p_value_bonferroni'] < 0.001 else \
              "**" if r['p_value_bonferroni'] < 0.01 else \
              "*" if r['p_value_bonferroni'] < 0.05 else "ns"
        print(f"{r['model']:<10} {int(r['snr_db']):>5} {r['mean_baseline_auc']:>9.4f} "
              f"{r['mean_corrupted_auc']:>9.4f} {r['mean_drop']:>+7.4f} "
              f"{r['pct_drop']:>5.1f}% {r['p_value']:.2e} {sig:>5}")

    return df_results


# ================================================================
# 2. CLIFF'S DELTA (EFFECT SIZE)
# ================================================================
def run_cliffs_delta(df, bl):
    """For each (model, SNR), compute effect size of corruption."""
    print("\n" + "=" * 70)
    print("  2. CLIFF'S DELTA (effect size)")
    print("=" * 70)

    rows = []
    models = sorted(df['model'].unique())
    snr_levels = sorted(df['snr_db'].unique(), reverse=True)

    for model in models:
        bl_model = bl[bl['model'] == model].set_index('file')['AUC_ROC']

        for snr in snr_levels:
            corrupted = df[(df['model'] == model) & (df['snr_db'] == snr)]
            corr_auc = corrupted.set_index('file')['AUC_ROC']

            common = bl_model.index.intersection(corr_auc.index)
            if len(common) < 10:
                continue

            baseline_vals = bl_model.loc[common].values
            corrupted_vals = corr_auc.loc[common].values

            delta, magnitude = cliffs_delta(baseline_vals, corrupted_vals)

            rows.append({
                'model': model,
                'snr_db': snr,
                'n': len(common),
                'cliffs_delta': round(delta, 4),
                'magnitude': magnitude,
            })

    df_results = pd.DataFrame(rows)

    print(f"\n{'Model':<10} {'SNR':>5} {'δ':>8} {'Magnitude':<12}")
    print("-" * 40)
    for _, r in df_results.iterrows():
        print(f"{r['model']:<10} {int(r['snr_db']):>5} {r['cliffs_delta']:>+8.4f} {r['magnitude']:<12}")

    return df_results


# ================================================================
# 3. FRIEDMAN TEST (model comparison per SNR)
# ================================================================
def run_friedman(df, bl):
    """Per SNR level: do models differ significantly?

    Friedman test is a non-parametric repeated-measures ANOVA.
    Each file is a "subject", each model is a "treatment".
    """
    print("\n" + "=" * 70)
    print("  3. FRIEDMAN TEST (model comparison per SNR level)")
    print("=" * 70)

    models = sorted(df['model'].unique())
    snr_levels = sorted(df['snr_db'].unique(), reverse=True)

    rows = []
    for snr in snr_levels:
        # Build matrix: files × models
        pivot = df[df['snr_db'] == snr].pivot_table(
            index='file', columns='model', values='AUC_ROC', aggfunc='mean')
        pivot = pivot.dropna()

        if len(pivot) < 10 or len(pivot.columns) < 2:
            continue

        # Get arrays per model (same order, same files)
        model_arrays = [pivot[m].values for m in pivot.columns if m in models]
        model_names = [m for m in pivot.columns if m in models]

        if len(model_arrays) < 2:
            continue

        stat, p_value = stats.friedmanchisquare(*model_arrays)

        # Mean ranks
        ranks = np.zeros_like(pivot[model_names].values)
        for i in range(len(pivot)):
            ranks[i] = stats.rankdata(-pivot[model_names].values[i])  # lower rank = better
        mean_ranks = ranks.mean(axis=0)

        rank_dict = {m: round(r, 3) for m, r in zip(model_names, mean_ranks)}
        best_model = min(rank_dict, key=rank_dict.get)

        rows.append({
            'snr_db': snr,
            'n_files': len(pivot),
            'n_models': len(model_names),
            'friedman_stat': round(stat, 2),
            'p_value': p_value,
            'significant': p_value < 0.05,
            'best_model': best_model,
            **{f'rank_{m}': rank_dict.get(m) for m in models},
        })

    df_results = pd.DataFrame(rows)

    # Print
    rank_cols = [f'rank_{m}' for m in models if f'rank_{m}' in df_results.columns]
    print(f"\n{'SNR':>5} {'n':>5} {'χ²':>8} {'p-value':>10} {'Sig?':>5} {'Best':<10} " +
          " ".join(f"{m:>8}" for m in models))
    print("-" * (50 + 9 * len(models)))
    for _, r in df_results.iterrows():
        sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
        rank_str = " ".join(f"{r.get(f'rank_{m}', 'N/A'):>8}" for m in models)
        print(f"{int(r['snr_db']):>5} {int(r['n_files']):>5} {r['friedman_stat']:>8.2f} "
              f"{r['p_value']:.2e} {sig:>5} {r['best_model']:<10} {rank_str}")

    # Nemenyi post-hoc (if significant)
    print("\n--- Nemenyi Post-hoc (where Friedman is significant) ---")
    for _, r in df_results.iterrows():
        if not r['significant']:
            continue
        snr = r['snr_db']
        pivot = df[df['snr_db'] == snr].pivot_table(
            index='file', columns='model', values='AUC_ROC', aggfunc='mean').dropna()
        model_names_avail = [m for m in models if m in pivot.columns]
        if len(model_names_avail) < 2:
            continue

        k = len(model_names_avail)
        n = len(pivot)

        # Critical difference (Nemenyi)
        # q_alpha for k groups at alpha=0.05
        # Using approximation from scipy
        q_alpha = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

        ranks = np.zeros((n, k))
        for i in range(n):
            ranks[i] = stats.rankdata(-pivot[model_names_avail].values[i])
        mean_ranks = ranks.mean(axis=0)

        print(f"\n  SNR={int(snr)}dB: CD={cd:.3f}")
        for i, m1 in enumerate(model_names_avail):
            for j, m2 in enumerate(model_names_avail):
                if j <= i:
                    continue
                diff = abs(mean_ranks[i] - mean_ranks[j])
                sig = "SIGNIFICANT" if diff > cd else "not significant"
                print(f"    {m1} vs {m2}: |{mean_ranks[i]:.3f} - {mean_ranks[j]:.3f}| = {diff:.3f} -> {sig}")

    return df_results


# ================================================================
# 4. RANKING STABILITY
# ================================================================
def run_ranking_stability(df, bl):
    """Does the best model change under different noise levels?"""
    print("\n" + "=" * 70)
    print("  4. RANKING STABILITY")
    print("=" * 70)

    models = sorted(df['model'].unique())
    snr_levels = sorted(df['snr_db'].unique(), reverse=True)

    rows = []

    # Baseline ranking
    bl_means = {}
    for model in models:
        bl_model = bl[bl['model'] == model]
        if not bl_model.empty:
            bl_means[model] = bl_model['AUC_ROC'].mean()

    if bl_means:
        bl_ranking = sorted(bl_means, key=bl_means.get, reverse=True)
        rows.append({
            'condition': 'baseline',
            'snr_db': None,
            **{f'rank_{i+1}': m for i, m in enumerate(bl_ranking)},
            **{f'auc_{m}': round(bl_means[m], 4) for m in models},
        })

    # Per-SNR ranking
    for snr in snr_levels:
        snr_means = {}
        for model in models:
            sub = df[(df['model'] == model) & (df['snr_db'] == snr)]
            if not sub.empty:
                snr_means[model] = sub['AUC_ROC'].mean()

        if snr_means:
            ranking = sorted(snr_means, key=snr_means.get, reverse=True)
            rows.append({
                'condition': f'snr_{int(snr)}dB',
                'snr_db': snr,
                **{f'rank_{i+1}': m for i, m in enumerate(ranking)},
                **{f'auc_{m}': round(snr_means[m], 4) for m in models if m in snr_means},
            })

    df_results = pd.DataFrame(rows)

    # Print
    rank_cols = [c for c in df_results.columns if c.startswith('rank_')]
    auc_cols = [f'auc_{m}' for m in models]
    print(f"\n{'Condition':<15} " + " ".join(f"{'#'+str(i+1):>10}" for i in range(len(models))) +
          " | " + " ".join(f"{m:>10}" for m in models))
    print("-" * (16 + 11 * len(models) * 2 + 3))
    for _, r in df_results.iterrows():
        ranks = " ".join(f"{r.get(f'rank_{i+1}', 'N/A'):>10}" for i in range(len(models)))
        aucs = " ".join(f"{r.get(f'auc_{m}', 0):>10.4f}" for m in models)
        print(f"{r['condition']:<15} {ranks} | {aucs}")

    # Check if ranking changes
    if len(df_results) > 1:
        first_places = df_results['rank_1'].dropna().tolist()
        n_changes = sum(1 for i in range(1, len(first_places)) if first_places[i] != first_places[i-1])
        print(f"\nRanking changes (#1 position): {n_changes}/{len(first_places)-1}")
        print(f"#1 model by condition: {dict(zip(df_results['condition'], df_results['rank_1']))}")

    return df_results


# ================================================================
# MAIN
# ================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df, bl = load_data()
    print(f"  Experiment: {len(df)} rows, models: {sorted(df['model'].unique())}")
    print(f"  Baseline: {len(bl)} rows")

    # 1. Wilcoxon
    wilcoxon_df = run_wilcoxon(df, bl)
    wilcoxon_df.to_csv(os.path.join(OUTPUT_DIR, "wilcoxon_results.csv"), index=False)

    # 2. Cliff's delta
    cliffs_df = run_cliffs_delta(df, bl)
    cliffs_df.to_csv(os.path.join(OUTPUT_DIR, "cliffs_delta_results.csv"), index=False)

    # 3. Friedman
    friedman_df = run_friedman(df, bl)
    friedman_df.to_csv(os.path.join(OUTPUT_DIR, "friedman_results.csv"), index=False)

    # 4. Ranking stability
    ranking_df = run_ranking_stability(df, bl)
    ranking_df.to_csv(os.path.join(OUTPUT_DIR, "ranking_stability.csv"), index=False)

    print(f"\n{'=' * 70}")
    print(f"  All results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
