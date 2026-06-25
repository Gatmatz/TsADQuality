"""
Statistical Tests for Spikes (Normal Only) Experiment

Tests:
  1. Wilcoxon signed-rank test -- per (model, fraction, multiplier):
     is the AUC drop from baseline statistically significant?
  2. Cliff's delta -- effect size per (model, fraction, multiplier)
  3. Friedman test -- per (fraction, multiplier): do models differ?
     + Nemenyi post-hoc for pairwise comparisons
  4. Ranking stability -- does the best model change under spikes?

Output:
  - wilcoxon_results.csv
  - cliffs_delta_results.csv
  - friedman_results.csv
  - ranking_stability.csv

Usage:
    python statistical_tests_spikes.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "spikes_normal_only")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "statistical_tests")


def load_data():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    df = df[df['error'].isna()]
    bl = pd.read_csv(BASELINE_CSV)
    return df, bl


def cliffs_delta(x, y):
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, 'negligible'
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
    print("\n" + "=" * 80)
    print("  1. WILCOXON SIGNED-RANK TEST (spikes vs baseline)")
    print("=" * 80)

    rows = []
    models = sorted(df['model'].unique())
    fractions = sorted(df['fraction'].unique())
    multipliers = sorted(df['multiplier'].unique())

    n_tests = len(models) * len(fractions) * len(multipliers)

    for model in models:
        bl_model = bl[bl['model'] == model].set_index('file')['AUC_ROC']

        for mult in multipliers:
            for frac in fractions:
                corrupted = df[(df['model'] == model) &
                               (df['fraction'] == frac) &
                               (df['multiplier'] == mult)]
                corr_auc = corrupted.set_index('file')['AUC_ROC']

                common = bl_model.index.intersection(corr_auc.index)
                if len(common) < 10:
                    continue

                baseline_vals = bl_model.loc[common].values
                corrupted_vals = corr_auc.loc[common].values
                diff = baseline_vals - corrupted_vals

                non_zero = diff[diff != 0]
                if len(non_zero) < 5:
                    rows.append({
                        'model': model, 'fraction': frac, 'multiplier': int(mult),
                        'n_paired': len(common),
                        'mean_baseline_auc': round(np.mean(baseline_vals), 4),
                        'mean_corrupted_auc': round(np.mean(corrupted_vals), 4),
                        'mean_drop': round(np.mean(diff), 4),
                        'median_drop': round(np.median(diff), 4),
                        'pct_drop': 0.0,
                        'wilcoxon_stat': None, 'p_value': 1.0,
                        'p_value_bonferroni': 1.0,
                        'significant_bonferroni': False,
                    })
                    continue

                stat, p_value = stats.wilcoxon(non_zero, alternative='two-sided')

                mean_drop = np.mean(diff)
                mean_bl = np.mean(baseline_vals)
                mean_corr = np.mean(corrupted_vals)
                pct_drop = (mean_drop / mean_bl * 100) if mean_bl > 0 else 0

                rows.append({
                    'model': model, 'fraction': frac, 'multiplier': int(mult),
                    'n_paired': len(common),
                    'mean_baseline_auc': round(mean_bl, 4),
                    'mean_corrupted_auc': round(mean_corr, 4),
                    'mean_drop': round(mean_drop, 4),
                    'median_drop': round(np.median(diff), 4),
                    'pct_drop': round(pct_drop, 2),
                    'wilcoxon_stat': round(stat, 1),
                    'p_value': p_value,
                    'p_value_bonferroni': min(p_value * n_tests, 1.0),
                    'significant_bonferroni': p_value < (0.05 / n_tests),
                })

    df_results = pd.DataFrame(rows)

    for mult in multipliers:
        sub = df_results[df_results['multiplier'] == int(mult)]
        print(f"\n--- Multiplier = {int(mult)} ---")
        print(f"{'Model':<10} {'Frac':>6} {'Baseline':>9} {'Corrupt':>9} "
              f"{'Drop':>7} {'%Drop':>6} {'p-value':>10} {'Sig?':>5}")
        print("-" * 72)
        for _, r in sub.iterrows():
            sig = "***" if r['p_value_bonferroni'] < 0.001 else \
                  "**" if r['p_value_bonferroni'] < 0.01 else \
                  "*" if r['p_value_bonferroni'] < 0.05 else "ns"
            print(f"{r['model']:<10} {r['fraction']:>6.2f} {r['mean_baseline_auc']:>9.4f} "
                  f"{r['mean_corrupted_auc']:>9.4f} {r['mean_drop']:>+7.4f} "
                  f"{r['pct_drop']:>5.1f}% {r['p_value']:.2e} {sig:>5}")

    return df_results


# ================================================================
# 2. CLIFF'S DELTA
# ================================================================
def run_cliffs_delta(df, bl):
    print("\n" + "=" * 80)
    print("  2. CLIFF'S DELTA (effect size)")
    print("=" * 80)

    rows = []
    models = sorted(df['model'].unique())
    fractions = sorted(df['fraction'].unique())
    multipliers = sorted(df['multiplier'].unique())

    for model in models:
        bl_model = bl[bl['model'] == model].set_index('file')['AUC_ROC']

        for mult in multipliers:
            for frac in fractions:
                corrupted = df[(df['model'] == model) &
                               (df['fraction'] == frac) &
                               (df['multiplier'] == mult)]
                corr_auc = corrupted.set_index('file')['AUC_ROC']

                common = bl_model.index.intersection(corr_auc.index)
                if len(common) < 10:
                    continue

                delta, magnitude = cliffs_delta(
                    bl_model.loc[common].values,
                    corr_auc.loc[common].values)

                rows.append({
                    'model': model, 'fraction': frac, 'multiplier': int(mult),
                    'n': len(common),
                    'cliffs_delta': round(delta, 4),
                    'magnitude': magnitude,
                })

    df_results = pd.DataFrame(rows)

    for mult in multipliers:
        sub = df_results[df_results['multiplier'] == int(mult)]
        print(f"\n--- Multiplier = {int(mult)} ---")
        print(f"{'Model':<10} {'Frac':>6} {'delta':>8} {'Magnitude':<12}")
        print("-" * 40)
        for _, r in sub.iterrows():
            print(f"{r['model']:<10} {r['fraction']:>6.2f} {r['cliffs_delta']:>+8.4f} {r['magnitude']:<12}")

    return df_results


# ================================================================
# 3. FRIEDMAN TEST
# ================================================================
def run_friedman(df, bl):
    print("\n" + "=" * 80)
    print("  3. FRIEDMAN TEST (model comparison per condition)")
    print("=" * 80)

    models = sorted(df['model'].unique())
    fractions = sorted(df['fraction'].unique())
    multipliers = sorted(df['multiplier'].unique())

    rows = []
    for mult in multipliers:
        for frac in fractions:
            sub = df[(df['fraction'] == frac) & (df['multiplier'] == mult)]
            pivot = sub.pivot_table(
                index='file', columns='model', values='AUC_ROC', aggfunc='mean')
            pivot = pivot.dropna()

            model_names = [m for m in models if m in pivot.columns]
            if len(pivot) < 10 or len(model_names) < 2:
                continue

            model_arrays = [pivot[m].values for m in model_names]
            stat, p_value = stats.friedmanchisquare(*model_arrays)

            # Mean ranks
            ranks = np.zeros((len(pivot), len(model_names)))
            for i in range(len(pivot)):
                ranks[i] = stats.rankdata(-pivot[model_names].values[i])
            mean_ranks = ranks.mean(axis=0)
            rank_dict = {m: round(r, 3) for m, r in zip(model_names, mean_ranks)}
            best_model = min(rank_dict, key=rank_dict.get)

            rows.append({
                'fraction': frac, 'multiplier': int(mult),
                'n_files': len(pivot), 'n_models': len(model_names),
                'friedman_stat': round(stat, 2),
                'p_value': p_value,
                'significant': p_value < 0.05,
                'best_model': best_model,
                **{f'rank_{m}': rank_dict.get(m) for m in models},
            })

    df_results = pd.DataFrame(rows)

    print(f"\n{'Mult':>5} {'Frac':>6} {'n':>5} {'chi2':>8} {'p-value':>10} {'Sig?':>5} {'Best':<10} " +
          " ".join(f"{m:>8}" for m in models))
    print("-" * (55 + 9 * len(models)))
    for _, r in df_results.iterrows():
        sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
        rank_str = " ".join(f"{r.get(f'rank_{m}', 'N/A'):>8}" for m in models)
        print(f"{int(r['multiplier']):>5} {r['fraction']:>6.2f} {int(r['n_files']):>5} "
              f"{r['friedman_stat']:>8.2f} {r['p_value']:.2e} {sig:>5} {r['best_model']:<10} {rank_str}")

    # Nemenyi post-hoc
    print("\n--- Nemenyi Post-hoc (where Friedman is significant) ---")
    for _, r in df_results.iterrows():
        if not r['significant']:
            continue
        frac, mult = r['fraction'], r['multiplier']
        sub = df[(df['fraction'] == frac) & (df['multiplier'] == mult)]
        pivot = sub.pivot_table(
            index='file', columns='model', values='AUC_ROC', aggfunc='mean').dropna()
        model_names_avail = [m for m in models if m in pivot.columns]
        if len(model_names_avail) < 2:
            continue

        k = len(model_names_avail)
        n = len(pivot)
        q_alpha = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

        ranks = np.zeros((n, k))
        for i in range(n):
            ranks[i] = stats.rankdata(-pivot[model_names_avail].values[i])
        mean_ranks = ranks.mean(axis=0)

        print(f"\n  frac={frac}, mult={int(mult)}: CD={cd:.3f}")
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
    print("\n" + "=" * 80)
    print("  4. RANKING STABILITY")
    print("=" * 80)

    models = sorted(df['model'].unique())
    fractions = sorted(df['fraction'].unique())
    multipliers = sorted(df['multiplier'].unique())

    rows = []

    # Baseline ranking
    bl_means = {}
    for model in models:
        bl_m = bl[bl['model'] == model]
        if not bl_m.empty:
            bl_means[model] = bl_m['AUC_ROC'].mean()

    if bl_means:
        bl_ranking = sorted(bl_means, key=bl_means.get, reverse=True)
        rows.append({
            'condition': 'baseline', 'fraction': None, 'multiplier': None,
            **{f'rank_{i+1}': m for i, m in enumerate(bl_ranking)},
            **{f'auc_{m}': round(bl_means[m], 4) for m in models},
        })

    for mult in multipliers:
        for frac in fractions:
            cond_means = {}
            for model in models:
                sub = df[(df['model'] == model) & (df['fraction'] == frac) & (df['multiplier'] == mult)]
                if not sub.empty:
                    cond_means[model] = sub['AUC_ROC'].mean()

            if cond_means:
                ranking = sorted(cond_means, key=cond_means.get, reverse=True)
                rows.append({
                    'condition': f'f{frac}_m{int(mult)}',
                    'fraction': frac, 'multiplier': int(mult),
                    **{f'rank_{i+1}': m for i, m in enumerate(ranking)},
                    **{f'auc_{m}': round(cond_means[m], 4) for m in models if m in cond_means},
                })

    df_results = pd.DataFrame(rows)

    print(f"\n{'Condition':<15} " + " ".join(f"{'#'+str(i+1):>10}" for i in range(len(models))) +
          " | " + " ".join(f"{m:>10}" for m in models))
    print("-" * (16 + 11 * len(models) * 2 + 3))
    for _, r in df_results.iterrows():
        ranks = " ".join(f"{r.get(f'rank_{i+1}', 'N/A'):>10}" for i in range(len(models)))
        aucs = " ".join(f"{r.get(f'auc_{m}', 0):>10.4f}" for m in models)
        print(f"{r['condition']:<15} {ranks} | {aucs}")

    if len(df_results) > 1:
        first_places = df_results['rank_1'].dropna().tolist()
        n_changes = sum(1 for i in range(1, len(first_places)) if first_places[i] != first_places[i-1])
        print(f"\nRanking changes (#1 position): {n_changes}/{len(first_places)-1}")

    return df_results


# ================================================================
# MAIN
# ================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df, bl = load_data()
    print(f"  Experiment: {len(df)} rows, models: {sorted(df['model'].unique())}")
    print(f"  Fractions: {sorted(df['fraction'].unique())}")
    print(f"  Multipliers: {sorted(df['multiplier'].unique())}")
    print(f"  Baseline: {len(bl)} rows")

    wilcoxon_df = run_wilcoxon(df, bl)
    wilcoxon_df.to_csv(os.path.join(OUTPUT_DIR, "wilcoxon_results.csv"), index=False)

    cliffs_df = run_cliffs_delta(df, bl)
    cliffs_df.to_csv(os.path.join(OUTPUT_DIR, "cliffs_delta_results.csv"), index=False)

    friedman_df = run_friedman(df, bl)
    friedman_df.to_csv(os.path.join(OUTPUT_DIR, "friedman_results.csv"), index=False)

    ranking_df = run_ranking_stability(df, bl)
    ranking_df.to_csv(os.path.join(OUTPUT_DIR, "ranking_stability.csv"), index=False)

    print(f"\n{'=' * 80}")
    print(f"  All results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
