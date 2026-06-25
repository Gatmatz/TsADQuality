"""
Statistical Tests for Missing Values (True Impact) Experiment

Tests:
  1. Wilcoxon signed-rank test -- per (model, missing_type, fraction, num_bursts):
     is the AUC drop from baseline statistically significant?
  2. Cliff's delta -- effect size per condition
  3. Friedman test -- per condition: do models differ?
     + Nemenyi post-hoc for pairwise comparisons
  4. Ranking stability -- does the best model change under missing data?
  5. Point vs Burst comparison -- is burst worse than point at same fraction?

Output:
  - wilcoxon_results.csv
  - cliffs_delta_results.csv
  - friedman_results.csv
  - ranking_stability.csv
  - point_vs_burst.csv

Usage:
    python statistical_tests_missing_true_impact.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_true_impact")
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


def build_conditions(df):
    """Build list of (missing_type, fraction, num_bursts) conditions."""
    conditions = []
    for mt in sorted(df['missing_type'].unique()):
        for frac in sorted(df['fraction'].unique()):
            if mt == 'point':
                conditions.append((mt, frac, 0))
            else:
                for nb in sorted(df[df['missing_type'] == mt]['num_bursts'].unique()):
                    conditions.append((mt, frac, int(nb)))
    return conditions


def condition_label(mt, frac, nb):
    if mt == 'point':
        return f"point_f{frac}"
    return f"burst_f{frac}_nb{nb}"


def get_condition_data(df, mt, frac, nb):
    if mt == 'point':
        return df[(df['missing_type'] == mt) & (df['fraction'] == frac)]
    return df[(df['missing_type'] == mt) & (df['fraction'] == frac) & (df['num_bursts'] == nb)]


# ================================================================
# 1. WILCOXON SIGNED-RANK TEST
# ================================================================
def run_wilcoxon(df, bl):
    print("\n" + "=" * 80)
    print("  1. WILCOXON SIGNED-RANK TEST (missing true impact vs baseline)")
    print("=" * 80)

    rows = []
    models = sorted(df['model'].unique())
    conditions = build_conditions(df)
    n_tests = len(models) * len(conditions)

    for model in models:
        bl_model = bl[bl['model'] == model].set_index('file')['AUC_ROC']

        for mt, frac, nb in conditions:
            corrupted = get_condition_data(df, mt, frac, nb)
            corrupted = corrupted[corrupted['model'] == model]
            corr_auc = corrupted.set_index('file')['AUC_ROC']

            common = bl_model.index.intersection(corr_auc.index)
            if len(common) < 10:
                continue

            baseline_vals = bl_model.loc[common].values
            corrupted_vals = corr_auc.loc[common].values
            diff = baseline_vals - corrupted_vals

            # Also get mean lost anomalies for this condition
            mean_lost = corrupted.set_index('file').loc[common, 'n_lost_anomalies'].mean() \
                if 'n_lost_anomalies' in corrupted.columns else 0

            non_zero = diff[diff != 0]
            if len(non_zero) < 5:
                rows.append({
                    'model': model, 'missing_type': mt, 'fraction': frac,
                    'num_bursts': nb if mt == 'burst' else None,
                    'condition': condition_label(mt, frac, nb),
                    'n_paired': len(common),
                    'mean_baseline_auc': round(np.mean(baseline_vals), 4),
                    'mean_corrupted_auc': round(np.mean(corrupted_vals), 4),
                    'mean_drop': round(np.mean(diff), 4),
                    'median_drop': round(np.median(diff), 4),
                    'pct_drop': 0.0,
                    'mean_lost_anomalies': round(mean_lost, 1),
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
                'model': model, 'missing_type': mt, 'fraction': frac,
                'num_bursts': nb if mt == 'burst' else None,
                'condition': condition_label(mt, frac, nb),
                'n_paired': len(common),
                'mean_baseline_auc': round(mean_bl, 4),
                'mean_corrupted_auc': round(mean_corr, 4),
                'mean_drop': round(mean_drop, 4),
                'median_drop': round(np.median(diff), 4),
                'pct_drop': round(pct_drop, 2),
                'mean_lost_anomalies': round(mean_lost, 1),
                'wilcoxon_stat': round(stat, 1),
                'p_value': p_value,
                'p_value_bonferroni': min(p_value * n_tests, 1.0),
                'significant_bonferroni': p_value < (0.05 / n_tests),
            })

    df_results = pd.DataFrame(rows)

    # Print point results
    point_res = df_results[df_results['missing_type'] == 'point']
    if not point_res.empty:
        print(f"\n--- POINT MISSING ---")
        print(f"{'Model':<10} {'Frac':>6} {'Baseline':>9} {'Corrupt':>9} "
              f"{'Drop':>7} {'%Drop':>6} {'Lost':>6} {'p-value':>10} {'Sig?':>5}")
        print("-" * 80)
        for _, r in point_res.iterrows():
            sig = "***" if r['p_value_bonferroni'] < 0.001 else \
                  "**" if r['p_value_bonferroni'] < 0.01 else \
                  "*" if r['p_value_bonferroni'] < 0.05 else "ns"
            print(f"{r['model']:<10} {r['fraction']:>6.2f} {r['mean_baseline_auc']:>9.4f} "
                  f"{r['mean_corrupted_auc']:>9.4f} {r['mean_drop']:>+7.4f} "
                  f"{r['pct_drop']:>5.1f}% {r['mean_lost_anomalies']:>6.1f} "
                  f"{r['p_value']:.2e} {sig:>5}")

    # Print burst results grouped by num_bursts
    burst_res = df_results[df_results['missing_type'] == 'burst']
    if not burst_res.empty:
        for nb in sorted(burst_res['num_bursts'].dropna().unique()):
            sub = burst_res[burst_res['num_bursts'] == nb]
            print(f"\n--- BURST MISSING (num_bursts={int(nb)}) ---")
            print(f"{'Model':<10} {'Frac':>6} {'Baseline':>9} {'Corrupt':>9} "
                  f"{'Drop':>7} {'%Drop':>6} {'Lost':>6} {'p-value':>10} {'Sig?':>5}")
            print("-" * 80)
            for _, r in sub.iterrows():
                sig = "***" if r['p_value_bonferroni'] < 0.001 else \
                      "**" if r['p_value_bonferroni'] < 0.01 else \
                      "*" if r['p_value_bonferroni'] < 0.05 else "ns"
                print(f"{r['model']:<10} {r['fraction']:>6.2f} {r['mean_baseline_auc']:>9.4f} "
                      f"{r['mean_corrupted_auc']:>9.4f} {r['mean_drop']:>+7.4f} "
                      f"{r['pct_drop']:>5.1f}% {r['mean_lost_anomalies']:>6.1f} "
                      f"{r['p_value']:.2e} {sig:>5}")

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
    conditions = build_conditions(df)

    for model in models:
        bl_model = bl[bl['model'] == model].set_index('file')['AUC_ROC']

        for mt, frac, nb in conditions:
            corrupted = get_condition_data(df, mt, frac, nb)
            corrupted = corrupted[corrupted['model'] == model]
            corr_auc = corrupted.set_index('file')['AUC_ROC']

            common = bl_model.index.intersection(corr_auc.index)
            if len(common) < 10:
                continue

            delta, magnitude = cliffs_delta(
                bl_model.loc[common].values,
                corr_auc.loc[common].values)

            rows.append({
                'model': model, 'missing_type': mt, 'fraction': frac,
                'num_bursts': nb if mt == 'burst' else None,
                'condition': condition_label(mt, frac, nb),
                'n': len(common),
                'cliffs_delta': round(delta, 4),
                'magnitude': magnitude,
            })

    df_results = pd.DataFrame(rows)

    # Print point
    point_res = df_results[df_results['missing_type'] == 'point']
    if not point_res.empty:
        print(f"\n--- POINT MISSING ---")
        print(f"{'Model':<10} {'Frac':>6} {'delta':>8} {'Magnitude':<12}")
        print("-" * 40)
        for _, r in point_res.iterrows():
            print(f"{r['model']:<10} {r['fraction']:>6.2f} {r['cliffs_delta']:>+8.4f} {r['magnitude']:<12}")

    # Print burst by nb
    burst_res = df_results[df_results['missing_type'] == 'burst']
    if not burst_res.empty:
        for nb in sorted(burst_res['num_bursts'].dropna().unique()):
            sub = burst_res[burst_res['num_bursts'] == nb]
            print(f"\n--- BURST MISSING (num_bursts={int(nb)}) ---")
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
    conditions = build_conditions(df)

    rows = []
    for mt, frac, nb in conditions:
        sub = get_condition_data(df, mt, frac, nb)
        pivot = sub.pivot_table(
            index='file', columns='model', values='AUC_ROC', aggfunc='mean')
        pivot = pivot.dropna()

        model_names = [m for m in models if m in pivot.columns]
        if len(pivot) < 10 or len(model_names) < 2:
            continue

        model_arrays = [pivot[m].values for m in model_names]
        stat, p_value = stats.friedmanchisquare(*model_arrays)

        # Mean ranks (lower is better -> rank descending AUC)
        ranks = np.zeros((len(pivot), len(model_names)))
        for i in range(len(pivot)):
            ranks[i] = stats.rankdata(-pivot[model_names].values[i])
        mean_ranks = ranks.mean(axis=0)
        rank_dict = {m: round(r, 3) for m, r in zip(model_names, mean_ranks)}
        best_model = min(rank_dict, key=rank_dict.get)

        rows.append({
            'missing_type': mt, 'fraction': frac,
            'num_bursts': nb if mt == 'burst' else None,
            'condition': condition_label(mt, frac, nb),
            'n_files': len(pivot), 'n_models': len(model_names),
            'friedman_stat': round(stat, 2),
            'p_value': p_value,
            'significant': p_value < 0.05,
            'best_model': best_model,
            **{f'rank_{m}': rank_dict.get(m) for m in models},
        })

    df_results = pd.DataFrame(rows)

    print(f"\n{'Condition':<22} {'n':>5} {'chi2':>8} {'p-value':>10} {'Sig?':>5} {'Best':<10} " +
          " ".join(f"{m:>8}" for m in models))
    print("-" * (65 + 9 * len(models)))
    for _, r in df_results.iterrows():
        sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
        rank_str = " ".join(f"{r.get(f'rank_{m}', 'N/A'):>8}" for m in models)
        print(f"{r['condition']:<22} {int(r['n_files']):>5} "
              f"{r['friedman_stat']:>8.2f} {r['p_value']:.2e} {sig:>5} {r['best_model']:<10} {rank_str}")

    # Nemenyi post-hoc
    print("\n--- Nemenyi Post-hoc (where Friedman is significant) ---")
    for _, r in df_results.iterrows():
        if not r['significant']:
            continue
        mt, frac = r['missing_type'], r['fraction']
        nb = r['num_bursts']
        sub = get_condition_data(df, mt, frac, nb if nb is not None and not pd.isna(nb) else 0)
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

        print(f"\n  {r['condition']}: CD={cd:.3f}")
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
    conditions = build_conditions(df)

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
            'condition': 'baseline', 'missing_type': None,
            'fraction': None, 'num_bursts': None,
            **{f'rank_{i+1}': m for i, m in enumerate(bl_ranking)},
            **{f'auc_{m}': round(bl_means[m], 4) for m in models},
        })

    for mt, frac, nb in conditions:
        cond_means = {}
        for model in models:
            sub = get_condition_data(df, mt, frac, nb)
            sub = sub[sub['model'] == model]
            if not sub.empty:
                cond_means[model] = sub['AUC_ROC'].mean()

        if cond_means:
            ranking = sorted(cond_means, key=cond_means.get, reverse=True)
            rows.append({
                'condition': condition_label(mt, frac, nb),
                'missing_type': mt, 'fraction': frac,
                'num_bursts': nb if mt == 'burst' else None,
                **{f'rank_{i+1}': m for i, m in enumerate(ranking)},
                **{f'auc_{m}': round(cond_means[m], 4) for m in models if m in cond_means},
            })

    df_results = pd.DataFrame(rows)

    print(f"\n{'Condition':<22} " + " ".join(f"{'#'+str(i+1):>10}" for i in range(len(models))) +
          " | " + " ".join(f"{m:>10}" for m in models))
    print("-" * (23 + 11 * len(models) * 2 + 3))
    for _, r in df_results.iterrows():
        ranks = " ".join(f"{r.get(f'rank_{i+1}', 'N/A'):>10}" for i in range(len(models)))
        aucs = " ".join(f"{r.get(f'auc_{m}', 0):>10.4f}" for m in models)
        print(f"{r['condition']:<22} {ranks} | {aucs}")

    if len(df_results) > 1:
        first_places = df_results['rank_1'].dropna().tolist()
        n_changes = sum(1 for i in range(1, len(first_places)) if first_places[i] != first_places[i-1])
        print(f"\nRanking changes (#1 position): {n_changes}/{len(first_places)-1}")

    return df_results


# ================================================================
# 5. POINT VS BURST COMPARISON
# ================================================================
def run_point_vs_burst(df, bl):
    print("\n" + "=" * 80)
    print("  5. POINT vs BURST COMPARISON (same fraction)")
    print("=" * 80)

    models = sorted(df['model'].unique())
    fractions = sorted(df['fraction'].unique())
    burst_nbs = sorted(df[df['missing_type'] == 'burst']['num_bursts'].unique())

    rows = []
    for model in models:
        for frac in fractions:
            point_data = df[(df['model'] == model) &
                            (df['missing_type'] == 'point') &
                            (df['fraction'] == frac)]
            point_auc = point_data.set_index('file')['AUC_ROC']

            for nb in burst_nbs:
                burst_data = df[(df['model'] == model) &
                                (df['missing_type'] == 'burst') &
                                (df['fraction'] == frac) &
                                (df['num_bursts'] == nb)]
                burst_auc = burst_data.set_index('file')['AUC_ROC']

                common = point_auc.index.intersection(burst_auc.index)
                if len(common) < 10:
                    continue

                p_vals = point_auc.loc[common].values
                b_vals = burst_auc.loc[common].values
                diff = p_vals - b_vals  # positive = point better (burst worse)

                non_zero = diff[diff != 0]
                if len(non_zero) < 5:
                    stat, p_value = None, 1.0
                else:
                    stat, p_value = stats.wilcoxon(non_zero, alternative='two-sided')
                    stat = round(stat, 1)

                mean_point_lost = point_data.set_index('file').loc[common, 'n_lost_anomalies'].mean()
                mean_burst_lost = burst_data.set_index('file').loc[common, 'n_lost_anomalies'].mean()

                rows.append({
                    'model': model, 'fraction': frac, 'num_bursts': int(nb),
                    'n_paired': len(common),
                    'mean_point_auc': round(np.mean(p_vals), 4),
                    'mean_burst_auc': round(np.mean(b_vals), 4),
                    'mean_diff': round(np.mean(diff), 4),
                    'point_better': np.mean(p_vals) > np.mean(b_vals),
                    'mean_point_lost': round(mean_point_lost, 1),
                    'mean_burst_lost': round(mean_burst_lost, 1),
                    'wilcoxon_stat': stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                })

    df_results = pd.DataFrame(rows)

    print(f"\n{'Model':<10} {'Frac':>6} {'nb':>4} {'Point AUC':>10} {'Burst AUC':>10} "
          f"{'Diff':>7} {'Winner':>8} {'P-Lost':>7} {'B-Lost':>7} {'p-value':>10} {'Sig?':>5}")
    print("-" * 95)
    for _, r in df_results.iterrows():
        winner = "Point" if r['point_better'] else "Burst"
        sig = "*" if r['significant'] else "ns"
        print(f"{r['model']:<10} {r['fraction']:>6.2f} {int(r['num_bursts']):>4} "
              f"{r['mean_point_auc']:>10.4f} {r['mean_burst_auc']:>10.4f} "
              f"{r['mean_diff']:>+7.4f} {winner:>8} "
              f"{r['mean_point_lost']:>7.1f} {r['mean_burst_lost']:>7.1f} "
              f"{r['p_value']:.2e} {sig:>5}")

    return df_results


# ================================================================
# MAIN
# ================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df, bl = load_data()
    print(f"  Experiment: {len(df)} rows, models: {sorted(df['model'].unique())}")
    print(f"  Missing types: {sorted(df['missing_type'].unique())}")
    print(f"  Fractions: {sorted(df['fraction'].unique())}")
    print(f"  Baseline: {len(bl)} rows")

    wilcoxon_df = run_wilcoxon(df, bl)
    wilcoxon_df.to_csv(os.path.join(OUTPUT_DIR, "wilcoxon_results.csv"), index=False)

    cliffs_df = run_cliffs_delta(df, bl)
    cliffs_df.to_csv(os.path.join(OUTPUT_DIR, "cliffs_delta_results.csv"), index=False)

    friedman_df = run_friedman(df, bl)
    friedman_df.to_csv(os.path.join(OUTPUT_DIR, "friedman_results.csv"), index=False)

    ranking_df = run_ranking_stability(df, bl)
    ranking_df.to_csv(os.path.join(OUTPUT_DIR, "ranking_stability.csv"), index=False)

    pvb_df = run_point_vs_burst(df, bl)
    pvb_df.to_csv(os.path.join(OUTPUT_DIR, "point_vs_burst.csv"), index=False)

    print(f"\n{'=' * 80}")
    print(f"  All results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
