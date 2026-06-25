"""
Corruption-as-Anomaly Overlap Analysis

For each corruption type with trackable locations (spikes, missing, freeze, swap),
measures what fraction of the model's top predictions fall on:
  1. TRUE ANOMALY points (correct detection)
  2. CORRUPTION points (model detects corruption as anomaly)
  3. NEITHER (unrelated false positives)

This PROVES the causal chain mechanism: "spikes steal the anomaly budget."

Corruptions analyzed (with localized injection):
  - Spikes: fraction={0.01, 0.05, 0.10, 0.20}, multiplier={3, 5, 10}
  - Freeze: fraction={0.05, 0.10, 0.20}, num_stucks=5
  - Swap: fraction={0.10, 0.20}, swap_length=1

Missing is EXCLUDED: true impact removes NaN points (score=0), so corruption
points never appear in top-K, making overlap always 0%.

Models: IForest, LOF, MP, AE

Output (results/experiments/corruption_overlap/):
  - checkpoint.csv          per-file raw results
  - overlap_summary.csv     aggregated per corruption x severity x model
  - overlap_heatmap.png     visual summary

Usage:
    python run_corruption_overlap.py                    # full run
    python run_corruption_overlap.py --test             # 3 files
    python run_corruption_overlap.py --models IForest   # one model
    python run_corruption_overlap.py --workers 4        # parallel
"""
import argparse
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

# ==========================================
# PATH CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TSB_UAD_PATH = os.path.join(PROJECT_ROOT, 'TSB-UAD')
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

for p in [PROJECT_ROOT, TSB_UAD_PATH, SRC_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.lof import LOF
from TSB_UAD.models.matrix_profile import MatrixProfile
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length

from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors
from data_loader import load_tsb_dataframe, load_pretrained_ae, remap_filepath

# ==========================================
# CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "final_subset.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "corruption_overlap")

# Top-K: what fraction of top-scoring points to analyze
TOP_K_FRACTIONS = [0.01, 0.05, 0.10]  # top 1%, 5%, 10% of points

# Corruption conditions to test
CORRUPTION_CONDITIONS = [
    # Clean baseline — no corruption, measures where model puts top-K on clean data
    {'type': 'clean', 'label': 'clean', 'params': {}},

    # Spikes: the worst corruption — does the model "see" the spikes?
    {'type': 'spikes', 'label': 'spikes_f0.01_m5',  'params': {'fraction': 0.01, 'multiplier': 5.0}},
    {'type': 'spikes', 'label': 'spikes_f0.05_m5',  'params': {'fraction': 0.05, 'multiplier': 5.0}},
    {'type': 'spikes', 'label': 'spikes_f0.10_m5',  'params': {'fraction': 0.10, 'multiplier': 5.0}},
    {'type': 'spikes', 'label': 'spikes_f0.20_m5',  'params': {'fraction': 0.20, 'multiplier': 5.0}},
    {'type': 'spikes', 'label': 'spikes_f0.10_m3',  'params': {'fraction': 0.10, 'multiplier': 3.0}},
    {'type': 'spikes', 'label': 'spikes_f0.10_m10', 'params': {'fraction': 0.10, 'multiplier': 10.0}},

    # Missing: EXCLUDED — true impact removes NaN points, so corruption
    # points always have score=0 and never appear in top-K (pct_corruption=0% always).

    # Freeze: does the model flag the flat segments?
    {'type': 'freeze', 'label': 'freeze_f0.05_ns5', 'params': {'fraction': 0.05, 'num_stucks': 5}},
    {'type': 'freeze', 'label': 'freeze_f0.10_ns5', 'params': {'fraction': 0.10, 'num_stucks': 5}},
    {'type': 'freeze', 'label': 'freeze_f0.20_ns5', 'params': {'fraction': 0.20, 'num_stucks': 5}},

    # Swap: values preserved — model shouldn't flag swap locations
    {'type': 'swap', 'label': 'swap_f0.10', 'params': {'fraction': 0.10, 'swap_length': 1}},
    {'type': 'swap', 'label': 'swap_f0.20', 'params': {'fraction': 0.20, 'swap_length': 1}},
]


def run_model(model_name, X, scaled_data, sw, file_name=None):
    """Run a model and return anomaly scores."""
    if model_name == 'AE':
        clf, _ = load_pretrained_ae(file_name, PROJECT_ROOT)
        clf.predict(scaled_data)
        return clf.decision_scores_, True
    elif model_name == 'IForest':
        clf = IForest(n_estimators=100, random_state=42)
        clf.fit(X)
        return clf.decision_scores_, False
    elif model_name == 'MP':
        clf = MatrixProfile(window=sw)
        clf.fit(scaled_data)
        return clf.decision_scores_, False
    elif model_name == 'LOF':
        clf = LOF(n_neighbors=20)
        clf.fit(X)
        return clf.decision_scores_, False
    else:
        raise ValueError(f"Unknown model: {model_name}")


def apply_corruption(df, corruption_type, params, seed=42):
    """Apply corruption and return (corrupted_df, corruption_mask)."""
    if corruption_type == 'clean':
        return df.copy(), np.zeros(len(df), dtype=bool)

    corruptor = TSCorruptor(
        df, value_col='value', label_col='is_anomaly',
        seed=seed, corruption_target='only_normal'
    )

    if corruption_type == 'spikes':
        ts_corruptor.injectors.inject_spikes(
            corruptor,
            fraction=params['fraction'],
            multiplier=params['multiplier'],
            sequential=False,
            sequence_length=1
        )
    elif corruption_type == 'freeze':
        n = len(df)
        num_stucks = params['num_stucks']
        stuck_length = max(1, int(params['fraction'] * n / num_stucks))
        ts_corruptor.injectors.inject_sensor_stuck(
            corruptor,
            num_stucks=num_stucks,
            stuck_length=stuck_length
        )
    elif corruption_type == 'swap':
        ts_corruptor.injectors.inject_swap(
            corruptor,
            fraction=params['fraction'],
            swap_length=params.get('swap_length', 1),
            max_distance=None
        )

    corrupted_df = corruptor.get_corrupted_df()
    corruption_mask = corruptor.corruption_mask.to_numpy().astype(bool)
    return corrupted_df, corruption_mask


def compute_overlap(full_score, labels, corruption_mask, top_k_fractions):
    """
    For each top-K fraction, compute what % of top predictions are:
      - true anomalies
      - corruption points
      - neither (random FP)
    """
    n = len(full_score)
    results = []

    for k_frac in top_k_fractions:
        k = max(1, int(k_frac * n))
        top_indices = np.argsort(full_score)[-k:]  # highest scores

        is_anomaly = labels[top_indices] == 1
        is_corruption = corruption_mask[top_indices]
        is_both = is_anomaly & is_corruption  # shouldn't happen if corruption_target='only_normal'
        is_neither = ~is_anomaly & ~is_corruption

        # A corrupted point that is also anomaly: count as anomaly (shouldn't happen with only_normal)
        n_anomaly = int(is_anomaly.sum())
        n_corruption_only = int(is_corruption.sum() - is_both.sum())
        n_neither = int(is_neither.sum())

        results.append({
            'top_k_pct': k_frac * 100,
            'top_k_n': k,
            'n_true_anomaly': n_anomaly,
            'n_corruption': n_corruption_only,
            'n_neither': n_neither,
            'pct_true_anomaly': 100.0 * n_anomaly / k,
            'pct_corruption': 100.0 * n_corruption_only / k,
            'pct_neither': 100.0 * n_neither / k,
        })

    return results


def process_single_job(job_args):
    """Process one file × one condition × all models."""
    file_path, condition, model_names, seed = job_args
    file_name = os.path.basename(file_path)
    results = []

    try:
        # 1. Load data
        df, file_name = load_tsb_dataframe(file_path)
        labels = df['is_anomaly'].to_numpy('int')
        n = len(df)

        # 2. Sliding window from clean data
        clean_scaled = StandardScaler().fit_transform(
            df['value'].to_numpy('float').reshape(-1, 1)
        ).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # 3. Apply corruption
        corrupted_df, corruption_mask = apply_corruption(
            df, condition['type'], condition['params'], seed=seed
        )
        corrupted_data = corrupted_df['value'].to_numpy('float')
        n_corrupted_points = int(corruption_mask.sum())

        # 4. Prepare model input
        sw_used = sliding_window
        scaled_data = StandardScaler().fit_transform(
            corrupted_data.reshape(-1, 1)
        ).flatten()
        X = Window(window=sw_used).convert(scaled_data).to_numpy()

        # 5. Run each model
        for model_name in model_names:
            try:
                score, is_full = run_model(model_name, X, scaled_data, sw_used, file_name)
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                    score.reshape(-1, 1)
                ).ravel()

                # Pad to full length of the (possibly shortened) series
                if is_full:
                    padded_score = score
                else:
                    padded_score = np.array(
                        [score[0]] * math.ceil((sw_used - 1) / 2) +
                        list(score) +
                        [score[-1]] * ((sw_used - 1) // 2)
                    )

                full_score = padded_score
                if len(full_score) > n:
                    full_score = full_score[:n]
                elif len(full_score) < n:
                    full_score = np.pad(full_score, (0, n - len(full_score)),
                                        mode='edge')

                if len(np.unique(full_score[full_score > 0])) <= 1:
                    results.append({
                        'file': file_name, 'condition': condition['label'],
                        'corruption_type': condition['type'],
                        'model': model_name, 'error': 'Invalid scores',
                    })
                    continue

                # 6. Compute overlap
                overlaps = compute_overlap(full_score, labels, corruption_mask,
                                           TOP_K_FRACTIONS)

                for ov in overlaps:
                    results.append({
                        'file': file_name,
                        'condition': condition['label'],
                        'corruption_type': condition['type'],
                        'model': model_name,
                        'n_points': n,
                        'n_corrupted_points': n_corrupted_points,
                        'n_anomaly_points': int(labels.sum()),
                        'corruption_pct': 100.0 * n_corrupted_points / n,
                        'anomaly_pct': 100.0 * labels.sum() / n,
                        'error': None,
                        **ov,
                    })

            except Exception as e:
                results.append({
                    'file': file_name, 'condition': condition['label'],
                    'corruption_type': condition['type'],
                    'model': model_name,
                    'error': str(e)[:200],
                })

    except Exception as e:
        results.append({
            'file': file_name, 'condition': condition['label'],
            'error': f"File load error: {str(e)[:200]}",
        })

    return results


def generate_summary(df):
    """Aggregate per condition x model x top_k, with budget_stolen vs clean baseline."""
    valid = df[df['error'].isna()].copy()

    # Basic aggregation
    summary = valid.groupby(['corruption_type', 'condition', 'model', 'top_k_pct']).agg(
        n_files=('file', 'nunique'),
        mean_pct_anomaly=('pct_true_anomaly', 'mean'),
        std_pct_anomaly=('pct_true_anomaly', 'std'),
        mean_pct_corruption=('pct_corruption', 'mean'),
        std_pct_corruption=('pct_corruption', 'std'),
        mean_pct_neither=('pct_neither', 'mean'),
        std_pct_neither=('pct_neither', 'std'),
        mean_corruption_pct=('corruption_pct', 'mean'),
    ).reset_index()

    # Compute budget_stolen: per-file (clean_pct_anomaly - corrupted_pct_anomaly), then mean
    clean = valid[valid['condition'] == 'clean'][['file', 'model', 'top_k_pct', 'pct_true_anomaly']].copy()
    clean = clean.rename(columns={'pct_true_anomaly': 'clean_pct_anomaly'})

    if not clean.empty:
        corrupted = valid[valid['condition'] != 'clean'].copy()
        merged = corrupted.merge(clean, on=['file', 'model', 'top_k_pct'], how='left')
        merged['budget_stolen'] = merged['clean_pct_anomaly'] - merged['pct_true_anomaly']

        budget = merged.groupby(['condition', 'model', 'top_k_pct']).agg(
            mean_budget_stolen=('budget_stolen', 'mean'),
            std_budget_stolen=('budget_stolen', 'std'),
        ).reset_index()

        summary = summary.merge(budget, on=['condition', 'model', 'top_k_pct'], how='left')

    return summary


def generate_heatmap(summary_df, output_dir):
    """Create heatmap: for top-5% predictions, what % are corruption points."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Filter to top_k=5%
    data = summary_df[summary_df['top_k_pct'] == 5.0].copy()
    if data.empty:
        print("  [WARN] No data for top_k=5%, skipping heatmap")
        return

    models = sorted(data['model'].unique())
    conditions = data['condition'].unique()

    # Sort conditions by corruption type then severity
    type_order = {'spikes': 0, 'freeze': 1, 'swap': 2}
    cond_sorted = sorted(conditions, key=lambda c: (
        type_order.get(data[data['condition'] == c]['corruption_type'].iloc[0], 99), c
    ))

    matrix = np.full((len(models), len(cond_sorted)), np.nan)
    anomaly_matrix = np.full((len(models), len(cond_sorted)), np.nan)

    for i, model in enumerate(models):
        for j, cond in enumerate(cond_sorted):
            row = data[(data['model'] == model) & (data['condition'] == cond)]
            if not row.empty:
                matrix[i, j] = row.iloc[0]['mean_pct_corruption']
                anomaly_matrix[i, j] = row.iloc[0]['mean_pct_anomaly']

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(cond_sorted) * 1.2), max(4, len(models) * 1.2)))

    # Plot 1: % corruption in top predictions
    ax = axes[0]
    im = ax.imshow(matrix, cmap='Reds', aspect='auto', vmin=0, vmax=80)
    ax.set_xticks(range(len(cond_sorted)))
    ax.set_xticklabels(cond_sorted, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_title('% of Top-5% Predictions at Corruption Points', fontsize=11, fontweight='bold')
    for i in range(len(models)):
        for j in range(len(cond_sorted)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 40 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Plot 2: % true anomaly in top predictions
    ax = axes[1]
    im2 = ax.imshow(anomaly_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=60)
    ax.set_xticks(range(len(cond_sorted)))
    ax.set_xticklabels(cond_sorted, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_title('% of Top-5% Predictions at True Anomalies', fontsize=11, fontweight='bold')
    for i in range(len(models)):
        for j in range(len(cond_sorted)):
            val = anomaly_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 30 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)
    fig.colorbar(im2, ax=ax, shrink=0.8)

    fig.suptitle('Corruption-as-Anomaly Overlap (Top-5% Predictions)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'overlap_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: overlap_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description='Corruption-as-Anomaly Overlap Analysis')
    parser.add_argument('--test', action='store_true', help='Quick test with 3 files')
    parser.add_argument('--models', nargs='+', default=['IForest', 'LOF', 'MP', 'AE'],
                        help='Models to test')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers')
    parser.add_argument('--conditions', nargs='+', default=None,
                        help='Specific condition labels to run (default: all)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load subset
    subset_df = pd.read_csv(SUBSET_CSV)
    if args.test:
        subset_df = subset_df.head(3)
    print(f"Files: {len(subset_df)} | Models: {args.models}")

    # Filter conditions
    conditions = CORRUPTION_CONDITIONS
    if args.conditions:
        conditions = [c for c in conditions if c['label'] in args.conditions]
    print(f"Conditions: {len(conditions)}")

    # Load checkpoint
    checkpoint_path = os.path.join(RESULTS_DIR, 'checkpoint.csv')
    done_keys = set()
    if os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        for _, row in existing.iterrows():
            if pd.notna(row.get('model')):
                done_keys.add((row['file'], row['condition'], row['model']))
        print(f"Checkpoint: {len(done_keys)} existing entries")

    # Build jobs
    jobs = []
    for _, meta in subset_df.iterrows():
        file_path = remap_filepath(meta['filepath'], PROJECT_ROOT)
        for condition in conditions:
            # Check which models still need to run
            models_needed = []
            for m in args.models:
                fname = os.path.basename(file_path)
                key = (fname, condition['label'], m)
                if key not in done_keys:
                    models_needed.append(m)

            if models_needed:
                jobs.append((file_path, condition, models_needed, 42))

    print(f"Jobs to run: {len(jobs)}")
    if not jobs:
        print("All done! Generating summary...")
        all_results = pd.read_csv(checkpoint_path)
    else:
        all_results_list = []

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_single_job, j): j for j in jobs}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    try:
                        results = future.result(timeout=300)
                        all_results_list.extend(results)
                    except Exception as e:
                        job = futures[future]
                        all_results_list.append({
                            'file': os.path.basename(job[0]),
                            'condition': job[1]['label'],
                            'error': str(e)[:200]
                        })
        else:
            for job in tqdm(jobs, desc="Processing"):
                try:
                    results = process_single_job(job)
                    all_results_list.extend(results)
                except Exception as e:
                    all_results_list.append({
                        'file': os.path.basename(job[0]),
                        'condition': job[1]['label'],
                        'error': str(e)[:200]
                    })

        # Save checkpoint (append)
        new_df = pd.DataFrame(all_results_list)
        if os.path.exists(checkpoint_path):
            existing = pd.read_csv(checkpoint_path)
            all_results = pd.concat([existing, new_df], ignore_index=True)
        else:
            all_results = new_df

        all_results.to_csv(checkpoint_path, index=False)
        print(f"\nCheckpoint saved: {len(all_results)} rows")

    # Summary
    print("\nGenerating summary...")
    summary = generate_summary(all_results)
    summary.to_csv(os.path.join(RESULTS_DIR, 'overlap_summary.csv'), index=False)
    print(f"Summary saved: {len(summary)} rows")

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS — Top-5% predictions")
    print("=" * 70)
    top5 = summary[summary['top_k_pct'] == 5.0].copy()
    if not top5.empty:
        # Show clean baseline first
        clean_data = top5[top5['condition'] == 'clean']
        if not clean_data.empty:
            print("\n  CLEAN BASELINE:")
            for _, row in clean_data.sort_values('model').iterrows():
                print(f"    {'clean':25s} {row['model']:8s}: "
                      f"anomaly={row['mean_pct_anomaly']:5.1f}%")

        for ctype in ['spikes', 'freeze', 'swap']:
            ct_data = top5[top5['corruption_type'] == ctype]
            if ct_data.empty:
                continue
            print(f"\n  {ctype.upper()}:")
            for _, row in ct_data.sort_values(['condition', 'model']).iterrows():
                budget_str = ""
                if 'mean_budget_stolen' in row and pd.notna(row.get('mean_budget_stolen')):
                    budget_str = f"  budget_stolen={row['mean_budget_stolen']:+5.1f}pp"
                print(f"    {row['condition']:25s} {row['model']:8s}: "
                      f"corruption={row['mean_pct_corruption']:5.1f}%  "
                      f"anomaly={row['mean_pct_anomaly']:5.1f}%  "
                      f"random={row['mean_pct_neither']:5.1f}%{budget_str}")

    # Heatmap
    print("\nGenerating heatmap...")
    generate_heatmap(summary, RESULTS_DIR)

    print(f"\nAll outputs saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
