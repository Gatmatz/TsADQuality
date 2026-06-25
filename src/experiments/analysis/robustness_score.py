"""
Corruption Robustness Score (CRS)

Computes a unified robustness score per model across all corruption types.

For each (model, corruption_type):
    R(M, C) = mean(AUC across all conditions) / baseline_AUC(M)

Overall:
    CRS(M) = mean( R(M, C) for all C )

Score interpretation:
    1.0  = no degradation (perfectly robust)
    0.5  = lost half its performance
    0.0  = completely destroyed

Usage:
    python robustness_score.py
    python robustness_score.py --models IForest PCA LOF MP
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))

RESULTS_BASE = os.path.join(project_root, "results", "experiments")
BASELINE_CSV = os.path.join(project_root, "results", "tables", "baseline_141_iforest.csv")
OUTPUT_DIR = os.path.join(project_root, "results", "analysis", "robustness")

# The 11 experiments we keep, with their parameter columns and checkpoint paths
EXPERIMENTS = {
    'white_noise_snr': {
        'label': 'White Noise (SNR)',
        'group_cols': ['snr_db'],
    },
    'missing_true_impact': {
        'label': 'Missing (True Impact)',
        'group_cols': ['fraction', 'num_bursts'],
    },
    'missing_masking': {
        'label': 'Missing (Masking)',
        'group_cols': ['fraction', 'num_bursts'],
    },
    'missing_positional': {
        'label': 'Missing (Positional)',
        'group_cols': ['fraction', 'target'],
    },
    'gilbert_elliott_true_impact': {
        'label': 'Gilbert-Elliott',
        'group_cols': ['alpha', 'beta'],
    },
    'freeze': {
        'label': 'Freeze',
        'group_cols': ['fraction', 'num_stucks'],
    },
    'swap_point': {
        'label': 'Point Swap',
        'group_cols': ['fraction'],
    },
    'swap_segment': {
        'label': 'Segment Swap',
        'group_cols': ['fraction', 'num_swaps'],
    },
    'swap_permutation': {
        'label': 'Permutation',
        'group_cols': ['n_segments'],
    },
    'spikes_normal_only': {
        'label': 'Spikes (Normal Only)',
        'group_cols': ['fraction', 'multiplier'],
    },
    'noise_position': {
        'label': 'Noise Position',
        'group_cols': ['snr_db', 'position_mode'],
    },
}


def load_baseline(model_name):
    """Load baseline AUC-ROC for a model."""
    # For now we only have IForest baseline
    # When other models are added, we'll need separate baseline files
    if model_name == 'IForest':
        csv_path = BASELINE_CSV
    else:
        csv_path = os.path.join(project_root, "results", "tables",
                                f"baseline_141_{model_name.lower()}.csv")

    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    return df['AUC_ROC'].mean()


def compute_robustness_for_experiment(exp_name, model_name, baseline_auc):
    """Compute R(M, C) for one experiment."""
    checkpoint = os.path.join(RESULTS_BASE, exp_name, "checkpoint.csv")
    if not os.path.exists(checkpoint):
        return None

    df = pd.read_csv(checkpoint)

    # Filter by model
    if 'model' in df.columns:
        df = df[df['model'] == model_name]

    # Filter errors
    if 'error' in df.columns:
        df = df[df['error'].isna()]

    if df.empty or 'AUC_ROC' not in df.columns:
        return None

    # Mean AUC across all conditions
    mean_auc = df['AUC_ROC'].mean()

    # Robustness = mean_auc / baseline
    robustness = mean_auc / baseline_auc if baseline_auc > 0 else 0

    # Also compute per-condition breakdown
    config = EXPERIMENTS[exp_name]
    group_cols = [c for c in config['group_cols'] if c in df.columns]

    conditions = []
    if group_cols:
        for name, group in df.groupby(group_cols):
            cond = dict(zip(group_cols, name)) if len(group_cols) > 1 else {group_cols[0]: name}
            cond['mean_AUC_ROC'] = group['AUC_ROC'].mean()
            cond['n_files'] = len(group)
            cond['robustness'] = cond['mean_AUC_ROC'] / baseline_auc
            conditions.append(cond)

    return {
        'experiment': exp_name,
        'label': config['label'],
        'model': model_name,
        'mean_auc': mean_auc,
        'baseline_auc': baseline_auc,
        'robustness': robustness,
        'n_rows': len(df),
        'n_conditions': len(conditions),
        'worst_condition_auc': min(c['mean_AUC_ROC'] for c in conditions) if conditions else mean_auc,
        'best_condition_auc': max(c['mean_AUC_ROC'] for c in conditions) if conditions else mean_auc,
        'conditions': conditions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['IForest'])
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for model_name in args.models:
        baseline_auc = load_baseline(model_name)
        if baseline_auc is None:
            print(f"[SKIP] {model_name}: no baseline found")
            continue

        print(f"\n{'='*60}")
        print(f"  Model: {model_name}  |  Baseline AUC-ROC: {baseline_auc:.4f}")
        print(f"{'='*60}")

        model_results = []

        for exp_name in EXPERIMENTS:
            result = compute_robustness_for_experiment(exp_name, model_name, baseline_auc)
            if result is None:
                print(f"  {EXPERIMENTS[exp_name]['label']:30s}  [NO DATA]")
                continue

            model_results.append(result)
            all_results.append(result)

            worst_r = result['worst_condition_auc'] / baseline_auc
            print(f"  {result['label']:30s}  R={result['robustness']:.3f}  "
                  f"(worst={worst_r:.3f}, AUC range: {result['worst_condition_auc']:.3f}-{result['best_condition_auc']:.3f})")

        if model_results:
            # Overall CRS
            crs = np.mean([r['robustness'] for r in model_results])
            # Worst-case CRS (using worst condition per experiment)
            crs_worst = np.mean([r['worst_condition_auc'] / baseline_auc for r in model_results])

            print(f"\n  CRS (mean):       {crs:.3f}")
            print(f"  CRS (worst-case): {crs_worst:.3f}")
            print(f"  Experiments:      {len(model_results)}")

    # Summary table
    if all_results:
        print(f"\n{'='*60}")
        print("  SUMMARY TABLE")
        print(f"{'='*60}")

        summary_rows = []
        for model_name in args.models:
            model_res = [r for r in all_results if r['model'] == model_name]
            if not model_res:
                continue

            baseline_auc = model_res[0]['baseline_auc']
            row = {'Model': model_name, 'Baseline': baseline_auc}

            for r in model_res:
                row[r['label']] = round(r['robustness'], 3)

            row['CRS (mean)'] = round(np.mean([r['robustness'] for r in model_res]), 3)
            row['CRS (worst)'] = round(np.mean([r['worst_condition_auc'] / baseline_auc for r in model_res]), 3)
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        print(df_summary.to_string(index=False))

        # Save
        df_summary.to_csv(os.path.join(OUTPUT_DIR, "robustness_summary.csv"), index=False)
        print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'robustness_summary.csv')}")

        # Per-experiment detail
        detail_rows = []
        for r in all_results:
            detail_rows.append({
                'model': r['model'],
                'experiment': r['experiment'],
                'label': r['label'],
                'baseline_auc': r['baseline_auc'],
                'mean_auc': round(r['mean_auc'], 4),
                'robustness': round(r['robustness'], 4),
                'worst_auc': round(r['worst_condition_auc'], 4),
                'best_auc': round(r['best_condition_auc'], 4),
                'n_conditions': r['n_conditions'],
            })
        pd.DataFrame(detail_rows).to_csv(os.path.join(OUTPUT_DIR, "robustness_detail.csv"), index=False)

        # Plot: robustness per experiment (bar chart)
        for model_name in args.models:
            model_res = [r for r in all_results if r['model'] == model_name]
            if not model_res:
                continue

            model_res.sort(key=lambda x: x['robustness'])
            labels = [r['label'] for r in model_res]
            values = [r['robustness'] for r in model_res]
            worst_values = [r['worst_condition_auc'] / r['baseline_auc'] for r in model_res]

            fig, ax = plt.subplots(figsize=(10, 6))
            y = range(len(labels))
            bars = ax.barh(y, values, height=0.4, label='Mean R(M,C)', color='#377eb8', alpha=0.8)
            ax.barh([yi + 0.4 for yi in y], worst_values, height=0.4,
                    label='Worst condition', color='#e41a1c', alpha=0.6)

            ax.set_yticks([yi + 0.2 for yi in y])
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel('Robustness Score (1.0 = no degradation)', fontsize=12)
            ax.set_title(f'{model_name}: Corruption Robustness Score per Experiment', fontsize=13)
            ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.3)
            ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.3)
            ax.set_xlim(0, 1.15)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2, axis='x')

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=9)

            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, f'robustness_{model_name}.png'), dpi=300)
            plt.close(fig)
            print(f"Saved: robustness_{model_name}.png")

        # Plot: radar/spider chart (if multiple models)
        if len(args.models) > 1:
            exp_labels = sorted(set(r['label'] for r in all_results))
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(len(exp_labels))
            width = 0.8 / len(args.models)
            colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']

            for i, model_name in enumerate(args.models):
                model_res = {r['label']: r['robustness'] for r in all_results if r['model'] == model_name}
                values = [model_res.get(l, 0) for l in exp_labels]
                ax.bar(x + i * width, values, width, label=model_name,
                       color=colors[i % len(colors)], alpha=0.8)

            ax.set_xticks(x + width * (len(args.models) - 1) / 2)
            ax.set_xticklabels(exp_labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Robustness Score', fontsize=12)
            ax.set_title('Corruption Robustness Score — Model Comparison', fontsize=13)
            ax.legend(fontsize=9)
            ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.3)
            ax.grid(True, alpha=0.2, axis='y')
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, 'robustness_comparison.png'), dpi=300)
            plt.close(fig)
            print("Saved: robustness_comparison.png")

    print("\nDone.")


if __name__ == '__main__':
    main()
