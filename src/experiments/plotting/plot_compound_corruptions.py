"""
Plot compound corruption experiment results.

Generates:
1. Heatmap: interaction effects (synergistic/additive/sub-additive) per combination × severity
2. Grouped bar: compound AUC drop vs sum of individual drops
3. Bar chart: mean AUC-ROC per condition type (baseline, singles, compounds)
4. Heatmap: AUC-ROC for all pair combinations × severity crosses
5. Triple corruption: AUC degradation surface

Usage:
    python plot_compound_corruptions.py
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "compound_corruptions")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

PAIR_NAMES = {
    'noise_missing': 'Noise + Missing',
    'noise_spikes': 'Noise + Spikes',
    'spikes_missing': 'Spikes + Missing',
    'missing_freeze': 'Missing + Freeze',
    'noise_ge_missing': 'Noise + GE Missing',
}
TRIPLE_NAME = {'noise_spikes_missing': 'Noise + Spikes + Missing'}

SEVERITY_ORDER = ['low', 'med', 'high']


def load_data():
    checkpoint = pd.read_csv(os.path.join(RESULTS_DIR, "checkpoint.csv"))
    interaction = pd.read_csv(os.path.join(RESULTS_DIR, "interaction_analysis.csv"))
    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    return checkpoint, interaction, summary


# ──────────────────────────────────────────────
# Plot 1: Interaction heatmap (pairs only)
# ──────────────────────────────────────────────
def plot_interaction_heatmap(interaction, model='IForest'):
    pairs = interaction[
        (interaction['model'] == model) &
        (interaction['combination_name'] != 'noise_spikes_missing')
    ].copy()

    if pairs.empty:
        print(f"[Skip] No pair interaction data for {model}")
        return

    combinations = list(PAIR_NAMES.keys())
    combinations = [c for c in combinations if c in pairs['combination_name'].unique()]
    sev_crosses = []
    for sa in SEVERITY_ORDER:
        for sb in SEVERITY_ORDER:
            sev_crosses.append((sa, sb))

    matrix = np.full((len(combinations), len(sev_crosses)), np.nan)
    for i, combo in enumerate(combinations):
        sub = pairs[pairs['combination_name'] == combo]
        for _, row in sub.iterrows():
            sa = row['severity_A'].split('_')[-1]  # e.g. noise_low -> low
            sb = row['severity_B'].split('_')[-1]
            if (sa, sb) in sev_crosses:
                j = sev_crosses.index((sa, sb))
                matrix[i, j] = row['interaction_pct']

    fig, ax = plt.subplots(figsize=(12, 5))

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(matrix, cmap='RdBu_r', norm=norm, aspect='auto')

    ax.set_yticks(range(len(combinations)))
    ax.set_yticklabels([PAIR_NAMES.get(c, c) for c in combinations], fontsize=10)

    xlabels = [f"{sa[0].upper()}+{sb[0].upper()}" for sa, sb in sev_crosses]
    ax.set_xticks(range(len(sev_crosses)))
    ax.set_xticklabels(xlabels, fontsize=8, rotation=45, ha='right')
    ax.set_xlabel('Severity (A + B): L=Low, M=Med, H=High', fontsize=10)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f"{val:.0f}%", ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Interaction Effect (%)', fontsize=10)

    ax.set_title(f'Interaction Effects: Synergistic (red) vs Sub-additive (blue) — {model}',
                 fontsize=12, fontweight='bold')

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f'interaction_heatmap_{model}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────
# Plot 2: Compound drop vs sum of individual drops
# ──────────────────────────────────────────────
def plot_compound_vs_additive(interaction, model='IForest'):
    pairs = interaction[
        (interaction['model'] == model) &
        (interaction['combination_name'] != 'noise_spikes_missing')
    ].copy()

    if pairs.empty:
        return

    # Average across severity levels per combination
    avg = pairs.groupby('combination_name').agg({
        'drop_compound': 'mean',
        'predicted_additive': 'mean',
        'interaction': 'mean',
    }).reset_index()

    combinations = [c for c in PAIR_NAMES.keys() if c in avg['combination_name'].values]
    avg = avg.set_index('combination_name').loc[combinations].reset_index()

    x = np.arange(len(combinations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width/2, avg['predicted_additive'], width,
                   label='Sum of Individual Drops', color='#4DBEEE', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, avg['drop_compound'], width,
                   label='Actual Compound Drop', color='#D95319', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Mean AUC-ROC Drop', fontsize=11)
    ax.set_title(f'Compound vs Additive Degradation — {model}', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_NAMES[c] for c in combinations], fontsize=9, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Annotate interaction
    for i, combo in enumerate(combinations):
        row = avg[avg['combination_name'] == combo].iloc[0]
        diff = row['interaction']
        label = "syn" if diff > 0.005 else ("sub" if diff < -0.005 else "≈add")
        color = '#D62728' if diff > 0.005 else ('#2CA02C' if diff < -0.005 else '#666666')
        y_pos = max(row['drop_compound'], row['predicted_additive']) + 0.003
        ax.text(i, y_pos, label, ha='center', fontsize=9, color=color, fontweight='bold')

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f'compound_vs_additive_{model}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────
# Plot 3: Overall AUC-ROC by condition type
# ──────────────────────────────────────────────
def plot_auc_by_condition_type(summary, model='IForest'):
    df = summary[summary['model'] == model].copy()
    if df.empty:
        return

    # Group by combination_name
    type_means = df.groupby('combination_name')['mean_AUC_ROC'].mean().reset_index()

    order = ['baseline', 'single'] + list(PAIR_NAMES.keys()) + list(TRIPLE_NAME.keys())
    labels = ['Baseline', 'Singles (avg)'] + list(PAIR_NAMES.values()) + list(TRIPLE_NAME.values())

    order = [o for o in order if o in type_means['combination_name'].values]
    labels_filtered = [labels[i] for i, o in enumerate(
        ['baseline', 'single'] + list(PAIR_NAMES.keys()) + list(TRIPLE_NAME.keys())
    ) if o in type_means['combination_name'].values]

    vals = [type_means[type_means['combination_name'] == o]['mean_AUC_ROC'].values[0] for o in order]

    colors = []
    for o in order:
        if o == 'baseline':
            colors.append('#2CA02C')
        elif o == 'single':
            colors.append('#4DBEEE')
        elif o in TRIPLE_NAME:
            colors.append('#7E2F8E')
        else:
            colors.append('#D95319')

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels_filtered, fontsize=9, rotation=25, ha='right')
    ax.set_ylabel('Mean AUC-ROC', fontsize=11)
    ax.set_title(f'Mean AUC-ROC by Condition Type — {model}', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    baseline_val = vals[0] if order[0] == 'baseline' else None
    if baseline_val:
        ax.axhline(y=baseline_val, color='green', linestyle='--', alpha=0.85, linewidth=2.0, zorder=5)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f'auc_by_condition_type_{model}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────
# Plot 4: Pair AUC heatmaps (one per combination)
# ──────────────────────────────────────────────
def plot_pair_heatmaps(summary, model='IForest'):
    df = summary[summary['model'] == model].copy()

    for combo_key, combo_label in PAIR_NAMES.items():
        sub = df[df['combination_name'] == combo_key].copy()
        if sub.empty:
            continue

        # Parse severity from condition name
        parts_list = []
        for _, row in sub.iterrows():
            cond = row['condition']
            parts = cond.split('+')
            if len(parts) == 2:
                sa = parts[0].split('_')[-1]  # e.g. noise_low -> low
                sb = parts[1].split('_')[-1]
                parts_list.append((sa, sb, row['mean_AUC_ROC']))

        if not parts_list:
            continue

        matrix = np.full((3, 3), np.nan)
        for sa, sb, val in parts_list:
            if sa in SEVERITY_ORDER and sb in SEVERITY_ORDER:
                i = SEVERITY_ORDER.index(sa)
                j = SEVERITY_ORDER.index(sb)
                matrix[i, j] = val

        # Get baseline
        baseline_row = df[df['combination_name'] == 'baseline']
        baseline_auc = baseline_row['mean_AUC_ROC'].values[0] if not baseline_row.empty else 0.7

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=matrix[~np.isnan(matrix)].min() - 0.02,
                       vmax=baseline_auc, aspect='auto')

        type_A = combo_key.split('_')[0].capitalize()
        types = combo_key.split('_')
        if len(types) == 2:
            type_B = types[1].capitalize()
        elif combo_key == 'noise_ge_missing':
            type_A = 'Noise'
            type_B = 'GE Missing'
        elif combo_key == 'missing_freeze':
            type_A = 'Missing'
            type_B = 'Freeze'

        ax.set_xticks(range(3))
        ax.set_xticklabels(['Low', 'Med', 'High'], fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels(['Low', 'Med', 'High'], fontsize=9)
        ax.set_xlabel(f'{type_B} Severity', fontsize=10)
        ax.set_ylabel(f'{type_A} Severity', fontsize=10)

        for i in range(3):
            for j in range(3):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                            fontsize=10, fontweight='bold',
                            color='white' if matrix[i, j] < (baseline_auc - 0.08) else 'black')

        ax.set_title(f'{combo_label}\n(Baseline: {baseline_auc:.3f})', fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8, label='AUC-ROC')

        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f'pair_heatmap_{combo_key}_{model}.png')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out}")


# ──────────────────────────────────────────────
# Plot 5: Interaction type distribution (pie/bar)
# ──────────────────────────────────────────────
def plot_interaction_distribution(interaction, model='IForest'):
    df = interaction[interaction['model'] == model].copy()
    if df.empty:
        return

    counts = df['interaction_type'].value_counts()

    colors_map = {
        'synergistic': '#D62728',
        'additive': '#FFD700',
        'sub-additive': '#2CA02C',
    }

    labels = counts.index.tolist()
    values = counts.values
    colors = [colors_map.get(l, '#999999') for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Pie chart
    axes[0].pie(values, labels=[f'{l}\n({v})' for l, v in zip(labels, values)],
                colors=colors, autopct='%1.0f%%', startangle=90,
                textprops={'fontsize': 10})
    axes[0].set_title('All Conditions', fontsize=11, fontweight='bold')

    # Per combination bar
    combo_counts = df.groupby(['combination_name', 'interaction_type']).size().unstack(fill_value=0)
    combo_order = list(PAIR_NAMES.keys()) + list(TRIPLE_NAME.keys())
    combo_order = [c for c in combo_order if c in combo_counts.index]
    combo_counts = combo_counts.reindex(combo_order)

    combo_labels = [PAIR_NAMES.get(c, TRIPLE_NAME.get(c, c)) for c in combo_order]

    x = np.arange(len(combo_order))
    bottom = np.zeros(len(combo_order))
    for itype in ['sub-additive', 'additive', 'synergistic']:
        if itype in combo_counts.columns:
            vals = combo_counts[itype].values
            axes[1].bar(x, vals, bottom=bottom, color=colors_map[itype],
                        label=itype, edgecolor='black', linewidth=0.5)
            bottom += vals

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(combo_labels, fontsize=8, rotation=30, ha='right')
    axes[1].set_ylabel('Count', fontsize=10)
    axes[1].set_title('Per Combination', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=8)

    fig.suptitle(f'Interaction Type Distribution — {model}', fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f'interaction_distribution_{model}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────
# Plot 6: Triple corruption — severity heatmap
# ──────────────────────────────────────────────
def plot_triple_heatmap(summary, model='IForest'):
    df = summary[(summary['model'] == model) &
                 (summary['combination_name'] == 'noise_spikes_missing')].copy()
    if df.empty:
        return

    baseline_row = summary[(summary['model'] == model) & (summary['combination_name'] == 'baseline')]
    baseline_auc = baseline_row['mean_AUC_ROC'].values[0] if not baseline_row.empty else 0.7

    # Parse: condition like noise_low+spikes_med+missing_high
    rows_data = []
    for _, row in df.iterrows():
        parts = row['condition'].split('+')
        if len(parts) == 3:
            sn = parts[0].split('_')[-1]
            ss = parts[1].split('_')[-1]
            sm = parts[2].split('_')[-1]
            rows_data.append({
                'noise': sn, 'spikes': ss, 'missing': sm,
                'auc': row['mean_AUC_ROC'],
                'drop': baseline_auc - row['mean_AUC_ROC']
            })

    if not rows_data:
        return

    tdf = pd.DataFrame(rows_data)

    # Create 3 heatmaps (one per missing severity)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for k, miss_sev in enumerate(SEVERITY_ORDER):
        sub = tdf[tdf['missing'] == miss_sev]
        matrix = np.full((3, 3), np.nan)
        for _, r in sub.iterrows():
            i = SEVERITY_ORDER.index(r['noise'])
            j = SEVERITY_ORDER.index(r['spikes'])
            matrix[i, j] = r['auc']

        vmin = tdf['auc'].min() - 0.02
        vmax = baseline_auc

        im = axes[k].imshow(matrix, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')

        axes[k].set_xticks(range(3))
        axes[k].set_xticklabels(['Low', 'Med', 'High'], fontsize=9)
        axes[k].set_yticks(range(3))
        if k == 0:
            axes[k].set_yticklabels(['Low', 'Med', 'High'], fontsize=9)
            axes[k].set_ylabel('Noise Severity', fontsize=10)
        axes[k].set_xlabel('Spikes Severity', fontsize=10)
        axes[k].set_title(f'Missing = {miss_sev.capitalize()}', fontsize=11, fontweight='bold')

        for i in range(3):
            for j in range(3):
                if not np.isnan(matrix[i, j]):
                    color = 'white' if matrix[i, j] < (vmin + (vmax - vmin) * 0.4) else 'black'
                    axes[k].text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                                 fontsize=9, fontweight='bold', color=color)

    fig.colorbar(im, ax=axes, shrink=0.8, label='AUC-ROC')
    fig.suptitle(f'Triple Corruption: Noise × Spikes × Missing — {model}\n(Baseline: {baseline_auc:.3f})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f'triple_heatmap_{model}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────
# Plot 7: Worst-case degradation ranking
# ──────────────────────────────────────────────
def plot_worst_case_ranking(summary, model='IForest'):
    df = summary[summary['model'] == model].copy()
    if df.empty:
        return

    baseline_row = df[df['combination_name'] == 'baseline']
    if baseline_row.empty:
        return
    baseline_auc = baseline_row['mean_AUC_ROC'].values[0]

    # Get worst (min AUC) per combination
    worst = df[df['combination_name'] != 'baseline'].groupby('combination_name')['mean_AUC_ROC'].min().reset_index()
    worst['drop'] = baseline_auc - worst['mean_AUC_ROC']
    worst = worst.sort_values('drop', ascending=True)

    all_names = {**PAIR_NAMES, **TRIPLE_NAME, 'single': 'Singles (worst)'}
    labels = [all_names.get(c, c) for c in worst['combination_name']]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#7E2F8E' if c in TRIPLE_NAME else '#D95319' if c in PAIR_NAMES else '#4DBEEE'
              for c in worst['combination_name']]

    bars = ax.barh(range(len(labels)), worst['drop'].values, color=colors,
                   edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Worst-Case AUC-ROC Drop', fontsize=11)
    ax.set_title(f'Worst-Case Degradation by Combination — {model}', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, worst['drop'].values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    legend_handles = [
        mpatches.Patch(color='#4DBEEE', label='Single'),
        mpatches.Patch(color='#D95319', label='Pair'),
        mpatches.Patch(color='#7E2F8E', label='Triple'),
    ]
    ax.legend(handles=legend_handles, fontsize=9)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f'worst_case_ranking_{model}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    checkpoint, interaction, summary = load_data()

    models = interaction['model'].unique()
    print(f"Models found: {models}")

    for model in models:
        print(f"\n{'='*50}")
        print(f"Generating plots for: {model}")
        print(f"{'='*50}")

        plot_interaction_heatmap(interaction, model)
        plot_compound_vs_additive(interaction, model)
        plot_auc_by_condition_type(summary, model)
        plot_pair_heatmaps(summary, model)
        plot_interaction_distribution(interaction, model)
        plot_triple_heatmap(summary, model)
        plot_worst_case_ranking(summary, model)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == '__main__':
    main()
