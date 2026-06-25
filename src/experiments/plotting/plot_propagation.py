"""Plot propagation analysis results.

Generates 5 plots:
  1. Bar chart: propagation ratio per model x corruption type
  2. Zone decay: line chart showing score diff across zones
  3. Heatmap: model x corruption -> propagation ratio
  4. Severity comparison: low/med/high per model
  5. Zone decay per corruption type (3 subplots)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "propagation")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

MODEL_COLORS = {
    'IForest': '#1f77b4',
    'LOF': '#d62728',
    'MP': '#2ca02c',
    'AE': '#ff7f0e',
}
MODEL_ORDER = ['IForest', 'LOF', 'MP', 'AE']
CORRUPTION_ORDER = ['noise', 'spikes', 'missing']
CORRUPTION_LABELS = {'noise': 'Noise', 'spikes': 'Spikes', 'missing': 'Missing'}
ZONE_LABELS = ['Corruption\nZone', 'Near\n(0-10%)', 'Mid\n(10-30%)', 'Far\n(>30%)']


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    compact = pd.read_csv(os.path.join(RESULTS_DIR, "propagation_compact.csv"))
    zones = pd.read_csv(os.path.join(RESULTS_DIR, "propagation_zones.csv"))
    summary = pd.read_csv(os.path.join(RESULTS_DIR, "propagation_summary.csv"))

    print("=== Propagation Analysis Plots ===")
    print(f"Compact: {len(compact)} rows")
    print(f"Zones: {len(zones)} rows")
    print(f"Summary: {len(summary)} rows")
    print()

    # Print compact table
    print("Propagation Ratios:")
    print(compact.to_string(index=False))
    print()

    # ================================================================
    # Plot 1: Bar chart — propagation ratio per model x corruption
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(CORRUPTION_ORDER))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, model in enumerate(MODEL_ORDER):
        ratios = []
        for ct in CORRUPTION_ORDER:
            row = compact[compact['corruption_type'] == ct]
            ratios.append(float(row[model].iloc[0]) if len(row) > 0 else 0)
        bars = ax.bar(x + offsets[i] * width, ratios, width,
                      label=model, color=MODEL_COLORS[model], edgecolor='white')
        for bar, val in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Corruption Type')
    ax.set_ylabel('Propagation Ratio')
    ax.set_title('Score Propagation Ratio by Model and Corruption Type')
    ax.set_xticks(x)
    ax.set_xticklabels([CORRUPTION_LABELS[c] for c in CORRUPTION_ORDER])
    ax.legend(loc='upper left')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No propagation')
    ax.set_ylim(0, max(compact[MODEL_ORDER].max().max() + 1.5, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "propagation_ratio_bar.png"), dpi=150)
    plt.close()
    print("Saved: propagation_ratio_bar.png")

    # ================================================================
    # Plot 2: Zone decay — all models, averaged across corruptions
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    zone_cols = ['mean_diff_corruption_zone', 'mean_diff_near', 'mean_diff_mid', 'mean_diff_far']

    for model in MODEL_ORDER:
        model_zones = zones[zones['model'] == model]
        means = [model_zones[col].mean() for col in zone_cols]
        ax.plot(range(4), means, 'o-', color=MODEL_COLORS[model],
                label=model, linewidth=2, markersize=8)

    ax.set_xticks(range(4))
    ax.set_xticklabels(ZONE_LABELS)
    ax.set_ylabel('Mean Score Difference (normalized)')
    ax.set_title('Score Difference Decay Across Zones')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "propagation_zone_decay.png"), dpi=150)
    plt.close()
    print("Saved: propagation_zone_decay.png")

    # ================================================================
    # Plot 3: Heatmap — model x corruption -> propagation ratio
    # ================================================================
    fig, ax = plt.subplots(figsize=(7, 4))

    heatmap_data = np.zeros((len(MODEL_ORDER), len(CORRUPTION_ORDER)))
    for i, model in enumerate(MODEL_ORDER):
        for j, ct in enumerate(CORRUPTION_ORDER):
            row = compact[compact['corruption_type'] == ct]
            heatmap_data[i, j] = float(row[model].iloc[0]) if len(row) > 0 else 0

    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(CORRUPTION_ORDER)))
    ax.set_xticklabels([CORRUPTION_LABELS[c] for c in CORRUPTION_ORDER])
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)

    for i in range(len(MODEL_ORDER)):
        for j in range(len(CORRUPTION_ORDER)):
            val = heatmap_data[i, j]
            color = 'white' if val > 4 else 'black'
            ax.text(j, i, f'{val:.1f}x', ha='center', va='center',
                    fontweight='bold', fontsize=12, color=color)

    ax.set_title('Propagation Ratio Heatmap')
    plt.colorbar(im, ax=ax, label='Propagation Ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "propagation_heatmap.png"), dpi=150)
    plt.close()
    print("Saved: propagation_heatmap.png")

    # ================================================================
    # Plot 4: Severity comparison — grouped bar per model
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    severity_order = ['low', 'med', 'high']
    severity_colors = {'low': '#a6d854', 'med': '#ffd92f', 'high': '#e41a1c'}

    for idx, ct in enumerate(CORRUPTION_ORDER):
        ax = axes[idx]
        ct_data = summary[summary['corruption_type'] == ct]

        x = np.arange(len(MODEL_ORDER))
        width = 0.25
        sev_offsets = [-1, 0, 1]

        for si, sev in enumerate(severity_order):
            ratios = []
            for model in MODEL_ORDER:
                row = ct_data[(ct_data['model'] == model) & (ct_data['severity'] == sev)]
                ratios.append(float(row['mean_ratio'].iloc[0]) if len(row) > 0 else 0)
            ax.bar(x + sev_offsets[si] * width, ratios, width,
                   label=sev.capitalize(), color=severity_colors[sev], edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=45)
        ax.set_title(CORRUPTION_LABELS[ct])
        ax.grid(True, alpha=0.3, axis='y')
        if idx == 0:
            ax.set_ylabel('Propagation Ratio')
            ax.legend()

    fig.suptitle('Propagation Ratio by Severity Level', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "propagation_severity.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: propagation_severity.png")

    # ================================================================
    # Plot 5: Zone decay per corruption type (3 subplots)
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, ct in enumerate(CORRUPTION_ORDER):
        ax = axes[idx]
        ct_zones = zones[zones['corruption_type'] == ct]

        for model in MODEL_ORDER:
            model_row = ct_zones[ct_zones['model'] == model]
            if len(model_row) == 0:
                continue
            vals = [float(model_row[col].iloc[0]) for col in zone_cols]
            ax.plot(range(4), vals, 'o-', color=MODEL_COLORS[model],
                    label=model, linewidth=2, markersize=7)

        ax.set_xticks(range(4))
        ax.set_xticklabels(ZONE_LABELS, fontsize=9)
        ax.set_title(CORRUPTION_LABELS[ct])
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel('Mean Score Difference')
            ax.legend()

    fig.suptitle('Zone Decay by Corruption Type', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "propagation_zone_decay_per_type.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: propagation_zone_decay_per_type.png")

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
