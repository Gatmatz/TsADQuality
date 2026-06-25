
import matplotlib.pyplot as plt
import os

# Data extracted from build_chapter4_spikes_v2.py
FRACTIONS = [0.01, 0.05, 0.10, 0.20]
MULTIPLIERS = [3.0, 5.0, 10.0]

AUC = {
    'IForest': {
        (0.01, 3.0): 0.669, (0.01, 5.0): 0.659, (0.01, 10.0): 0.640,
        (0.05, 3.0): 0.579, (0.05, 5.0): 0.522, (0.05, 10.0): 0.437,
        (0.10, 3.0): 0.498, (0.10, 5.0): 0.428, (0.10, 10.0): 0.329,
        (0.20, 3.0): 0.393, (0.20, 5.0): 0.306, (0.20, 10.0): 0.209,
    },
    'LOF': {
        (0.01, 3.0): 0.506, (0.01, 5.0): 0.499, (0.01, 10.0): 0.512,
        (0.05, 3.0): 0.374, (0.05, 5.0): 0.340, (0.05, 10.0): 0.331,
        (0.10, 3.0): 0.355, (0.10, 5.0): 0.323, (0.10, 10.0): 0.306,
        (0.20, 3.0): 0.331, (0.20, 5.0): 0.296, (0.20, 10.0): 0.288,
    },
    'MP': {
        (0.01, 3.0): 0.641, (0.01, 5.0): 0.639, (0.01, 10.0): 0.638,
        (0.05, 3.0): 0.456, (0.05, 5.0): 0.442, (0.05, 10.0): 0.439,
        (0.10, 3.0): 0.384, (0.10, 5.0): 0.373, (0.10, 10.0): 0.378,
        (0.20, 3.0): 0.321, (0.20, 5.0): 0.318, (0.20, 10.0): 0.345,
    },
    'AE': {
        (0.01, 3.0): 0.607, (0.01, 5.0): 0.581, (0.01, 10.0): 0.567,
        (0.05, 3.0): 0.539, (0.05, 5.0): 0.515, (0.05, 10.0): 0.501,
        (0.10, 3.0): 0.490, (0.10, 5.0): 0.466, (0.10, 10.0): 0.447,
        (0.20, 3.0): 0.403, (0.20, 5.0): 0.374, (0.20, 10.0): 0.340,
    },
}
BASELINE_AUC = {'IForest': 0.695, 'LOF': 0.674, 'MP': 0.731, 'AE': 0.725}
COLORS = {'IForest': '#1f77b4', 'LOF': '#2ca02c', 'MP': '#d62728', 'AE': '#9467bd'}
MARKERS = {'IForest': 'o', 'LOF': 's', 'MP': '^', 'AE': 'D'}

plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Subplot 1: Effect of Fraction
for alg in ['IForest', 'LOF', 'MP', 'AE']:
    # Average over multipliers for each fraction
    y_vals = [sum(AUC[alg][(f, m)] for m in MULTIPLIERS) / len(MULTIPLIERS) for f in FRACTIONS]
    x_vals = [f * 100 for f in FRACTIONS]
    axes[0].plot(x_vals, y_vals, label=alg, color=COLORS[alg], marker=MARKERS[alg], linewidth=2.5, markersize=8)
    # Baseline
    axes[0].axhline(y=BASELINE_AUC[alg], color=COLORS[alg], linestyle='--', alpha=0.3, linewidth=1)

axes[0].set_title('(a) Effect of Fraction', fontweight='bold')
axes[0].set_xlabel('Fraction (%)')
axes[0].set_ylabel('Mean AUC-ROC')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].set_xticks(x_vals)

# Subplot 2: Effect of Multiplier
for alg in ['IForest', 'LOF', 'MP', 'AE']:
    # Average over fractions for each multiplier
    y_vals = [sum(AUC[alg][(f, m)] for f in FRACTIONS) / len(FRACTIONS) for m in MULTIPLIERS]
    axes[1].plot(MULTIPLIERS, y_vals, label=alg, color=COLORS[alg], marker=MARKERS[alg], linewidth=2.5, markersize=8)
    # Baseline
    axes[1].axhline(y=BASELINE_AUC[alg], color=COLORS[alg], linestyle='--', alpha=0.3, linewidth=1)

axes[1].set_title('(b) Effect of Multiplier', fontweight='bold')
axes[1].set_xlabel('Multiplier')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].set_xticks(MULTIPLIERS)

# Add a horizontal line for random performance
for ax in axes:
    ax.axhline(y=0.5, color='black', linestyle='-', alpha=0.2, label='Random (0.5)')

# Global Legend
handles, labels = axes[0].get_legend_handles_labels()
# Remove duplicates in legend
unique_labels = dict(zip(labels, handles))
fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=5)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('spikes_performance_academic.png', dpi=300, bbox_inches='tight')
print("Plot saved as spikes_performance_academic.png")
