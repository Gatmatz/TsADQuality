"""
Centralized plotting style for thesis figures.

Import this module in every plot_*.py script to ensure:
1. Consistent algorithm colors across all figures.
2. Consistent, visible baseline lines (Toliopoulos feedback).

Usage:
    from _style import ALGO_COLORS, ALGO_MARKERS, baseline_kwargs, apply_style

    apply_style()  # call once at top of script
    ax.plot(..., color=ALGO_COLORS['IForest'], marker=ALGO_MARKERS['IForest'], label='IForest')
    ax.axhline(y=base, color=ALGO_COLORS['IForest'], **baseline_kwargs())
"""
import matplotlib as mpl

# ---------- Algorithm palette (Tableau / colorblind-friendly) ----------
ALGO_COLORS = {
    'IForest':         '#1f77b4',  # blue
    'Isolation Forest':'#1f77b4',
    'IF':              '#1f77b4',
    'LOF':             '#d62728',  # red
    'MP':              '#2ca02c',  # green
    'Matrix Profile':  '#2ca02c',
    'AE':              '#ff7f0e',  # orange
    'Autoencoder':     '#ff7f0e',
    'PCA':             '#9467bd',  # purple (rare)
}

ALGO_MARKERS = {
    'IForest':         'o',
    'Isolation Forest':'o',
    'IF':              'o',
    'LOF':             's',
    'MP':              '^',
    'Matrix Profile':  '^',
    'AE':              'D',
    'Autoencoder':     'D',
    'PCA':             'v',
}

# ---------- Baseline line style ----------
def baseline_kwargs(color='black', label=None):
    """Standard kwargs for ax.axhline(...) baseline lines.

    Returns kwargs that produce a prominent dashed baseline (linewidth=2,
    alpha=0.85, zorder above default curves so the line is readable).
    """
    kw = dict(
        color=color,
        linestyle='--',
        linewidth=2.0,     # was 1
        alpha=0.85,        # was 0.5
        zorder=5,          # above default curves
    )
    if label is not None:
        kw['label'] = label
    return kw

# ---------- Global matplotlib defaults ----------
def apply_style():
    """Apply consistent matplotlib defaults across all thesis figures."""
    mpl.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 1.8,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })
