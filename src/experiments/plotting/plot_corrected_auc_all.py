"""Generate max(AUC, 1-AUC) and high/low baseline split plots."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_141_iforest.csv")
RESULTS_BASE = os.path.join(PROJECT_ROOT, "results", "experiments")


def corrected(auc):
    return max(auc, 1 - auc)


def setup_ax(ax, ylabel, title):
    ax.set_xlabel('Fraction', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    baseline = pd.read_csv(BASELINE_CSV)
    baseline_mean = baseline['AUC_ROC'].mean()
    baseline_corr = baseline['AUC_ROC'].apply(corrected).mean()
    high_files = set(baseline[baseline['AUC_ROC'] >= 0.5]['file'].tolist())
    low_files = set(baseline[baseline['AUC_ROC'] < 0.5]['file'].tolist())
    baseline_high = baseline[baseline['AUC_ROC'] >= 0.5]['AUC_ROC'].mean()
    baseline_low = baseline[baseline['AUC_ROC'] < 0.5]['AUC_ROC'].mean()

    print(f"High baseline files: {len(high_files)}, Low: {len(low_files)}")

    # --- 1. Spikes normal only ---
    exp_dir = os.path.join(RESULTS_BASE, "spikes_normal_only")
    plots_dir = os.path.join(exp_dir, "plots")
    df = pd.read_csv(os.path.join(exp_dir, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]
    df['AUC_corr'] = df['AUC_ROC'].apply(corrected)

    MULT_COLORS = {3.0: '#377eb8', 5.0: '#4daf4a', 10.0: '#e41a1c'}
    MULT_MARKERS = {3.0: 'o', 5.0: '^', 10.0: 's'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for mult in sorted(df['multiplier'].unique()):
        sub = df[df['multiplier'] == mult]
        lbl = f'mult={int(mult)}x'
        mk = MULT_MARKERS.get(mult, 'o')
        cl = MULT_COLORS.get(mult, 'gray')
        # All - AUC-ROC
        m = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        axes[0, 0].plot(m['fraction'], m['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        # All - max(AUC, 1-AUC)
        mc = sub.groupby('fraction')['AUC_corr'].mean().reset_index().sort_values('fraction')
        axes[0, 1].plot(mc['fraction'], mc['AUC_corr'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        # High baseline
        sub_h = sub[sub['file'].isin(high_files)]
        if not sub_h.empty:
            mh = sub_h.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            axes[1, 0].plot(mh['fraction'], mh['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        # Low baseline
        sub_l = sub[sub['file'].isin(low_files)]
        if not sub_l.empty:
            ml = sub_l.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            axes[1, 1].plot(ml['fraction'], ml['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)

    axes[0, 0].axhline(y=baseline_mean, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[0, 0], 'AUC-ROC', 'AUC-ROC (141 αρχεια)')
    axes[0, 1].axhline(y=baseline_corr, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[0, 1], 'max(AUC, 1-AUC)', 'max(AUC, 1-AUC) (141 αρχεια)')
    axes[1, 0].axhline(y=baseline_high, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[1, 0], 'AUC-ROC', f'{len(high_files)} αρχεια (baseline >= 0.5)')
    axes[1, 1].axhline(y=baseline_low, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[1, 1], 'AUC-ROC', f'{len(low_files)} αρχεια (baseline < 0.5)')

    fig.suptitle('Spikes (Normal Only)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'spikes_normal_corrected_auc.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: spikes_normal_corrected_auc.png")

    # --- 2. MNAR point ---
    exp_dir = os.path.join(RESULTS_BASE, "missing_mnar")
    plots_dir = os.path.join(exp_dir, "plots")
    df = pd.read_csv(os.path.join(exp_dir, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]
    df['AUC_corr'] = df['AUC_ROC'].apply(corrected)

    MECH_COLORS = {'mcar': '#377eb8', 'mnar_extreme': '#e41a1c', 'mnar_high': '#ff7f00'}
    MECH_MARKERS = {'mcar': 'o', 'mnar_extreme': 's', 'mnar_high': '^'}
    MECH_LABELS = {'mcar': 'MCAR', 'mnar_extreme': 'MNAR (extreme)', 'mnar_high': 'MNAR (high)'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for mech in ['mcar', 'mnar_extreme', 'mnar_high']:
        sub = df[df['mechanism'] == mech]
        if sub.empty: continue
        lbl = MECH_LABELS[mech]
        mk = MECH_MARKERS[mech]
        cl = MECH_COLORS[mech]
        m = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        axes[0, 0].plot(m['fraction'], m['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        mc = sub.groupby('fraction')['AUC_corr'].mean().reset_index().sort_values('fraction')
        axes[0, 1].plot(mc['fraction'], mc['AUC_corr'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        sub_h = sub[sub['file'].isin(high_files)]
        if not sub_h.empty:
            mh = sub_h.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            axes[1, 0].plot(mh['fraction'], mh['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        sub_l = sub[sub['file'].isin(low_files)]
        if not sub_l.empty:
            ml = sub_l.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            axes[1, 1].plot(ml['fraction'], ml['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)

    axes[0, 0].axhline(y=baseline_mean, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[0, 0], 'AUC-ROC', 'AUC-ROC (141 αρχεια)')
    axes[0, 1].axhline(y=baseline_corr, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[0, 1], 'max(AUC, 1-AUC)', 'max(AUC, 1-AUC) (141 αρχεια)')
    axes[1, 0].axhline(y=baseline_high, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[1, 0], 'AUC-ROC', f'{len(high_files)} αρχεια (baseline >= 0.5)')
    axes[1, 1].axhline(y=baseline_low, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[1, 1], 'AUC-ROC', f'{len(low_files)} αρχεια (baseline < 0.5)')

    fig.suptitle('MCAR vs MNAR (Point)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'mnar_corrected_auc.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: mnar_corrected_auc.png")

    # --- 3. MNAR burst ---
    exp_dir = os.path.join(RESULTS_BASE, "missing_mnar_burst")
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(exp_dir, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]
    df['AUC_corr'] = df['AUC_ROC'].apply(corrected)

    MECH_COLORS2 = {'mnar_extreme_burst': '#e41a1c', 'mnar_high_burst': '#ff7f00'}
    MECH_MARKERS2 = {'mnar_extreme_burst': 's', 'mnar_high_burst': '^'}
    MECH_LABELS2 = {'mnar_extreme_burst': 'MNAR extreme', 'mnar_high_burst': 'MNAR high'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for mech in ['mnar_extreme_burst', 'mnar_high_burst']:
        sub = df[(df['mechanism'] == mech) & (df['num_bursts'] == 1)]
        if sub.empty: continue
        lbl = MECH_LABELS2[mech]
        mk = MECH_MARKERS2[mech]
        cl = MECH_COLORS2[mech]
        m = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        axes[0, 0].plot(m['fraction'], m['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        mc = sub.groupby('fraction')['AUC_corr'].mean().reset_index().sort_values('fraction')
        axes[0, 1].plot(mc['fraction'], mc['AUC_corr'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        sub_h = sub[sub['file'].isin(high_files)]
        if not sub_h.empty:
            mh = sub_h.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            axes[1, 0].plot(mh['fraction'], mh['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        sub_l = sub[sub['file'].isin(low_files)]
        if not sub_l.empty:
            ml = sub_l.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            axes[1, 1].plot(ml['fraction'], ml['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)

    axes[0, 0].axhline(y=baseline_mean, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[0, 0], 'AUC-ROC', 'AUC-ROC (141 αρχεια)')
    axes[0, 1].axhline(y=baseline_corr, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[0, 1], 'max(AUC, 1-AUC)', 'max(AUC, 1-AUC) (141 αρχεια)')
    axes[1, 0].axhline(y=baseline_high, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[1, 0], 'AUC-ROC', f'{len(high_files)} αρχεια (baseline >= 0.5)')
    axes[1, 1].axhline(y=baseline_low, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[1, 1], 'AUC-ROC', f'{len(low_files)} αρχεια (baseline < 0.5)')

    fig.suptitle('MNAR Burst (nb=1)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'mnar_burst_corrected_auc.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: mnar_burst_corrected_auc.png")

    # --- 4. Permutation ---
    exp_dir = os.path.join(RESULTS_BASE, "swap_permutation")
    plots_dir = os.path.join(exp_dir, "plots")
    df = pd.read_csv(os.path.join(exp_dir, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]
    df['AUC_corr'] = df['AUC_ROC'].apply(corrected)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # All - AUC-ROC
    means = df.groupby('n_segments')['AUC_ROC'].mean().reset_index().sort_values('n_segments')
    axes[0, 0].bar(range(len(means)), means['AUC_ROC'], color='#377eb8', alpha=0.8)
    axes[0, 0].set_xticks(range(len(means)))
    axes[0, 0].set_xticklabels([f'N={int(n)}' for n in means['n_segments']])
    axes[0, 0].axhline(y=baseline_mean, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    axes[0, 0].set_ylabel('AUC-ROC', fontsize=11)
    axes[0, 0].set_title('AUC-ROC (141 αρχεια)', fontsize=11)
    axes[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # All - max(AUC, 1-AUC)
    means_c = df.groupby('n_segments')['AUC_corr'].mean().reset_index().sort_values('n_segments')
    axes[0, 1].bar(range(len(means_c)), means_c['AUC_corr'], color='#377eb8', alpha=0.8)
    axes[0, 1].set_xticks(range(len(means_c)))
    axes[0, 1].set_xticklabels([f'N={int(n)}' for n in means_c['n_segments']])
    axes[0, 1].axhline(y=baseline_corr, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    axes[0, 1].set_ylabel('max(AUC, 1-AUC)', fontsize=11)
    axes[0, 1].set_title('max(AUC, 1-AUC) (141 αρχεια)', fontsize=11)
    axes[0, 1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # High baseline
    df_h = df[df['file'].isin(high_files)]
    means_h = df_h.groupby('n_segments')['AUC_ROC'].mean().reset_index().sort_values('n_segments')
    axes[1, 0].bar(range(len(means_h)), means_h['AUC_ROC'], color='#377eb8', alpha=0.8)
    axes[1, 0].set_xticks(range(len(means_h)))
    axes[1, 0].set_xticklabels([f'N={int(n)}' for n in means_h['n_segments']])
    axes[1, 0].axhline(y=baseline_high, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    axes[1, 0].set_ylabel('AUC-ROC', fontsize=11)
    axes[1, 0].set_title(f'{len(high_files)} αρχεια (baseline >= 0.5)', fontsize=11)
    axes[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Low baseline
    df_l = df[df['file'].isin(low_files)]
    means_l = df_l.groupby('n_segments')['AUC_ROC'].mean().reset_index().sort_values('n_segments')
    axes[1, 1].bar(range(len(means_l)), means_l['AUC_ROC'], color='#377eb8', alpha=0.8)
    axes[1, 1].set_xticks(range(len(means_l)))
    axes[1, 1].set_xticklabels([f'N={int(n)}' for n in means_l['n_segments']])
    axes[1, 1].axhline(y=baseline_low, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    axes[1, 1].set_ylabel('AUC-ROC', fontsize=11)
    axes[1, 1].set_title(f'{len(low_files)} αρχεια (baseline < 0.5)', fontsize=11)
    axes[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Permutation', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'permutation_corrected_auc.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: permutation_corrected_auc.png")

    # --- 5. Freeze ---
    exp_dir = os.path.join(RESULTS_BASE, "freeze")
    plots_dir = os.path.join(exp_dir, "plots")
    df = pd.read_csv(os.path.join(exp_dir, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]

    NS_COLORS = {1: '#e41a1c', 3: '#4daf4a', 5: '#984ea3', 10: '#ff7f00'}
    NS_MARKERS = {1: 's', 3: '^', 5: 'D', 10: 'v'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ns in sorted(df['num_stucks'].unique()):
        ns = int(ns)
        sub = df[df['num_stucks'] == ns]
        lbl = f'ns={ns}'
        mk = NS_MARKERS.get(ns, 'o')
        cl = NS_COLORS.get(ns, 'gray')
        # All
        m = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        axes[0].plot(m['fraction'], m['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        # High
        sub_h = sub[sub['file'].isin(high_files)]
        if not sub_h.empty:
            mh = sub_h.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            axes[1].plot(mh['fraction'], mh['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        # Low
        sub_l = sub[sub['file'].isin(low_files)]
        if not sub_l.empty:
            ml = sub_l.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            axes[2].plot(ml['fraction'], ml['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)

    axes[0].axhline(y=baseline_mean, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[0], 'AUC-ROC', 'AUC-ROC (141 αρχεια)')
    axes[1].axhline(y=baseline_high, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[1], 'AUC-ROC', f'{len(high_files)} αρχεια (baseline >= 0.5)')
    axes[2].axhline(y=baseline_low, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[2], 'AUC-ROC', f'{len(low_files)} αρχεια (baseline < 0.5)')

    fig.suptitle('Freeze (Sensor Stuck)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'freeze_high_vs_low.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: freeze_high_vs_low.png")

    # --- 6. Swap Point ---
    exp_dir = os.path.join(RESULTS_BASE, "swap_point")
    plots_dir = os.path.join(exp_dir, "plots")
    df = pd.read_csv(os.path.join(exp_dir, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_i, (files, title, bl) in enumerate([
        (None, 'AUC-ROC (141 αρχεια)', baseline_mean),
        (high_files, f'{len(high_files)} αρχεια (baseline >= 0.5)', baseline_high),
        (low_files, f'{len(low_files)} αρχεια (baseline < 0.5)', baseline_low),
    ]):
        sub = df if files is None else df[df['file'].isin(files)]
        m = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        axes[ax_i].plot(m['fraction'], m['AUC_ROC'], marker='o', color='#e41a1c', linewidth=2, markersize=7)
        axes[ax_i].axhline(y=bl, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
        axes[ax_i].axhline(y=0.5, color='gray', linestyle=':', linewidth=2.0, alpha=0.85, zorder=5)
        axes[ax_i].set_xlabel('Fraction', fontsize=11)
        axes[ax_i].set_ylabel('AUC-ROC', fontsize=11)
        axes[ax_i].set_title(title, fontsize=11)
        axes[ax_i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        axes[ax_i].grid(True, alpha=0.3)

    fig.suptitle('Point Swap', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'swap_point_high_vs_low.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: swap_point_high_vs_low.png")

    # --- 7. Swap Segment ---
    exp_dir = os.path.join(RESULTS_BASE, "swap_segment")
    plots_dir = os.path.join(exp_dir, "plots")
    df = pd.read_csv(os.path.join(exp_dir, "checkpoint.csv"))
    df = df[(df['model'] == 'IForest') & (df['error'].isna())]

    NS_COLORS_SW = {1: '#e41a1c', 3: '#4daf4a', 5: '#984ea3', 10: '#ff7f00'}
    NS_MARKERS_SW = {1: 's', 3: '^', 5: 'D', 10: 'v'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ns in sorted(df['num_swaps'].unique()):
        ns = int(ns)
        sub = df[df['num_swaps'] == ns]
        lbl = f'ns={ns}'
        mk = NS_MARKERS_SW.get(ns, 'o')
        cl = NS_COLORS_SW.get(ns, 'gray')
        m = sub.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
        axes[0].plot(m['fraction'], m['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        sub_h = sub[sub['file'].isin(high_files)]
        if not sub_h.empty:
            mh = sub_h.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            axes[1].plot(mh['fraction'], mh['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)
        sub_l = sub[sub['file'].isin(low_files)]
        if not sub_l.empty:
            ml = sub_l.groupby('fraction')['AUC_ROC'].mean().reset_index().sort_values('fraction')
            axes[2].plot(ml['fraction'], ml['AUC_ROC'], marker=mk, color=cl, label=lbl, linewidth=2, markersize=6)

    axes[0].axhline(y=baseline_mean, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[0], 'AUC-ROC', 'AUC-ROC (141 αρχεια)')
    axes[1].axhline(y=baseline_high, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[1], 'AUC-ROC', f'{len(high_files)} αρχεια (baseline >= 0.5)')
    axes[2].axhline(y=baseline_low, color='black', linestyle='--', linewidth=2.0, alpha=0.85, zorder=5)
    setup_ax(axes[2], 'AUC-ROC', f'{len(low_files)} αρχεια (baseline < 0.5)')

    fig.suptitle('Segment Swap', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'swap_segment_high_vs_low.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: swap_segment_high_vs_low.png")

    print("\nDone.")


if __name__ == '__main__':
    main()
