"""
FP vs FN Decomposition Analysis

For each corruption type x severity x model, decomposes the performance drop into:
  - Precision drop -> more False Positives (corruption mistaken for anomaly)
  - Recall drop    -> more False Negatives (real anomalies missed)

This reveals HOW each corruption damages detection, not just how much.

No new experiments needed — uses existing summary CSVs.

Output (results/experiments/fp_fn_decomposition/):
  - fp_fn_decomposition.csv     (full table)
  - fp_fn_by_corruption.csv     (aggregated per corruption type x model)
  - dominant_error_type.csv     (which error dominates per combination)
  - Console summary

Usage:
    python fp_fn_decomposition.py
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "fp_fn_decomposition")

BASELINE_CSV = os.path.join(PROJECT_ROOT, "results", "tables", "baseline_final_subset.csv")
NOISE_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "white_noise_snr", "summary.csv")
SPIKES_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "spikes_normal_only", "summary.csv")
MISSING_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "missing_true_impact", "summary.csv")
FREEZE_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "freeze", "summary.csv")
SWAP_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "swap", "summary.csv")
GE_SUMMARY = os.path.join(PROJECT_ROOT, "results", "experiments", "gilbert_elliott_true_impact", "summary.csv")


def load_baseline():
    bl = pd.read_csv(BASELINE_CSV)
    bl['model'] = bl['model'].replace('Autoencoder', 'AE')
    baseline = bl.groupby('model').agg(
        bl_Precision=('Precision', 'mean'),
        bl_Recall=('Recall', 'mean'),
        bl_F1=('F1', 'mean'),
        bl_AUC_ROC=('AUC_ROC', 'mean'),
        bl_AUC_PR=('AUC_PR', 'mean'),
    ).reset_index()
    return baseline


def load_noise():
    df = pd.read_csv(NOISE_SUMMARY)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'corruption': 'noise',
            'condition': f"SNR={int(r['snr_db'])}dB",
            'severity_order': -r['snr_db'],  # lower SNR = higher severity
            'model': r['model'],
            'Precision': r['mean_Precision'],
            'Recall': r['mean_Recall'],
            'F1': r['mean_F'],
            'AUC_ROC': r['mean_AUC_ROC'],
            'AUC_PR': r['mean_AUC_PR'],
        })
    return pd.DataFrame(rows)


def load_spikes():
    df = pd.read_csv(SPIKES_SUMMARY)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'corruption': 'spikes',
            'condition': f"frac={r['fraction']},mult={int(r['multiplier'])}",
            'severity_order': r['fraction'] * r['multiplier'],
            'model': r['model'],
            'Precision': r['mean_Precision'],
            'Recall': r['mean_Recall'],
            'F1': r['mean_F'],
            'AUC_ROC': r['mean_AUC_ROC'],
            'AUC_PR': r['mean_AUC_PR'],
        })
    return pd.DataFrame(rows)


def load_missing():
    df = pd.read_csv(MISSING_SUMMARY)
    rows = []
    for _, r in df.iterrows():
        mtype = r['missing_type']
        frac = r['fraction']
        if mtype == 'point':
            cond = f"point,frac={frac}"
        else:
            nb = int(r['num_bursts'])
            cond = f"burst,frac={frac},nb={nb}"
        rows.append({
            'corruption': f'missing_{mtype}',
            'condition': cond,
            'severity_order': frac,
            'model': r['model'],
            'Precision': r['mean_Precision'],
            'Recall': r['mean_Recall'],
            'F1': r['mean_F'],
            'AUC_ROC': r['mean_AUC_ROC'],
            'AUC_PR': r['mean_AUC_PR'],
        })
    return pd.DataFrame(rows)


def load_freeze():
    if not os.path.exists(FREEZE_SUMMARY):
        return pd.DataFrame()
    df = pd.read_csv(FREEZE_SUMMARY)
    if 'mean_Precision' not in df.columns:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'corruption': 'freeze',
            'condition': r.get('condition', str(r.get('num_stucks', ''))),
            'severity_order': r.get('freeze_fraction', r.get('num_stucks', 0)),
            'model': r['model'],
            'Precision': r['mean_Precision'],
            'Recall': r['mean_Recall'],
            'F1': r['mean_F'],
            'AUC_ROC': r['mean_AUC_ROC'],
            'AUC_PR': r['mean_AUC_PR'],
        })
    return pd.DataFrame(rows)


def load_swap():
    if not os.path.exists(SWAP_SUMMARY):
        return pd.DataFrame()
    df = pd.read_csv(SWAP_SUMMARY)
    if 'mean_Precision' not in df.columns:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'corruption': 'swap',
            'condition': r.get('condition', ''),
            'severity_order': r.get('fraction', 0),
            'model': r['model'],
            'Precision': r['mean_Precision'],
            'Recall': r['mean_Recall'],
            'F1': r['mean_F'],
            'AUC_ROC': r['mean_AUC_ROC'],
            'AUC_PR': r['mean_AUC_PR'],
        })
    return pd.DataFrame(rows)


def classify_error(prec_drop, rec_drop):
    """Classify dominant error type."""
    if abs(prec_drop) < 0.01 and abs(rec_drop) < 0.01:
        return 'negligible'
    ratio = prec_drop / (rec_drop + 1e-10)
    if ratio > 2.0:
        return 'FP-dominant'
    elif ratio < 0.5:
        return 'FN-dominant'
    else:
        return 'balanced'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    baseline = load_baseline()
    print("Baseline:")
    print(baseline.to_string(index=False))
    print()

    # Load all experiments
    dfs = []
    for loader, name in [(load_noise, 'noise'), (load_spikes, 'spikes'),
                          (load_missing, 'missing'), (load_freeze, 'freeze'),
                          (load_swap, 'swap')]:
        df = loader()
        if not df.empty:
            dfs.append(df)
            print(f"  Loaded {name}: {len(df)} rows")

    if not dfs:
        print("No data loaded!")
        return

    all_data = pd.concat(dfs, ignore_index=True)

    # Merge with baseline
    merged = all_data.merge(baseline, on='model', how='left')

    # Compute drops
    merged['precision_drop'] = merged['bl_Precision'] - merged['Precision']
    merged['recall_drop'] = merged['bl_Recall'] - merged['Recall']
    merged['f1_drop'] = merged['bl_F1'] - merged['F1']
    merged['auc_drop'] = merged['bl_AUC_ROC'] - merged['AUC_ROC']

    # Relative drops (percentage of baseline)
    merged['precision_drop_pct'] = 100 * merged['precision_drop'] / (merged['bl_Precision'] + 1e-10)
    merged['recall_drop_pct'] = 100 * merged['recall_drop'] / (merged['bl_Recall'] + 1e-10)

    # Classify error type
    merged['dominant_error'] = merged.apply(
        lambda r: classify_error(r['precision_drop'], r['recall_drop']), axis=1)

    # Round for display
    for col in ['precision_drop', 'recall_drop', 'f1_drop', 'auc_drop',
                'precision_drop_pct', 'recall_drop_pct',
                'Precision', 'Recall', 'bl_Precision', 'bl_Recall']:
        merged[col] = merged[col].round(4)

    # Save full table
    merged.to_csv(os.path.join(OUTPUT_DIR, "fp_fn_decomposition.csv"), index=False)

    # ===============================================
    # Aggregated per corruption type x model
    # ===============================================
    agg = merged.groupby(['corruption', 'model']).agg(
        n_conditions=('condition', 'count'),
        mean_precision_drop=('precision_drop', 'mean'),
        mean_recall_drop=('recall_drop', 'mean'),
        mean_auc_drop=('auc_drop', 'mean'),
        mean_precision_drop_pct=('precision_drop_pct', 'mean'),
        mean_recall_drop_pct=('recall_drop_pct', 'mean'),
    ).round(4).reset_index()

    agg['dominant_error'] = agg.apply(
        lambda r: classify_error(r['mean_precision_drop'], r['mean_recall_drop']), axis=1)

    agg.to_csv(os.path.join(OUTPUT_DIR, "fp_fn_by_corruption.csv"), index=False)

    # ===============================================
    # Print results
    # ===============================================
    print("\n" + "=" * 85)
    print("  FP vs FN Decomposition: Corruption Type x Model")
    print("=" * 85)
    print(f"\n{'Corruption':<16} {'Model':<10} {'Prec drop':>10} {'Rec drop':>10} "
          f"{'AUC drop':>10} {'Dominant':>14}")
    print("-" * 75)

    for _, r in agg.sort_values(['corruption', 'model']).iterrows():
        print(f"{r['corruption']:<16} {r['model']:<10} "
              f"{r['mean_precision_drop']:>+10.4f} {r['mean_recall_drop']:>+10.4f} "
              f"{r['mean_auc_drop']:>+10.4f} {r['dominant_error']:>14}")

    # ===============================================
    # Dominant error summary
    # ===============================================
    print("\n" + "=" * 85)
    print("  Dominant Error Type Summary")
    print("=" * 85)

    dom_summary = agg.groupby(['corruption', 'dominant_error']).size().unstack(fill_value=0)
    print(dom_summary.to_string())

    # Key findings
    print("\n" + "=" * 85)
    print("  Key Findings")
    print("=" * 85)

    for corr in agg['corruption'].unique():
        sub = agg[agg['corruption'] == corr]
        avg_prec = sub['mean_precision_drop'].mean()
        avg_rec = sub['mean_recall_drop'].mean()
        dominant = classify_error(avg_prec, avg_rec)
        print(f"\n  {corr}:")
        print(f"    Avg Precision drop: {avg_prec:+.4f}")
        print(f"    Avg Recall drop:    {avg_rec:+.4f}")
        print(f"    -> {dominant}")
        if dominant == 'FP-dominant':
            print(f"    Corruption creates false alarms (corruption looks like anomalies)")
        elif dominant == 'FN-dominant':
            print(f"    Corruption masks real anomalies (anomalies become harder to detect)")
        else:
            print(f"    Both FP and FN increase comparably")

    print(f"\n\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
