"""
Generate multi-metric tables for all experiments.
Outputs: results/experiments/<exp>/multimetric_table.csv
        results/experiments/<exp>/multimetric_markdown.txt

Run from project root:
    python scripts/generate_multimetric_tables.py
"""
import pandas as pd
import os

# Map experiment → grouping columns (the dimensions of the experiment)
EXPERIMENTS = {
    'white_noise_snr':              ['snr_db'],
    'spikes_normal_only':           ['fraction', 'multiplier'],
    'missing_mnar':                 ['fraction', 'mechanism'],
    'missing_mnar_burst':           ['fraction', 'num_bursts', 'mechanism'],
    'missing_true_impact':          ['missing_type', 'fraction', 'num_bursts'],
    'gilbert_elliott_true_impact':  ['p_good_to_bad', 'p_bad_to_good'],
    'swap_point':                   ['fraction'],
    'swap_segment':                 ['fraction', 'num_swaps'],
    'swap_permutation':             ['n_segments'],
    'freeze':                       ['fraction', 'num_stucks'],
}

METRICS = [
    ('mean_AUC_ROC',           'AUC-ROC'),
    ('mean_AUC_PR',            'AUC-PR'),
    ('mean_R_AUC_ROC',         'R-AUC-ROC'),
    ('mean_R_AUC_PR',          'R-AUC-PR'),
    ('mean_VUS_ROC',           'VUS-ROC'),
    ('mean_Affiliation_Recall','Aff-Recall'),
]

ROOT = 'results/experiments'

for exp, group_cols in EXPERIMENTS.items():
    csv_path = f'{ROOT}/{exp}/summary.csv'
    if not os.path.exists(csv_path):
        print(f'[skip] {exp} (no summary.csv)')
        continue

    df = pd.read_csv(csv_path)
    keep = group_cols + ['model'] + [m for m, _ in METRICS if m in df.columns]
    available = [c for c in keep if c in df.columns]
    out = df[available].copy()

    # Round metrics to 3 decimals
    for m, _ in METRICS:
        if m in out.columns:
            out[m] = out[m].round(3)

    # Rename columns for display
    rename_map = dict(METRICS)
    out = out.rename(columns=rename_map)

    # Save CSV
    out_csv = f'{ROOT}/{exp}/multimetric_table.csv'
    out.to_csv(out_csv, index=False)

    # Save markdown for easy paste in docx
    out_md = f'{ROOT}/{exp}/multimetric_markdown.txt'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(f'# Multi-metric table: {exp}\n\n')
        f.write(out.to_markdown(index=False))
    print(f'[ok] {exp}: {len(out)} rows -> {out_csv}')

print('\nDone. Open multimetric_markdown.txt for each experiment and paste into thesis.')
