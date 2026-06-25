import os
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPECIAL_FOLDERS = ['YAHOO', 'ECG', 'GHL', 'MITDB', 'SVDB', 'Occupancy',
                   'NASA-SMAP', 'NASA-MSL']

ALGO_COLS = ['IFOREST', 'LOF', 'MP', 'NORMA', 'IFOREST1', 'HBOS',
             'OCSVM', 'PCA', 'AE', 'CNN', 'LSTM', 'POLY']

TARGET_TOTAL = 147


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_file_index(base_dir):
    rows = []
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.out') or f.endswith('.txt'):
                fullpath = os.path.join(root, f)
                folder = os.path.basename(os.path.dirname(fullpath))
                rows.append({'filename': f, 'folder_disk': folder,
                             'filepath': fullpath})
    return pd.DataFrame(rows)


def _largest_remainder(total, weights):
    s = sum(weights)
    if s == 0:
        return [0] * len(weights)
    exact = [total * w / s for w in weights]
    floors = [int(e) for e in exact]
    remainders = [e - f for e, f in zip(exact, floors)]
    deficit = total - sum(floors)
    indices = sorted(range(len(remainders)),
                     key=lambda i: (-remainders[i], -weights[i]))
    for i in indices[:deficit]:
        floors[i] += 1
    return floors


def _sqrt_allocation(folder_counts, target):
    sqrt_vals = np.sqrt(folder_counts.values).tolist()
    folders = folder_counts.index.tolist()
    avail = folder_counts.values.tolist()

    alloc = _largest_remainder(target, sqrt_vals)

    for i in range(len(alloc)):
        alloc[i] = max(1, min(alloc[i], avail[i]))

    for _ in range(100):
        diff = sum(alloc) - target
        if diff == 0:
            break
        if diff > 0:
            order = sorted(range(len(alloc)), key=lambda i: -alloc[i])
            for i in order:
                if diff <= 0:
                    break
                if alloc[i] > 1:
                    alloc[i] -= 1
                    diff -= 1
        else:
            caps = [avail[i] - alloc[i] for i in range(len(alloc))]
            order = sorted(range(len(caps)), key=lambda i: -caps[i])
            for i in order:
                if diff >= 0:
                    break
                if caps[i] > 0:
                    alloc[i] += 1
                    diff += 1

    return pd.Series(alloc, index=folders)


def _sample_folder(group, n_target, rng):
    if len(group) <= n_target:
        return group

    diffs = ['Easy', 'Medium', 'Hard']
    counts = [len(group[group['difficulty'] == d]) for d in diffs]

    diff_alloc = _largest_remainder(n_target, counts)
    diff_alloc = [min(a, c) for a, c in zip(diff_alloc, counts)]

    shortfall = n_target - sum(diff_alloc)
    if shortfall > 0:
        caps = [counts[i] - diff_alloc[i] for i in range(3)]
        order = sorted(range(3), key=lambda i: -caps[i])
        for i in order:
            add = min(caps[i], shortfall)
            diff_alloc[i] += add
            shortfall -= add
            if shortfall <= 0:
                break

    sampled = []
    for d, n in zip(diffs, diff_alloc):
        if n <= 0:
            continue
        sub = group[group['difficulty'] == d]
        if n >= len(sub):
            sampled.append(sub)
            continue

        types = sub['type_an'].value_counts()
        if len(types) > 1 and n >= len(types):
            type_alloc = _largest_remainder(n, types.values.tolist())
            type_alloc = [max(1, min(a, c))
                          for a, c in zip(type_alloc, types.values)]
            excess = sum(type_alloc) - n
            if excess > 0:
                order = sorted(range(len(type_alloc)),
                               key=lambda i: -type_alloc[i])
                for i in order:
                    if excess <= 0:
                        break
                    rm = min(type_alloc[i] - 1, excess)
                    type_alloc[i] -= rm
                    excess -= rm
            deficit = n - sum(type_alloc)
            if deficit > 0:
                caps_t = [types.values[i] - type_alloc[i]
                          for i in range(len(type_alloc))]
                order = sorted(range(len(caps_t)), key=lambda i: -caps_t[i])
                for i in order:
                    add = min(caps_t[i], deficit)
                    type_alloc[i] += add
                    deficit -= add
                    if deficit <= 0:
                        break

            for t_name, t_n in zip(types.index, type_alloc):
                if t_n > 0:
                    t_sub = sub[sub['type_an'] == t_name]
                    sampled.append(t_sub.sample(n=t_n, random_state=rng))
        else:
            sampled.append(sub.sample(n=n, random_state=rng))

    return pd.concat(sampled) if sampled else pd.DataFrame()


# ---------------------------------------------------------------------------
# Meta <-> disk matching
# ---------------------------------------------------------------------------

def _match_meta_to_disk(df_meta, df_disk):
    disk_lookup = {}
    for _, row in df_disk.iterrows():
        disk_lookup[row['filename']] = (row['filepath'], row['folder_disk'])
        stem = os.path.splitext(row['filename'])[0]
        disk_lookup.setdefault(f"_stem_{stem}",
                               (row['filepath'], row['folder_disk']))

    def _nasa_meta_to_disk_stem(meta_filename):
        stem = os.path.splitext(meta_filename)[0].replace('_data', '')
        for prefix in ['SMAP', 'MSL']:
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        return stem + '.test'

    matched_rows = []
    for _, meta_row in df_meta.iterrows():
        mf = meta_row['filename']
        filepath = folder_disk = None

        if mf in disk_lookup:
            filepath, folder_disk = disk_lookup[mf]
        else:
            stem = os.path.splitext(mf)[0]
            ext = os.path.splitext(mf)[1]
            alt_name = stem + ('.out' if ext == '.txt' else '.txt')
            if alt_name in disk_lookup:
                filepath, folder_disk = disk_lookup[alt_name]
                mf = alt_name
            elif meta_row['dataset'] in ('NASA_SMAP', 'NASA_MSL'):
                nasa_stem = _nasa_meta_to_disk_stem(meta_row['filename'])
                key = f"_stem_{nasa_stem}"
                if key in disk_lookup:
                    filepath, folder_disk = disk_lookup[key]
                    mf = nasa_stem + '.out'

        if filepath:
            row = meta_row.to_dict()
            row['filename_disk'] = mf
            row['filepath'] = filepath
            row['folder_disk'] = folder_disk
            matched_rows.append(row)

    return pd.DataFrame(matched_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_robust_subset(output_path=None):
    if output_path is None:
        output_path = PROJECT_ROOT / "results" / "tables" / "final_subset.csv"

    meta_csv = (PROJECT_ROOT / "TSB-UAD" / "result" / "accuracy_table"
                / "mergedTable_AUC_ROC.csv")
    data_dir = PROJECT_ROOT / "TSB-UAD" / "data" / "TSB-UAD-Public"

    # 1. Load meta table
    print("[Step 1/7] Loading TSB-UAD accuracy meta table ...")
    df_meta = pd.read_csv(meta_csv)
    df_meta['mean_AUC_ROC'] = df_meta[ALGO_COLS].mean(axis=1)
    print(f"  Meta datasets: {len(df_meta)}")

    # 2. Scan disk
    print(f"[Step 2/7] Scanning {data_dir} ...")
    df_disk = _build_file_index(data_dir)
    print(f"  Files on disk: {len(df_disk)}")

    # 3. Match meta -> disk
    print("[Step 3/7] Matching meta -> disk ...")
    df = _match_meta_to_disk(df_meta, df_disk)
    df['folder'] = df['dataset'].str.replace('_', '-')
    print(f"  Matched: {len(df)}")

    print("[Step 4/7] Filtering ...")
    #    Normal folders:  length [1000, 100 000] AND anomaly ratio <= 20 percent
    #    Special folders: keep regardless
    cond_length = df['data_len'].between(1000, 100_000)
    cond_ratio = (df['ratio'].isna()) | (df['ratio'] <= 0.20)
    cond_special = df['folder'].isin(SPECIAL_FOLDERS)
    df_filtered = df[(cond_length & cond_ratio) | cond_special].copy()
    print(f"  After filtering: {len(df_filtered)}")

    # 5. Tercile-based difficulty
    print("[Step 5/7] Computing difficulty thresholds ...")
    q33 = df_filtered['mean_AUC_ROC'].quantile(0.33)
    q66 = df_filtered['mean_AUC_ROC'].quantile(0.66)
    df_filtered['difficulty'] = np.select(
        [df_filtered['mean_AUC_ROC'] >= q66,
         df_filtered['mean_AUC_ROC'] <= q33],
        ['Easy', 'Hard'], default='Medium')

    print(f"  Difficulty thresholds: Hard <= {q33:.4f} < Medium < {q66:.4f} <= Easy")
    pool_dist = df_filtered['difficulty'].value_counts().to_dict()
    print(f"  Pool: {pool_dist}")

    # 6. Sqrt-proportional allocation
    print("[Step 6/7] Computing sqrt-proportional allocation ...")
    folder_counts = df_filtered.groupby('folder').size().sort_values(
        ascending=False)
    allocations = _sqrt_allocation(folder_counts, TARGET_TOTAL)

    # Cap MITDB at 3 (its 650K-point series are very expensive)
    if "MITDB" in allocations.index and allocations["MITDB"] > 3:
        allocations["MITDB"] = 3

    print(f"\n  Allocation (target ~{TARGET_TOTAL}):")
    for f in allocations.sort_values(ascending=False).index:
        print(f"    {f:15s}: {allocations[f]:3d} / {folder_counts[f]:3d}")


    # 7. Stratified sampling per folder (difficulty + anomaly type diversity)
    print("[Step 7/7] Stratified sampling per folder ...")
    rng = np.random.RandomState(42)
    parts = []
    n_folders = len(allocations)
    for idx, folder in enumerate(allocations.index, 1):
        grp = df_filtered[df_filtered['folder'] == folder]
        sampled = _sample_folder(grp, allocations[folder], rng)
        parts.append(sampled)
        print(f"  [{idx:2d}/{n_folders}] {folder:15s}  "+
              f"sampled {len(sampled):3d} / {len(grp):3d}")

    final = pd.concat(parts, ignore_index=True)

    # 8. Format output
    final = final[[
        'filename_disk', 'folder', 'mean_AUC_ROC', 'difficulty',
        'data_len', 'ratio', 'type_an', 'filepath'
    ]].rename(columns={'filename_disk': 'filename_actual'})
    final.insert(0, 'baseline_name', final['filename_actual'])

    # 9. Diagnostics
    sep = '=' * 60
    print(f"\n{sep}")
    print(f"Final subset: {len(final)} datasets")

    print("\nFolder x Difficulty:")
    ct = pd.crosstab(final['folder'], final['difficulty'])
    for c in ['Easy', 'Medium', 'Hard']:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[['Easy', 'Medium', 'Hard']]
    ct['Total'] = ct.sum(axis=1)
    print(ct.to_string())

    print("\nDifficulty totals:")
    print(final['difficulty'].value_counts().sort_index().to_string())

    print("\nAnomaly type totals:")
    print(final['type_an'].value_counts().to_string())

    print("\nFolder x Anomaly Type:")
    print(pd.crosstab(final['folder'], final['type_an']).to_string())

    # Representativeness vs full filtered benchmark
    print("\nRepresentativeness (benchmark vs subset):")
    for col, label in [('difficulty', 'Difficulty'),
                       ('type_an', 'Anomaly type')]:
        bm = df_filtered[col].value_counts(normalize=True) * 100
        sub = final[col].value_counts(normalize=True) * 100
        print(f"  {label}:")
        for k in bm.index:
            b = bm[k]
            s = sub.get(k, 0)
            print(f"    {k:10s}  benchmark {b:5.1f}%   subset {s:5.1f}%")

    # 10. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    create_robust_subset()
