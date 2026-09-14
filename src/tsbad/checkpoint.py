"""Per-model checkpoints: one CSV per model, so runs of different models never clobber each other.

A legacy combined `checkpoint.csv` (written before the per-model split) is still read.
"""
import glob
import os

import pandas as pd

KEY = ('file', 'condition', 'seed', 'model')


def ckpt_path(results_dir, model):
    return os.path.join(results_dir, f"checkpoint_{model}.csv")


def legacy_path(results_dir):
    return os.path.join(results_dir, "checkpoint.csv")


def row_key(r):
    # key on `condition` (a string): a clean anchor stores its parameters as None, which pandas
    # reads back as NaN, so a parameter-based key would never match on resume and clean rows
    # would be re-run and duplicated
    return f"{r['file']}|{r['condition']}|{r['seed']}|{r['model']}"


def load_checkpoints(results_dir, models, key=row_key):
    """Previous successful results for `models` only, plus the legacy combined file.

    Returns (records, completed_keys), de-duplicated on `key` (the legacy file may overlap a
    per-model file).
    """
    records, paths = [], [ckpt_path(results_dir, m) for m in models]
    legacy = legacy_path(results_dir)
    if os.path.exists(legacy):
        paths.append(legacy)
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
            if 'model' in df.columns:
                df = df[df['model'].isin(models)]
            ok = df[df['error'].isna()] if 'error' in df.columns else df
            records.extend(ok.to_dict('records'))
        except Exception as e:
            print(f"Warning: could not read {os.path.basename(p)}: {e}")
    seen, uniq, keys = set(), [], set()
    for r in records:
        k = key(r)
        if k in seen:
            continue
        seen.add(k); keys.add(k); uniq.append(r)
    return uniq, keys


def save_checkpoints(results_dir, all_results, models):
    """Write results back, one file per model. Only touches this run's models."""
    if not all_results:
        return
    df = pd.DataFrame(all_results)
    for m in models:
        sub = df[df['model'] == m]
        if not sub.empty:
            sub.to_csv(ckpt_path(results_dir, m), index=False)


def load_all_for_summary(results_dir, key_cols=KEY):
    """Every checkpoint on disk (all models, plus legacy), de-duplicated."""
    paths = sorted(glob.glob(os.path.join(results_dir, "checkpoint_*.csv")))
    legacy = legacy_path(results_dir)
    if os.path.exists(legacy):
        paths.append(legacy)
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if set(key_cols).issubset(df.columns):
        df = df.drop_duplicates(subset=list(key_cols), keep='first')
    return df
