import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
TABLES_DIR = RESULTS_DIR / "tables"


CANONICAL_COLUMNS = [
    "file",
    "model",
    "experiment",
    "corruption_family",
    "condition",
    "frontier_group",
    "axis_name",
    "axis_value",
    "auc_corrupted",
    "fraction",
    "num_bursts",
    "missing_type",
    "num_stucks",
    "snr_db",
    "multiplier",
    "combination_name",
    "severity_score",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _empty_canonical() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _filter_success(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "error" not in df.columns:
        return df
    err = df["error"]
    mask = err.isna() | (err.astype(str).str.strip() == "")
    return df.loc[mask].copy()


def _with_default_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[CANONICAL_COLUMNS]


def _safe_int_text(value) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def _severity_from_condition_string(condition: str) -> float:
    if pd.isna(condition):
        return np.nan
    levels = {"low": 0.0, "med": 0.5, "high": 1.0}
    found = re.findall(r"_(low|med|high)(?:\b|_|$|\+)", str(condition))
    if not found:
        return np.nan
    return float(np.mean([levels[x] for x in found]))


def _load_white_noise() -> pd.DataFrame:
    df = _filter_success(_read_csv_if_exists(EXPERIMENTS_DIR / "white_noise_snr" / "checkpoint.csv"))
    if df.empty or not {"file", "model", "AUC_ROC", "snr_db"}.issubset(df.columns):
        return _empty_canonical()

    snr_min = float(df["snr_db"].min())
    snr_max = float(df["snr_db"].max())
    denom = (snr_max - snr_min) if (snr_max - snr_min) > 1e-12 else 1.0

    out = pd.DataFrame(
        {
            "file": df["file"],
            "model": df["model"],
            "experiment": "white_noise_snr",
            "corruption_family": "white_noise",
            "condition": df["condition"] if "condition" in df.columns else ("snr_" + df["snr_db"].astype(str) + "dB"),
            "frontier_group": "white_noise",
            "axis_name": "snr_inverse_norm",
            "axis_value": (snr_max - df["snr_db"]) / denom,
            "auc_corrupted": df["AUC_ROC"],
            "snr_db": df["snr_db"],
        }
    )
    out["severity_score"] = out["axis_value"]
    return _with_default_columns(out)


def _load_missing_true_impact() -> pd.DataFrame:
    df = _filter_success(_read_csv_if_exists(EXPERIMENTS_DIR / "missing_true_impact" / "checkpoint.csv"))
    required = {"file", "model", "AUC_ROC", "missing_type", "fraction", "num_bursts"}
    if df.empty or not required.issubset(df.columns):
        return _empty_canonical()

    frontier_group = np.where(
        df["missing_type"].astype(str) == "point",
        "missing_point",
        "missing_burst_nb" + df["num_bursts"].apply(_safe_int_text),
    )

    out = pd.DataFrame(
        {
            "file": df["file"],
            "model": df["model"],
            "experiment": "missing_true_impact",
            "corruption_family": "missing",
            "condition": df["condition"] if "condition" in df.columns else (
                df["missing_type"].astype(str) + "_f" + df["fraction"].astype(str) + "_nb" + df["num_bursts"].astype(str)
            ),
            "frontier_group": frontier_group,
            "axis_name": "fraction",
            "axis_value": df["fraction"],
            "auc_corrupted": df["AUC_ROC"],
            "fraction": df["fraction"],
            "num_bursts": df["num_bursts"],
            "missing_type": df["missing_type"],
            "severity_score": df["fraction"],
        }
    )
    return _with_default_columns(out)


def _load_freeze() -> pd.DataFrame:
    df = _filter_success(_read_csv_if_exists(EXPERIMENTS_DIR / "freeze" / "checkpoint.csv"))
    required = {"file", "model", "AUC_ROC", "fraction", "num_stucks"}
    if df.empty or not required.issubset(df.columns):
        return _empty_canonical()

    out = pd.DataFrame(
        {
            "file": df["file"],
            "model": df["model"],
            "experiment": "freeze",
            "corruption_family": "freeze",
            "condition": df["condition"] if "condition" in df.columns else (
                "freeze_f" + df["fraction"].astype(str) + "_ns" + df["num_stucks"].astype(str)
            ),
            "frontier_group": "freeze_ns" + df["num_stucks"].apply(_safe_int_text),
            "axis_name": "fraction",
            "axis_value": df["fraction"],
            "auc_corrupted": df["AUC_ROC"],
            "fraction": df["fraction"],
            "num_stucks": df["num_stucks"],
            "severity_score": df["fraction"],
        }
    )
    return _with_default_columns(out)


def _load_spikes() -> pd.DataFrame:
    candidates = [
        ("spikes_normal_only", EXPERIMENTS_DIR / "spikes_normal_only" / "checkpoint.csv"),
        ("spikes_no_zscore", EXPERIMENTS_DIR / "spikes_no_zscore" / "checkpoint.csv"),
    ]
    selected_df = None
    selected_experiment = None

    for experiment_name, path in candidates:
        df = _filter_success(_read_csv_if_exists(path))
        if df.empty or not {"file", "model", "AUC_ROC", "fraction", "multiplier"}.issubset(df.columns):
            continue
        selected_df = df
        selected_experiment = experiment_name
        break

    if selected_df is None:
        return _empty_canonical()

    df = selected_df
    if "sequential" in df.columns:
        seq = df["sequential"].fillna(False).astype(bool)
        seq_len = df["sequence_length"] if "sequence_length" in df.columns else pd.Series(1, index=df.index)
        seq_len_series = pd.Series(seq_len, index=df.index).apply(_safe_int_text)
        mode = pd.Series(np.where(seq, "burst_seq" + seq_len_series, "point"), index=df.index)
        frontier = "spikes_" + mode.astype(str) + "_mult" + df["multiplier"].astype(str)
    else:
        frontier = "spikes_mult" + df["multiplier"].astype(str)

    out = pd.DataFrame(
        {
            "file": df["file"],
            "model": df["model"],
            "experiment": selected_experiment,
            "corruption_family": "spikes",
            "condition": df["condition"] if "condition" in df.columns else (
                "spikes_f" + df["fraction"].astype(str) + "_m" + df["multiplier"].astype(str)
            ),
            "frontier_group": frontier,
            "axis_name": "fraction",
            "axis_value": df["fraction"],
            "auc_corrupted": df["AUC_ROC"],
            "fraction": df["fraction"],
            "multiplier": df["multiplier"],
            "severity_score": df["fraction"],
        }
    )
    return _with_default_columns(out)


def _load_compound() -> pd.DataFrame:
    df = _filter_success(_read_csv_if_exists(EXPERIMENTS_DIR / "compound_corruptions" / "checkpoint.csv"))
    required = {"file", "model", "AUC_ROC", "condition"}
    if df.empty or not required.issubset(df.columns):
        return _empty_canonical()

    # Keep only corrupted conditions, skip the explicit baseline rows.
    if "condition_type" in df.columns:
        df = df[df["condition_type"].astype(str) != "baseline"].copy()
    df = df[df["condition"].astype(str).str.lower() != "clean"].copy()
    if df.empty:
        return _empty_canonical()

    if "combination_name" in df.columns:
        combo = df["combination_name"].copy()
    else:
        combo = pd.Series("compound", index=df.index)
    if "combination_name" in df.columns:
        single_mask = df["combination_name"].astype(str) == "single"
        if single_mask.any():
            single_type = df.loc[single_mask, "condition"].astype(str).str.extract(r"^([a-z_]+)_(low|med|high)_only")[0]
            combo = combo.copy()
            combo.loc[single_mask] = "single_" + single_type.fillna("unknown")

    severity = df["severity_score"] if "severity_score" in df.columns else df["condition"].map(_severity_from_condition_string)

    out = pd.DataFrame(
        {
            "file": df["file"],
            "model": df["model"],
            "experiment": "compound_corruptions",
            "corruption_family": "compound",
            "condition": df["condition"],
            "frontier_group": "compound_" + combo.astype(str),
            "axis_name": "severity_score",
            "axis_value": severity,
            "auc_corrupted": df["AUC_ROC"],
            "fraction": df["missing_fraction"] if "missing_fraction" in df.columns else np.nan,
            "num_stucks": df["freeze_num_stucks"] if "freeze_num_stucks" in df.columns else np.nan,
            "snr_db": df["noise_snr_db"] if "noise_snr_db" in df.columns else np.nan,
            "multiplier": df["spikes_multiplier"] if "spikes_multiplier" in df.columns else np.nan,
            "combination_name": combo,
            "severity_score": severity,
        }
    )
    return _with_default_columns(out)


def load_baseline_table() -> pd.DataFrame:
    # Preferred baseline table with all models.
    full_path = TABLES_DIR / "baseline_final_subset.csv"
    if full_path.exists():
        df = pd.read_csv(full_path)
        needed = {"file", "model", "AUC_ROC"}
        if needed.issubset(df.columns):
            out = df.rename(columns={"AUC_ROC": "baseline_auc"}).copy()
            for col in ["folder", "n_anomalies", "anomaly_ratio"]:
                if col not in out.columns:
                    out[col] = np.nan
            return out[["file", "model", "baseline_auc", "folder", "n_anomalies", "anomaly_ratio"]]

    # Fallback IForest-only baseline.
    fallback_path = TABLES_DIR / "baseline_141_iforest.csv"
    if fallback_path.exists():
        df = pd.read_csv(fallback_path)
        if {"file", "AUC_ROC"}.issubset(df.columns):
            out = df.copy()
            out["model"] = "IForest"
            out = out.rename(columns={"AUC_ROC": "baseline_auc"})
            for col in ["folder", "n_anomalies", "anomaly_ratio"]:
                if col not in out.columns:
                    out[col] = np.nan
            return out[["file", "model", "baseline_auc", "folder", "n_anomalies", "anomaly_ratio"]]

    raise FileNotFoundError("No baseline table found in results\\tables (expected baseline_final_subset.csv).")


def load_subset_features() -> pd.DataFrame:
    subset_path = TABLES_DIR / "final_subset.csv"
    if not subset_path.exists():
        return pd.DataFrame(columns=["file", "length", "subset_ratio", "difficulty", "type_an", "folder"])

    df = pd.read_csv(subset_path).copy()
    rename_map = {
        "baseline_name": "file",
        "data_len": "length",
        "ratio": "subset_ratio",
    }
    df = df.rename(columns=rename_map)
    for col in ["file", "length", "subset_ratio", "difficulty", "type_an", "folder"]:
        if col not in df.columns:
            df[col] = np.nan
    return df[["file", "length", "subset_ratio", "difficulty", "type_an", "folder"]].drop_duplicates()


def build_delta_dataset(
    include_experiments: Optional[Iterable[str]] = None,
    quick_files: int = 0,
) -> pd.DataFrame:
    loaders = {
        "white_noise_snr": _load_white_noise,
        "missing_true_impact": _load_missing_true_impact,
        "freeze": _load_freeze,
        "spikes": _load_spikes,
        "compound_corruptions": _load_compound,
    }

    selected = set(include_experiments) if include_experiments is not None else set(loaders.keys())
    frames = []
    for name, loader in loaders.items():
        if name not in selected and not (name == "spikes" and "spikes" in selected):
            continue
        df = loader()
        if not df.empty:
            frames.append(df)
            print(f"[INFO] Loaded {name}: {len(df)} rows")
        else:
            print(f"[WARN] No usable rows for {name}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["file", "model", "auc_corrupted"]).copy()

    baseline = load_baseline_table()
    subset_features = load_subset_features()

    df = df.merge(baseline, on=["file", "model"], how="left")
    missing_baseline = int(df["baseline_auc"].isna().sum())
    if missing_baseline > 0:
        print(f"[WARN] Dropping {missing_baseline} rows without matching baseline_auc")
    df = df.dropna(subset=["baseline_auc"]).copy()

    df = df.merge(
        subset_features,
        on="file",
        how="left",
        suffixes=("", "_subset"),
    )

    # Prefer subset folder/type/difficulty if canonical fields are missing.
    for col in ["folder", "difficulty", "type_an"]:
        subset_col = f"{col}_subset"
        if subset_col in df.columns:
            df[col] = df[col].fillna(df[subset_col])
            df.drop(columns=[subset_col], inplace=True)

    df["delta_auc"] = df["auc_corrupted"] - df["baseline_auc"]

    # Average duplicate rows (e.g., multiple seeds) at file-level per condition.
    group_cols = [
        "file",
        "model",
        "experiment",
        "corruption_family",
        "condition",
        "frontier_group",
        "axis_name",
        "axis_value",
        "fraction",
        "num_bursts",
        "missing_type",
        "num_stucks",
        "snr_db",
        "multiplier",
        "combination_name",
        "severity_score",
        "folder",
        "n_anomalies",
        "anomaly_ratio",
        "baseline_auc",
        "length",
        "subset_ratio",
        "difficulty",
        "type_an",
    ]
    for col in group_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            auc_corrupted=("auc_corrupted", "mean"),
            delta_auc=("delta_auc", "mean"),
        )
        .copy()
    )

    if quick_files and quick_files > 0:
        keep_files = sorted(df["file"].dropna().unique())[:quick_files]
        df = df[df["file"].isin(keep_files)].copy()
        print(f"[INFO] Quick mode active: keeping {len(keep_files)} files")

    return df


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.empty(n, dtype=float)
    for i, pv in enumerate(sorted_p):
        adjusted_sorted[i] = (n - i) * pv
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted
