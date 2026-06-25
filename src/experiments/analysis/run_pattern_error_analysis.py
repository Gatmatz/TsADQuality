"""
Pattern-based error analysis with rerun for 3 corruption families:
  - white noise (SNR)
  - spikes (normal-only)
  - missing (true impact)

This script reruns scoring and computes pattern-wise FN/TP behavior using
the same thresholding logic as `get_metrics(..., metric="all")`:
    pred = score > (mean(score) + 3*std(score))

Outputs (results/experiments/pattern_error_analysis/):
  - checkpoint.csv               (pattern-level rows)
  - pattern_summary.csv          (condition × model × pattern)
  - model_pattern_summary.csv    (model × pattern)
  - worst_cases.csv              (highest FN-rate rows)
  - includes pattern_auc_roc / pattern_auc_pr in checkpoint and summaries

Usage examples:
  python run_pattern_error_analysis.py
  python run_pattern_error_analysis.py --models IForest LOF MP --workers 4
  python run_pattern_error_analysis.py --experiments white_noise spikes missing --full-grid
  python run_pattern_error_analysis.py --test
"""
import argparse
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

# ==========================================
# PATH CONFIGURATION
# ==========================================
CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = Path(CURRENT_FILE).resolve().parents[3]
TSB_UAD_PATH = PROJECT_ROOT / "TSB-UAD"
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TSB_UAD_PATH) not in sys.path:
    sys.path.insert(0, str(TSB_UAD_PATH))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_loader import load_pretrained_ae, load_tsb_dataframe, remap_filepath  # noqa: E402
from ts_corruptor.core import TSCorruptor  # noqa: E402
import ts_corruptor.injectors  # noqa: E402
from TSB_UAD.models.feature import Window  # noqa: E402
from TSB_UAD.models.iforest import IForest  # noqa: E402
from TSB_UAD.models.lof import LOF  # noqa: E402
from TSB_UAD.models.matrix_profile import MatrixProfile  # noqa: E402
from TSB_UAD.models.pca import PCA  # noqa: E402
from TSB_UAD.utils.slidingWindows import find_length  # noqa: E402

# ==========================================
# CONFIGURATION
# ==========================================
SUBSET_CSV = PROJECT_ROOT / "results" / "tables" / "final_subset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments" / "pattern_error_analysis"

FULL_WHITE_NOISE_SNRS = [40, 30, 20, 10, 5, 0, -5, -10, -20]
CORE_WHITE_NOISE_SNRS = [20, 10, 5]

FULL_SPIKE_FRACTIONS = [0.01, 0.05, 0.10, 0.20]
FULL_SPIKE_MULTIPLIERS = [3.0, 5.0, 10.0]
CORE_SPIKE_FRACTIONS = [0.01, 0.05, 0.10]
CORE_SPIKE_MULTIPLIERS = [3.0, 5.0, 10.0]

FULL_MISSING_FRACTIONS = [0.01, 0.05, 0.10, 0.20]
FULL_MISSING_BURSTS = [1, 3, 5, 10,20]
CORE_MISSING_FRACTIONS = [0.01, 0.05, 0.10]
CORE_MISSING_BURSTS = [3]

PATTERNS = [
    "sudden_spike",
    "gradual_change",
    "seasonal_anomaly",
    "contextual_anomaly",
]


def find_anomaly_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    in_seg = False
    start = 0
    for i, y in enumerate(labels):
        if y == 1 and not in_seg:
            in_seg = True
            start = i
        elif y == 0 and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(labels) - 1))
    return segments


def classify_patterns(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """
    Heuristic anomaly-pattern labeling on ground-truth anomaly segments.
    """
    n = len(values)
    pattern_labels = np.array(["normal"] * n, dtype=object)
    segments = find_anomaly_segments(labels)
    if not segments:
        return pattern_labels, []

    starts = np.array([s for s, _ in segments], dtype=float)
    periodic = False
    if len(starts) >= 4:
        diffs = np.diff(starts)
        d_mean = float(np.mean(diffs))
        d_std = float(np.std(diffs))
        if d_mean > 8 and d_std / (d_mean + 1e-12) < 0.25:
            periodic = True

    segment_meta = []
    for idx, (s, e) in enumerate(segments):
        seg = values[s : e + 1]
        seg_len = len(seg)

        ctx_w = max(10, seg_len)
        left = values[max(0, s - ctx_w) : s]
        right = values[e + 1 : min(n, e + 1 + ctx_w)]
        if len(left) + len(right) > 0:
            context = np.concatenate([left, right])
        else:
            context = values

        ctx_med = float(np.median(context))
        ctx_std = float(np.std(context))
        if ctx_std < 1e-12:
            ctx_std = 1.0

        amp = float(np.max(np.abs(seg - ctx_med))) if seg_len > 0 else 0.0
        amp_z = amp / ctx_std

        if seg_len >= 3:
            x = np.arange(seg_len, dtype=float)
            if float(np.std(seg)) > 1e-10:
                corr = float(np.corrcoef(x, seg)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            slope = float(np.polyfit(x, seg, 1)[0])
        else:
            corr = 0.0
            slope = 0.0
        slope_norm = abs(slope) / ctx_std

        if periodic and seg_len >= 3:
            pattern = "seasonal_anomaly"
        elif seg_len <= 3 and amp_z >= 3.0:
            pattern = "sudden_spike"
        elif seg_len >= 8 and abs(corr) >= 0.80 and slope_norm >= 0.05:
            pattern = "gradual_change"
        else:
            pattern = "contextual_anomaly"

        pattern_labels[s : e + 1] = pattern
        segment_meta.append(
            {
                "segment_id": idx,
                "start": s,
                "end": e,
                "length": seg_len,
                "amplitude_z": amp_z,
                "trend_corr": corr,
                "slope_norm": slope_norm,
                "pattern": pattern,
            }
        )

    return pattern_labels, segment_meta


def run_model_scores(
    model_name: str,
    scaled_data: np.ndarray,
    sliding_window: int,
    canonical_name: str,
) -> np.ndarray:
    if model_name == "AE":
        clf, _meta = load_pretrained_ae(canonical_name, str(PROJECT_ROOT))
        clf.predict(scaled_data)
        score = clf.decision_scores_
        full_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        if len(full_score) > len(scaled_data):
            full_score = full_score[: len(scaled_data)]
        elif len(full_score) < len(scaled_data):
            full_score = np.pad(full_score, (0, len(scaled_data) - len(full_score)), mode="edge")
        return full_score

    if model_name == "MP":
        clf = MatrixProfile(window=sliding_window)
        clf.fit(scaled_data)
        score = clf.decision_scores_
    else:
        X = Window(window=sliding_window).convert(scaled_data).to_numpy()
        if model_name == "IForest":
            clf = IForest(n_estimators=100, random_state=42)
        elif model_name == "PCA":
            n_components = min(10, X.shape[1] - 1) if X.shape[1] > 1 else 1
            clf = PCA(n_components=n_components)
        elif model_name == "LOF":
            n_neighbors = min(20, max(2, len(X) - 1))
            clf = LOF(n_neighbors=n_neighbors)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        clf.fit(X)
        score = clf.decision_scores_

    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    full_score = np.array(
        [score[0]] * math.ceil((sliding_window - 1) / 2)
        + list(score)
        + [score[-1]] * ((sliding_window - 1) // 2)
    )

    if len(full_score) > len(scaled_data):
        full_score = full_score[: len(scaled_data)]
    elif len(full_score) < len(scaled_data):
        full_score = np.pad(full_score, (0, len(scaled_data) - len(full_score)), mode="edge")
    return full_score


def _apply_white_noise(df: pd.DataFrame, snr_db: float, seed: int) -> tuple[np.ndarray, dict]:
    corruptor = TSCorruptor(df, value_col="value", label_col="is_anomaly", seed=seed)
    ts_corruptor.injectors.inject_white_noise_snr(corruptor, snr_db=snr_db)
    out = corruptor.get_corrupted_df()["value"].to_numpy("float")
    return out, {"snr_db": float(snr_db), "condition": f"snr_{int(snr_db)}dB"}


def _apply_spikes(df: pd.DataFrame, fraction: float, multiplier: float, seed: int) -> tuple[np.ndarray, dict]:
    corruptor = TSCorruptor(
        df,
        value_col="value",
        label_col="is_anomaly",
        seed=seed,
        corruption_target="only_normal",
    )
    ts_corruptor.injectors.inject_spikes(
        corruptor,
        fraction=fraction,
        multiplier=multiplier,
        sequential=False,
        sequence_length=1,
    )
    out = corruptor.get_corrupted_df()["value"].to_numpy("float")
    return out, {
        "fraction": float(fraction),
        "multiplier": float(multiplier),
        "condition": f"frac_{fraction}_mult_{multiplier}",
    }


def _apply_missing(
    df: pd.DataFrame,
    missing_type: str,
    fraction: float,
    num_bursts: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    n = len(df)
    corruptor = TSCorruptor(df, value_col="value", label_col="is_anomaly", seed=seed)
    if missing_type == "point":
        ts_corruptor.injectors.inject_point_missing(corruptor, fraction=fraction)
        burst_length = 0
    else:
        burst_length = max(1, int(fraction * n / num_bursts))
        ts_corruptor.injectors.inject_burst_missing(
            corruptor,
            num_bursts=num_bursts,
            burst_length=burst_length,
        )
    out = corruptor.get_corrupted_df()["value"].to_numpy("float")
    if missing_type == "point":
        condition = f"point_frac_{fraction}"
    else:
        condition = f"burst_frac_{fraction}_nb_{num_bursts}"
    return out, {
        "missing_type": missing_type,
        "fraction": float(fraction),
        "num_bursts": int(num_bursts),
        "burst_length": int(burst_length),
        "condition": condition,
    }


def _pattern_rows_from_eval(
    eval_scores: np.ndarray,
    eval_labels: np.ndarray,
    eval_patterns: np.ndarray,
    base_meta: dict,
) -> list[dict]:
    score_mu = float(np.mean(eval_scores))
    score_sigma = float(np.std(eval_scores))
    threshold = score_mu + 3.0 * score_sigma
    preds = eval_scores > threshold

    tp_total = int(np.sum((preds == 1) & (eval_labels == 1)))
    fn_total = int(np.sum((preds == 0) & (eval_labels == 1)))
    fp_total = int(np.sum((preds == 1) & (eval_labels == 0)))
    tn_total = int(np.sum((preds == 0) & (eval_labels == 0)))

    rows = []
    for patt in PATTERNS:
        mask = (eval_labels == 1) & (eval_patterns == patt)
        anomaly_points = int(np.sum(mask))
        tp_p = int(np.sum((preds == 1) & mask))
        fn_p = int(np.sum((preds == 0) & mask))
        recall_p = float(tp_p / anomaly_points) if anomaly_points > 0 else np.nan
        fn_rate_p = float(fn_p / anomaly_points) if anomaly_points > 0 else np.nan

        # Pattern-wise AUC: this pattern's anomaly points (positives)
        # versus all normal points (negatives) in the same evaluation set.
        neg_mask = eval_labels == 0
        eval_mask_pattern = mask | neg_mask
        n_eval_pattern = int(np.sum(eval_mask_pattern))
        n_pos_pattern = int(np.sum(mask))
        n_neg_pattern = int(np.sum(neg_mask))

        auc_roc_pattern = np.nan
        auc_pr_pattern = np.nan
        if n_pos_pattern > 0 and n_neg_pattern > 0:
            y_pattern = mask[eval_mask_pattern].astype(int)
            s_pattern = eval_scores[eval_mask_pattern]
            finite = np.isfinite(s_pattern)
            y_pattern = y_pattern[finite]
            s_pattern = s_pattern[finite]
            if len(np.unique(y_pattern)) == 2:
                try:
                    auc_roc_pattern = float(roc_auc_score(y_pattern, s_pattern))
                except Exception:
                    auc_roc_pattern = np.nan
                try:
                    auc_pr_pattern = float(average_precision_score(y_pattern, s_pattern))
                except Exception:
                    auc_pr_pattern = np.nan

        row = {
            **base_meta,
            "pattern": patt,
            "anomaly_points": anomaly_points,
            "n_eval_pattern": n_eval_pattern,
            "n_pos_pattern": n_pos_pattern,
            "n_neg_pattern": n_neg_pattern,
            "tp_pattern": tp_p,
            "fn_pattern": fn_p,
            "pattern_recall": recall_p,
            "pattern_fn_rate": fn_rate_p,
            "pattern_auc_roc": auc_roc_pattern,
            "pattern_auc_pr": auc_pr_pattern,
            "threshold": threshold,
            "score_mean": score_mu,
            "score_std": score_sigma,
            "tp_total": tp_total,
            "fp_total": fp_total,
            "fn_total": fn_total,
            "tn_total": tn_total,
        }
        rows.append(row)
    return rows


def process_single_job(job: dict) -> dict:
    try:
        remapped = remap_filepath(job["filepath"], str(PROJECT_ROOT))
        df, canonical_name = load_tsb_dataframe(remapped)
        values = df["value"].to_numpy("float")
        labels = df["is_anomaly"].to_numpy("int")

        pattern_labels, segment_meta = classify_patterns(values, labels)
        seg_counts = {p: 0 for p in PATTERNS}
        for sm in segment_meta:
            if sm["pattern"] in seg_counts:
                seg_counts[sm["pattern"]] += 1

        clean_scaled = StandardScaler().fit_transform(values.reshape(-1, 1)).flatten()
        base_sw = max(int(find_length(clean_scaled)), 10)

        exp = job["experiment"]
        if exp == "white_noise":
            corrupted, cond_meta = _apply_white_noise(df, job["snr_db"], job["seed"])

            scaled_data = StandardScaler().fit_transform(corrupted.reshape(-1, 1)).flatten()
            full_score = run_model_scores(job["model"], scaled_data, base_sw, canonical_name)

            eval_scores = full_score
            eval_labels = labels
            eval_patterns = pattern_labels

        elif exp == "spikes":
            corrupted, cond_meta = _apply_spikes(df, job["fraction"], job["multiplier"], job["seed"])

            scaled_data = StandardScaler().fit_transform(corrupted.reshape(-1, 1)).flatten()
            full_score = run_model_scores(job["model"], scaled_data, base_sw, canonical_name)

            eval_scores = full_score
            eval_labels = labels
            eval_patterns = pattern_labels

        elif exp == "missing":
            corrupted, cond_meta = _apply_missing(
                df,
                job["missing_type"],
                job["fraction"],
                job["num_bursts"],
                job["seed"],
            )
            nan_mask = np.isnan(corrupted)
            masked_normal = nan_mask & (labels == 0)
            masked_anomaly = nan_mask & (labels == 1)

            model_data = corrupted[~nan_mask]
            n_kept = len(model_data)
            if n_kept < base_sw + 10:
                return {
                    "status": "skipped",
                    "error": f"Too few points after masking: {n_kept}",
                    "job_id": job["job_id"],
                }

            sw = min(base_sw, n_kept // 4)
            sw = max(sw, 10)
            scaled_data = StandardScaler().fit_transform(model_data.reshape(-1, 1)).flatten()
            score_kept = run_model_scores(job["model"], scaled_data, sw, canonical_name)

            full_score = np.full(len(labels), np.nan, dtype=float)
            full_score[~nan_mask] = score_kept
            full_score[masked_anomaly] = 0.0

            eval_mask = ~masked_normal
            eval_scores = full_score[eval_mask]
            eval_labels = labels[eval_mask]
            eval_patterns = pattern_labels[eval_mask]

            cond_meta["n_kept"] = int(n_kept)
            cond_meta["n_lost_anomalies"] = int(np.sum(masked_anomaly))
            cond_meta["actual_missing_rate"] = float(np.sum(nan_mask) / len(labels))
        else:
            return {"status": "error", "error": f"Unknown experiment: {exp}", "job_id": job["job_id"]}

        base_meta = {
            "experiment": exp,
            "file": canonical_name,
            "folder": job["folder"],
            "model": job["model"],
            "seed": job["seed"],
            "job_id": job["job_id"],
            **cond_meta,
            **{f"n_segments_{p}": seg_counts[p] for p in PATTERNS},
        }
        rows = _pattern_rows_from_eval(eval_scores, eval_labels, eval_patterns, base_meta)
        return {"status": "success", "rows": rows, "job_id": job["job_id"]}

    except Exception:
        return {
            "status": "error",
            "error": traceback.format_exc(),
            "job_id": job.get("job_id", "unknown"),
        }


def build_jobs(args: argparse.Namespace, subset_df: pd.DataFrame) -> list[dict]:
    files_df = subset_df.copy()
    if args.max_files and args.max_files > 0:
        files_df = files_df.head(args.max_files)
    if args.test:
        files_df = files_df.head(3)

    if args.full_grid and not args.test:
        white_snrs = FULL_WHITE_NOISE_SNRS
        spike_fractions = FULL_SPIKE_FRACTIONS
        spike_multipliers = FULL_SPIKE_MULTIPLIERS
        missing_fractions = FULL_MISSING_FRACTIONS
        missing_bursts = FULL_MISSING_BURSTS
    else:
        white_snrs = CORE_WHITE_NOISE_SNRS
        spike_fractions = CORE_SPIKE_FRACTIONS
        spike_multipliers = CORE_SPIKE_MULTIPLIERS
        missing_fractions = CORE_MISSING_FRACTIONS
        missing_bursts = CORE_MISSING_BURSTS

    if args.test:
        white_snrs = [20, 5]
        spike_fractions = [0.05]
        spike_multipliers = [5.0]
        missing_fractions = [0.05]
        missing_bursts = [3]

    jobs = []
    for _, r in files_df.iterrows():
        filepath = r["filepath"]
        folder = r["folder"]

        for model in args.models:
            for seed in args.seeds:
                if "white_noise" in args.experiments:
                    for snr in white_snrs:
                        jid = f"white_noise::{os.path.basename(filepath)}::{model}::{seed}::snr_{snr}"
                        jobs.append(
                            {
                                "job_id": jid,
                                "experiment": "white_noise",
                                "filepath": filepath,
                                "folder": folder,
                                "model": model,
                                "seed": seed,
                                "snr_db": snr,
                            }
                        )

                if "spikes" in args.experiments:
                    for frac in spike_fractions:
                        for mult in spike_multipliers:
                            jid = f"spikes::{os.path.basename(filepath)}::{model}::{seed}::f{frac}_m{mult}"
                            jobs.append(
                                {
                                    "job_id": jid,
                                    "experiment": "spikes",
                                    "filepath": filepath,
                                    "folder": folder,
                                    "model": model,
                                    "seed": seed,
                                    "fraction": frac,
                                    "multiplier": mult,
                                }
                            )

                if "missing" in args.experiments:
                    for frac in missing_fractions:
                        jid = f"missing::{os.path.basename(filepath)}::{model}::{seed}::point_f{frac}"
                        jobs.append(
                            {
                                "job_id": jid,
                                "experiment": "missing",
                                "filepath": filepath,
                                "folder": folder,
                                "model": model,
                                "seed": seed,
                                "missing_type": "point",
                                "fraction": frac,
                                "num_bursts": 0,
                            }
                        )
                        for nb in missing_bursts:
                            jid = f"missing::{os.path.basename(filepath)}::{model}::{seed}::burst_f{frac}_nb{nb}"
                            jobs.append(
                                {
                                    "job_id": jid,
                                    "experiment": "missing",
                                    "filepath": filepath,
                                    "folder": folder,
                                    "model": model,
                                    "seed": seed,
                                    "missing_type": "burst",
                                    "fraction": frac,
                                    "num_bursts": nb,
                                }
                            )
    return jobs


def _save_outputs(df: pd.DataFrame, top_k: int) -> None:
    checkpoint_path = OUTPUT_DIR / "checkpoint.csv"
    df.to_csv(checkpoint_path, index=False)

    for col in [
        "pattern_auc_roc",
        "pattern_auc_pr",
        "n_eval_pattern",
        "n_pos_pattern",
        "n_neg_pattern",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    group_cols = ["experiment", "model", "pattern", "condition"]
    if "missing_type" in df.columns:
        group_cols.append("missing_type")

    pattern_summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("file", "count"),
            n_files=("file", "nunique"),
            anomaly_points=("anomaly_points", "sum"),
            tp_pattern=("tp_pattern", "sum"),
            fn_pattern=("fn_pattern", "sum"),
            fp_total=("fp_total", "sum"),
            fn_total=("fn_total", "sum"),
            mean_pattern_recall=("pattern_recall", "mean"),
            mean_pattern_fn_rate=("pattern_fn_rate", "mean"),
            mean_pattern_auc_roc=("pattern_auc_roc", "mean"),
            mean_pattern_auc_pr=("pattern_auc_pr", "mean"),
        )
        .reset_index()
    )
    pattern_summary["micro_pattern_recall"] = pattern_summary.apply(
        lambda x: (x["tp_pattern"] / (x["tp_pattern"] + x["fn_pattern"]))
        if (x["tp_pattern"] + x["fn_pattern"]) > 0
        else np.nan,
        axis=1,
    )
    pattern_summary.to_csv(OUTPUT_DIR / "pattern_summary.csv", index=False)

    model_pattern_summary = (
        df.groupby(["experiment", "model", "pattern"], dropna=False)
        .agg(
            n_rows=("file", "count"),
            n_files=("file", "nunique"),
            anomaly_points=("anomaly_points", "sum"),
            tp_pattern=("tp_pattern", "sum"),
            fn_pattern=("fn_pattern", "sum"),
            mean_pattern_recall=("pattern_recall", "mean"),
            mean_pattern_fn_rate=("pattern_fn_rate", "mean"),
            mean_pattern_auc_roc=("pattern_auc_roc", "mean"),
            mean_pattern_auc_pr=("pattern_auc_pr", "mean"),
        )
        .reset_index()
    )
    model_pattern_summary["micro_pattern_recall"] = model_pattern_summary.apply(
        lambda x: (x["tp_pattern"] / (x["tp_pattern"] + x["fn_pattern"]))
        if (x["tp_pattern"] + x["fn_pattern"]) > 0
        else np.nan,
        axis=1,
    )
    model_pattern_summary.to_csv(OUTPUT_DIR / "model_pattern_summary.csv", index=False)

    worst = df[df["anomaly_points"] > 0].copy()
    worst = worst.sort_values(["pattern_fn_rate", "fn_pattern"], ascending=[False, False]).head(top_k)
    worst.to_csv(OUTPUT_DIR / "worst_cases.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pattern-based error analysis reruns")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["white_noise", "spikes", "missing"],
        choices=["white_noise", "spikes", "missing"],
        help="Corruption families to run",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["IForest"],
        choices=["IForest", "LOF", "MP", "PCA", "AE"],
        help="Models to run",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Random seeds",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files (0 = all)")
    parser.add_argument("--full-grid", action="store_true", help="Run full condition grids (default is core grid)")
    parser.add_argument("--test", action="store_true", help="Tiny smoke run")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k worst rows")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing checkpoint and recompute all jobs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subset_df = pd.read_csv(SUBSET_CSV)
    jobs = build_jobs(args, subset_df)

    checkpoint_path = OUTPUT_DIR / "checkpoint.csv"
    existing_rows = []
    completed = set()
    if checkpoint_path.exists() and not args.force_recompute:
        try:
            old = pd.read_csv(checkpoint_path)
            if {"pattern_auc_roc", "pattern_auc_pr"}.issubset(set(old.columns)):
                existing_rows = old.to_dict("records")
                if "job_id" in old.columns:
                    completed = set(old["job_id"].dropna().astype(str).tolist())
                print(f"[Resume] Loaded {len(existing_rows)} rows from checkpoint")
            else:
                print("[Resume] Old checkpoint schema detected (no pattern AUC columns). Recomputing all jobs.")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}")

    jobs = [j for j in jobs if j["job_id"] not in completed]

    print("\n" + "=" * 72)
    print("  Pattern-based Error Analysis")
    print("=" * 72)
    print(f"Experiments: {args.experiments}")
    print(f"Models:      {args.models}")
    print(f"Seeds:       {args.seeds}")
    print(f"Workers:     {args.workers}")
    print(f"Output:      {OUTPUT_DIR}")
    print(f"Jobs total:  {len(jobs)}")
    print("=" * 72)

    all_rows = list(existing_rows)
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_job, job): job for job in jobs}
            with tqdm(total=len(jobs), desc="pattern-analysis") as pbar:
                for fut in as_completed(futures):
                    res = fut.result()
                    if res["status"] == "success":
                        all_rows.extend(res["rows"])
                    elif res["status"] in {"error", "skipped"}:
                        print(f"[{res['status'].upper()}] {res.get('job_id', 'unknown')}: {res.get('error', '')[:200]}")
                    pbar.update(1)

    if all_rows:
        df = pd.DataFrame(all_rows)
        _save_outputs(df, args.top_k)
        print("\nSaved:")
        print(f"- {OUTPUT_DIR / 'checkpoint.csv'}")
        print(f"- {OUTPUT_DIR / 'pattern_summary.csv'}")
        print(f"- {OUTPUT_DIR / 'model_pattern_summary.csv'}")
        print(f"- {OUTPUT_DIR / 'worst_cases.csv'}")
    else:
        print("No rows produced.")


if __name__ == "__main__":
    main()

