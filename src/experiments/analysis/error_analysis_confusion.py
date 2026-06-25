"""
Confusion-matrix style error analysis from existing corruption checkpoints.

This script computes TP/FP/FN/TN per file-condition-model for:
  - white noise (SNR)
  - spikes (normal-only)
  - missing (true impact)

Important:
  We reconstruct confusion counts from stored Precision/Recall and label counts.
  Precision/Recall in these checkpoints come from `get_metrics(..., metric="all")`,
  which uses threshold-based predictions internally. So this is threshold-consistent
  with your current pipeline without rerunning all experiments.

Outputs per experiment (under <results_dir>/error_analysis_confusion/):
  - confusion_file_level.csv
  - confusion_condition_summary.csv
  - confusion_model_summary.csv
  - worst_fp_cases.csv
  - worst_fn_cases.csv
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Path setup so we can import shared loaders
CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = Path(CURRENT_FILE).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_loader import load_tsb_file, remap_filepath  # noqa: E402


EXPERIMENTS = {
    "white_noise": {
        "label": "White Noise (SNR)",
        "results_dir": PROJECT_ROOT / "results" / "experiments" / "white_noise_snr",
        "checkpoint": "checkpoint.csv",
        "condition_cols": ["snr_db", "condition"],
    },
    "spikes": {
        "label": "Spikes (normal-only)",
        "results_dir": PROJECT_ROOT / "results" / "experiments" / "spikes_normal_only",
        "checkpoint": "checkpoint.csv",
        "condition_cols": ["fraction", "multiplier", "condition"],
    },
    "missing": {
        "label": "Missing (true impact)",
        "results_dir": PROJECT_ROOT / "results" / "experiments" / "missing_true_impact",
        "checkpoint": "checkpoint.csv",
        "condition_cols": ["missing_type", "fraction", "num_bursts", "condition"],
    },
}

SUBSET_CSV = PROJECT_ROOT / "results" / "tables" / "final_subset.csv"


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den and den > 0 else np.nan


def _build_file_label_stats() -> dict:
    """
    Build per-file label stats from final subset using canonical file names.
    Returns:
      { canonical_name: {"n_total": int, "n_anom": int, "n_normal": int} }
    """
    df_meta = pd.read_csv(SUBSET_CSV)
    filepaths = df_meta["filepath"].tolist()
    stats = {}

    for fp in filepaths:
        remapped = remap_filepath(fp, str(PROJECT_ROOT))
        data, labels, canonical = load_tsb_file(remapped)
        n_total = int(len(labels))
        n_anom = int(np.sum(labels))
        n_normal = int(n_total - n_anom)
        stats[canonical] = {
            "n_total": n_total,
            "n_anom": n_anom,
            "n_normal": n_normal,
        }

    return stats


def _compute_counts_from_precision_recall(
    precision: float,
    recall: float,
    n_pos: int,
    n_neg: int,
) -> tuple[int, int, int, int]:
    """
    Reconstruct TP/FP/FN/TN from precision/recall and class counts.
    """
    precision = float(precision) if pd.notna(precision) else 0.0
    recall = float(recall) if pd.notna(recall) else 0.0

    tp = int(round(recall * n_pos))
    tp = min(max(tp, 0), n_pos)
    fn = int(max(n_pos - tp, 0))

    if precision > 0:
        fp = int(round(tp * (1.0 / precision - 1.0)))
    else:
        fp = 0 if tp == 0 else n_neg
    fp = min(max(fp, 0), n_neg)

    tn = int(max(n_neg - fp, 0))
    return tp, fp, fn, tn


def _prepare_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()

    for c in ["Precision", "Recall", "F", "AUC_ROC", "AUC_PR"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["snr_db", "fraction", "multiplier", "num_bursts", "seed", "n_kept", "n_lost_anomalies"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _resolve_class_counts(row: pd.Series, file_stats: dict, exp_key: str) -> tuple[int, int]:
    """
    Returns (n_pos, n_neg) for evaluation space used by each experiment.
    """
    f = row["file"]
    if f not in file_stats:
        return 0, 0

    n_pos = int(file_stats[f]["n_anom"])
    n_neg = int(file_stats[f]["n_normal"])

    # Missing true impact excludes masked normal points from evaluation,
    # but keeps masked anomalies as positives with score=0.
    if exp_key == "missing" and ("n_kept" in row.index):
        n_kept = int(row["n_kept"]) if pd.notna(row["n_kept"]) else None
        n_lost_anom = int(row["n_lost_anomalies"]) if ("n_lost_anomalies" in row.index and pd.notna(row["n_lost_anomalies"])) else 0
        if n_kept is not None:
            kept_anom = max(n_pos - n_lost_anom, 0)
            eval_neg = max(n_kept - kept_anom, 0)
            n_neg = int(eval_neg)

    return n_pos, n_neg


def _run_experiment(exp_key: str, file_stats: dict, top_k: int) -> None:
    cfg = EXPERIMENTS[exp_key]
    ckpt_path = cfg["results_dir"] / cfg["checkpoint"]
    if not ckpt_path.exists():
        print(f"[SKIP] Missing checkpoint: {ckpt_path}")
        return

    print("\n" + "=" * 72)
    print(f"  Confusion Error Analysis: {cfg['label']}")
    print("=" * 72)

    df = pd.read_csv(ckpt_path)
    df = _prepare_checkpoint(df)
    if df.empty:
        print("No successful rows.")
        return

    rows = []
    for _, r in df.iterrows():
        file_name = r.get("file")
        if file_name not in file_stats:
            continue

        n_pos, n_neg = _resolve_class_counts(r, file_stats, exp_key)
        if n_pos <= 0 and n_neg <= 0:
            continue

        tp, fp, fn, tn = _compute_counts_from_precision_recall(
            r.get("Precision", 0.0),
            r.get("Recall", 0.0),
            n_pos,
            n_neg,
        )

        out = {
            "file": file_name,
            "model": r.get("model"),
            "seed": r.get("seed"),
            "n_pos_eval": n_pos,
            "n_neg_eval": n_neg,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision_est": _safe_div(tp, tp + fp),
            "recall_est": _safe_div(tp, tp + fn),
            "f1_est": _safe_div(2 * tp, 2 * tp + fp + fn),
            "fpr_est": _safe_div(fp, fp + tn),
            "fnr_est": _safe_div(fn, fn + tp),
            "AUC_ROC": r.get("AUC_ROC", np.nan),
            "AUC_PR": r.get("AUC_PR", np.nan),
            "Precision": r.get("Precision", np.nan),
            "Recall": r.get("Recall", np.nan),
            "F": r.get("F", np.nan),
        }

        for c in cfg["condition_cols"]:
            if c in r.index:
                out[c] = r[c]
        rows.append(out)

    df_conf = pd.DataFrame(rows)
    if df_conf.empty:
        print("No matched rows after file alignment.")
        return

    condition_cols = [c for c in cfg["condition_cols"] if c in df_conf.columns]
    group_cols = ["model"] + condition_cols

    cond_summary = (
        df_conf.groupby(group_cols, dropna=False)
        .agg(
            n_files=("file", "nunique"),
            TP=("TP", "sum"),
            FP=("FP", "sum"),
            FN=("FN", "sum"),
            TN=("TN", "sum"),
            mean_auc_roc=("AUC_ROC", "mean"),
            mean_auc_pr=("AUC_PR", "mean"),
        )
        .reset_index()
    )
    cond_summary["precision_micro"] = cond_summary.apply(lambda x: _safe_div(x["TP"], x["TP"] + x["FP"]), axis=1)
    cond_summary["recall_micro"] = cond_summary.apply(lambda x: _safe_div(x["TP"], x["TP"] + x["FN"]), axis=1)
    cond_summary["f1_micro"] = cond_summary.apply(
        lambda x: _safe_div(2 * x["TP"], 2 * x["TP"] + x["FP"] + x["FN"]), axis=1
    )
    cond_summary["fpr_micro"] = cond_summary.apply(lambda x: _safe_div(x["FP"], x["FP"] + x["TN"]), axis=1)
    cond_summary["fnr_micro"] = cond_summary.apply(lambda x: _safe_div(x["FN"], x["FN"] + x["TP"]), axis=1)

    model_summary = (
        df_conf.groupby(["model"], dropna=False)
        .agg(
            n_rows=("file", "count"),
            TP=("TP", "sum"),
            FP=("FP", "sum"),
            FN=("FN", "sum"),
            TN=("TN", "sum"),
            mean_auc_roc=("AUC_ROC", "mean"),
            mean_auc_pr=("AUC_PR", "mean"),
        )
        .reset_index()
    )
    model_summary["precision_micro"] = model_summary.apply(lambda x: _safe_div(x["TP"], x["TP"] + x["FP"]), axis=1)
    model_summary["recall_micro"] = model_summary.apply(lambda x: _safe_div(x["TP"], x["TP"] + x["FN"]), axis=1)
    model_summary["f1_micro"] = model_summary.apply(
        lambda x: _safe_div(2 * x["TP"], 2 * x["TP"] + x["FP"] + x["FN"]), axis=1
    )
    model_summary["fpr_micro"] = model_summary.apply(lambda x: _safe_div(x["FP"], x["FP"] + x["TN"]), axis=1)
    model_summary["fnr_micro"] = model_summary.apply(lambda x: _safe_div(x["FN"], x["FN"] + x["TP"]), axis=1)

    worst_fp = df_conf.sort_values("FP", ascending=False).head(top_k)
    worst_fn = df_conf.sort_values("FN", ascending=False).head(top_k)

    out_dir = cfg["results_dir"] / "error_analysis_confusion"
    os.makedirs(out_dir, exist_ok=True)
    df_conf.to_csv(out_dir / "confusion_file_level.csv", index=False)
    cond_summary.to_csv(out_dir / "confusion_condition_summary.csv", index=False)
    model_summary.to_csv(out_dir / "confusion_model_summary.csv", index=False)
    worst_fp.to_csv(out_dir / "worst_fp_cases.csv", index=False)
    worst_fn.to_csv(out_dir / "worst_fn_cases.csv", index=False)

    print(f"File-level rows: {len(df_conf)}")
    print(f"Condition summary rows: {len(cond_summary)}")
    print(f"Model summary rows: {len(model_summary)}")
    print(f"Saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Confusion-matrix error analysis from checkpoints")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        choices=list(EXPERIMENTS.keys()),
        help="Experiments to analyze",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Top-k worst FP/FN cases to save")
    args = parser.parse_args()

    print("Building per-file anomaly stats from subset...")
    file_stats = _build_file_label_stats()
    print(f"Loaded stats for {len(file_stats)} files.")

    for exp_key in args.experiments:
        _run_experiment(exp_key, file_stats, args.top_k)

    print("\nDone.")


if __name__ == "__main__":
    main()

