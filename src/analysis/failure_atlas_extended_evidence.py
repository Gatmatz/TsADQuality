"""Extended evidence layers for the Corruption Failure Atlas.

This script integrates existing secondary analyses into a compact evidence
package:
- anomaly-pattern vulnerability,
- segment/anomaly-geometry vulnerability,
- transferability/entropy links,
- score-budget stealing.

It does not modify or rerun any experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "failure_atlas_extended"

MODELS = ["IForest", "LOF", "MP", "AE"]
CORRUPTION_ORDER = ["noise", "spikes", "missing", "freeze", "swap", "compound"]
MODEL_ALIASES = {"Autoencoder": "AE", "MatrixProfile": "MP", "Matrix Profile": "MP"}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_model(value: Any) -> str:
    if pd.isna(value):
        return ""
    return MODEL_ALIASES.get(str(value), str(value))


def normalize_family(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    if text.startswith("white noise") or text.startswith("white_noise") or text.startswith("noise") or "snr" in text:
        return "noise"
    if text.startswith("spike") or "spikes" in text:
        return "spikes"
    if text.startswith("missing") or "missing" in text:
        return "missing"
    if text.startswith("sensor freeze") or text.startswith("freeze") or "freeze" in text:
        return "freeze"
    if text.startswith("segment swap") or text.startswith("swap") or "swap" in text:
        return "swap"
    if "compound" in text:
        return "compound"
    return text


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def build_pattern_vulnerability() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ROOT / "results" / "experiments" / "pattern_error_analysis" / "model_pattern_summary.csv"
    df = read_csv(path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["model"] = df["model"].map(normalize_model)
    df["corruption"] = df["experiment"].map(normalize_family)
    df = df[df["model"].isin(MODELS)]
    df = df[numeric(df["anomaly_points"]).fillna(0) > 0].copy()
    for col in ["mean_pattern_recall", "mean_pattern_fn_rate", "micro_pattern_recall", "mean_pattern_auc_roc"]:
        if col in df.columns:
            df[col] = numeric(df[col])

    detail = df[
        [
            "corruption",
            "model",
            "pattern",
            "n_files",
            "anomaly_points",
            "mean_pattern_recall",
            "mean_pattern_fn_rate",
            "micro_pattern_recall",
            "mean_pattern_auc_roc",
            "mean_pattern_auc_pr",
        ]
    ].sort_values(["corruption", "model", "mean_pattern_fn_rate"], ascending=[True, True, False])

    worst = (
        detail.dropna(subset=["mean_pattern_fn_rate"])
        .sort_values(["corruption", "model", "mean_pattern_fn_rate"], ascending=[True, True, False])
        .groupby(["corruption", "model"], as_index=False)
        .first()
    )
    worst = worst.rename(
        columns={
            "pattern": "most_vulnerable_pattern",
            "mean_pattern_recall": "worst_pattern_recall",
            "mean_pattern_fn_rate": "worst_pattern_fn_rate",
            "micro_pattern_recall": "worst_micro_recall",
        }
    )
    worst["claim_use"] = "Shows which anomaly pattern is most vulnerable for each detector-corruption pair."
    return detail, worst


def build_segment_vulnerability() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ROOT / "results" / "experiments" / "segment_vulnerability" / "worst_vulnerable.csv"
    df = read_csv(path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["model"] = df["model"].map(normalize_model)
    df["corruption"] = df["corruption_type"].map(normalize_family)
    df = df[df["model"].isin(MODELS)].copy()
    for col in ["detection_rate", "clean_rate", "detection_drop", "rank_drop", "detection_drop_pct", "n_segments"]:
        if col in df.columns:
            df[col] = numeric(df[col])

    detail_cols = [
        "condition",
        "corruption",
        "model",
        "combined_bin",
        "length_bin",
        "amplitude_bin",
        "n_segments",
        "detection_rate",
        "clean_rate",
        "detection_drop",
        "detection_drop_pct",
        "rank_drop",
    ]
    detail = df[[c for c in detail_cols if c in df.columns]].sort_values(
        ["corruption", "model", "detection_drop_pct"], ascending=[True, True, False]
    )
    worst = (
        detail.dropna(subset=["detection_drop_pct"])
        .sort_values(["corruption", "model", "detection_drop_pct"], ascending=[True, True, False])
        .groupby(["corruption", "model"], as_index=False)
        .first()
    )
    worst = worst.rename(
        columns={
            "combined_bin": "most_vulnerable_segment_bin",
            "detection_drop_pct": "worst_detection_drop_pct",
            "detection_drop": "worst_detection_drop",
        }
    )
    worst["claim_use"] = "Shows which anomaly geometry loses detectability under each corruption."
    return detail, worst


def build_transferability_entropy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entropy_path = ROOT / "results" / "analysis" / "failure_transferability_v2" / "entropy_failure_link.csv"
    pair_path = ROOT / "results" / "analysis" / "failure_transferability_v2" / "per_pair_summary.csv"
    severity_path = ROOT / "results" / "analysis" / "failure_transferability_v2" / "severity_trends.csv"
    entropy = read_csv(entropy_path)
    pairs = read_csv(pair_path)
    severity = read_csv(severity_path)

    if not entropy.empty:
        entropy["corruption"] = entropy["experiment"].map(normalize_family)
        entropy["significant_0_05"] = numeric(entropy["p_value"]) < 0.05
        entropy["claim_use"] = "Links dataset uncertainty/entropy to shared detector failures."

    if not pairs.empty:
        pairs["corruption"] = pairs["experiment"].map(normalize_family)
        pairs["claim_use"] = "Quantifies whether failures transfer across detector pairs."

    if not severity.empty and "experiment" in severity.columns:
        severity["corruption"] = severity["experiment"].map(normalize_family)

    return entropy, pairs, severity


def build_budget_stealing() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ROOT / "results" / "experiments" / "corruption_overlap" / "overlap_summary.csv"
    df = read_csv(path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["model"] = df["model"].map(normalize_model)
    df["corruption"] = df["corruption_type"].map(normalize_family)
    df = df[df["model"].isin(MODELS) & ~df["corruption"].eq("clean")].copy()
    for col in [
        "mean_pct_anomaly",
        "mean_pct_corruption",
        "mean_pct_neither",
        "mean_corruption_pct",
        "mean_budget_stolen",
        "top_k_pct",
    ]:
        if col in df.columns:
            df[col] = numeric(df[col])

    detail_cols = [
        "corruption",
        "condition",
        "model",
        "top_k_pct",
        "n_files",
        "mean_pct_anomaly",
        "mean_pct_corruption",
        "mean_pct_neither",
        "mean_corruption_pct",
        "mean_budget_stolen",
    ]
    detail = df[[c for c in detail_cols if c in df.columns]].sort_values(
        ["corruption", "model", "mean_pct_corruption"], ascending=[True, True, False]
    )

    grouped = (
        detail.groupby(["corruption", "model"], as_index=False)
        .agg(
            max_pct_corruption_in_top_scores=("mean_pct_corruption", "max"),
            mean_pct_corruption_in_top_scores=("mean_pct_corruption", "mean"),
            max_budget_stolen=("mean_budget_stolen", "max"),
            mean_true_anomaly_share=("mean_pct_anomaly", "mean"),
            n_overlap_conditions=("condition", "count"),
        )
        .sort_values(["corruption", "max_pct_corruption_in_top_scores"], ascending=[True, False])
    )
    grouped["claim_use"] = "Shows corrupted regions stealing top anomaly-score budget from true anomalies."
    return detail, grouped


def build_unified_summary(
    pattern_worst: pd.DataFrame,
    segment_worst: pd.DataFrame,
    entropy: pd.DataFrame,
    budget_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for corruption in CORRUPTION_ORDER:
        p = pattern_worst[pattern_worst["corruption"].eq(corruption)] if not pattern_worst.empty else pd.DataFrame()
        s = segment_worst[segment_worst["corruption"].eq(corruption)] if not segment_worst.empty else pd.DataFrame()
        e = entropy[entropy["corruption"].eq(corruption)] if not entropy.empty else pd.DataFrame()
        b = budget_summary[budget_summary["corruption"].eq(corruption)] if not budget_summary.empty else pd.DataFrame()

        worst_pattern = ""
        if not p.empty:
            top = p.sort_values("worst_pattern_fn_rate", ascending=False).iloc[0]
            worst_pattern = f"{top['model']}:{top['most_vulnerable_pattern']} FN={top['worst_pattern_fn_rate']:.3f}"

        worst_segment = ""
        if not s.empty:
            top = s.sort_values("worst_detection_drop_pct", ascending=False).iloc[0]
            worst_segment = f"{top['model']}:{top['most_vulnerable_segment_bin']} drop={top['worst_detection_drop_pct']:.1f}%"

        entropy_link = ""
        if not e.empty:
            sig = e[e["significant_0_05"]].sort_values("spearman_rho", ascending=False)
            if not sig.empty:
                top = sig.iloc[0]
                entropy_link = f"{top['failure_type']} rho={top['spearman_rho']:.3f}, p={top['p_value']:.2g}"

        budget = ""
        if not b.empty:
            top = b.sort_values("max_pct_corruption_in_top_scores", ascending=False).iloc[0]
            budget = f"{top['model']} max corrupted top-score share={top['max_pct_corruption_in_top_scores']:.1f}%"

        rows.append(
            {
                "corruption": corruption,
                "pattern_vulnerability_signal": worst_pattern,
                "segment_vulnerability_signal": worst_segment,
                "transferability_entropy_signal": entropy_link,
                "budget_stealing_signal": budget,
                "paper_use": "Extension layer for anomaly-pattern, segment-geometry, shared-failure, and score-budget mechanisms.",
            }
        )
    return pd.DataFrame(rows)


def plot_pattern_worst(pattern_worst: pd.DataFrame) -> None:
    if pattern_worst.empty:
        return
    plot_df = pattern_worst.dropna(subset=["worst_pattern_fn_rate"]).copy()
    plot_df["label"] = plot_df["corruption"] + "\n" + plot_df["model"]
    plot_df = plot_df.sort_values("worst_pattern_fn_rate", ascending=False).head(16)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(plot_df["label"], plot_df["worst_pattern_fn_rate"], color="#d64045")
    ax.set_ylabel("Worst pattern FN rate")
    ax.set_title("Worst Anomaly-Pattern Vulnerability")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "pattern_vulnerability_top.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_segment_worst(segment_worst: pd.DataFrame) -> None:
    if segment_worst.empty:
        return
    plot_df = segment_worst.dropna(subset=["worst_detection_drop_pct"]).copy()
    plot_df["label"] = plot_df["corruption"] + "\n" + plot_df["model"]
    plot_df = plot_df.sort_values("worst_detection_drop_pct", ascending=False).head(16)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(plot_df["label"], plot_df["worst_detection_drop_pct"], color="#f78154")
    ax.set_ylabel("Worst detection drop (%)")
    ax.set_title("Worst Segment-Geometry Vulnerability")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "segment_vulnerability_top.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_budget_summary(budget_summary: pd.DataFrame) -> None:
    if budget_summary.empty:
        return
    plot_df = budget_summary.dropna(subset=["max_pct_corruption_in_top_scores"]).copy()
    plot_df["label"] = plot_df["corruption"] + "\n" + plot_df["model"]
    plot_df = plot_df.sort_values("max_pct_corruption_in_top_scores", ascending=False).head(16)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(plot_df["label"], plot_df["max_pct_corruption_in_top_scores"], color="#4c78a8")
    ax.set_ylabel("Max corrupted share in top scores (%)")
    ax.set_title("Score-Budget Stealing by Corrupted Regions")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "budget_stealing_top.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme() -> None:
    text = """# Failure Atlas Extended Evidence

Generated by `src/analysis/failure_atlas_extended_evidence.py`.

This folder adds secondary evidence layers to the Corruption Failure Atlas:

- `pattern_vulnerability_layer.csv`: pattern-level recall/FN-rate evidence.
- `pattern_vulnerability_worst.csv`: most vulnerable anomaly pattern per corruption/model.
- `segment_vulnerability_layer.csv`: segment-geometry detection drops.
- `segment_vulnerability_worst.csv`: most vulnerable segment bin per corruption/model.
- `transferability_entropy_layer.csv`: entropy/consensus links to shared failures.
- `failure_transferability_pair_layer.csv`: pairwise detector failure overlap.
- `budget_stealing_layer.csv`: top-score overlap between corrupted regions and true anomalies.
- `budget_stealing_summary.csv`: corruption/model summary of score-budget stealing.
- `extended_evidence_summary.csv`: compact paper-facing summary.

Recommended use:

Use pattern and segment vulnerability as the main extension. Use transferability and budget stealing as supporting evidence. Avoid turning this into a cleaning paper.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    pattern_detail, pattern_worst = build_pattern_vulnerability()
    segment_detail, segment_worst = build_segment_vulnerability()
    entropy, transfer_pairs, severity = build_transferability_entropy()
    budget_detail, budget_summary = build_budget_stealing()
    summary = build_unified_summary(pattern_worst, segment_worst, entropy, budget_summary)

    pattern_detail.to_csv(OUT / "pattern_vulnerability_layer.csv", index=False)
    pattern_worst.to_csv(OUT / "pattern_vulnerability_worst.csv", index=False)
    segment_detail.to_csv(OUT / "segment_vulnerability_layer.csv", index=False)
    segment_worst.to_csv(OUT / "segment_vulnerability_worst.csv", index=False)
    entropy.to_csv(OUT / "transferability_entropy_layer.csv", index=False)
    transfer_pairs.to_csv(OUT / "failure_transferability_pair_layer.csv", index=False)
    severity.to_csv(OUT / "failure_transferability_severity_layer.csv", index=False)
    budget_detail.to_csv(OUT / "budget_stealing_layer.csv", index=False)
    budget_summary.to_csv(OUT / "budget_stealing_summary.csv", index=False)
    summary.to_csv(OUT / "extended_evidence_summary.csv", index=False)

    plot_pattern_worst(pattern_worst)
    plot_segment_worst(segment_worst)
    plot_budget_summary(budget_summary)
    write_readme()

    print(f"Wrote extended evidence outputs to {OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
