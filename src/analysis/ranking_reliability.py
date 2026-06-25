"""
Ranking reliability and failure transferability analysis.

This script consolidates existing model-disagreement and failure-transferability
outputs to test whether data-quality defects merely reduce absolute
performance or also undermine detector-ranking conclusions.

Outputs:
  results/analysis/ranking_reliability/ranking_stability_summary.csv
  results/analysis/ranking_reliability/rank_condition_table.csv
  results/analysis/ranking_reliability/disagreement_amplification.csv
  results/analysis/ranking_reliability/failure_transferability_summary.csv
  results/analysis/ranking_reliability/failure_transferability_pairwise.csv
  results/analysis/ranking_reliability/data_vs_model_failure_modes.csv
  results/analysis/ranking_reliability/*.png
  results/analysis/ranking_reliability/README.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
MODEL_DISAGREEMENT_DIR = ANALYSIS_DIR / "model_disagreement"
FAILURE_TRANSFER_DIR = ANALYSIS_DIR / "failure_transferability_v2"
OUTPUT_DIR = ANALYSIS_DIR / "ranking_reliability"

EXPERIMENT_DIR_MAP = {
    "white_noise_snr": "White Noise",
    "spikes_normal_only": "Spikes",
    "missing_true_impact": "Missing Data",
    "freeze": "Sensor Freeze",
    "swap_segment": "Segment Swap",
}

EXPERIMENT_ORDER = [
    "White Noise",
    "Spikes",
    "Missing Data",
    "Sensor Freeze",
    "Segment Swap",
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v - min_v < 1e-12:
        score = pd.Series(1.0, index=series.index)
    else:
        score = ((values - min_v) / (max_v - min_v)).clip(0.0, 1.0)
    return 1.0 - score if invert else score


def load_rank_condition_table() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for folder, label in EXPERIMENT_DIR_MAP.items():
        path = MODEL_DISAGREEMENT_DIR / folder / "rank_stability.csv"
        df = _read_csv(path)
        if df.empty:
            continue
        required = {"condition", "kendall_tau_mean", "kendall_tau_median", "n_files"}
        if not required.issubset(df.columns):
            print(f"[WARN] Skipping rank stability for {folder}: missing columns")
            continue
        df = df.copy()
        df["experiment"] = label
        df["kendall_tau_mean"] = pd.to_numeric(df["kendall_tau_mean"], errors="coerce")
        df["kendall_tau_median"] = pd.to_numeric(df["kendall_tau_median"], errors="coerce")
        df["kendall_tau_std"] = pd.to_numeric(df.get("kendall_tau_std", np.nan), errors="coerce")
        df["n_files"] = pd.to_numeric(df["n_files"], errors="coerce")
        # Lower Kendall tau means less reliable ranking.
        df["ranking_instability"] = 1.0 - ((df["kendall_tau_mean"] + 1.0) / 2.0)
        frames.append(df)
        print(f"[INFO] Loaded rank stability {folder}: {len(df)} rows")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["experiment", "kendall_tau_mean"])


def summarize_ranking_stability(rank_conditions: pd.DataFrame) -> pd.DataFrame:
    if rank_conditions.empty:
        return pd.DataFrame()

    summary = (
        rank_conditions.groupby("experiment", as_index=False)
        .agg(
            n_conditions=("condition", "count"),
            mean_kendall_tau=("kendall_tau_mean", "mean"),
            median_kendall_tau=("kendall_tau_median", "median"),
            worst_kendall_tau=("kendall_tau_mean", "min"),
            mean_ranking_instability=("ranking_instability", "mean"),
            worst_condition=("condition", lambda s: s.iloc[rank_conditions.loc[s.index, "kendall_tau_mean"].argmin()]),
        )
        .copy()
    )
    summary["ranking_unreliability_score"] = _minmax(summary["mean_kendall_tau"], invert=True)
    summary["experiment"] = pd.Categorical(summary["experiment"], EXPERIMENT_ORDER, ordered=True)
    return summary.sort_values("experiment")


def load_disagreement_amplification() -> pd.DataFrame:
    path = MODEL_DISAGREEMENT_DIR / "cross_experiment_summary.csv"
    df = _read_csv(path)
    if df.empty:
        return df
    for col in ["baseline_std", "max_corrupted_std", "disagreement_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["experiment"] = df["experiment"].replace(
        {
            "Spikes (normal region)": "Spikes",
            "White Noise (SNR)": "White Noise",
            "Missing Data (true impact)": "Missing Data",
        }
    )
    df["disagreement_amplification_score"] = _minmax(df["disagreement_ratio"])
    df["experiment"] = pd.Categorical(df["experiment"], EXPERIMENT_ORDER, ordered=True)
    return df.sort_values("experiment")


def load_failure_transferability() -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_path = FAILURE_TRANSFER_DIR / "per_pair_summary.csv"
    pair = _read_csv(pair_path)
    if pair.empty:
        return pd.DataFrame(), pd.DataFrame()

    pair = pair.copy()
    pair["jaccard"] = pd.to_numeric(pair["jaccard"], errors="coerce")
    pair["failure_type"] = pair["failure_type"].astype(str)
    pair["model_pair"] = pair["model_a"].astype(str) + "-" + pair["model_b"].astype(str)

    summary = (
        pair.groupby(["experiment", "failure_type"], as_index=False)
        .agg(
            mean_jaccard=("jaccard", "mean"),
            median_jaccard=("jaccard", "median"),
            min_jaccard=("jaccard", "min"),
            max_jaccard=("jaccard", "max"),
            n_pairs=("jaccard", "count"),
        )
        .copy()
    )

    overall = (
        pair.groupby("experiment", as_index=False)
        .agg(
            transferability_jaccard=("jaccard", "mean"),
            transferability_jaccard_min=("jaccard", "min"),
            transferability_jaccard_max=("jaccard", "max"),
            n_pairs=("jaccard", "count"),
        )
        .copy()
    )
    overall["transferability_score"] = _minmax(overall["transferability_jaccard"])

    summary = summary.merge(
        overall[["experiment", "transferability_jaccard", "transferability_score"]],
        on="experiment",
        how="left",
    )
    summary["experiment"] = pd.Categorical(summary["experiment"], EXPERIMENT_ORDER, ordered=True)
    pair["experiment"] = pd.Categorical(pair["experiment"], EXPERIMENT_ORDER, ordered=True)
    return summary.sort_values(["experiment", "failure_type"]), pair.sort_values(["experiment", "failure_type", "model_pair"])


def classify_failure_modes(
    ranking_summary: pd.DataFrame,
    disagreement: pd.DataFrame,
    transfer_summary: pd.DataFrame,
) -> pd.DataFrame:
    if ranking_summary.empty:
        return pd.DataFrame()

    transfer_overall = (
        transfer_summary.groupby("experiment", as_index=False)
        .agg(
            transferability_jaccard=("transferability_jaccard", "mean"),
            transferability_score=("transferability_score", "mean"),
        )
        if not transfer_summary.empty
        else pd.DataFrame(columns=["experiment", "transferability_jaccard", "transferability_score"])
    )

    out = ranking_summary.merge(
        disagreement[["experiment", "disagreement_ratio", "disagreement_amplification_score"]],
        on="experiment",
        how="left",
    ).merge(
        transfer_overall,
        on="experiment",
        how="left",
    )

    out["ranking_unreliable"] = out["mean_kendall_tau"] < 0.5
    out["high_transferability"] = out["transferability_jaccard"] >= 0.65
    out["high_disagreement_amplification"] = out["disagreement_ratio"] >= 1.10

    def classify(row: pd.Series) -> str:
        if bool(row["ranking_unreliable"]) and bool(row["high_transferability"]):
            return "ranking-unstable data-level failure"
        if bool(row["ranking_unreliable"]) and not bool(row["high_transferability"]):
            return "ranking-unstable model-dependent failure"
        if bool(row["high_transferability"]):
            return "ranking-stable but shared data-level failure"
        return "mostly model-specific or lower-transfer failure"

    out["failure_mode"] = out.apply(classify, axis=1)
    out["interpretation"] = out["failure_mode"].map(
        {
            "ranking-unstable data-level failure": "Corruption changes model ordering and failures overlap across detectors; changing detector alone is unlikely to solve the issue.",
            "ranking-unstable model-dependent failure": "Corruption changes model ordering but failures are less shared; detector behavior matters strongly.",
            "ranking-stable but shared data-level failure": "Model ordering is relatively stable, but the same files fail across detectors; data-level diagnostics remain important.",
            "mostly model-specific or lower-transfer failure": "Failures are less transferable and rankings are comparatively stable; model-specific robustness dominates.",
        }
    )

    out["experiment"] = pd.Categorical(out["experiment"], EXPERIMENT_ORDER, ordered=True)
    return out.sort_values("experiment")


def save_csv(df: pd.DataFrame, name: str) -> None:
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    print(f"[OK] {path} ({len(df)} rows)")


def _ordered_experiments(values: Iterable[str]) -> list[str]:
    vals = [str(v) for v in values]
    ordered = [v for v in EXPERIMENT_ORDER if v in vals]
    ordered.extend([v for v in vals if v not in ordered])
    return ordered


def plot_ranking_stability(ranking_summary: pd.DataFrame) -> None:
    if ranking_summary.empty:
        return
    data = ranking_summary.copy()
    data["experiment"] = data["experiment"].astype(str)
    data = data.set_index("experiment").loc[_ordered_experiments(data["experiment"])].reset_index()

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = ["#b85450" if v < 0.5 else "#4f7cac" for v in data["mean_kendall_tau"]]
    ax.bar(data["experiment"], data["mean_kendall_tau"], color=colors)
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1, label="low reliability threshold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Kendall tau vs clean ranking")
    ax.set_title("Detector Ranking Reliability Under Data Defects")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False)
    for i, value in enumerate(data["mean_kendall_tau"]):
        ax.text(i, value + 0.025, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ranking_stability_bar.png", dpi=220)
    plt.close(fig)


def plot_disagreement(disagreement: pd.DataFrame) -> None:
    if disagreement.empty:
        return
    data = disagreement.copy()
    data["experiment"] = data["experiment"].astype(str)
    data = data.set_index("experiment").loc[_ordered_experiments(data["experiment"])].reset_index()

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(data["experiment"], data["disagreement_ratio"], color="#7a8f3a")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="clean-data disagreement")
    ax.set_ylabel("Disagreement ratio")
    ax.set_title("Disagreement Amplification Across Detectors")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False)
    for i, value in enumerate(data["disagreement_ratio"]):
        ax.text(i, value + 0.015, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "disagreement_amplification_bar.png", dpi=220)
    plt.close(fig)


def plot_transferability(transfer_summary: pd.DataFrame) -> None:
    if transfer_summary.empty:
        return
    pivot = transfer_summary.pivot(index="experiment", columns="failure_type", values="mean_jaccard")
    pivot.index = pivot.index.astype(str)
    pivot = pivot.loc[_ordered_experiments(pivot.index)]
    failure_cols = [c for c in ["fp", "fn"] if c in pivot.columns]
    pivot = pivot[failure_cols]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.upper() for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Cross-Model Failure Transferability")
    fig.colorbar(image, ax=ax, label="Mean Jaccard overlap")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "failure_transferability_heatmap.png", dpi=220)
    plt.close(fig)


def plot_reliability_three_panel(
    ranking_summary: pd.DataFrame,
    disagreement: pd.DataFrame,
    transfer_summary: pd.DataFrame,
) -> None:
    if ranking_summary.empty or disagreement.empty or transfer_summary.empty:
        return

    rank = ranking_summary.copy()
    rank["experiment"] = rank["experiment"].astype(str)
    rank = rank.set_index("experiment").loc[_ordered_experiments(rank["experiment"])].reset_index()

    dis = disagreement.copy()
    dis["experiment"] = dis["experiment"].astype(str)
    dis = dis.set_index("experiment").loc[_ordered_experiments(dis["experiment"])].reset_index()

    transfer = (
        transfer_summary.groupby("experiment", as_index=False)
        .agg(mean_jaccard=("mean_jaccard", "mean"))
        .copy()
    )
    transfer["experiment"] = transfer["experiment"].astype(str)
    transfer = transfer.set_index("experiment").loc[_ordered_experiments(transfer["experiment"])].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(rank["experiment"], rank["mean_kendall_tau"], color="#4f7cac")
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_title("A. Ranking Stability")
    axes[0].set_ylabel("Mean Kendall tau")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(dis["experiment"], dis["disagreement_ratio"], color="#7a8f3a")
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_title("B. Disagreement Amplification")
    axes[1].set_ylabel("Std ratio vs clean")
    axes[1].tick_params(axis="x", rotation=35)

    axes[2].bar(transfer["experiment"], transfer["mean_jaccard"], color="#b85450")
    axes[2].set_title("C. Failure Transferability")
    axes[2].set_ylabel("Mean Jaccard")
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", rotation=35)

    fig.suptitle("Ranking Reliability Under Data Quality Defects", y=1.04, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ranking_reliability_three_panel.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows available._"

    def fmt(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value)

    cols = [str(c) for c in df.columns]
    rows = [[fmt(v) for v in row] for row in df.to_numpy()]
    widths = [max(len(cols[i]), *(len(row[i]) for row in rows)) for i in range(len(cols))]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    return "\n".join([
        render(cols),
        "| " + " | ".join("-" * w for w in widths) + " |",
        *(render(row) for row in rows),
    ])


def write_readme(
    ranking_summary: pd.DataFrame,
    disagreement: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    modes: pd.DataFrame,
) -> None:
    top_rank = ranking_summary[
        ["experiment", "mean_kendall_tau", "worst_kendall_tau", "worst_condition"]
    ].copy()
    top_dis = disagreement[["experiment", "disagreement_ratio"]].copy()
    top_transfer = (
        transfer_summary.groupby("experiment", as_index=False)
        .agg(mean_jaccard=("mean_jaccard", "mean"))
        .copy()
    )
    mode_cols = ["experiment", "failure_mode", "interpretation"]
    lines = [
        "# Ranking Reliability and Failure Transferability",
        "",
        "Generated by `python src/analysis/ranking_reliability.py`.",
        "",
        "This analysis tests whether data-quality defects only reduce absolute performance or also undermine detector-ranking conclusions.",
        "",
        "## Outputs",
        "",
        "- `ranking_stability_summary.csv`: Kendall-tau stability of corrupted rankings against clean rankings.",
        "- `rank_condition_table.csv`: per-condition ranking stability.",
        "- `disagreement_amplification.csv`: cross-model AUC dispersion under corruption versus clean data.",
        "- `failure_transferability_summary.csv`: cross-model Jaccard overlap of failure sets.",
        "- `data_vs_model_failure_modes.csv`: qualitative classification of data-level versus model-dependent failures.",
        "",
        "## Ranking Stability",
        "",
        dataframe_to_markdown(top_rank),
        "",
        "## Disagreement Amplification",
        "",
        dataframe_to_markdown(top_dis),
        "",
        "## Failure Transferability",
        "",
        dataframe_to_markdown(top_transfer),
        "",
        "## Failure-Mode Interpretation",
        "",
        dataframe_to_markdown(modes[mode_cols]) if not modes.empty else "_No modes available._",
    ]
    path = OUTPUT_DIR / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {path}")


def main() -> None:
    ensure_output_dir()

    rank_conditions = load_rank_condition_table()
    ranking_summary = summarize_ranking_stability(rank_conditions)
    disagreement = load_disagreement_amplification()
    transfer_summary, transfer_pairwise = load_failure_transferability()
    modes = classify_failure_modes(ranking_summary, disagreement, transfer_summary)

    save_csv(rank_conditions, "rank_condition_table.csv")
    save_csv(ranking_summary, "ranking_stability_summary.csv")
    save_csv(disagreement, "disagreement_amplification.csv")
    save_csv(transfer_summary, "failure_transferability_summary.csv")
    save_csv(transfer_pairwise, "failure_transferability_pairwise.csv")
    save_csv(modes, "data_vs_model_failure_modes.csv")

    plot_ranking_stability(ranking_summary)
    plot_disagreement(disagreement)
    plot_transferability(transfer_summary)
    plot_reliability_three_panel(ranking_summary, disagreement, transfer_summary)
    write_readme(ranking_summary, disagreement, transfer_summary, modes)

    print("\nReliability summary:")
    if not modes.empty:
        print(
            modes[
                [
                    "experiment",
                    "mean_kendall_tau",
                    "disagreement_ratio",
                    "transferability_jaccard",
                    "failure_mode",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
