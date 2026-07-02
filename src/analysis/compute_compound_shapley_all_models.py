"""Compute Shapley attribution for compound corruptions across all models.

The original compound pipeline can only compute Shapley rows for a model when
all coalitions are present:

    clean, noise, spikes, missing,
    noise+spikes, noise+missing, spikes+missing,
    noise+spikes+missing

In the current results, LOF/MP/AE have the compound rows but their spike-only
single rows are absent from compound_corruptions/summary.csv. This script can
fill those spike-only anchors from spikes_normal_only/summary.csv and writes
new all-model Shapley outputs without overwriting the legacy IForest-only files.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOUND_DIR = PROJECT_ROOT / "results" / "experiments" / "compound_corruptions"
PLOT_DIR = PROJECT_ROOT / "results" / "plots" / "compound_chapter"
COMPOUND_SUMMARY_PATH = COMPOUND_DIR / "summary.csv"
SPIKES_NORMAL_SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "experiments" / "spikes_normal_only" / "summary.csv"
)

MODELS = ["IForest", "LOF", "MP", "AE"]
MODEL_ALIASES = {"Autoencoder": "AE", "ME": "MP"}
PLAYERS = ["noise", "spikes", "missing"]
LEVELS = ["low", "med", "high"]
METRICS = [
    ("mean_AUC_ROC", "auc_roc", "AUC-ROC"),
    ("mean_AUC_PR", "auc_pr", "AUC-PR"),
    ("mean_Recall", "recall", "Recall"),
]
SPIKE_LEVEL_TO_PARAMS = {
    "low": (0.05, 3.0),
    "med": (0.10, 3.0),
    "high": (0.10, 10.0),
}
CORRUPTION_COLORS = {
    "noise": "#4c78a8",
    "spikes": "#f58518",
    "missing": "#54a24b",
}
MISSING_COALITION_COLUMNS = [
    "metric",
    "model",
    "noise_severity",
    "spikes_severity",
    "missing_severity",
    "missing_condition",
]


def normalize_models(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model"] = df["model"].replace(MODEL_ALIASES)
    return df


def load_compound_summary() -> pd.DataFrame:
    if not COMPOUND_SUMMARY_PATH.exists():
        raise FileNotFoundError(COMPOUND_SUMMARY_PATH)
    df = normalize_models(pd.read_csv(COMPOUND_SUMMARY_PATH))
    df["input_source"] = "compound_corruptions/summary.csv"
    return df


def load_spike_single_anchors() -> pd.DataFrame:
    if not SPIKES_NORMAL_SUMMARY_PATH.exists():
        raise FileNotFoundError(SPIKES_NORMAL_SUMMARY_PATH)

    spikes = normalize_models(pd.read_csv(SPIKES_NORMAL_SUMMARY_PATH))
    rows: List[Dict[str, object]] = []
    for severity, (fraction, multiplier) in SPIKE_LEVEL_TO_PARAMS.items():
        selected = spikes[
            spikes["fraction"].eq(fraction) & spikes["multiplier"].eq(multiplier)
        ]
        for _, row in selected.iterrows():
            out: Dict[str, object] = {
                "condition": f"spikes_{severity}_only",
                "combination_name": "single",
                "condition_type": "single",
                "model": row["model"],
                "n": row.get("n_runs", np.nan),
                "input_source": "spikes_normal_only/summary.csv",
            }
            for metric_col, _, _ in METRICS:
                out[metric_col] = row.get(metric_col, np.nan)
            rows.append(out)
    return pd.DataFrame(rows)


def merge_inputs(spike_anchor_mode: str) -> pd.DataFrame:
    df = load_compound_summary()
    spike_anchors = load_spike_single_anchors()

    if spike_anchor_mode == "existing-only":
        merged = df
    elif spike_anchor_mode == "fill-missing-normal-only":
        merged = df.copy()
        for _, row in spike_anchors.iterrows():
            exists = (
                merged["condition"].eq(row["condition"])
                & merged["model"].eq(row["model"])
            ).any()
            if not exists:
                merged = pd.concat([merged, pd.DataFrame([row])], ignore_index=True)
    elif spike_anchor_mode == "replace-with-normal-only":
        spike_condition = df["condition"].astype(str).str.match(
            r"^spikes_(low|med|high)_only$"
        )
        merged = pd.concat(
            [df.loc[~spike_condition].copy(), spike_anchors],
            ignore_index=True,
        )
    else:
        raise ValueError(f"Unknown spike anchor mode: {spike_anchor_mode}")

    return normalize_models(merged)


def coalition_condition(
    coalition: Iterable[str], noise_level: str, spikes_level: str, missing_level: str
) -> str:
    coalition_set = frozenset(coalition)
    if not coalition_set:
        return "clean"

    parts = []
    if "noise" in coalition_set:
        parts.append(f"noise_{noise_level}")
    if "spikes" in coalition_set:
        parts.append(f"spikes_{spikes_level}")
    if "missing" in coalition_set:
        parts.append(f"missing_{missing_level}")
    return "+".join(parts) + ("_only" if len(parts) == 1 else "")


def build_value_lookup(df: pd.DataFrame, metric_col: str) -> Dict[Tuple[str, str], float]:
    values = (
        df[df["model"].isin(MODELS)]
        .groupby(["condition", "model"], dropna=False)[metric_col]
        .mean()
    )
    return {
        (str(condition), str(model)): float(value)
        for (condition, model), value in values.items()
        if pd.notna(value)
    }


def shapley_weight(subset_size: int, n_players: int) -> float:
    return (
        math.factorial(subset_size)
        * math.factorial(n_players - subset_size - 1)
        / math.factorial(n_players)
    )


def compute_metric_shapley(
    df: pd.DataFrame, metric_col: str, metric_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    value_lookup = build_value_lookup(df, metric_col)
    source_lookup = (
        df[df["model"].isin(MODELS)]
        .groupby(["condition", "model"], dropna=False)["input_source"]
        .agg(lambda s: ";".join(sorted(set(map(str, s)))))
        .to_dict()
    )

    detail_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []
    all_players = frozenset(PLAYERS)

    for model in MODELS:
        for noise_level, spikes_level, missing_level in itertools.product(
            LEVELS, repeat=3
        ):
            coalition_names = {
                frozenset(coalition): coalition_condition(
                    coalition, noise_level, spikes_level, missing_level
                )
                for size in range(len(PLAYERS) + 1)
                for coalition in itertools.combinations(PLAYERS, size)
            }

            coalition_values: Dict[frozenset[str], float] = {}
            coalition_sources: Dict[frozenset[str], str] = {}
            complete = True
            for coalition, condition in coalition_names.items():
                value = value_lookup.get((condition, model))
                if value is None:
                    missing_rows.append(
                        {
                            "metric": metric_name,
                            "model": model,
                            "noise_severity": noise_level,
                            "spikes_severity": spikes_level,
                            "missing_severity": missing_level,
                            "missing_condition": condition,
                        }
                    )
                    complete = False
                    break
                coalition_values[coalition] = value
                coalition_sources[coalition] = source_lookup.get(
                    (condition, model), "unknown"
                )
            if not complete:
                continue

            baseline = coalition_values[frozenset()]
            damage = {
                coalition: baseline - value
                for coalition, value in coalition_values.items()
            }
            total_damage = damage[all_players]
            triple_value = coalition_values[all_players]

            for corruption in PLAYERS:
                other_players = [p for p in PLAYERS if p != corruption]
                shapley_value = 0.0

                for size in range(len(other_players) + 1):
                    for subset_tuple in itertools.combinations(other_players, size):
                        subset = frozenset(subset_tuple)
                        subset_with_corruption = subset | {corruption}
                        marginal = (
                            damage[subset_with_corruption] - damage[subset]
                        )
                        shapley_value += shapley_weight(size, len(PLAYERS)) * marginal

                pair_without = all_players - {corruption}
                recovery_if_fixed = (
                    coalition_values[pair_without] - coalition_values[all_players]
                )
                pct = (
                    shapley_value / total_damage * 100.0
                    if abs(total_damage) > 1e-12
                    else np.nan
                )

                detail_rows.append(
                    {
                        "metric": metric_name,
                        "metric_col": metric_col,
                        "model": model,
                        "noise_severity": noise_level,
                        "spikes_severity": spikes_level,
                        "missing_severity": missing_level,
                        "corruption": corruption,
                        "corruption_severity": {
                            "noise": noise_level,
                            "spikes": spikes_level,
                            "missing": missing_level,
                        }[corruption],
                        "baseline_value": baseline,
                        "triple_value": triple_value,
                        "total_damage": total_damage,
                        "shapley_value": shapley_value,
                        "recovery_if_fixed": recovery_if_fixed,
                        "shapley_pct": pct,
                        "clean_source": coalition_sources[frozenset()],
                        "noise_source": coalition_sources[frozenset(["noise"])],
                        "spikes_source": coalition_sources[frozenset(["spikes"])],
                        "missing_source": coalition_sources[frozenset(["missing"])],
                        "noise_spikes_source": coalition_sources[
                            frozenset(["noise", "spikes"])
                        ],
                        "noise_missing_source": coalition_sources[
                            frozenset(["noise", "missing"])
                        ],
                        "spikes_missing_source": coalition_sources[
                            frozenset(["spikes", "missing"])
                        ],
                        "triple_source": coalition_sources[all_players],
                    }
                )

    detail = pd.DataFrame(detail_rows)
    missing = pd.DataFrame(missing_rows, columns=MISSING_COALITION_COLUMNS)
    return detail, missing


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    return (
        detail.groupby(["metric", "metric_col", "model", "corruption"], as_index=False)
        .agg(
            n=("shapley_value", "size"),
            mean_shapley=("shapley_value", "mean"),
            mean_recovery=("recovery_if_fixed", "mean"),
            mean_pct=("shapley_pct", "mean"),
            mean_total_damage=("total_damage", "mean"),
        )
        .sort_values(["metric", "model", "mean_shapley"], ascending=[True, True, False])
    )


def write_split_metric_files(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    for _, slug, _ in METRICS:
        metric_detail = detail[detail["metric"].eq(slug)].copy()
        metric_summary = summary[summary["metric"].eq(slug)].copy()
        metric_detail.to_csv(
            COMPOUND_DIR / f"shapley_ablation_all_models_{slug}.csv",
            index=False,
        )
        metric_summary.to_csv(
            COMPOUND_DIR / f"shapley_summary_all_models_{slug}.csv",
            index=False,
        )


def plot_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for _, slug, title in METRICS:
        metric_summary = summary[summary["metric"].eq(slug)].copy()
        if metric_summary.empty:
            continue

        pivot = (
            metric_summary.pivot(
                index="model", columns="corruption", values="mean_pct"
            )
            .reindex(index=MODELS, columns=PLAYERS)
            .fillna(0.0)
        )

        x = np.arange(len(MODELS))
        width = 0.24
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for idx, corruption in enumerate(PLAYERS):
            ax.bar(
                x + (idx - 1) * width,
                pivot[corruption].to_numpy(),
                width=width,
                label=corruption,
                color=CORRUPTION_COLORS[corruption],
                alpha=0.9,
            )

        ax.axhline(33.333, color="#555555", linestyle="--", linewidth=1, alpha=0.65)
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)
        ax.set_ylabel("Mean Shapley share of total damage (%)")
        ax.set_title(f"Compound Shapley attribution by model ({title})")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        fig.tight_layout()
        out_path = PLOT_DIR / f"fig_compound_shapley_all_models_{slug}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def print_console_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("No complete coalitions found.")
        return

    for _, slug, title in METRICS:
        metric_summary = summary[summary["metric"].eq(slug)].copy()
        if metric_summary.empty:
            continue
        pivot = (
            metric_summary.pivot(
                index="model", columns="corruption", values="mean_pct"
            )
            .reindex(index=MODELS, columns=PLAYERS)
            .round(1)
        )
        print(f"\n=== {title}: mean Shapley % of total damage ===")
        print(pivot.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute all-model Shapley outputs for compound corruptions."
    )
    parser.add_argument(
        "--spike-anchor-mode",
        choices=[
            "existing-only",
            "fill-missing-normal-only",
            "replace-with-normal-only",
        ],
        default="replace-with-normal-only",
        help=(
            "How to handle spike-only single anchors. The default replaces the "
            "legacy IForest-only spike anchors with the corrected all-model "
            "spikes_normal_only anchors for a consistent all-model table."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    COMPOUND_DIR.mkdir(parents=True, exist_ok=True)

    inputs = merge_inputs(args.spike_anchor_mode)

    all_detail: List[pd.DataFrame] = []
    all_missing: List[pd.DataFrame] = []
    for metric_col, slug, _ in METRICS:
        detail, missing = compute_metric_shapley(inputs, metric_col, slug)
        all_detail.append(detail)
        all_missing.append(missing)

    detail_df = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
    missing_df = pd.concat(all_missing, ignore_index=True) if all_missing else pd.DataFrame()
    summary_df = summarize(detail_df)

    detail_path = COMPOUND_DIR / "shapley_ablation_all_models.csv"
    summary_path = COMPOUND_DIR / "shapley_summary_all_models.csv"
    missing_path = COMPOUND_DIR / "shapley_missing_coalitions_all_models.csv"
    input_audit_path = COMPOUND_DIR / "shapley_input_audit_all_models.csv"

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    missing_df.to_csv(missing_path, index=False)

    audit_cols = [
        "condition",
        "combination_name",
        "condition_type",
        "model",
        "n",
        "input_source",
    ] + [metric_col for metric_col, _, _ in METRICS]
    inputs[audit_cols].drop_duplicates().sort_values(
        ["model", "condition"]
    ).to_csv(input_audit_path, index=False)

    write_split_metric_files(detail_df, summary_df)
    plot_summary(summary_df)
    print_console_summary(summary_df)
    print("\nWrote:")
    print(f"  {detail_path}")
    print(f"  {summary_path}")
    print(f"  {missing_path}")
    print(f"  {input_audit_path}")
    print(f"Spike anchor mode: {args.spike_anchor_mode}")


if __name__ == "__main__":
    main()
