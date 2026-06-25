"""Strict validation layer for the Corruption Failure Atlas.

Creates reviewer-facing artifacts that make the atlas more auditable:
- rulebook for failure-mode assignment,
- bootstrap confidence intervals for available row-level evidence,
- evidence ablation/stability table,
- formal same-AUC/different-mechanism tests.

The script reads existing outputs only and writes new files under
results/analysis/failure_atlas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "failure_atlas"

MODELS = ["IForest", "LOF", "MP", "AE"]
CORRUPTIONS = ["noise", "spikes", "freeze", "missing", "swap", "compound"]
RNG = np.random.default_rng(20260501)

MODEL_ALIASES = {"Autoencoder": "AE", "MatrixProfile": "MP", "Matrix Profile": "MP"}

PRIMARY_INTERNAL_METRIC = {
    "IForest": ("separation_gap_change_pct", "path-length separation change (%)"),
    "LOF": ("kdist_gap_change_pct", "local-density gap change (%)"),
    "MP": ("pct_nn_changed", "nearest-neighbor changed (%)"),
    "AE": ("error_ratio_change_pct", "reconstruction contrast change (%)"),
}

FAILURE_RULES = {
    ("IForest", "noise"): {
        "failure_mode": "global score/path shift",
        "rule": "AUC drop >= 0.05 AND multiscale profile = global AND path-length separation decreases materially.",
        "primary_metric": "separation_gap_change_pct",
        "threshold": "<= -15%",
    },
    ("IForest", "spikes"): {
        "failure_mode": "rarity flooding / score inversion",
        "rule": "AUC drop >= 0.15 AND ranking inversion = true AND multiscale profile = global AND path-length separation decreases.",
        "primary_metric": "separation_gap_change_pct",
        "threshold": "<= -15%",
    },
    ("IForest", "freeze"): {
        "failure_mode": "path-length separation drift",
        "rule": "Freeze condition with measured multiscale propagation and path-length separation drift; AUC loss may remain moderate.",
        "primary_metric": "separation_gap_change_pct",
        "threshold": "directional decrease",
    },
    ("IForest", "missing"): {
        "failure_mode": "availability-driven score distortion",
        "rule": "Causal/ranking/location evidence indicates missingness damage; multiscale propagation is not claimed.",
        "primary_metric": "location contrast / AUC drop",
        "threshold": "location contrast >= 0.30 or AUC drop >= 0.05",
    },
    ("IForest", "swap"): {
        "failure_mode": "temporal distortion, stable ranking",
        "rule": "Ranking-stable shared failure plus detector-internal signature; multiscale propagation is not claimed.",
        "primary_metric": "separation_gap_change_pct",
        "threshold": "directional change",
    },
    ("IForest", "compound"): {
        "failure_mode": "mixed-corruption amplification",
        "rule": "Compound AUC drop observed; mechanism remains compound-level unless internal/multiscale evidence is added.",
        "primary_metric": "compound AUC drop",
        "threshold": ">= 0.10",
    },
    ("LOF", "noise"): {
        "failure_mode": "density contrast collapse",
        "rule": "AUC drop >= 0.05 AND local-density gap changes materially.",
        "primary_metric": "kdist_gap_change_pct",
        "threshold": "absolute change >= 15%",
    },
    ("LOF", "spikes"): {
        "failure_mode": "local density pollution",
        "rule": "AUC drop >= 0.15 AND ranking inversion = true AND local-density gap changes materially.",
        "primary_metric": "kdist_gap_change_pct",
        "threshold": "absolute change >= 15%",
    },
    ("LOF", "freeze"): {
        "failure_mode": "neighborhood flattening",
        "rule": "Freeze condition with density/neighborhood signature and measured multiscale propagation.",
        "primary_metric": "kdist_gap_change_pct",
        "threshold": "directional change",
    },
    ("LOF", "missing"): {
        "failure_mode": "neighborhood availability disruption",
        "rule": "Missingness causes performance/ranking/location damage; multiscale propagation is not claimed.",
        "primary_metric": "location contrast / AUC drop",
        "threshold": "location contrast >= 0.30 or AUC drop >= 0.05",
    },
    ("LOF", "swap"): {
        "failure_mode": "boundary-sensitive neighborhood shift",
        "rule": "Swap has ranking-stable shared failure plus local-density internal signature.",
        "primary_metric": "kdist_gap_change_pct",
        "threshold": "absolute change >= 15%",
    },
    ("LOF", "compound"): {
        "failure_mode": "density pollution under mixtures",
        "rule": "Compound AUC drop observed; mechanism remains compound-level unless internal/multiscale evidence is added.",
        "primary_metric": "compound AUC drop",
        "threshold": ">= 0.10",
    },
    ("MP", "noise"): {
        "failure_mode": "nearest-neighbor instability",
        "rule": "AUC drop >= 0.10 AND nearest-neighbor changed percentage is high.",
        "primary_metric": "pct_nn_changed",
        "threshold": ">= 20%",
    },
    ("MP", "spikes"): {
        "failure_mode": "discord saturation / NN instability",
        "rule": "AUC drop >= 0.15 AND ranking inversion = true AND nearest-neighbor changed percentage is high.",
        "primary_metric": "pct_nn_changed",
        "threshold": ">= 20%",
    },
    ("MP", "freeze"): {
        "failure_mode": "motif smoothing / low-complexity trap",
        "rule": "Freeze condition with NN/window instability and measured multiscale propagation.",
        "primary_metric": "pct_nn_changed",
        "threshold": ">= 10%",
    },
    ("MP", "missing"): {
        "failure_mode": "subsequence availability loss",
        "rule": "Missingness causes performance/ranking/location damage; multiscale propagation is not claimed.",
        "primary_metric": "location contrast / AUC drop",
        "threshold": "location contrast >= 0.30 or AUC drop >= 0.05",
    },
    ("MP", "swap"): {
        "failure_mode": "window/block mismatch",
        "rule": "Swap has ranking-stable shared failure plus NN/window internal signature.",
        "primary_metric": "pct_nn_changed",
        "threshold": ">= 10%",
    },
    ("MP", "compound"): {
        "failure_mode": "NN regime shift under mixtures",
        "rule": "Compound AUC drop observed; mechanism remains compound-level unless internal/multiscale evidence is added.",
        "primary_metric": "compound AUC drop",
        "threshold": ">= 0.10",
    },
    ("AE", "noise"): {
        "failure_mode": "reconstruction contrast degradation",
        "rule": "AUC drop >= 0.10 AND reconstruction contrast changes materially.",
        "primary_metric": "error_ratio_change_pct",
        "threshold": "absolute change >= 15%",
    },
    ("AE", "spikes"): {
        "failure_mode": "reconstruction saturation",
        "rule": "Spike condition with reconstruction contrast signature and anomaly-aware normal-region pollution.",
        "primary_metric": "error_ratio_change_pct",
        "threshold": "absolute change >= 15%",
    },
    ("AE", "freeze"): {
        "failure_mode": "low-complexity reconstruction trap",
        "rule": "Freeze condition with reconstruction contrast drift and measured multiscale propagation.",
        "primary_metric": "error_ratio_change_pct",
        "threshold": "directional change",
    },
    ("AE", "missing"): {
        "failure_mode": "reconstruction availability loss",
        "rule": "Missingness causes location-dependent anomaly erasure; multiscale propagation is not claimed.",
        "primary_metric": "location contrast / AUC drop",
        "threshold": "location contrast >= 0.30",
    },
    ("AE", "swap"): {
        "failure_mode": "fixed-reference temporal mismatch",
        "rule": "Swap has ranking-stable shared failure plus reconstruction/internal signature.",
        "primary_metric": "error_ratio_change_pct",
        "threshold": "directional change",
    },
    ("AE", "compound"): {
        "failure_mode": "contrast collapse under mixtures",
        "rule": "Compound AUC drop observed; mechanism remains compound-level unless internal/multiscale evidence is added.",
        "primary_metric": "compound AUC drop",
        "threshold": ">= 0.10",
    },
}


def normalize_model(value: Any) -> str:
    if pd.isna(value):
        return ""
    return MODEL_ALIASES.get(str(value), str(value))


def normalize_family(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    if text.startswith("noise") or "noise" in text or "snr" in text:
        return "noise"
    if text.startswith("spike") or "spike" in text:
        return "spikes"
    if text.startswith("freeze") or "freeze" in text:
        return "freeze"
    if text.startswith("missing") or "missing" in text or "ge_" in text:
        return "missing"
    if text.startswith("swap") or "swap" in text:
        return "swap"
    return text


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def bootstrap_mean_ci(values: pd.Series | np.ndarray, n_boot: int = 2000) -> tuple[float, float, float, int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    if n == 1:
        return float(arr[0]), float(arr[0]), float(arr[0]), 1
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), int(n)


def bootstrap_delta_ci(left: pd.Series, right: pd.Series, n_boot: int = 2000) -> tuple[float, float, float, int, int]:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan, np.nan, int(a.size), int(b.size)
    ia = RNG.integers(0, a.size, size=(n_boot, a.size))
    ib = RNG.integers(0, b.size, size=(n_boot, b.size))
    deltas = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(a.mean() - b.mean()), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)), int(a.size), int(b.size)


def build_rulebook(atlas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in atlas.iterrows():
        key = (row["model"], row["corruption"])
        rule = FAILURE_RULES[key]
        rows.append(
            {
                "model": row["model"],
                "corruption": row["corruption"],
                "failure_mode": rule["failure_mode"],
                "assignment_rule": rule["rule"],
                "primary_diagnostic_metric": rule["primary_metric"],
                "threshold_or_condition": rule["threshold"],
                "required_evidence_layers": row["evidence_layers"],
                "evidence_strength": row["evidence_strength"],
                "mechanism_support": row["mechanism_support"],
                "propagation_evidence_status": row["propagation_evidence_status"],
            }
        )
    return pd.DataFrame(rows)


def build_bootstrap_intervals() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    prop_path = ROOT / "results" / "experiments" / "propagation_multiscale" / "propagation_results.csv"
    if prop_path.exists():
        prop = pd.read_csv(prop_path)
        prop["model"] = prop["model"].map(normalize_model)
        prop["corruption"] = prop["corruption_type"].map(normalize_family)
        for (corruption, model), group in prop.groupby(["corruption", "model"]):
            if model not in MODELS:
                continue
            for metric in ["propagation_ratio", "mean_diff_far", "mean_diff_corruption_zone", "AUC_ROC_drop"]:
                if metric in group.columns:
                    mean, lo, hi, n = bootstrap_mean_ci(numeric(group[metric]))
                    rows.append(
                        {
                            "evidence_source": "propagation_multiscale/raw",
                            "corruption": corruption,
                            "model": model,
                            "metric": metric,
                            "mean": mean,
                            "ci95_low": lo,
                            "ci95_high": hi,
                            "n": n,
                            "bootstrap_unit": "row/file-condition",
                        }
                    )

    internal_path = ROOT / "results" / "experiments" / "internal_analysis" / "advanced_analysis" / "merged_internal_performance_rows.csv"
    if internal_path.exists():
        internal = pd.read_csv(internal_path)
        internal["model"] = internal["model"].map(normalize_model)
        internal["corruption_family"] = internal["corruption"].map(normalize_family)
        for (corruption, model), group in internal.groupby(["corruption_family", "model"]):
            if model not in MODELS or corruption not in CORRUPTIONS:
                continue
            metric, metric_label = PRIMARY_INTERNAL_METRIC[model]
            if metric not in group.columns:
                continue
            mean, lo, hi, n = bootstrap_mean_ci(numeric(group[metric]))
            rows.append(
                {
                    "evidence_source": "internal_analysis/merged_rows",
                    "corruption": corruption,
                    "model": model,
                    "metric": metric_label,
                    "mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n": n,
                    "bootstrap_unit": "file-condition",
                }
            )

    ranking_path = ROOT / "results" / "analysis" / "ranking_reliability" / "rank_condition_table.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        exp_map = {
            "White Noise": "noise",
            "Spikes": "spikes",
            "Missing Data": "missing",
            "Sensor Freeze": "freeze",
            "Segment Swap": "swap",
        }
        ranking["corruption"] = ranking["experiment"].map(exp_map)
        for corruption, group in ranking.dropna(subset=["corruption"]).groupby("corruption"):
            for metric in ["kendall_tau_mean", "ranking_instability", "pct_inversions"]:
                if metric in group.columns:
                    mean, lo, hi, n = bootstrap_mean_ci(numeric(group[metric]))
                    rows.append(
                        {
                            "evidence_source": "ranking_reliability/condition_table",
                            "corruption": corruption,
                            "model": "all-model-ranking",
                            "metric": metric,
                            "mean": mean,
                            "ci95_low": lo,
                            "ci95_high": hi,
                            "n": n,
                            "bootstrap_unit": "condition",
                        }
                    )

    anomaly_path = ROOT / "results" / "experiments" / "anomaly_aware_corruption" / "summary.csv"
    if anomaly_path.exists():
        anomaly = pd.read_csv(anomaly_path)
        anomaly["model"] = anomaly["model"].map(normalize_model)
        anomaly["corruption"] = anomaly["corruption_type"].map(normalize_family)
        for (corruption, model), group in anomaly.groupby(["corruption", "model"]):
            if model not in MODELS:
                continue
            normal = numeric(group[group["target_mode"].eq("only_normal")]["mean_AUC_ROC"])
            anom = numeric(group[group["target_mode"].eq("only_anomaly")]["mean_AUC_ROC"])
            delta, lo, hi, n_normal, n_anom = bootstrap_delta_ci(normal, anom)
            rows.append(
                {
                    "evidence_source": "anomaly_aware/summary_conditions",
                    "corruption": corruption,
                    "model": model,
                    "metric": "AUC only_normal minus only_anomaly",
                    "mean": delta,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n": min(n_normal, n_anom),
                    "bootstrap_unit": "condition-summary-row",
                }
            )

    return pd.DataFrame(rows)


def build_ablation_stability(atlas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_cols = {
        "causal-chain performance": "has_causal_evidence",
        "detector-internal signature": "has_internal_evidence",
        "multiscale propagation": "has_multiscale_propagation_evidence",
        "ranking reliability": "has_ranking_evidence",
        "anomaly-aware location": "has_location_evidence",
        "compound interaction": "has_compound_evidence",
    }

    rows = []
    for _, row in atlas.iterrows():
        full_strength = row["evidence_strength"]
        full_count = int(row["evidence_layer_count"])
        for omitted, col in source_cols.items():
            if not to_bool(row[col]):
                new_count = full_count
                affected = False
            else:
                new_count = full_count - 1
                affected = True
            if new_count >= 4:
                new_strength = "strong"
            elif new_count >= 2:
                new_strength = "medium"
            elif new_count >= 1:
                new_strength = "weak"
            else:
                new_strength = "unsupported"

            if new_strength == full_strength:
                status = "unchanged"
            elif new_strength == "unsupported":
                status = "unsupported_without_source"
            else:
                status = f"downgraded_to_{new_strength}"

            rows.append(
                {
                    "model": row["model"],
                    "corruption": row["corruption"],
                    "failure_mode": row["failure_mode"],
                    "omitted_evidence_source": omitted,
                    "source_was_present": affected,
                    "full_evidence_strength": full_strength,
                    "ablated_evidence_strength": new_strength,
                    "full_layer_count": full_count,
                    "ablated_layer_count": new_count,
                    "stability_status": status,
                    "label_defensible_after_ablation": new_strength in {"strong", "medium"},
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail[detail["source_was_present"]]
        .groupby("omitted_evidence_source", as_index=False)
        .agg(
            affected_cells=("failure_mode", "count"),
            defensible_cells=("label_defensible_after_ablation", "sum"),
            unchanged_cells=("stability_status", lambda s: int((s == "unchanged").sum())),
        )
    )
    summary["defensible_rate"] = summary["defensible_cells"] / summary["affected_cells"]
    summary["unchanged_rate"] = summary["unchanged_cells"] / summary["affected_cells"]
    return detail, summary


def get_internal_values(corruption: str, model: str) -> pd.Series:
    path = ROOT / "results" / "experiments" / "internal_analysis" / "advanced_analysis" / "merged_internal_performance_rows.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    internal = pd.read_csv(path)
    internal["model"] = internal["model"].map(normalize_model)
    internal["corruption_family"] = internal["corruption"].map(normalize_family)
    metric, _ = PRIMARY_INTERNAL_METRIC[model]
    subset = internal[(internal["corruption_family"].eq(corruption)) & (internal["model"].eq(model))]
    if metric not in subset.columns:
        return pd.Series(dtype=float)
    return numeric(subset[metric]).dropna()


def metric_threshold_result(model: str, mean: float, lo: float, hi: float) -> tuple[bool | float, str]:
    if not np.isfinite(mean):
        return np.nan, "not testable"
    if model == "IForest":
        supported = hi <= -15.0
        status = "CI entirely <= -15% path-separation threshold" if supported else "CI does not fully cross path-separation threshold"
    elif model == "LOF":
        supported = lo >= 15.0 or hi <= -15.0
        status = "CI entirely outside +/-15% density-gap threshold" if supported else "CI overlaps +/-15% density-gap threshold"
    elif model == "MP":
        supported = lo >= 20.0
        status = "CI entirely >= 20% NN-change threshold" if supported else "CI does not fully exceed 20% NN-change threshold"
    elif model == "AE":
        supported = lo >= 15.0 or hi <= -15.0
        status = "CI entirely outside +/-15% reconstruction-contrast threshold" if supported else "CI overlaps +/-15% reconstruction-contrast threshold"
    else:
        supported = np.nan
        status = "unknown model"
    return supported, status


def build_same_auc_mechanism_tests(same_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, pair in same_pairs.iterrows():
        corruption = pair["corruption"]
        model_a = pair["model_a"]
        model_b = pair["model_b"]
        values_a = get_internal_values(corruption, model_a)
        values_b = get_internal_values(corruption, model_b)
        mean_a, lo_a, hi_a, n_a = bootstrap_mean_ci(values_a)
        mean_b, lo_b, hi_b, n_b = bootstrap_mean_ci(values_b)
        auc_delta = float(pair["auc_drop_delta"])
        auc_similar = auc_delta <= 0.03
        supported_a, status_a = metric_threshold_result(model_a, mean_a, lo_a, hi_a)
        supported_b, status_b = metric_threshold_result(model_b, mean_b, lo_b, hi_b)

        if n_a == 0 or n_b == 0:
            mechanism_test_status = "not testable: internal row-level evidence missing"
            mechanism_distinct = np.nan
        elif bool(supported_a) and bool(supported_b):
            mechanism_test_status = "both detector-specific diagnostic rules supported by bootstrap CI"
            mechanism_distinct = True
        else:
            mechanism_test_status = "one or both diagnostic rules not fully supported by bootstrap CI"
            mechanism_distinct = False

        rows.append(
            {
                "corruption": corruption,
                "model_a": model_a,
                "model_b": model_b,
                "auc_drop_a": pair["auc_drop_a"],
                "auc_drop_b": pair["auc_drop_b"],
                "auc_drop_delta": auc_delta,
                "auc_similar_rule": "absolute delta <= 0.03",
                "auc_statistically_similar": auc_similar,
                "metric_a": PRIMARY_INTERNAL_METRIC[model_a][1],
                "metric_b": PRIMARY_INTERNAL_METRIC[model_b][1],
                "metric_a_mean": mean_a,
                "metric_a_ci95_low": lo_a,
                "metric_a_ci95_high": hi_a,
                "metric_b_mean": mean_b,
                "metric_b_ci95_low": lo_b,
                "metric_b_ci95_high": hi_b,
                "n_a": n_a,
                "n_b": n_b,
                "metric_a_rule_supported_by_ci": supported_a,
                "metric_b_rule_supported_by_ci": supported_b,
                "metric_a_rule_status": status_a,
                "metric_b_rule_status": status_b,
                "cross_metric_comparison_note": "Metrics are detector-specific and not compared on a shared numeric scale.",
                "mechanism_distinct_by_ci": mechanism_distinct,
                "mechanism_test_status": mechanism_test_status,
                "mechanism_contrast": pair["mechanism_contrast"],
            }
        )

    return pd.DataFrame(rows)


def write_method_note() -> None:
    text = """# Strict Failure Atlas Validation

Generated by `src/analysis/failure_atlas_strict_validation.py`.

This layer makes the Corruption Failure Atlas stricter in four ways:

1. Failure-mode labels are tied to explicit assignment rules in `failure_mode_rulebook.csv`.
2. Available row-level metrics receive bootstrap 95% confidence intervals in `bootstrap_evidence_intervals.csv`.
3. The atlas is stress-tested by evidence-source ablation in `atlas_ablation_stability.csv` and `atlas_ablation_summary.csv`.
4. The claim that similar AUC drops hide different mechanisms is formalized in `same_auc_mechanism_tests.csv`.

Important scope note:

- Multiscale propagation bootstrap intervals are available for noise, spikes, and freeze only.
- Causal-chain AUC/CRI values are summary-level in the current artifacts, so they are used as measured summaries rather than bootstrapped raw estimates.
- Mechanism tests use detector-specific internal metrics; when a pair has no row-level internal metric for one side, the test is marked as not testable rather than inferred.
"""
    (OUT / "STRICT_VALIDATION_README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_csv(OUT / "failure_atlas_cells.csv")
    same_pairs = pd.read_csv(OUT / "same_auc_different_mechanisms.csv")

    rulebook = build_rulebook(atlas)
    bootstrap = build_bootstrap_intervals()
    ablation_detail, ablation_summary = build_ablation_stability(atlas)
    same_tests = build_same_auc_mechanism_tests(same_pairs)

    rulebook.to_csv(OUT / "failure_mode_rulebook.csv", index=False)
    bootstrap.to_csv(OUT / "bootstrap_evidence_intervals.csv", index=False)
    ablation_detail.to_csv(OUT / "atlas_ablation_stability.csv", index=False)
    ablation_summary.to_csv(OUT / "atlas_ablation_summary.csv", index=False)
    same_tests.to_csv(OUT / "same_auc_mechanism_tests.csv", index=False)
    write_method_note()

    print(f"Wrote strict validation outputs to {OUT}")
    print("Rulebook rows:", len(rulebook))
    print("Bootstrap rows:", len(bootstrap))
    print("Same-AUC tests:", len(same_tests))
    print(ablation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
