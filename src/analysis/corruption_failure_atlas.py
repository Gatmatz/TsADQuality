"""Build a Corruption Failure Atlas from existing experiment outputs.

This script is intentionally read-only with respect to experiment artifacts. It
creates a new analysis layer that links aggregate degradation to detector-level
failure signatures, with multiscale propagation as the propagation evidence.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "analysis" / "failure_atlas"

MODELS = ["IForest", "LOF", "MP", "AE"]
CORRUPTIONS = ["noise", "spikes", "freeze", "missing", "swap", "compound"]

MODEL_ALIASES = {
    "Autoencoder": "AE",
    "MatrixProfile": "MP",
    "Matrix Profile": "MP",
}

CORRUPTION_LABELS = {
    "noise": "Noise",
    "spikes": "Spikes",
    "freeze": "Freeze",
    "missing": "Missing",
    "swap": "Swap",
    "compound": "Compound",
}

RANKING_EXPERIMENT_TO_CORRUPTION = {
    "White Noise": "noise",
    "Spikes": "spikes",
    "Missing Data": "missing",
    "Sensor Freeze": "freeze",
    "Segment Swap": "swap",
}

SEVERITY_ORDER = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
SEVERITY_COLORS = ["#4f9d69", "#f2c14e", "#f78154", "#d64045"]

FAILURE_MODE_LABELS = {
    ("IForest", "noise"): "global score/path shift",
    ("IForest", "spikes"): "rarity flooding / score inversion",
    ("IForest", "freeze"): "path-length separation drift",
    ("IForest", "missing"): "availability-driven score distortion",
    ("IForest", "swap"): "temporal distortion, stable ranking",
    ("IForest", "compound"): "mixed-corruption amplification",
    ("LOF", "noise"): "density contrast collapse",
    ("LOF", "spikes"): "local density pollution",
    ("LOF", "freeze"): "neighborhood flattening",
    ("LOF", "missing"): "neighborhood availability disruption",
    ("LOF", "swap"): "boundary-sensitive neighborhood shift",
    ("LOF", "compound"): "density pollution under mixtures",
    ("MP", "noise"): "nearest-neighbor instability",
    ("MP", "spikes"): "discord saturation / NN instability",
    ("MP", "freeze"): "motif smoothing / low-complexity trap",
    ("MP", "missing"): "subsequence availability loss",
    ("MP", "swap"): "window/block mismatch",
    ("MP", "compound"): "NN regime shift under mixtures",
    ("AE", "noise"): "reconstruction contrast degradation",
    ("AE", "spikes"): "reconstruction saturation",
    ("AE", "freeze"): "low-complexity reconstruction trap",
    ("AE", "missing"): "reconstruction availability loss",
    ("AE", "swap"): "fixed-reference temporal mismatch",
    ("AE", "compound"): "contrast collapse under mixtures",
}

TARGETED_IMPLICATIONS = {
    "noise": "Prioritize denoising or robust scaling before detector scoring.",
    "spikes": "Suppress artificial point outliers before they enter the detector.",
    "freeze": "Detect flat/frozen segments and isolate them before model scoring.",
    "missing": "Separate anomaly erasure from normal-point missingness; imputation alone may not recover erased anomaly evidence.",
    "swap": "Validate ordering and segment boundaries before window-based scoring.",
    "compound": "Treat mixed corruptions as interacting data-quality failures, not as independent single corruptions.",
}


@dataclass(frozen=True)
class SourcePaths:
    causal: Path = RESULTS / "analysis" / "causal_chains" / "causal_chain_summary.csv"
    ranking: Path = RESULTS / "analysis" / "ranking_reliability" / "data_vs_model_failure_modes.csv"
    internal: Path = RESULTS / "experiments" / "internal_analysis" / "aggregate_summary.csv"
    propagation_summary: Path = RESULTS / "experiments" / "propagation_multiscale" / "propagation_summary.csv"
    propagation_zones: Path = RESULTS / "experiments" / "propagation_multiscale" / "propagation_zones.csv"
    anomaly_aware: Path = RESULTS / "experiments" / "anomaly_aware_corruption" / "summary.csv"
    compound_summary: Path = RESULTS / "experiments" / "compound_corruptions" / "summary.csv"
    compound_interaction: Path = RESULTS / "experiments" / "compound_corruptions" / "interaction_summary.csv"
    compound_shapley: Path = RESULTS / "experiments" / "compound_corruptions" / "shapley_summary.csv"


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_model(model: Any) -> str:
    if pd.isna(model):
        return ""
    text = str(model)
    return MODEL_ALIASES.get(text, text)


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


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bounded_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def robust_median(series: pd.Series) -> float | None:
    values = numeric(series).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    lo, hi = values.quantile([0.05, 0.95])
    clipped = values.clip(lo, hi)
    return bounded_float(clipped.median())


def classify_multiscale_profile(row: pd.Series) -> str:
    zone = float(row.get("mean_diff_corruption_zone", 0.0) or 0.0)
    far = float(row.get("mean_diff_far", 0.0) or 0.0)
    ratio = float(row.get("mean_ratio", 0.0) or 0.0)
    far_fraction = far / zone if zone > 0 else 0.0

    if far >= 0.09 or far_fraction >= 0.38:
        return "global"
    if far >= 0.04 or far_fraction >= 0.20 or ratio >= 3.0:
        return "moderate"
    return "local"


def classify_multiscale_strength(row: pd.Series) -> str:
    far = float(row.get("mean_diff_far", 0.0) or 0.0)
    ratio = float(row.get("mean_ratio", 0.0) or 0.0)
    if ratio >= 5.0 or far >= 0.09:
        return "high"
    if ratio >= 3.0 or far >= 0.04:
        return "moderate"
    return "low"


def severity_from_evidence(
    auc_drop: float | None,
    cri: float | None,
    ranking_inverted: bool | None,
    propagation_strength: str | None,
    internal_severe: bool,
) -> str:
    drop = 0.0 if auc_drop is None else max(0.0, auc_drop)
    cri_value = 0.0 if cri is None else max(0.0, cri)

    score = 0
    if drop >= 0.25:
        score = max(score, 3)
    elif drop >= 0.15:
        score = max(score, 2)
    elif drop >= 0.05:
        score = max(score, 1)

    if ranking_inverted and drop >= 0.18:
        score = max(score, 3)
    elif ranking_inverted and drop >= 0.10:
        score = max(score, 2)

    if propagation_strength == "high":
        score = max(score, 2 if drop >= 0.05 else 1)
    elif propagation_strength == "moderate":
        score = max(score, 1)

    if internal_severe:
        score = max(score, 2 if drop >= 0.05 else 1)

    if cri_value >= 0.60 and drop >= 0.05:
        score = max(score, 2)
    elif cri_value >= 0.45:
        score = max(score, 1)

    for name, value in SEVERITY_ORDER.items():
        if value == score:
            return name
    return "low"


def build_multiscale_profiles(paths: SourcePaths) -> pd.DataFrame:
    summary = read_csv_if_exists(paths.propagation_summary)
    zones = read_csv_if_exists(paths.propagation_zones)
    if summary.empty:
        return pd.DataFrame()

    summary["model"] = summary["model"].map(normalize_model)
    summary["corruption"] = summary["corruption_type"].map(normalize_family)
    grouped_summary = (
        summary[summary["model"].isin(MODELS)]
        .groupby(["corruption", "model"], as_index=False)
        .agg(
            mean_ratio=("mean_ratio", "mean"),
            max_ratio=("mean_ratio", "max"),
            mean_diff_far=("mean_diff_far", "mean"),
            n_multiscale_rows=("n", "sum"),
        )
    )

    if not zones.empty:
        zones["model"] = zones["model"].map(normalize_model)
        zones["corruption"] = zones["corruption_type"].map(normalize_family)
        grouped_zones = (
            zones[zones["model"].isin(MODELS)]
            .groupby(["corruption", "model"], as_index=False)
            .agg(
                mean_diff_corruption_zone=("mean_diff_corruption_zone", "mean"),
                mean_diff_near=("mean_diff_near", "mean"),
                mean_diff_mid=("mean_diff_mid", "mean"),
                zone_mean_diff_far=("mean_diff_far", "mean"),
            )
        )
        profiles = grouped_summary.merge(grouped_zones, on=["corruption", "model"], how="left")
        profiles["mean_diff_far"] = profiles["mean_diff_far"].fillna(profiles["zone_mean_diff_far"])
        profiles = profiles.drop(columns=[c for c in ["zone_mean_diff_far"] if c in profiles.columns])
    else:
        profiles = grouped_summary
        profiles["mean_diff_corruption_zone"] = np.nan
        profiles["mean_diff_near"] = np.nan
        profiles["mean_diff_mid"] = np.nan

    profiles["far_to_corrupted_zone_ratio"] = profiles.apply(
        lambda r: (
            r["mean_diff_far"] / r["mean_diff_corruption_zone"]
            if pd.notna(r["mean_diff_far"])
            and pd.notna(r["mean_diff_corruption_zone"])
            and r["mean_diff_corruption_zone"] > 0
            else np.nan
        ),
        axis=1,
    )
    profiles["multiscale_profile"] = profiles.apply(classify_multiscale_profile, axis=1)
    profiles["propagation_strength"] = profiles.apply(classify_multiscale_strength, axis=1)
    return profiles


def build_internal_profiles(paths: SourcePaths) -> pd.DataFrame:
    internal = read_csv_if_exists(paths.internal)
    if internal.empty:
        return pd.DataFrame()

    internal["model"] = internal["model"].map(normalize_model)
    internal["corruption"] = internal["corruption"].map(normalize_family)
    internal = internal[internal["model"].isin(MODELS) & internal["corruption"].isin(CORRUPTIONS)]

    metric_by_model = {
        "AE": ("error_ratio_change_pct_mean", "reconstruction contrast change (%)"),
        "IForest": ("separation_gap_change_pct_mean", "path-length separation change (%)"),
        "LOF": ("kdist_gap_change_pct_mean", "local-density gap change (%)"),
        "MP": ("pct_nn_changed_mean", "nearest-neighbor changed (%)"),
    }

    rows: list[dict[str, Any]] = []
    for (corruption, model), group in internal.groupby(["corruption", "model"]):
        metric_col, metric_name = metric_by_model[model]
        metric_median = robust_median(group[metric_col]) if metric_col in group.columns else None
        aux_col = "nn_dist_gap_change_pct_mean" if model == "MP" else None
        aux_median = robust_median(group[aux_col]) if aux_col and aux_col in group.columns else None

        if model == "AE":
            severe = metric_median is not None and metric_median <= -8.0
        elif model == "IForest":
            severe = metric_median is not None and metric_median <= -15.0
        elif model == "LOF":
            severe = metric_median is not None and abs(metric_median) >= 15.0
        else:
            severe = metric_median is not None and metric_median >= 20.0

        rows.append(
            {
                "corruption": corruption,
                "model": model,
                "internal_metric": metric_name,
                "internal_change_median": metric_median,
                "internal_aux_change_median": aux_median,
                "internal_severe": severe,
                "internal_n_conditions": int(group.shape[0]),
            }
        )

    return pd.DataFrame(rows)


def build_ranking_profiles(paths: SourcePaths) -> pd.DataFrame:
    ranking = read_csv_if_exists(paths.ranking)
    if ranking.empty:
        return pd.DataFrame()
    ranking["corruption"] = ranking["experiment"].map(RANKING_EXPERIMENT_TO_CORRUPTION)
    keep_cols = [
        "corruption",
        "mean_kendall_tau",
        "ranking_unreliability_score",
        "disagreement_ratio",
        "transferability_jaccard",
        "failure_mode",
        "interpretation",
    ]
    return ranking[[c for c in keep_cols if c in ranking.columns]].dropna(subset=["corruption"])


def build_causal_profiles(paths: SourcePaths) -> pd.DataFrame:
    causal = read_csv_if_exists(paths.causal)
    if causal.empty:
        return pd.DataFrame()
    causal["model"] = causal["model"].map(normalize_model)
    causal["corruption"] = causal["corruption"].map(normalize_family)
    causal = causal[causal["model"].isin(MODELS) & causal["corruption"].isin(CORRUPTIONS)]
    causal["ranking_inverted"] = causal["ranking_inverted"].astype(str).str.lower().eq("true")
    return causal[
        [
            "corruption",
            "model",
            "CRI",
            "precision_drop_pct",
            "recall_drop_pct",
            "mean_auc_drop",
            "ranking_inverted",
            "most_vulnerable",
        ]
    ]


def build_anomaly_aware_profiles(paths: SourcePaths) -> pd.DataFrame:
    anomaly = read_csv_if_exists(paths.anomaly_aware)
    if anomaly.empty:
        return pd.DataFrame()

    anomaly["model"] = anomaly["model"].map(normalize_model)
    anomaly["corruption"] = anomaly["corruption_type"].map(normalize_family)
    anomaly = anomaly[anomaly["model"].isin(MODELS)]
    grouped = (
        anomaly.groupby(["corruption", "model", "target_mode"], as_index=False)
        .agg(mean_auc=("mean_AUC_ROC", "mean"))
    )

    rows: list[dict[str, Any]] = []
    for (corruption, model), group in grouped.groupby(["corruption", "model"]):
        modes = {row.target_mode: row.mean_auc for row in group.itertuples(index=False)}
        only_normal = modes.get("only_normal")
        only_anomaly = modes.get("only_anomaly")
        mixed = modes.get("mixed")
        contrast = None
        interpretation = "not measured"
        if only_normal is not None and only_anomaly is not None:
            contrast = only_normal - only_anomaly
            if corruption == "spikes" and contrast < -0.15:
                interpretation = "normal-region corruption creates artificial anomalies"
            elif corruption == "missing" and contrast > 0.15:
                interpretation = "anomaly-region corruption erases anomaly evidence"
            else:
                interpretation = "location sensitivity present but weaker"

        rows.append(
            {
                "corruption": corruption,
                "model": model,
                "auc_only_normal": only_normal,
                "auc_only_anomaly": only_anomaly,
                "auc_mixed": mixed,
                "location_auc_contrast_normal_minus_anomaly": contrast,
                "location_interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows)


def build_compound_profiles(paths: SourcePaths) -> pd.DataFrame:
    summary = read_csv_if_exists(paths.compound_summary)
    interactions = read_csv_if_exists(paths.compound_interaction)
    shapley = read_csv_if_exists(paths.compound_shapley)
    if summary.empty:
        return pd.DataFrame()

    summary["model"] = summary["model"].map(normalize_model)
    summary = summary[summary["model"].isin(MODELS)]

    baselines = (
        summary[summary["condition_type"].eq("baseline")]
        .groupby("model")["mean_AUC_ROC"]
        .mean()
        .to_dict()
    )
    compound = summary[summary["condition_type"].eq("compound")].copy()
    compound["baseline_auc"] = compound["model"].map(baselines)
    compound["auc_drop"] = compound["baseline_auc"] - numeric(compound["mean_AUC_ROC"])
    compound_profiles = (
        compound.groupby("model", as_index=False)
        .agg(
            mean_auc_drop=("auc_drop", "mean"),
            max_auc_drop=("auc_drop", "max"),
            worst_compound=("condition", lambda s: str(s.iloc[int(np.argmax(compound.loc[s.index, "auc_drop"].fillna(-np.inf).to_numpy()))])),
            n_compound_conditions=("condition", "count"),
        )
    )

    if not interactions.empty:
        interactions["model"] = interactions["model"].map(normalize_model)
        inter_profiles = (
            interactions[interactions["model"].isin(MODELS)]
            .groupby("model", as_index=False)
            .agg(
                mean_interaction_pct=("mean_interaction_pct", "mean"),
                max_synergistic_cases=("n_synergistic", "max"),
                n_interaction_rows=("combination_name", "count"),
            )
        )
        compound_profiles = compound_profiles.merge(inter_profiles, on="model", how="left")

    if not shapley.empty:
        shapley["model"] = shapley["model"].map(normalize_model)
        top_shapley = (
            shapley.sort_values(["model", "mean_pct"], ascending=[True, False])
            .groupby("model", as_index=False)
            .first()[["model", "corruption", "mean_pct"]]
            .rename(columns={"corruption": "top_shapley_driver", "mean_pct": "top_shapley_pct"})
        )
        compound_profiles = compound_profiles.merge(top_shapley, on="model", how="left")

    compound_profiles["corruption"] = "compound"
    return compound_profiles


def evidence_confidence(
    corruption: str,
    has_causal: bool,
    has_internal: bool,
    has_multiscale: bool,
    has_ranking: bool,
    has_location: bool,
    has_compound: bool,
) -> str:
    score = sum([has_causal, has_internal, has_multiscale, has_ranking, has_location, has_compound])
    if corruption in {"noise", "spikes", "freeze"} and has_multiscale and score >= 4:
        return "high"
    if corruption == "missing" and has_causal and has_ranking and has_location:
        return "medium-high"
    if corruption == "swap" and has_causal and has_internal and has_ranking:
        return "medium-high"
    if corruption == "compound" and has_compound:
        return "medium"
    if score >= 3:
        return "medium"
    return "low"


def evidence_audit_fields(
    corruption: str,
    has_causal: bool,
    has_internal: bool,
    has_multiscale: bool,
    has_ranking: bool,
    has_location: bool,
    has_compound: bool,
) -> dict[str, Any]:
    layers = []
    if has_causal:
        layers.append("causal-chain performance")
    if has_internal:
        layers.append("detector-internal signature")
    if has_multiscale:
        layers.append("multiscale propagation")
    if has_ranking:
        layers.append("ranking reliability")
    if has_location:
        layers.append("anomaly-aware location")
    if has_compound:
        layers.append("compound interaction")

    count = len(layers)
    if count >= 4:
        strength = "strong"
    elif count >= 2:
        strength = "medium"
    else:
        strength = "weak"

    if has_multiscale:
        propagation_evidence_status = "measured in multiscale"
    elif corruption in {"noise", "spikes", "freeze"}:
        propagation_evidence_status = "missing unexpectedly"
    else:
        propagation_evidence_status = "not measured in multiscale"

    if has_internal and (has_causal or has_compound) and (has_multiscale or has_ranking or has_location):
        mechanism_support = "directly evidence-backed"
    elif (has_causal or has_compound) and (has_ranking or has_location):
        mechanism_support = "indirectly evidence-backed"
    elif has_compound:
        mechanism_support = "compound-level evidence only"
    else:
        mechanism_support = "interpretive / needs more evidence"

    return {
        "has_causal_evidence": has_causal,
        "has_internal_evidence": has_internal,
        "has_multiscale_propagation_evidence": has_multiscale,
        "has_ranking_evidence": has_ranking,
        "has_location_evidence": has_location,
        "has_compound_evidence": has_compound,
        "evidence_layer_count": count,
        "evidence_layers": "; ".join(layers) if layers else "none",
        "evidence_strength": strength,
        "propagation_evidence_status": propagation_evidence_status,
        "mechanism_support": mechanism_support,
    }


def build_atlas(paths: SourcePaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    multiscale = build_multiscale_profiles(paths)
    internal = build_internal_profiles(paths)
    ranking = build_ranking_profiles(paths)
    causal = build_causal_profiles(paths)
    anomaly_aware = build_anomaly_aware_profiles(paths)
    compound = build_compound_profiles(paths)

    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for corruption in CORRUPTIONS:
            causal_row = select_one(causal, corruption, model)
            internal_row = select_one(internal, corruption, model)
            multiscale_row = select_one(multiscale, corruption, model)
            ranking_row = select_one_by_corruption(ranking, corruption)
            anomaly_row = select_one(anomaly_aware, corruption, model)
            compound_row = select_one(compound, corruption, model)
            has_causal = causal_row is not None
            has_internal = internal_row is not None and bool(get_value(internal_row, "internal_metric"))
            has_multiscale = multiscale_row is not None
            has_ranking = ranking_row is not None
            has_location = anomaly_row is not None
            has_compound = compound_row is not None
            audit = evidence_audit_fields(
                corruption=corruption,
                has_causal=has_causal,
                has_internal=has_internal,
                has_multiscale=has_multiscale,
                has_ranking=has_ranking,
                has_location=has_location,
                has_compound=has_compound,
            )

            auc_drop = get_value(causal_row, "mean_auc_drop")
            cri = get_value(causal_row, "CRI")
            ranking_inverted = get_value(causal_row, "ranking_inverted")
            if corruption == "compound":
                auc_drop = get_value(compound_row, "mean_auc_drop")
                cri = None
                ranking_inverted = None

            propagation_strength = get_value(multiscale_row, "propagation_strength")
            internal_severe = bool(get_value(internal_row, "internal_severe") or False)
            severity = severity_from_evidence(
                bounded_float(auc_drop),
                bounded_float(cri),
                bool(ranking_inverted) if ranking_inverted is not None else None,
                propagation_strength,
                internal_severe,
            )

            rows.append(
                {
                    "model": model,
                    "corruption": corruption,
                    "corruption_label": CORRUPTION_LABELS[corruption],
                    "failure_mode": FAILURE_MODE_LABELS[(model, corruption)],
                    "severity": severity,
                    "severity_score": SEVERITY_ORDER[severity],
                    "confidence": evidence_confidence(
                        corruption=corruption,
                        has_causal=has_causal,
                        has_internal=has_internal,
                        has_multiscale=has_multiscale,
                        has_ranking=has_ranking,
                        has_location=has_location,
                        has_compound=has_compound,
                    ),
                    **audit,
                    "mean_auc_drop": auc_drop,
                    "CRI": cri,
                    "precision_drop_pct": get_value(causal_row, "precision_drop_pct"),
                    "recall_drop_pct": get_value(causal_row, "recall_drop_pct"),
                    "ranking_inverted": ranking_inverted,
                    "ranking_failure_mode": get_value(ranking_row, "failure_mode"),
                    "ranking_unreliability_score": get_value(ranking_row, "ranking_unreliability_score"),
                    "transferability_jaccard": get_value(ranking_row, "transferability_jaccard"),
                    "multiscale_profile": get_value(multiscale_row, "multiscale_profile") or "not measured",
                    "propagation_strength": propagation_strength or "not measured",
                    "propagation_ratio": get_value(multiscale_row, "mean_ratio"),
                    "propagation_far_diff": get_value(multiscale_row, "mean_diff_far"),
                    "far_to_corrupted_zone_ratio": get_value(multiscale_row, "far_to_corrupted_zone_ratio"),
                    "internal_metric": get_value(internal_row, "internal_metric"),
                    "internal_change_median": get_value(internal_row, "internal_change_median"),
                    "internal_aux_change_median": get_value(internal_row, "internal_aux_change_median"),
                    "internal_severe": internal_severe,
                    "location_interpretation": get_value(anomaly_row, "location_interpretation") or "not measured",
                    "location_auc_contrast_normal_minus_anomaly": get_value(
                        anomaly_row, "location_auc_contrast_normal_minus_anomaly"
                    ),
                    "compound_worst_condition": get_value(compound_row, "worst_compound"),
                    "compound_max_auc_drop": get_value(compound_row, "max_auc_drop"),
                    "compound_mean_interaction_pct": get_value(compound_row, "mean_interaction_pct"),
                    "compound_top_shapley_driver": get_value(compound_row, "top_shapley_driver"),
                    "compound_top_shapley_pct": get_value(compound_row, "top_shapley_pct"),
                    "targeted_implication": TARGETED_IMPLICATIONS[corruption],
                    "evidence_summary": make_evidence_summary(
                        auc_drop=auc_drop,
                        cri=cri,
                        ranking_inverted=ranking_inverted,
                        multiscale_profile=get_value(multiscale_row, "multiscale_profile"),
                        propagation_strength=propagation_strength,
                        internal_metric=get_value(internal_row, "internal_metric"),
                        internal_change=get_value(internal_row, "internal_change_median"),
                        location=get_value(anomaly_row, "location_interpretation"),
                        compound_driver=get_value(compound_row, "top_shapley_driver"),
                    ),
                }
            )

    atlas = pd.DataFrame(rows)
    evidence = atlas.drop(columns=["failure_mode", "severity", "severity_score"])
    return atlas, evidence, multiscale, build_transition_chains(), build_same_drop_different_mechanisms(atlas)


def build_same_drop_different_mechanisms(atlas: pd.DataFrame, max_delta: float = 0.03) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for corruption, group in atlas.groupby("corruption"):
        usable = group.dropna(subset=["mean_auc_drop"]).copy()
        if usable.shape[0] < 2:
            continue
        records = usable.to_dict("records")
        for idx, left in enumerate(records):
            for right in records[idx + 1 :]:
                delta = abs(float(left["mean_auc_drop"]) - float(right["mean_auc_drop"]))
                if delta > max_delta:
                    continue
                if left["failure_mode"] == right["failure_mode"]:
                    continue
                rows.append(
                    {
                        "corruption": corruption,
                        "model_a": left["model"],
                        "model_b": right["model"],
                        "auc_drop_a": left["mean_auc_drop"],
                        "auc_drop_b": right["mean_auc_drop"],
                        "auc_drop_delta": delta,
                        "failure_mode_a": left["failure_mode"],
                        "failure_mode_b": right["failure_mode"],
                        "mechanism_contrast": f"{left['model']}: {left['failure_mode']} vs {right['model']}: {right['failure_mode']}",
                        "evidence_strength_a": left["evidence_strength"],
                        "evidence_strength_b": right["evidence_strength"],
                        "paper_claim_use": "Same corruption and similar AUC drop, but different diagnosed failure mechanism.",
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["mean_pair_drop"] = (numeric(result["auc_drop_a"]) + numeric(result["auc_drop_b"])) / 2.0
    return result.sort_values(["mean_pair_drop", "auc_drop_delta"], ascending=[False, True])


def select_one(df: pd.DataFrame, corruption: str, model: str) -> pd.Series | None:
    if df.empty or "corruption" not in df.columns or "model" not in df.columns:
        return None
    subset = df[(df["corruption"].eq(corruption)) & (df["model"].eq(model))]
    if subset.empty:
        return None
    return subset.iloc[0]


def select_one_by_corruption(df: pd.DataFrame, corruption: str) -> pd.Series | None:
    if df.empty or "corruption" not in df.columns:
        return None
    subset = df[df["corruption"].eq(corruption)]
    if subset.empty:
        return None
    return subset.iloc[0]


def get_value(row: pd.Series | None, key: str) -> Any:
    if row is None or key not in row.index:
        return None
    value = row[key]
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_evidence_summary(
    auc_drop: Any,
    cri: Any,
    ranking_inverted: Any,
    multiscale_profile: Any,
    propagation_strength: Any,
    internal_metric: Any,
    internal_change: Any,
    location: Any,
    compound_driver: Any,
) -> str:
    parts: list[str] = []
    if auc_drop is not None and not pd.isna(auc_drop):
        parts.append(f"AUC drop={float(auc_drop):.3f}")
    if cri is not None and not pd.isna(cri):
        parts.append(f"CRI={float(cri):.3f}")
    if ranking_inverted is not None:
        parts.append(f"ranking inverted={bool(ranking_inverted)}")
    if multiscale_profile:
        parts.append(f"multiscale propagation={multiscale_profile}/{propagation_strength}")
    if internal_metric and internal_change is not None:
        parts.append(f"{internal_metric} median={float(internal_change):.2f}")
    if location and location != "not measured":
        parts.append(f"location={location}")
    if compound_driver:
        parts.append(f"compound driver={compound_driver}")
    return "; ".join(parts)


def build_transition_chains() -> pd.DataFrame:
    rows = [
        {
            "corruption": "noise",
            "transition_chain": "noise injection -> distance/score inflation -> internal gap degradation -> ranking loss",
            "detectors_most_exposed": "MP, AE, LOF; IForest when propagation becomes global",
            "cleaning_implication": TARGETED_IMPLICATIONS["noise"],
        },
        {
            "corruption": "spikes",
            "transition_chain": "artificial spikes -> rarity flooding/local pollution -> score inversion or discord saturation -> AUC collapse",
            "detectors_most_exposed": "LOF, MP, IForest",
            "cleaning_implication": TARGETED_IMPLICATIONS["spikes"],
        },
        {
            "corruption": "freeze",
            "transition_chain": "flat segment -> low-complexity structure -> motif/reconstruction trap -> moderate shared degradation",
            "detectors_most_exposed": "MP, AE, IForest",
            "cleaning_implication": TARGETED_IMPLICATIONS["freeze"],
        },
        {
            "corruption": "missing",
            "transition_chain": "missingness -> availability loss/anomaly erasure -> detector-specific information loss -> partial or non-recoverable damage",
            "detectors_most_exposed": "all, with location-dependent severity",
            "cleaning_implication": TARGETED_IMPLICATIONS["missing"],
        },
        {
            "corruption": "swap",
            "transition_chain": "segment reorder -> temporal misalignment -> NN/window or reconstruction mismatch -> ranking-stable shared failure",
            "detectors_most_exposed": "MP, AE, LOF",
            "cleaning_implication": TARGETED_IMPLICATIONS["swap"],
        },
        {
            "corruption": "compound",
            "transition_chain": "multiple corruptions -> interaction/overlap -> dominant corruption plus amplification -> mixed failure mode",
            "detectors_most_exposed": "depends on mixture; missing/noise/spikes dominate existing compound evidence",
            "cleaning_implication": TARGETED_IMPLICATIONS["compound"],
        },
    ]
    return pd.DataFrame(rows)


def plot_atlas(atlas: pd.DataFrame, out_path: Path) -> None:
    pivot = atlas.pivot(index="model", columns="corruption", values="severity_score").loc[MODELS, CORRUPTIONS]
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.imshow(pivot.to_numpy(), cmap=ListedColormap(SEVERITY_COLORS), vmin=0, vmax=3)

    ax.set_xticks(np.arange(len(CORRUPTIONS)))
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_xticklabels([CORRUPTION_LABELS[c] for c in CORRUPTIONS], fontsize=12)
    ax.set_yticklabels(MODELS, fontsize=12)
    ax.set_title("Corruption Failure Atlas", fontsize=18, pad=18, weight="bold")

    for row_idx, model in enumerate(MODELS):
        for col_idx, corruption in enumerate(CORRUPTIONS):
            cell = atlas[(atlas["model"].eq(model)) & (atlas["corruption"].eq(corruption))].iloc[0]
            mechanism = "\n".join(textwrap.wrap(cell["failure_mode"], width=22))
            propagation = cell["multiscale_profile"]
            confidence = cell["confidence"]
            marker = "" if propagation == "not measured" else f"\n[{propagation}]"
            text = f"{mechanism}{marker}\nconf: {confidence}"
            color = "white" if cell["severity_score"] >= 2 else "#1f1f1f"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8.5, color=color)

    ax.set_xticks(np.arange(-0.5, len(CORRUPTIONS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=SEVERITY_COLORS[idx], label=name)
        for name, idx in SEVERITY_ORDER.items()
    ]
    ax.legend(
        handles=legend_handles,
        title="Failure severity",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_multiscale_profiles(multiscale: pd.DataFrame, out_path: Path) -> None:
    if multiscale.empty:
        return
    plot_df = multiscale[multiscale["corruption"].isin(["noise", "spikes", "freeze"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    x = np.arange(len(plot_df))
    labels = [f"{r.model}\n{CORRUPTION_LABELS[r.corruption]}" for r in plot_df.itertuples()]

    axes[0].bar(x, plot_df["mean_ratio"], color="#4c78a8")
    axes[0].set_title("Mean propagation ratio")
    axes[0].set_ylabel("ratio")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    axes[1].bar(x, plot_df["mean_diff_far"], color="#f58518")
    axes[1].set_title("Far-zone score difference")
    axes[1].set_ylabel("mean far diff")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.suptitle("Multiscale Propagation Evidence Used by the Atlas", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(atlas: pd.DataFrame, out_path: Path) -> None:
    severity_counts = atlas["severity"].value_counts().reindex(SEVERITY_ORDER.keys(), fill_value=0)
    lines = [
        "# Corruption Failure Atlas",
        "",
        "This folder is generated by `src/analysis/corruption_failure_atlas.py`.",
        "",
        "The atlas converts existing corruption experiments into a detector-by-corruption diagnostic map. It does not rerun or modify any existing experiment.",
        "",
        "## Evidence layers",
        "",
        "- AUC/CRI/ranking inversion from `results/analysis/causal_chains/causal_chain_summary.csv`.",
        "- Detector-specific internal signatures from `results/experiments/internal_analysis/aggregate_summary.csv`.",
        "- Propagation evidence from `results/experiments/propagation_multiscale/*` only.",
        "- Ranking reliability from `results/analysis/ranking_reliability/data_vs_model_failure_modes.csv`.",
        "- Location sensitivity from `results/experiments/anomaly_aware_corruption/summary.csv`.",
        "- Compound evidence from `results/experiments/compound_corruptions/*`.",
        "",
        "## Important limitation",
        "",
        "Multiscale propagation evidence exists for noise, spikes, and freeze. Missing, swap, and compound cells are marked as `not measured` for multiscale propagation unless future multiscale runs are added.",
        "",
        "## Severity counts",
        "",
    ]
    for severity, count in severity_counts.items():
        lines.append(f"- {severity}: {int(count)} cells")
    lines.extend(
        [
            "",
            "## Paper claim supported by this artifact",
            "",
            "Aggregate AUC hides qualitatively different failure modes: detectors can suffer similar degradation while failing through different internal, ranking, and propagation pathways.",
            "",
            "## Main outputs",
            "",
            "- `failure_atlas_cells.csv`: paper-facing atlas table.",
            "- `failure_atlas_evidence.csv`: expanded evidence columns for every cell.",
            "- `multiscale_propagation_profiles.csv`: propagation profiles derived from multiscale runs.",
            "- `failure_transition_chains.csv`: corruption-level causal transition narratives.",
            "- `same_auc_different_mechanisms.csv`: pairs where similar AUC drops hide different mechanisms.",
            "- `failure_atlas_heatmap.png`: visual atlas.",
            "- `multiscale_propagation_profiles.png`: propagation evidence figure.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = SourcePaths()
    atlas, evidence, multiscale, transitions, same_drop = build_atlas(paths)

    atlas.to_csv(OUT / "failure_atlas_cells.csv", index=False)
    evidence.to_csv(OUT / "failure_atlas_evidence.csv", index=False)
    multiscale.to_csv(OUT / "multiscale_propagation_profiles.csv", index=False)
    transitions.to_csv(OUT / "failure_transition_chains.csv", index=False)
    same_drop.to_csv(OUT / "same_auc_different_mechanisms.csv", index=False)

    plot_atlas(atlas, OUT / "failure_atlas_heatmap.png")
    plot_multiscale_profiles(multiscale, OUT / "multiscale_propagation_profiles.png")
    write_readme(atlas, OUT / "README.md")

    print(f"Wrote atlas outputs to {OUT}")
    print(atlas[["model", "corruption", "failure_mode", "severity", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
