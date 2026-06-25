"""
Corruption-Aware Training for Time-Series Anomaly Detection
============================================================
Novel experiment: Does training on corrupted data make the AE more robust?

Compares 3 training strategies:
  1. clean_train    — pretrained AE on clean data (existing baseline)
  2. matched_train  — AE retrained on data with SAME corruption as test
  3. multi_train    — AE retrained on data with MIX of all corruption types

For each strategy, the AE is tested on corrupted data and AUC is compared.
IForest/LOF/MP are included as reference (they always fit on whatever they see).

Key research question:
  "Can corruption-aware data augmentation during training improve
   anomaly detection robustness, analogous to corruption robustness
   in computer vision (Hendrycks & Dietterich, 2019)?"
"""

import os
import sys
import math
import json
import argparse
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "TSB-UAD"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.lof import LOF
from TSB_UAD.models.matrix_profile import MatrixProfile
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics
from TSB_UAD.models.AE_2 import AE_MLP2

from data_loader import load_tsb_file, remap_filepath
from ts_corruptor.core import TSCorruptor
import ts_corruptor.injectors

# ── config ─────────────────────────────────────────────────────────
SUBSET_CSV = PROJECT_ROOT / "results" / "tables" / "final_subset.csv"
PRETRAINED_DIR = PROJECT_ROOT / "results" / "pretrained_models" / "ae"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "corruption_aware_training"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AE_EPOCHS = 100
AE_VERBOSE = 0

# Test corruption conditions
TEST_CONDITIONS = [
    {"name": "noise_snr10",  "type": "noise",   "params": {"snr_db": 10}},
    {"name": "noise_snr0",   "type": "noise",   "params": {"snr_db": 0}},
    {"name": "missing_5pct", "type": "missing", "params": {"fraction": 0.05}},
    {"name": "missing_10pct","type": "missing", "params": {"fraction": 0.10}},
    {"name": "spikes_5pct",  "type": "spikes",  "params": {"fraction": 0.05, "multiplier": 5.0}},
    {"name": "freeze_5pct",  "type": "freeze",  "params": {"fraction": 0.05, "num_stucks": 5}},
]

# Multi-corruption augmentation (all at once, low severity)
MULTI_AUG_CORRUPTIONS = [
    {"type": "noise",   "params": {"snr_db": 20}},
    {"type": "missing", "params": {"fraction": 0.02}},
    {"type": "spikes",  "params": {"fraction": 0.01, "multiplier": 3.0}},
]

CLASSICAL_MODELS = ["IForest", "LOF", "MP"]


def get_train_length(file_path, folder, data_length):
    """Return training split length (same as pretrain_ae.py)."""
    import re
    filename = os.path.basename(file_path)

    if folder in ("NASA-SMAP", "NASA-MSL"):
        train_path = file_path.replace(".test.", ".train.")
        if os.path.exists(train_path):
            return len(pd.read_csv(train_path, header=None))
        return int(0.10 * data_length)

    if folder == "KDD21":
        m = re.search(r"_UCR_Anomaly_.+?_(\d+)_\d+_\d+\.out$", filename)
        if m:
            return int(m.group(1))
        return int(0.10 * data_length)

    if folder == "YAHOO":
        return int(0.30 * data_length)

    return int(0.10 * data_length)


def apply_corruption(data, labels, corruption_type, params, seed=42):
    """Apply a corruption to a numpy array, return corrupted array."""
    df = pd.DataFrame({"value": data, "is_anomaly": labels})
    corruptor = TSCorruptor(df, value_col="value", label_col="is_anomaly", seed=seed)

    if corruption_type == "noise":
        ts_corruptor.injectors.inject_white_noise_snr(corruptor, snr_db=params["snr_db"])
    elif corruption_type == "missing":
        ts_corruptor.injectors.inject_point_missing(corruptor, fraction=params["fraction"])
    elif corruption_type == "spikes":
        ts_corruptor.injectors.inject_spikes(
            corruptor, fraction=params["fraction"],
            multiplier=params["multiplier"], target="only_normal"
        )
    elif corruption_type == "freeze":
        ts_corruptor.injectors.inject_sensor_stuck(
            corruptor, fraction=params["fraction"],
            num_stucks=params["num_stucks"]
        )

    corrupted = corruptor.get_corrupted_df()["value"].to_numpy(float)

    # Handle NaNs from missing data — linear interpolation for training
    if np.isnan(corrupted).any():
        nans = np.isnan(corrupted)
        if nans.all():
            return data.copy()  # fallback
        corrupted = pd.Series(corrupted).interpolate(method="linear").ffill().bfill().to_numpy()

    return corrupted


def apply_multi_corruption(data, labels, seed=42):
    """Apply multiple corruption types at low severity (data augmentation)."""
    corrupted = data.copy()
    for aug in MULTI_AUG_CORRUPTIONS:
        corrupted = apply_corruption(
            corrupted, labels, aug["type"], aug["params"], seed=seed
        )
    return corrupted


def train_ae(train_data, full_data, sliding_window):
    """Train a fresh AE and return it."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    clf = AE_MLP2(slidingWindow=sliding_window, epochs=AE_EPOCHS, verbose=AE_VERBOSE)
    clf.fit(train_data, full_data)
    tf.keras.backend.clear_session()
    return clf


def score_classical(model_name, scaled_data, sliding_window):
    """Fit and score a classical model, return full_score."""
    X = Window(window=sliding_window).convert(scaled_data).to_numpy()

    if model_name == "IForest":
        clf = IForest(n_estimators=100, random_state=42)
        clf.fit(X)
        score = clf.decision_scores_
    elif model_name == "LOF":
        clf = LOF(n_neighbors=20)
        clf.fit(X)
        score = clf.decision_scores_
    elif model_name == "MP":
        clf = MatrixProfile(window=sliding_window)
        clf.fit(scaled_data)
        score = clf.decision_scores_
    else:
        raise ValueError(f"Unknown model: {model_name}")

    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        score.reshape(-1, 1)
    ).ravel()

    # Pad to full length
    full_score = np.array(
        [score[0]] * math.ceil((sliding_window - 1) / 2)
        + list(score)
        + [score[-1]] * ((sliding_window - 1) // 2)
    )
    return full_score


def evaluate_scores(full_score, labels, sliding_window):
    """Compute metrics from scores."""
    if np.isnan(full_score).any() or len(np.unique(full_score)) <= 1:
        return None
    metrics = get_metrics(full_score, labels, metric="all", slidingWindow=sliding_window)
    return metrics


def process_file(file_path, folder, condition, strategy):
    """
    Process one (file, condition, strategy) combination.

    Returns a list of result dicts (one per model).
    """
    canonical_name = os.path.basename(file_path)
    cond_name = condition["name"]
    cond_type = condition["type"]
    cond_params = condition["params"]

    try:
        # ── 1. Load clean data ──
        data, labels, canonical_name = load_tsb_file(file_path)
        train_length = get_train_length(file_path, folder, len(data))

        # Sliding window from clean signal
        clean_scaled = StandardScaler().fit_transform(
            data.reshape(-1, 1)
        ).flatten()
        sliding_window = max(int(find_length(clean_scaled)), 10)

        # ── 2. Corrupt TEST data (full series) ──
        corrupted_full = apply_corruption(data, labels, cond_type, cond_params, seed=0)
        corrupted_scaled = StandardScaler().fit_transform(
            corrupted_full.reshape(-1, 1)
        ).flatten()

        results = []

        # ── 3. AE with different training strategies ──
        if strategy == "clean_train":
            # Use pretrained model
            from tensorflow.keras.models import load_model
            model_path = PRETRAINED_DIR / canonical_name / "model.keras"
            if not model_path.exists():
                return [{"file": canonical_name, "condition": cond_name,
                         "strategy": strategy, "model": "AE",
                         "error": "pretrained model not found"}]

            meta_path = PRETRAINED_DIR / canonical_name / "metadata.json"
            with open(meta_path) as f:
                meta = json.load(f)

            clf = AE_MLP2(slidingWindow=meta["sliding_window"], epochs=0, verbose=0)
            clf.model_ = load_model(str(model_path))
            clf.predict(corrupted_scaled)

            import tensorflow as tf
            tf.keras.backend.clear_session()

            metrics = evaluate_scores(clf.decision_scores_, labels, sliding_window)
            if metrics:
                metrics.update({"file": canonical_name, "condition": cond_name,
                                "strategy": "clean_train", "model": "AE", "error": None})
                results.append(metrics)

        elif strategy == "matched_train":
            # Corrupt training data with SAME corruption as test
            train_data_clean = data[:train_length]
            train_labels = labels[:train_length]
            train_corrupted = apply_corruption(
                train_data_clean, train_labels, cond_type, cond_params, seed=1
            )
            train_scaled = StandardScaler().fit_transform(
                train_corrupted.reshape(-1, 1)
            ).flatten()

            if len(train_scaled) < sliding_window:
                return [{"file": canonical_name, "condition": cond_name,
                         "strategy": strategy, "model": "AE",
                         "error": "train too short"}]

            clf = train_ae(train_scaled, corrupted_scaled, sliding_window)
            metrics = evaluate_scores(clf.decision_scores_, labels, sliding_window)
            if metrics:
                metrics.update({"file": canonical_name, "condition": cond_name,
                                "strategy": "matched_train", "model": "AE", "error": None})
                results.append(metrics)

        elif strategy == "multi_train":
            # Corrupt training data with MIX of all corruption types
            train_data_clean = data[:train_length]
            train_labels = labels[:train_length]
            train_corrupted = apply_multi_corruption(train_data_clean, train_labels, seed=1)
            train_scaled = StandardScaler().fit_transform(
                train_corrupted.reshape(-1, 1)
            ).flatten()

            if len(train_scaled) < sliding_window:
                return [{"file": canonical_name, "condition": cond_name,
                         "strategy": strategy, "model": "AE",
                         "error": "train too short"}]

            clf = train_ae(train_scaled, corrupted_scaled, sliding_window)
            metrics = evaluate_scores(clf.decision_scores_, labels, sliding_window)
            if metrics:
                metrics.update({"file": canonical_name, "condition": cond_name,
                                "strategy": "multi_train", "model": "AE", "error": None})
                results.append(metrics)

        # ── 4. Classical models (reference — always fit on corrupted data) ──
        if strategy == "clean_train":
            # Only run classical models once (with the clean_train strategy pass)
            for model_name in CLASSICAL_MODELS:
                try:
                    full_score = score_classical(model_name, corrupted_scaled, sliding_window)
                    metrics = evaluate_scores(full_score, labels, sliding_window)
                    if metrics:
                        metrics.update({
                            "file": canonical_name, "condition": cond_name,
                            "strategy": "fit_on_corrupted", "model": model_name,
                            "error": None,
                        })
                        results.append(metrics)
                except Exception as e:
                    results.append({
                        "file": canonical_name, "condition": cond_name,
                        "strategy": "fit_on_corrupted", "model": model_name,
                        "error": str(e),
                    })

        return results

    except Exception as e:
        return [{"file": canonical_name, "condition": cond_name,
                 "strategy": strategy, "model": "AE",
                 "error": traceback.format_exc()[:300]}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run on 5 files only")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Suppress TF
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[GPU] {len(gpus)} GPU(s) found")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Load file list
    df_meta = pd.read_csv(SUBSET_CSV)
    files = list(zip(
        [remap_filepath(fp, str(PROJECT_ROOT)) for fp in df_meta["filepath"]],
        df_meta["folder"],
    ))

    if args.test:
        files = files[:5]
        print("*** TEST MODE (5 files) ***")

    # Resume support
    checkpoint_path = RESULTS_DIR / "checkpoint.csv"
    done_keys = set()
    if args.resume and checkpoint_path.exists():
        df_done = pd.read_csv(checkpoint_path)
        for _, row in df_done.iterrows():
            done_keys.add((row["file"], row["condition"], row["strategy"], row["model"]))
        print(f"Resuming: {len(done_keys)} results already done")

    strategies = ["clean_train", "matched_train", "multi_train"]

    # Build job list
    jobs = []
    for fp, folder in files:
        for cond in TEST_CONDITIONS:
            for strat in strategies:
                jobs.append((fp, folder, cond, strat))

    total = len(jobs)
    print(f"\n{'='*60}")
    print(f"  Corruption-Aware Training Experiment")
    print(f"{'='*60}")
    print(f"  Files:       {len(files)}")
    print(f"  Conditions:  {len(TEST_CONDITIONS)}")
    print(f"  Strategies:  {len(strategies)}")
    print(f"  Total jobs:  {total}")
    print(f"  Output:      {RESULTS_DIR}")
    print(f"{'='*60}\n")

    all_results = []
    if args.resume and checkpoint_path.exists():
        all_results = pd.read_csv(checkpoint_path).to_dict("records")

    n_done = 0
    n_new = 0
    for fp, folder, cond, strat in tqdm(jobs, desc="Running"):
        canonical = os.path.basename(fp)
        # Skip if already done (for AE results; classical only run once with clean_train)
        key = (canonical, cond["name"], strat, "AE")
        if key in done_keys:
            n_done += 1
            continue

        results = process_file(fp, folder, cond, strat)
        for r in results:
            rkey = (r.get("file"), r.get("condition"), r.get("strategy"), r.get("model"))
            if rkey not in done_keys:
                all_results.append(r)
                done_keys.add(rkey)
                n_new += 1

        # Periodic save
        if n_new > 0 and n_new % 50 == 0:
            pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)

    # Final save
    pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)
    print(f"\nCompleted: {n_new} new results ({n_done} skipped)")

    # ── Summary ──
    df = pd.read_csv(checkpoint_path)
    df_valid = df[df["error"].isna()].copy()

    if not df_valid.empty:
        summary = df_valid.groupby(["condition", "strategy", "model"])["AUC_ROC"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        summary = summary.sort_values(["condition", "model", "strategy"])

        summary_path = RESULTS_DIR / "summary.csv"
        summary.to_csv(summary_path, index=False)

        print(f"\n{'='*60}")
        print("  SUMMARY: Mean AUC_ROC by strategy")
        print(f"{'='*60}")
        print(summary.to_string(index=False))
        print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
