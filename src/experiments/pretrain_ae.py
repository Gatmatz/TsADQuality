"""
Pre-train AE models on clean data — train once, use in all experiments.

Trains AE_MLP2 on the clean (uncorrupted) portion of each time series
and saves the keras model + metadata. Experiment scripts then load the
pre-trained model and call predict() only, avoiding redundant retraining.

Training logic follows the TSB-UAD paper (VLDB 2022) exactly:
  - StandardScaler on full clean data
  - find_length for sliding window
  - Training split per dataset type:
      * NASA-SMAP/MSL: use the actual .train. file (anomaly-free)
      * KDD21 (UCR): parse train length from filename pattern
      * YAHOO: 30% of full data
      * All others: 10% of full data
  - fit(train_data, scaled_data)
  - epochs=100, EarlyStopping patience=5

Usage:
    python pretrain_ae.py                    # full run
    python pretrain_ae.py --test             # 3 files only
    python pretrain_ae.py --force            # retrain even if model exists
    python pretrain_ae.py --workers 4        # parallel (careful with TF memory)
"""
import os
import sys
import json
import argparse
import hashlib
import time
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import traceback

# ==========================================
# PATH CONFIGURATION
# ==========================================
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
tsb_uad_path = os.path.join(project_root, 'TSB-UAD')
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path: sys.path.insert(0, project_root)
if tsb_uad_path not in sys.path: sys.path.insert(0, tsb_uad_path)
if src_path not in sys.path: sys.path.insert(0, src_path)

from data_loader import load_tsb_file, remap_filepath

# ==========================================
# CONFIGURATION
# ==========================================
SUBSET_CSV = os.path.join(project_root, "results", "tables", "final_subset.csv")
PRETRAINED_DIR = os.path.join(project_root, "results", "pretrained_models", "ae")
EPOCHS = 100
VERBOSE = 0


def get_train_length(file_path, folder, data_length):
    """Return training length following the TSB-UAD paper methodology.

    The paper uses different strategies per dataset type:
      - NASA-SMAP/MSL: anomaly-free .train. file exists → train length = len(train file)
      - KDD21 (UCR Anomaly Archive): filename encodes train length
        Pattern: {id}_UCR_Anomaly_{name}_{trainlen}_{anomstart}_{anomend}.out
      - YAHOO: 30% of total data
      - All others: 10% of total data

    For NASA datasets, data_loader.load_tsb_file already concatenates
    train+test (train first), so train_length = len(train_file) gives
    the correct split point.

    Returns
    -------
    train_length : int
    method : str  (for logging which strategy was used)
    """
    filename = os.path.basename(file_path)

    # --- NASA-SMAP / NASA-MSL: use actual .train. file length ---
    if folder in ('NASA-SMAP', 'NASA-MSL'):
        train_path = file_path.replace('.test.', '.train.')
        if os.path.exists(train_path):
            df_train = pd.read_csv(train_path, header=None)
            train_len = len(df_train)
            return train_len, f'NASA .train. file ({train_len} rows)'
        # Fallback if .train. file missing (shouldn't happen)
        train_len = int(0.10 * data_length)
        return train_len, 'NASA fallback 10% (.train. not found)'

    # --- KDD21: parse train length from filename ---
    if folder == 'KDD21':
        # Pattern: {id}_UCR_Anomaly_{name}_{trainlen}_{anomstart}_{anomend}.out
        m = re.search(r'_UCR_Anomaly_.+?_(\d+)_\d+_\d+\.out$', filename)
        if m:
            train_len = int(m.group(1))
            return train_len, f'KDD21 filename parse ({train_len})'
        # Fallback
        train_len = int(0.10 * data_length)
        return train_len, 'KDD21 fallback 10% (parse failed)'

    # --- YAHOO: 30% ---
    if folder == 'YAHOO':
        train_len = int(0.30 * data_length)
        return train_len, 'YAHOO 30%'

    # --- All others: 10% ---
    train_len = int(0.10 * data_length)
    return train_len, f'{folder} 10%'


def pretrain_single(file_path, folder):
    """Train AE on clean data for a single file and save the model.

    Follows the TSB-UAD paper methodology:
      1. load_tsb_file (handles NASA train+test concatenation)
      2. StandardScaler on full clean data
      3. find_length for sliding window
      4. get_train_length: NASA→.train. file, KDD21→filename, YAHOO→30%, others→10%
      5. train_data = scaled_data[:train_length]
      6. clf.fit(train_data, scaled_data)
    """
    # Suppress TF logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    from sklearn.preprocessing import StandardScaler
    from TSB_UAD.models.AE_2 import AE_MLP2
    from TSB_UAD.utils.slidingWindows import find_length

    canonical_name = os.path.basename(file_path)

    try:
        # 1. Load data (handles NASA train+test concatenation)
        data, label, canonical_name = load_tsb_file(file_path)

        # 2. StandardScaler on full clean data
        scaled_data = StandardScaler().fit_transform(
            data.reshape(-1, 1)
        ).flatten()

        # 3. Sliding window from clean signal
        sliding_window = max(int(find_length(scaled_data)), 10)

        # 4. Training length (paper-compatible per-dataset strategy)
        train_length, train_method = get_train_length(file_path, folder, len(data))
        train_data = scaled_data[:train_length]

        # Guard: need enough train data
        if len(train_data) < sliding_window:
            return {
                'status': 'skipped',
                'file': canonical_name,
                'reason': f'train_data ({len(train_data)}) < sliding_window ({sliding_window})'
            }

        # 5. Train AE
        clf = AE_MLP2(slidingWindow=sliding_window, epochs=EPOCHS, verbose=VERBOSE)
        clf.fit(train_data, scaled_data)

        # 6. Verify model works: check decision_scores_
        if not hasattr(clf, 'decision_scores_') or clf.decision_scores_ is None:
            return {'status': 'error', 'file': canonical_name, 'error': 'No decision_scores_ after fit'}

        if len(clf.decision_scores_) != len(scaled_data):
            return {
                'status': 'error', 'file': canonical_name,
                'error': f'Score length mismatch: {len(clf.decision_scores_)} vs {len(scaled_data)}'
            }

        # 7. Save model
        model_dir = os.path.join(PRETRAINED_DIR, canonical_name)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.keras")
        clf.model_.save(model_path)

        # 8. Compute hash of training data for integrity checking
        data_hash = hashlib.sha256(train_data.tobytes()).hexdigest()[:16]

        # 9. Save metadata
        metadata = {
            'canonical_name': canonical_name,
            'folder': folder,
            'sliding_window': sliding_window,
            'train_length': train_length,
            'total_length': len(data),
            'train_ratio': round(train_length / len(data), 4),
            'train_method': train_method,
            'epochs': EPOCHS,
            'data_hash': data_hash,
            'tensorflow_version': tf.__version__,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }

        meta_path = os.path.join(model_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # 10. Verify: load model back and predict
        loaded_model = tf.keras.models.load_model(model_path)
        clf_verify = AE_MLP2(slidingWindow=sliding_window, epochs=0, verbose=0)
        clf_verify.model_ = loaded_model
        clf_verify.predict(scaled_data)

        # Compare scores
        max_diff = np.max(np.abs(clf.decision_scores_ - clf_verify.decision_scores_))
        if max_diff > 1e-5:
            return {
                'status': 'error', 'file': canonical_name,
                'error': f'Verification failed: max score diff = {max_diff}'
            }

        # Clean up TF session
        tf.keras.backend.clear_session()

        return {
            'status': 'success',
            'file': canonical_name,
            'folder': folder,
            'sliding_window': sliding_window,
            'train_length': train_length,
            'train_method': train_method,
            'model_path': model_path,
            'metadata_path': meta_path,
        }

    except Exception as e:
        return {
            'status': 'error',
            'file': canonical_name,
            'error': traceback.format_exc()
        }


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Pre-train AE models on clean data")
    parser.add_argument('--test', action='store_true', help='Run on 3 files only')
    parser.add_argument('--force', action='store_true', help='Retrain even if model exists')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers (default: 1, be careful with TF memory)')
    args = parser.parse_args()

    # GPU detection
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[GPU] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[GPU] No GPU found — training on CPU")

    os.makedirs(PRETRAINED_DIR, exist_ok=True)

    if not os.path.exists(SUBSET_CSV):
        print(f"[ERROR] Subset CSV not found: {SUBSET_CSV}")
        return

    # Load file list with folder info
    df_meta = pd.read_csv(SUBSET_CSV)
    file_paths = df_meta['filepath'].tolist()
    folders = df_meta['folder'].tolist()
    file_paths = [remap_filepath(fp, project_root) for fp in file_paths]

    if args.test:
        file_paths = file_paths[:3]
        folders = folders[:3]
        print("!!! TEST MODE !!!")

    # Check which models already exist
    jobs = []
    skipped = 0
    for fp, folder in zip(file_paths, folders):
        canonical = os.path.basename(fp)
        model_path = os.path.join(PRETRAINED_DIR, canonical, "model.keras")
        meta_path = os.path.join(PRETRAINED_DIR, canonical, "metadata.json")

        if not args.force and os.path.exists(model_path) and os.path.exists(meta_path):
            skipped += 1
            continue
        jobs.append((fp, folder))

    print(f"\n{'=' * 60}")
    print(f"  AE Pre-training (TSB-UAD paper methodology)")
    print(f"{'=' * 60}")
    print(f"  Total files:  {len(file_paths)}")
    print(f"  Already done: {skipped}")
    print(f"  To train:     {len(jobs)}")
    print(f"  Epochs:       {EPOCHS}")
    print(f"  GPU:          {'Yes (' + gpus[0].name + ')' if gpus else 'No (CPU)'}")
    print(f"  Train split:  NASA=.train. file | KDD21=filename | YAHOO=30% | others=10%")
    print(f"  Output:       {PRETRAINED_DIR}")
    print(f"{'=' * 60}\n")

    if not jobs:
        print("All models already trained!")
    else:
        results = []
        for fp, folder in tqdm(jobs, desc="Pre-training AE"):
            result = pretrain_single(fp, folder)
            results.append(result)

            if result['status'] == 'success':
                pass
            elif result['status'] == 'skipped':
                print(f"\n  [SKIP] {result['file']}: {result['reason']}")
            else:
                print(f"\n  [ERROR] {result['file']}: {result.get('error', 'unknown')[:200]}")

        # Summary
        n_success = sum(1 for r in results if r['status'] == 'success')
        n_error = sum(1 for r in results if r['status'] == 'error')
        n_skip = sum(1 for r in results if r['status'] == 'skipped')
        print(f"\nResults: {n_success} success, {n_skip} skipped, {n_error} errors")

    # Build manifest from all saved models
    manifest_rows = []
    for canonical in os.listdir(PRETRAINED_DIR):
        meta_path = os.path.join(PRETRAINED_DIR, canonical, "metadata.json")
        model_path = os.path.join(PRETRAINED_DIR, canonical, "model.keras")
        if os.path.exists(meta_path) and os.path.exists(model_path):
            with open(meta_path) as f:
                meta = json.load(f)
            manifest_rows.append({
                'canonical_name': meta['canonical_name'],
                'folder': meta['folder'],
                'sliding_window': meta['sliding_window'],
                'train_length': meta['train_length'],
                'total_length': meta['total_length'],
                'train_ratio': meta['train_ratio'],
                'train_method': meta.get('train_method', 'unknown'),
                'model_path': model_path,
                'metadata_path': meta_path,
            })

    if manifest_rows:
        manifest_path = os.path.join(PRETRAINED_DIR, "manifest.csv")
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        print(f"Manifest: {len(manifest_rows)} models -> {manifest_path}")

    print(f"\n[Done] Pre-trained models saved to {PRETRAINED_DIR}")


if __name__ == "__main__":
    main()
