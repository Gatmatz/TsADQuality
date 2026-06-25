import warnings
warnings.filterwarnings('ignore')
import math
import os
import gc
import time
import sys
import numpy as np
import pandas as pd
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# PATH CONFIGURATION
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
tsb_uad_path = os.path.join(project_root, 'TSB-UAD')
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path: sys.path.insert(0, project_root)
if tsb_uad_path not in sys.path: sys.path.insert(0, tsb_uad_path)
if src_path not in sys.path: sys.path.insert(0, src_path)

# Disable TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'


def process_single_file(job_args):
    """Process a single file using the shared loader (handles NASA concatenation)."""
    file_path, selected_models, folder, proj_root = job_args

    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.config.set_visible_devices([], 'GPU')
    try:
        tf.keras.utils.disable_interactive_logging()
    except:
        pass

    from TSB_UAD.models.iforest import IForest
    from TSB_UAD.models.lof import LOF
    from TSB_UAD.models.pca import PCA
    from TSB_UAD.models.matrix_profile import MatrixProfile
    from TSB_UAD.models.feature import Window
    from TSB_UAD.utils.slidingWindows import find_length
    from TSB_UAD.vus.metrics import get_metrics
    from data_loader import load_tsb_file, load_pretrained_ae

    file_results = []
    try:
        data, label, filename = load_tsb_file(file_path)
    except Exception as e:
        return []

    if label.sum() == 0:
        return []

    # 1. Z-score normalization (TSB-AD standard)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).ravel()

    sliding_window = find_length(scaled_data)
    sliding_window = max(int(sliding_window), 10)

    X = Window(window=sliding_window).convert(scaled_data).to_numpy()

    all_models = {
        "IForest": IForest(n_estimators=100, random_state=42),
        "MP": MatrixProfile(window=sliding_window),
        "AE": None,  # loaded from pretrained
        "LOF": LOF(n_neighbors=20),
        "PCA": PCA(n_components=10)
    }
    if selected_models:
        models = {k: v for k, v in all_models.items() if k in selected_models}
    else:
        models = all_models

    for name, clf in models.items():
        try:
            if name == "MP":
                clf.fit(scaled_data)
                score = clf.decision_scores_
            elif name == "AE":
                # Load pretrained AE (consistent with corruption experiments)
                ae_clf, meta = load_pretrained_ae(filename, proj_root)
                ae_clf.predict(scaled_data)
                score = ae_clf.decision_scores_
            else:
                clf.fit(X)
                score = clf.decision_scores_

            # Post-processing
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

            # Padding: AE already pads internally to len(data), others need it
            if name == "AE":
                full_score = score
            else:
                full_score = np.array([score[0]]*math.ceil((sliding_window-1)/2) + list(score) + [score[-1]]*((sliding_window-1)//2))

            metrics = get_metrics(full_score, label, metric="all", slidingWindow=sliding_window)

            file_results.append({
                "file": filename,
                "folder": folder,
                "model": name,
                "n_anomalies": int(label.sum()),
                "anomaly_ratio" : round(label.sum() / len(label), 4),
                "AUC_ROC": metrics["AUC_ROC"],
                "AUC_PR": metrics["AUC_PR"],
                "VUS_ROC": metrics["VUS_ROC"],
                "VUS_PR": metrics["VUS_PR"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F"],
            })
        except Exception as e:
            # print(f"Error model {name} on {filename}: {e}")
            continue

    del models, X, scaled_data, data, label
    gc.collect()
    return file_results

def run_tsb_full_zscore(meta_csv, data_root, output_csv, n_workers=10, subset_csv=None, selected_models=None):
    """Main execution function.

    Uses subset_csv (final_subset.csv) as the authoritative file list.
    Falls back to meta_csv discovery only if subset_csv is not provided.
    """
    if subset_csv and os.path.exists(subset_csv):
        from data_loader import remap_filepath
        df_subset = pd.read_csv(subset_csv)
        to_process_paths = [remap_filepath(p, project_root) for p in df_subset['filepath'].tolist()]
        to_process_paths = [p for p in to_process_paths if os.path.exists(p)]
        # Build folder lookup from subset
        folder_map = {}
        for _, row in df_subset.iterrows():
            fname = os.path.basename(remap_filepath(row['filepath'], project_root))
            folder_map[fname] = row['folder']
        print(f"[INFO] Subset mode: {len(to_process_paths)} files from {subset_csv}")
    else:
        folder_map = {}
        # 1. Identify files from meta CSV
        if not os.path.exists(meta_csv):
            print(f"[ERROR] Meta CSV {meta_csv} not found.")
            return

        df_meta = pd.read_csv(meta_csv)
        benchmark_filenames = set(df_meta['filename'].unique())
        print(f"[INFO] Total benchmark files in meta: {len(benchmark_filenames)}")

        # 2. Locate files on disk
        file_map = {}
        print(f"[INFO] Scanning {data_root} for files (searching all subfolders)...")
        for root, dirs, files in os.walk(data_root):
            for f in files:
                file_map[f] = os.path.join(root, f)

        to_process_paths = []
        for f in benchmark_filenames:
            if f in file_map:
                to_process_paths.append(file_map[f])
                continue
            base, ext = os.path.splitext(f)
            alt_ext = ('.out' if ext.lower() == '.txt' else '.txt')
            found = False
            for current_base in [base, base.replace('-', '_'), base.replace('_', '-')]:
                for current_ext in [ext, alt_ext]:
                    alt_f = current_base + current_ext
                    if alt_f in file_map:
                        to_process_paths.append(file_map[alt_f])
                        found = True
                        break
                if found: break

        print(f"[INFO] Found {len(to_process_paths)} files on disk.")

    # 3. Handle Checkpoints
    all_results = []
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            all_results = existing_df.to_dict('records')
            done_files = set(existing_df['file'].unique())
            print(f"[INFO] Found {len(done_files)} files already processed in output.")
            to_process_paths = [fp for fp in to_process_paths if os.path.basename(fp) not in done_files]
        except Exception as e:
            print(f"[WARNING] Could not read existing output: {e}")

    if not to_process_paths:
        print("[INFO] All benchmark files are already processed.")
        return

    n_models = len(selected_models) if selected_models else 5
    total_jobs = len(to_process_paths) * n_models
    model_names = selected_models if selected_models else ['IForest', 'MP', 'AE', 'LOF', 'PCA']
    print(f"[INFO] Starting: {len(to_process_paths)} files x {n_models} models ({model_names}) = {total_jobs} jobs, {n_workers} workers")
    total_start_time = time.time()

    # Sort by size (optional, but good for quick feedback)
    to_process_paths.sort(key=lambda x: os.path.getsize(x))

    completed_jobs = 0
    completed_files = 0

    # Build job list: (file_path, selected_models, folder, pretrained_dir)
    jobs = []
    for fp in to_process_paths:
        fname = os.path.basename(fp)
        folder = folder_map.get(fname, os.path.basename(os.path.dirname(fp)))
        jobs.append((fp, selected_models, folder, project_root))

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_file, job): job[0] for job in jobs}

        for future in concurrent.futures.as_completed(futures):
            try:
                file_results = future.result()
                completed_files += 1
                n_done = len(file_results) if file_results else 0
                completed_jobs += n_done

                if file_results:
                    all_results.extend(file_results)
                    if completed_jobs % 10 < n_done:
                        pd.DataFrame(all_results).to_csv(output_csv, index=False)

                elapsed = time.time() - total_start_time
                avg_per_job = elapsed / max(completed_jobs, 1)
                rem_jobs = total_jobs - completed_jobs
                rem_time = avg_per_job * rem_jobs
                pct = completed_jobs / total_jobs * 100

                print(f"[PROGRESS] {completed_jobs}/{total_jobs} jobs ({pct:.0f}%) | "
                      f"{completed_files}/{len(to_process_paths)} files | "
                      f"Last: {os.path.basename(futures[future])} ({n_done} ok) | "
                      f"ETA: {rem_time/60:.0f}min", flush=True)
            except Exception as e:
                completed_files += 1
                print(f"[ERROR] Worker failed for {futures[future]}: {e}", flush=True)

    pd.DataFrame(all_results).to_csv(output_csv, index=False)
    print(f"[FINISH] Processing complete. Results saved to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    default_subset = os.path.join(project_root, 'results', 'tables', 'final_subset.csv')
    parser.add_argument('--subset', type=str, default=default_subset,
                        help='Path to subset CSV. Default: final_subset.csv (141 files).')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: results/tables/baseline_final_subset.csv)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to run (e.g. --models IForest PCA). Default: all.')
    args = parser.parse_args()

    sel_models = args.models  # None = all, or list like ['IForest']
    if sel_models:
        print(f"[INFO] Running only: {sel_models}")

    meta_csv = os.path.join(project_root, 'TSB-UAD', 'result', 'accuracy_table', 'mergedTable_AUC_ROC.csv')
    data_root = os.path.join(project_root, 'TSB-UAD', 'data')
    output_csv = args.output or os.path.join(project_root, 'results', 'tables', 'baseline_final_subset.csv')

    run_tsb_full_zscore(meta_csv, data_root, output_csv, n_workers=args.workers, subset_csv=args.subset, selected_models=sel_models)
