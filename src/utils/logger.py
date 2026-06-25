import os
import csv
from datetime import datetime
import json

def log_experiment(model_name: str, experiment_name: str, dataset_name: str, parameters: dict, metrics: dict):
    """
    Logs experiment results to a specific CSV file grouped by model and experiment type.
    
    Args:
        model_name (str): The name of the target model (e.g., 'IForest', 'PCA', 'Chronos')
        experiment_name (str): The type of experiment (e.g., 'WhiteNoise', 'Adversarial_Distillation')
        dataset_name (str): The name of the dataset run (e.g., 'Daphnet_S01R02E0.csv')
        parameters (dict): The parameters used for the run (e.g., {'alpha': 0.01, 'target': 'PCA'})
        metrics (dict): The full dictionary of metrics (AUC_ROC, Precision, L2_Norm, AUC_Drop, etc.)
    """
    # Assuming scripts run from within thesis_timeseries/src/... 
    # Let's find the project root dynamically
    current_dir = os.path.abspath(__file__)
    # Go up from logger.py -> utils -> src -> thesis_timeseries
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    output_dir = os.path.join(project_root, 'outputs', 'metrics', model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = os.path.join(output_dir, f"{experiment_name}.csv")
    
    # Prepare the row
    row = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Dataset_Name': dataset_name,
        'Parameters': json.dumps(parameters)
    }
    # Add all metrics to the row
    row.update(metrics)
    
    file_exists = os.path.isfile(csv_file)
    
    # We want to manage headers dynamically in case new metrics are added later.
    # If the file exists, we should read the existing headers to avoid mismatch,
    # but for simplicity, we assume metrics dict keys are consistent per experiment.
    headers = list(row.keys())
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        # If file exists but headers don't match exactly, DictWriter might complain or just write.
        # It's safest to define fieldnames based on the current row.
        # If order changes, DictWriter handles mapping keys to columns if we read existing headers.
        if file_exists:
            with open(csv_file, mode='r', newline='', encoding='utf-8') as read_f:
                reader = csv.reader(read_f)
                try:
                    existing_headers = next(reader)
                    # Add any new headers that weren't there before
                    for h in headers:
                        if h not in existing_headers:
                            existing_headers.append(h)
                    headers = existing_headers
                except StopIteration:
                    file_exists = False
                    
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        
        # Write the row
        writer.writerow(row)
        
    print(f"  [Logger] Saved results to outputs/metrics/{model_name}/{experiment_name}.csv")
