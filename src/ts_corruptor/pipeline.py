import os
import json
from pathlib import Path
import pandas as pd
from .core import TSCorruptor
from . import injectors

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

class CorruptionPipeline:
    """
    Executes a sequence of corruption steps across a large dataset array.
    """
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = _PROJECT_ROOT / "results" / "corrupted_data"
        self.output_dir = output_dir
        self.log = []

    def load_dataset(self, csv_path):
        """Loads a single TSB-UAD dataset file."""
        # Typically no header if it's .out / .txt 
        df = pd.read_csv(csv_path, header=None, names=['value', 'is_anomaly'])
        return df

    def run_experiment(self, config, files_df):
        """
        Runs the specified corruption pipeline across all files in a metadata dataframe.
        
        config: dict
            A dictionary outlining the pipeline steps and targets.
            Example:
            {
                "experiment_name": "missing_5pct_near",
                "target": "near_anomaly",
                "steps": [
                    {"type": "missing", "fraction": 0.05}
                ]
            }
            
        files_df: pd.DataFrame
            The `robust_subset_TSB.csv` frame containing `filepath`, `baseline_name`, etc.
        """
        target_dir = os.path.join(self.output_dir, config['experiment_name'])
        os.makedirs(target_dir, exist_ok=True)
        print(f"Starting experiment: {config['experiment_name']}...")
        print(f"Saving to: {target_dir}")
        print(f"Target logic: {config.get('target', 'global')}\n")
        
        stats_log = []

        for index, row in files_df.iterrows():
            filepath = row['filepath']
            filename = row['baseline_name']
            folder = row['folder']
            
            try:
                # 1. Load Data
                df_raw = self.load_dataset(filepath)
                
                # 2. Init Corruptor
                corruptor = TSCorruptor(df_raw, corruption_target=config.get('target', 'global'), window_size=50, seed=42)
                
                # 3. Apply Steps sequentially
                for step in config['steps']:
                    stype = step.get('type')
                    if not stype or stype not in injectors.INJECTORS:
                        print(f"Warning: Corruptor type '{stype}' not found in registry. Skipping.")
                        continue
                        
                    # Extract params excluding 'type'
                    params = {k: v for k, v in step.items() if k != 'type'}
                    
                    # Call the registered function
                    inject_func = injectors.INJECTORS[stype]
                    inject_func(corruptor, **params)
                        
                # 4. Save Corrupted DataFrame to disk
                final_df = corruptor.get_corrupted_df()
                
                # Using folder_filename format to prevent collision
                out_name = f"{folder}_{filename}"
                out_path = os.path.join(target_dir, out_name)
                final_df.to_csv(out_path, index=False, header=False)
                
                # 5. Record Tracking stats
                report = corruptor.get_corruption_report()
                
                stats_log.append({
                    'original_file': filepath,
                    'output_file': out_name,
                    'folder': folder,
                    'baseline_name': filename,
                    'summary': report['summary']
                })
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
        # Export Experiment JSON Logs
        log_path = os.path.join(target_dir, "experiment_log.json")
        with open(log_path, 'w') as f:
            json.dump({
                "config": config,
                "dataset_stats": stats_log
            }, f, indent=4)
            
        print(f"Experiment completed. Processed {len(stats_log)} files.")
        return stats_log
