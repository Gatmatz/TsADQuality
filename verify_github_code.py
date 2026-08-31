import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

project_root = os.path.dirname(os.path.abspath(__file__))
tsb_ad_path = os.path.join(project_root, 'TSB-AD')
sys.path.insert(0, tsb_ad_path)

from TSB_AD.model_wrapper import run_Unsupervise_AD
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

file_list = pd.read_csv(os.path.join(tsb_ad_path, 'Datasets', 'File_List', 'TSB-AD-U-Eva.csv'))['file_name'].values
Optimal_Det_HP = Optimal_Uni_algo_HP_dict['IForest']
print('Optimal_Det_HP: ', Optimal_Det_HP)

vus_pr_list = []

for filename in tqdm(file_list):
    file_path = os.path.join(tsb_ad_path, 'Datasets', 'TSB-AD-U', filename)
    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    
    slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
    
    output = run_Unsupervise_AD('IForest', data, **Optimal_Det_HP)
    
    try:
        evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
        vus_pr = evaluation_result.get('VUS-PR')
        vus_pr_list.append(vus_pr)
    except:
        pass

print(f"Mean VUS-PR using EXACT Run_Detector_U.py loop: {np.mean(vus_pr_list)}")

# Compare to uni_mergedTable
df_merged = pd.read_csv(os.path.join(tsb_ad_path, 'benchmark_exp', 'benchmark_eval_results', 'uni_mergedTable_VUS-PR.csv'))
print(f"Mean VUS-PR in published table: {df_merged['IForest'].mean()}")
