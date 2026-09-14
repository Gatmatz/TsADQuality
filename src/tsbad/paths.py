"""Filesystem layout shared by every TSB-AD corruption runner."""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TSB_AD_PATH = PROJECT_ROOT / 'TSB-AD'
SRC_PATH = PROJECT_ROOT / 'src'

# Same entries, same final order as the per-runner bootstrap this replaces:
# src, TSB-AD, project root at the front of sys.path.
for _p in (PROJECT_ROOT, TSB_AD_PATH, SRC_PATH):
    _s = str(_p)
    while _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

# Full 350-series official eval list (the leaderboard set)
FILE_LIST_CSV = PROJECT_ROOT / "TSB-AD" / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
DATA_DIR = PROJECT_ROOT / "TSB-AD" / "Datasets" / "TSB-AD-U"
RESULTS_ROOT = PROJECT_ROOT / "results" / "experiments"


def results_dir(name):
    return RESULTS_ROOT / name


def resolve(path):
    """Paths given on the command line are relative to the project root, not the CWD."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_file_list(path):
    """Accept either the official TSB-AD eval list ('file_name') or a subset table ('file')."""
    df = pd.read_csv(path)
    for col in ('file_name', 'file', 'filename'):
        if col in df.columns:
            return df[col].astype(str).tolist()
    raise ValueError(f"{path}: no 'file_name' or 'file' column")
