"""Materializes corrupted copies of every dataset to disk, one compressed `.npz` per
(corruption_type, parameter combination, dataset), for every non-missing
corruption type and every parameter combination swept in `CORRUPTION_SETTINGS`.

Each `.npz` holds two arrays: `Data` (float32) and `Label` (int8), matching the
source CSVs' two columns. Load one back with e.g.:
    with np.load(path) as f:
        data, label = f["Data"], f["Label"]

Output layout:
    <project_root>/corrupted/<corruption_type>/<param_dirname>/<dataset_name>.npz

This is a one-off materialization step (e.g. for inspecting or sharing corrupted
data on disk); it's independent of the actual experiment sweep
(`tsadquality/utils/run_experiment.py`), which corrupts data in-memory on the fly
and never writes corrupted CSVs anywhere.

Missing-value corruption types (POINT_MISSING, BURST_MISSING, GILBERT_ELLIOTT) are
excluded for now, matching `tsadquality.utils.general_utils.get_experimental_params`.

Run with:
    uv run python -m tsadquality.corruption.save_corrupted_datasets
"""

from itertools import product
from pathlib import Path

from tsadquality.enums.data import CorruptionType
from tsadquality.reproducibility import ReproducibleOperations

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data" / "TSB-AD-U"
_EVA_LIST_PATH = _PROJECT_ROOT / "tsadquality" / "data" / "TSB-AD-U-Eva.csv"
_OUTPUT_DIR = _PROJECT_ROOT / "corrupted"

_VALUE_COL = "Data"
_LABEL_COL = "Label"

# Fixed so the datasets saved by this script are reproducible on their own,
# independent of the `RANDOM_SEEDS` used by the actual experiment sweep.
_RANDOM_SEED = 42

_EXCLUDED_CORRUPTION_TYPES = {
    CorruptionType.POINT_MISSING,
    CorruptionType.BURST_MISSING,
    CorruptionType.GILBERT_ELLIOTT,
}


def _dataset_names() -> list[str]:
    """Every dataset name in `data/TSB-AD-U` that's also listed in TSB-AD-U-Eva.csv."""
    with _EVA_LIST_PATH.open() as f:
        eva_names = {Path(line.strip()).stem for line in f.readlines()[1:] if line.strip()}
    return sorted(
        csv_path.stem for csv_path in _DATA_DIR.glob("*.csv") if csv_path.stem in eva_names
    )


def _param_combinations(corruption_type: CorruptionType) -> list[dict]:
    """All kwarg combinations for `corruption_type`, from the cartesian product of
    its swept values in `CORRUPTION_SETTINGS`. Mirrors
    `run_experiment._corruption_param_combinations`.
    """
    from tsadquality.environment.corruption import CORRUPTION_SETTINGS

    params = CORRUPTION_SETTINGS.get(corruption_type, {})
    if not params:
        return [{}]

    names = list(params.keys())
    return [dict(zip(names, combo)) for combo in product(*(params[name] for name in names))]


def _param_dirname(params: dict) -> str:
    if not params:
        return "default"
    return "_".join(f"{name}={value}" for name, value in sorted(params.items()))


def main() -> None:
    import numpy as np
    import pandas as pd

    ReproducibleOperations.set_random_seed(_RANDOM_SEED)

    dataset_names = _dataset_names()
    corruption_types = [c for c in CorruptionType if c not in _EXCLUDED_CORRUPTION_TYPES]

    for corruption_type in corruption_types:
        for params in _param_combinations(corruption_type):
            output_dir = _OUTPUT_DIR / str(corruption_type) / _param_dirname(params)
            output_dir.mkdir(parents=True, exist_ok=True)

            for dataset_name in dataset_names:
                df = pd.read_csv(_DATA_DIR / f"{dataset_name}.csv")

                corruptor = corruption_type.get_class()(
                    df, value_col=_VALUE_COL, label_col=_LABEL_COL
                )
                corruptor.inject(**params)
                corrupted_df = corruptor.get_corrupted_df()

                np.savez_compressed(
                    output_dir / f"{dataset_name}.npz",
                    Data=corrupted_df[_VALUE_COL].to_numpy(dtype="float32"),
                    Label=corrupted_df[_LABEL_COL].to_numpy(dtype="int8"),
                )

            print(
                f"Saved {len(dataset_names)} datasets for "
                f"{corruption_type}/{_param_dirname(params)}"
            )


if __name__ == "__main__":
    main()
