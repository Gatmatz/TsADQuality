import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import warnings
from itertools import product

from tsadquality.experiment import Experiment

warnings.filterwarnings("ignore")  # mitigates synthcity's annoying verbosity


from tsadquality.enums.data import CorruptionType, DataPerfectness
from tsadquality.reproducibility import ReproducibleOperations
from tsadquality.utils.general_utils import get_experimental_params
from tsadquality.logging.logging import get_logger


def _corruption_param_combinations(corruption_settings: dict, corruption_type: CorruptionType):
    """All kwarg combinations for `corruption_type`, from the cartesian product of its swept values."""
    params = corruption_settings.get(str(corruption_type), {})
    if not params:
        yield {}
        return

    names = list(params.keys())
    for combination in product(*(params[name] for name in names)):
        yield dict(zip(names, combination))


def _parse_shard_args() -> tuple[int, int]:
    """Parses --shard-index/--shard-count so multiple `run_experiment.py` processes
    can split the sweep between them without ever picking up the same experiment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based index of this worker among --shard-count parallel workers.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of parallel workers splitting the sweep.",
    )
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError(
            f"--shard-index ({args.shard_index}) must be in [0, --shard-count) "
            f"(--shard-count={args.shard_count})."
        )
    return args.shard_index, args.shard_count


def _in_shard(key_parts: list, shard_index: int, shard_count: int) -> bool:
    """Deterministically assigns the experiment identified by `key_parts` to exactly
    one of `shard_count` workers, independent of process/thread scheduling. Uses a
    stable hash (not the builtin `hash()`, which is randomized per-process for str)
    so the same experiment always lands in the same shard across all workers.
    """
    import hashlib
    import json

    if shard_count == 1:
        return True

    digest = hashlib.sha256(json.dumps(key_parts, sort_keys=True, default=str).encode()).digest()
    return int.from_bytes(digest[:8], "big") % shard_count == shard_index


LOG = get_logger(__file__)

SHARD_INDEX, SHARD_COUNT = _parse_shard_args()
if SHARD_COUNT > 1:
    LOG.info(f"Running as shard {SHARD_INDEX}/{SHARD_COUNT}.")

experimental_params = get_experimental_params()

# First, generate all perfect data
for random_seed in experimental_params.get("random_seeds"):
    ReproducibleOperations.set_random_seed(random_seed)
    ReproducibleOperations.seed_everything()
    for ts_name in experimental_params.get("ts_names"):
        for detector in experimental_params.get("detectors"):
            if not _in_shard(
                [random_seed, ts_name, str(DataPerfectness.PERFECT), None, None, str(detector)],
                SHARD_INDEX,
                SHARD_COUNT,
            ):
                continue

            experiment = None
            try:
                experiment = Experiment(
                    timeseries=ts_name,
                    detector=detector,
                    corruption_type=None,
                    data_perfectness=DataPerfectness.PERFECT,  # only perfect data at first
                )

                experiment.run()  # force-compute the regression datasets
            except Exception as e:
                LOG.error(
                    f"The experiment {experiment!s} failed but I will continue to the next one."
                    + f"Error: {e}.",
                    extra={"experiment_id": str(experiment)},
                )
                # exit(0)
                continue

# exit(0)

# Then, generate all corrupted experiments, sweeping every parameter combination
# that CORRUPTION_SETTINGS defines for each corruption type
corruption_settings = experimental_params.get("corruption_settings")
for random_seed in experimental_params.get("random_seeds"):
    ReproducibleOperations.set_random_seed(random_seed)
    ReproducibleOperations.seed_everything()
    for ts_name in experimental_params.get("ts_names"):
        for detector in experimental_params.get("detectors"):
            for corruption_type in experimental_params.get("corruption_types"):
                for corruption_params in _corruption_param_combinations(
                    corruption_settings, corruption_type
                ):
                    if not _in_shard(
                        [
                            random_seed,
                            ts_name,
                            str(DataPerfectness.IMPERFECT),
                            str(corruption_type),
                            corruption_params,
                            str(detector),
                        ],
                        SHARD_INDEX,
                        SHARD_COUNT,
                    ):
                        continue

                    experiment = None
                    try:
                        experiment = Experiment(
                            timeseries=ts_name,
                            detector=detector,
                            corruption_type=corruption_type,
                            corruption_params=corruption_params,
                            data_perfectness=DataPerfectness.IMPERFECT,
                        )
                        experiment.run()

                    except Exception as e:
                        LOG.error(
                            f"The experiment failed but I will continue to the next one. Error: {e}",
                            extra={"experiment_id": str(experiment)},
                        )
                        continue