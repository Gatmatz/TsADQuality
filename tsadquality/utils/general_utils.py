from collections.abc import Callable
from typing import Any

def timed_computation(
    computation: Callable, params: dict[str, Any], precision: int = 2
) -> tuple[Any, float]:
    from timeit import default_timer as timer

    start = timer()
    result = computation(**params)
    end = timer()
    return result, round(end - start, precision)  # tuple: result, execution_time

def get_experimental_params() -> dict[str, Any]:
    import copy
    import random
    from pathlib import Path
    from pprint import pp

    from tsadquality.enums.data import CorruptionType, DataPerfectness
    from tsadquality.enums.detectors import DETECTORS
    from tsadquality.enums.evaluators import EvaluationMethod
    from tsadquality.environment import RANDOM_SEEDS
    from tsadquality.environment.corruption import CORRUPTION_SETTINGS

    project_root = Path(__file__).resolve().parents[2]
    ts_names = [
        csv_path.stem for csv_path in sorted((project_root / "data" / "TSB-AD-U").glob("*.csv"))
    ]
    # ts_names = [name for name in ts_names if name == "001_NAB_id_1_Facility_tr_1007_1st_2014"]

    random_seeds = copy.deepcopy(RANDOM_SEEDS)

    # Shuffle the ordering of the sweep deterministically, using a private RNG seeded
    # from the (unshuffled) configured seeds. This must not touch the global `random`
    # state, since experiments seed reproducibility through `ReproducibleOperations`
    # instead, and must not depend on `random_seeds` after it's shuffled below, or the
    # ordering itself would stop being reproducible run-to-run.
    shuffle_rng = random.Random(random_seeds[0] if random_seeds else 0)

    shuffle_rng.shuffle(random_seeds)
    pp(f"{random_seeds=}")

    shuffle_rng.shuffle(ts_names)
    pp(f"{ts_names=}", compact=True)
    print()

    detectors = copy.deepcopy(DETECTORS)
    shuffle_rng.shuffle(detectors)
    pp(f"{detectors=}", compact=True)
    print()

    corruption_types = list(CorruptionType)
    shuffle_rng.shuffle(corruption_types)
    pp(f"{corruption_types=}", compact=True)
    print()

    data_perfectness_levels = [DataPerfectness.PERFECT, DataPerfectness.IMPERFECT]
    shuffle_rng.shuffle(data_perfectness_levels)
    pp(f"{data_perfectness_levels=}", compact=True)
    print()

    evaluation_methods = list(EvaluationMethod)
    shuffle_rng.shuffle(evaluation_methods)
    pp(f"{evaluation_methods=}", compact=True)
    print()

    corruption_settings = {
        str(corruption_type): dict(params)
        for corruption_type, params in CORRUPTION_SETTINGS.items()
    }
    pp(f"{corruption_settings=}", compact=True)
    print()

    return {
        "random_seeds": random_seeds,
        "ts_names": ts_names,
        "detectors": detectors,
        "corruption_types": corruption_types,
        "data_perfectness_levels": data_perfectness_levels,
        "evaluation_methods": evaluation_methods,
        "corruption_settings": corruption_settings,
    }
