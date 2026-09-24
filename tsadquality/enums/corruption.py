"""Corruption constants shared across the pipeline: where each corruption type is
applied, and the parameter field layout of `Experiment`'s experiment id
(`_get_experiment_id_parts`/`from_str`).
"""

from tsadquality.enums.data import CorruptionType
from tsadquality.enums.EasilyStringifyableEnum import EasilyStringifyableEnum


class CorruptionTarget(EasilyStringifyableEnum):
    """Which points a corruption may touch, relative to the labelled anomalies.
    Values are the `corruption_target` strings `BaseCorruptor` understands."""

    GLOBAL = "global"
    ONLY_NORMAL = "only_normal"
    OVERLAPPING_ANOMALY = "overlapping_anomaly"
    NEAR_ANOMALY = "near_anomaly"
    BEFORE_ANOMALY = "before_anomaly"
    AFTER_ANOMALY = "after_anomaly"
    FAR_FROM_ANOMALY = "far_from_anomaly"


# Every corruption type hits the whole series, except spikes: on anomalies they would
# push anomalous points further out and make them easier to detect, not harder.
CORRUPTION_TARGETS: dict[CorruptionType, CorruptionTarget] = {
    CorruptionType.POINT_MISSING: CorruptionTarget.GLOBAL,
    CorruptionType.BURST_MISSING: CorruptionTarget.GLOBAL,
    CorruptionType.GILBERT_ELLIOTT: CorruptionTarget.GLOBAL,
    CorruptionType.NOISE_SNR: CorruptionTarget.GLOBAL,
    CorruptionType.SPIKE: CorruptionTarget.ONLY_NORMAL,
    CorruptionType.STUCK: CorruptionTarget.GLOBAL,
    CorruptionType.POINT_SWAP: CorruptionTarget.GLOBAL,
    CorruptionType.SEGMENT_SWAP: CorruptionTarget.GLOBAL,
    CorruptionType.PERMUTATION_SWAP: CorruptionTarget.GLOBAL,
}

# Fixed positions for corruption params in the experiment id, matching the
# column order of the `experiments` table in postgres/init.sql. A field not
# used by a given corruption type is rendered as `Experiment._NULL`.
CORRUPTION_PARAM_FIELDS: tuple[str, ...] = (
    "snr_db",
    "fraction",
    "multiplier",
    "num_swaps",
    "num_permutations",
    "stuck_blocks",
    "num_bursts",
    "mechanism",
    "a",
    "b",
)

CORRUPTION_PARAM_TYPES: dict[str, type] = {
    "snr_db": float,
    "fraction": float,
    "multiplier": float,
    "num_swaps": int,
    "num_permutations": int,
    "stuck_blocks": int,
    "num_bursts": int,
    "mechanism": str,
    "a": float,
    "b": float,
}
