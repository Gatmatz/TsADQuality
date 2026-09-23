"""Corruption parameter field layout shared by `Experiment`'s experiment-id
(de)serialization (`_get_experiment_id_parts`/`from_str`).
"""

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
