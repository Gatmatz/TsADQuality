"""Read-only Postgres accessors for the `plots` package.

Every function here goes through `PostgresClient.query()`, which only accepts
`SELECT`/`WITH` statements (see `PostgresClient.query()`), so nothing in this
module can write to the database.
"""

from typing import TYPE_CHECKING, Any

from tsadquality.enums.data import CorruptionType, DataPerfectness
from tsadquality.enums.detectors import DetectorModel

if TYPE_CHECKING:
    import pandas as pd

_EXPERIMENTS_TABLE = "experiments"
_EVALUATIONS_TABLE = "evaluations"


def _escape(value: str) -> str:
    return value.replace("'", "''")


def _build_where(
    *,
    dataset_name: str | None,
    detector: DetectorModel | None,
    corruption_type: CorruptionType | None,
    data_perfectness: DataPerfectness | None,
    table_alias: str | None = None,
) -> str:
    prefix = f"{table_alias}." if table_alias else ""
    conditions: list[str] = []

    if dataset_name is not None:
        conditions.append(f"{prefix}dataset_name = '{_escape(dataset_name)}'")
    if detector is not None:
        conditions.append(f"{prefix}detector = '{_escape(str(detector))}'")
    if corruption_type is not None:
        conditions.append(f"{prefix}corruption_type = '{_escape(str(corruption_type))}'")
    if data_perfectness is not None:
        conditions.append(f"{prefix}data_perfectness = '{_escape(str(data_perfectness))}'")

    if not conditions:
        return ""
    return "WHERE " + " AND ".join(conditions)


def fetch_experiments(
    dataset_name: str | None = None,
    detector: DetectorModel | None = None,
    corruption_type: CorruptionType | None = None,
    data_perfectness: DataPerfectness | None = None,
) -> "pd.DataFrame":
    """Fetches rows from the `experiments` table, optionally filtered.

    Passing `None` (the default) for any filter skips it, so calling this
    with no arguments returns every row in `experiments`.
    """
    from tsadquality.data.clients.PostgresClient import PostgresClient

    where_clause = _build_where(
        dataset_name=dataset_name,
        detector=detector,
        corruption_type=corruption_type,
        data_perfectness=data_perfectness,
    )
    return PostgresClient.query(f"SELECT * FROM {_EXPERIMENTS_TABLE} {where_clause}")


def fetch_evaluations(
    dataset_name: str | None = None,
    detector: DetectorModel | None = None,
    corruption_type: CorruptionType | None = None,
    data_perfectness: DataPerfectness | None = None,
    with_corruption_params: bool = False,
) -> "pd.DataFrame":
    """Fetches rows from the `evaluations` table, optionally filtered.

    Passing `None` (the default) for any filter skips it, so calling this
    with no arguments returns every row in `evaluations`. When
    `with_corruption_params` is set, the result is joined against
    `experiments` on `experiment_id` to also bring in the corruption
    parameter columns (`snr_db`, `fraction`, `multiplier`, ...), which
    `evaluations` doesn't carry on its own.
    """
    from tsadquality.data.clients.PostgresClient import PostgresClient

    if not with_corruption_params:
        where_clause = _build_where(
            dataset_name=dataset_name,
            detector=detector,
            corruption_type=corruption_type,
            data_perfectness=data_perfectness,
        )
        return PostgresClient.query(f"SELECT * FROM {_EVALUATIONS_TABLE} {where_clause}")

    where_clause = _build_where(
        dataset_name=dataset_name,
        detector=detector,
        corruption_type=corruption_type,
        data_perfectness=data_perfectness,
        table_alias="e",
    )
    experiment_columns = ", ".join(
        f"x.{column} AS x_{column}"
        for column in (
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
    )
    return PostgresClient.query(f"""
        SELECT e.*, {experiment_columns}
        FROM {_EVALUATIONS_TABLE} e
        JOIN {_EXPERIMENTS_TABLE} x ON x.experiment_id = e.experiment_id
        {where_clause}
    """)
