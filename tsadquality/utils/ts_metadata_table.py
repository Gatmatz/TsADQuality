#!/usr/bin/env python3
"""Compute each timeseries' window length and periodicity and upload them to postgres."""

from pathlib import Path

import pandas as pd
from TSB_AD.utils.slidingWindows import find_length, find_length_rank

from tsadquality.data.clients.PostgresClient import PostgresClient
from tsadquality.logging.logging import get_logger

LOG = get_logger(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "TSB-AD-U"

TS_METADATA_TABLE = "ts_metadata"


def compute_ts_metadata(csv_path: Path) -> tuple[int, int]:
    data = pd.read_csv(csv_path)["Data"].to_numpy(dtype=float)
    return int(find_length(data)), int(find_length_rank(data, rank=1))


def main():
    PostgresClient.create_table(
        table_name=TS_METADATA_TABLE,
        schema=(
            "dataset TEXT PRIMARY KEY, "
            "window_length INTEGER NOT NULL, "
            "periodicity INTEGER NOT NULL"
        ),
    )

    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        dataset = csv_path.stem
        window_length, periodicity = compute_ts_metadata(csv_path)
        PostgresClient.execute_upsert_query(
            table_name=TS_METADATA_TABLE,
            query_params={
                "dataset": dataset,
                "window_length": window_length,
                "periodicity": periodicity,
            },
            conflict_columns=["dataset"],
        )
        LOG.info(f"{dataset}: window_length={window_length}, periodicity={periodicity}")


if __name__ == "__main__":
    main()
