"""Deletes rows from the `experiments` and `evaluations` tables for datasets
that are not listed in tsadquality/data/TSB-AD-U-Eva.csv.

This script only PRINTS what it would delete by default (dry run). Pass
--execute to actually perform the deletion.

NOTE: This script is not meant to be run automatically -- review the dry-run
output first.
"""

import argparse
from pathlib import Path

from sqlalchemy import create_engine, text

from tsadquality.environment.postgres import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_MAPPED_PORT,
    POSTGRES_PASSWORD,
    POSTGRES_USER,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EVA_LIST_PATH = _PROJECT_ROOT / "tsadquality" / "data" / "TSB-AD-U-Eva.csv"


def get_allowed_dataset_names() -> set[str]:
    with _EVA_LIST_PATH.open() as f:
        return {Path(line.strip()).stem for line in f.readlines()[1:] if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete the rows. Without this flag, only a dry-run count is printed.",
    )
    args = parser.parse_args()

    allowed_dataset_names = get_allowed_dataset_names()
    print(f"Datasets kept ({len(allowed_dataset_names)}): {sorted(allowed_dataset_names)}")

    engine = create_engine(
        url=f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_MAPPED_PORT}/{POSTGRES_DB}",
        echo=False,
    )

    with engine.connect() as connection:
        for table_name in ("experiments", "evaluations"):
            select_query = text(
                f"""
                SELECT dataset_name, COUNT(*) AS row_count
                FROM {table_name}
                WHERE dataset_name IS NULL OR dataset_name NOT IN :allowed_dataset_names
                GROUP BY dataset_name
                ORDER BY dataset_name
                """
            ).bindparams(allowed_dataset_names=tuple(allowed_dataset_names))
            rows = connection.execute(select_query).fetchall()

            total_rows = sum(row.row_count for row in rows)
            print(f"\n[{table_name}] {total_rows} rows across {len(rows)} dataset(s) to delete:")
            for row in rows:
                print(f"  {row.dataset_name!r}: {row.row_count} rows")

            if args.execute:
                delete_query = text(
                    f"""
                    DELETE FROM {table_name}
                    WHERE dataset_name IS NULL OR dataset_name NOT IN :allowed_dataset_names
                    """
                ).bindparams(allowed_dataset_names=tuple(allowed_dataset_names))
                result = connection.execute(delete_query)
                connection.commit()
                print(f"[{table_name}] Deleted {result.rowcount} rows.")
            else:
                print(f"[{table_name}] Dry run only, nothing deleted. Pass --execute to delete.")


if __name__ == "__main__":
    main()
