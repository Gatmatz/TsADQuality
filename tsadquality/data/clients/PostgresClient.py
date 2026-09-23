from typing import Any

import pandas as pd
from sqlalchemy import create_engine

from tsadquality.environment.postgres import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_MAPPED_PORT,
    POSTGRES_PASSWORD,
    POSTGRES_USER,
)
from tsadquality.logging.logging import get_logger

LOG = get_logger(__file__)


class SingletonPostgresClient(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class _PostgresClient:
    _engine = create_engine(
        url=f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_MAPPED_PORT}/{POSTGRES_DB}",
        echo=False,
        pool_pre_ping=True,
    )


class PostgresClient(_PostgresClient, metaclass=SingletonPostgresClient):
    @classmethod
    def create_table(cls, table_name: str, schema: str) -> None:
        from sqlalchemy import text

        query = text(f"""CREATE TABLE IF NOT EXISTS {table_name} ({schema})""")
        with cls._engine.connect() as connection:
            connection.execute(query)
            connection.commit()

    @classmethod
    def execute_insert_query(
        cls,
        table_name: str,
        query_params: dict[str, Any],
    ):
        from sqlalchemy import text

        from tsadquality.environment import EXECUTION_PROFILE

        query_params["execution_profile"] = EXECUTION_PROFILE
        field_names_list = list(query_params.keys())
        value_indicators_list = [":" + field_name for field_name in field_names_list]

        field_names = ", ".join(field_names_list)
        value_indicators = ", ".join(value_indicators_list)

        query = text(
            f"""INSERT INTO {table_name} ({field_names}) VALUES ({value_indicators})"""
        )
        with cls._engine.connect() as connection:
            connection.execute(query, query_params)
            connection.commit()

    @classmethod
    def execute_upsert_query(
        cls,
        table_name: str,
        query_params: dict[str, Any],
        conflict_columns: list[str],
    ):
        from sqlalchemy import text

        field_names_list = list(query_params.keys())
        value_indicators_list = [":" + field_name for field_name in field_names_list]
        update_columns = [
            field_name
            for field_name in field_names_list
            if field_name not in conflict_columns
        ]

        field_names = ", ".join(field_names_list)
        value_indicators = ", ".join(value_indicators_list)
        conflict_target = ", ".join(conflict_columns)
        update_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_columns)

        query = text(
            f"""INSERT INTO {table_name} ({field_names}) VALUES ({value_indicators})
            ON CONFLICT ({conflict_target}) DO UPDATE SET {update_clause}"""
        )
        with cls._engine.connect() as connection:
            connection.execute(query, query_params)
            connection.commit()

    @classmethod
    def write_runtime_error(
        cls,
        experiment_id: str,
        file_path: str,
        error_message: str,
        errors_table_name: str = "errors",
    ):
        try:
            query_params = {
                "experiment_id": experiment_id,
                "file_path": file_path,
                "error_message": error_message,
            }
            cls.execute_insert_query(
                table_name=errors_table_name, query_params=query_params
            )
            LOG.info(
                f"Wrote runtime error for experiment {experiment_id} in '{errors_table_name}'"
            )
        except Exception as e:
            LOG.error(
                f"Failed to write runtime error for experiment {experiment_id}. Error: {e}"
            )
            raise

    @classmethod
    def write_experiment(
        cls,
        experiment_id: str,
        dataset_name: str,
        random_seed: str,
        data_perfectness: str,
        corruption_type: str | None,
        detector: str,
        execution_time: float,
        corruption_params: dict[str, Any] | None = None,
        experiment_results_table_name: str = "experiments",
    ):
        try:
            query_params = {
                "experiment_id": experiment_id,
                "dataset_name": dataset_name,
                "random_seed": random_seed,
                "data_perfectness": data_perfectness,
                "corruption_type": corruption_type,
                "detector": detector,
                "execution_time": execution_time,
            }

            if corruption_params:
                query_params.update(corruption_params)

            cls.execute_insert_query(
                table_name=experiment_results_table_name, query_params=query_params
            )
            LOG.info(
                f"Wrote experiment {experiment_id} in '{experiment_results_table_name}'"
            )
        except Exception as e:
            LOG.error(f"Failed to write experiment {experiment_id}. Error: {e}")
            raise

    @classmethod
    def write_evaluation(
        cls,
        experiment_id: str,
        dataset_name: str,
        random_seed: str,
        data_perfectness: str,
        corruption_type: str | None,
        detector: str,
        metrics: dict[str, float],
        execution_time: float,
        evaluations_table_name: str = "evaluations",
    ):
        import math

        # NaN (e.g. an internal metric on a series with no anomaly windows) is stored
        # as NULL, not Postgres' NaN, so aggregates like AVG() skip it.
        metrics = {
            name: None if value is None or math.isnan(value) else float(value)
            for name, value in metrics.items()
        }
        try:
            query_params = {
                "experiment_id": experiment_id,
                "dataset_name": dataset_name,
                "random_seed": random_seed,
                "data_perfectness": data_perfectness,
                "corruption_type": corruption_type,
                "detector": detector,
                "execution_time": execution_time,
                **metrics,
            }

            cls.execute_insert_query(
                table_name=evaluations_table_name, query_params=query_params
            )
            LOG.info(
                f"Wrote evaluation {metrics} for experiment {experiment_id} in '{evaluations_table_name}'"
            )
        except Exception as e:
            LOG.error(
                f"Failed to write evaluation for experiment {experiment_id}. Error: {e}"
            )
            raise

    @classmethod
    def experiment_exists(
        cls,
        experiment_id: str,
        experiments_table_name: str = "experiments",
        experiment_id_column_name: str = "experiment_id",
    ) -> bool:
        """Checks if an experiment with the specific experiment id exists.

        Args:
            experiment_id (str): The experiment id to check for existence.

        Returns:
            bool: True if it exists, else False.
        """
        from sqlalchemy import text

        try:
            query = text(f"""
                SELECT 1 FROM {experiments_table_name} \
                WHERE {experiment_id_column_name} = :experiment_id \
                LIMIT 1 
            """)
            with cls._engine.connect() as connection:
                result = connection.execute(query, {"experiment_id": experiment_id})
                exists = result.scalar() is not None
                LOG.info(f"Checked existence of evaluation {experiment_id}: {exists}")
                return exists
        except Exception as e:
            LOG.error(
                f"Failed to check existence of experiment {experiment_id}. Error: {e}"
            )
            raise

    @classmethod
    def query(cls, sql: str) -> pd.DataFrame:
        """Executes a raw SQL query and returns the result as a pandas DataFrame.

        Args:
            sql: The SQL query string to execute.

        Returns:
            pd.DataFrame: The query result.
        """
        normalized_sql = sql.strip().lower()
        if not (
            normalized_sql.startswith("select") or normalized_sql.startswith("with")
        ):
            raise ValueError("Only SELECT queries are allowed for the query method.")
        from sqlalchemy import text

        with cls._engine.connect() as connection:
            return pd.read_sql_query(text(sql), connection)