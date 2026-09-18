from pathlib import Path
from typing import Any, Self

from tsadquality.enums.data import CorruptionType, DataPerfectness
from tsadquality.enums.detectors import DetectorModel
from tsadquality.enums.evaluators import EvaluationMethod
from tsadquality.logging.logging import get_logger

LOG = get_logger(__file__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Experiment:
    _delimiter: str = "#"
    _NULL: str = "NULL"
    _value_col: str = "Data"
    _label_col: str = "Label"

    # Fixed positions for corruption params in the experiment id, matching the
    # column order of the `experiments` table in postgres/init.sql. A field not
    # used by a given corruption type is rendered as `_NULL`.
    _corruption_param_fields: tuple[str, ...] = (
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
    _corruption_param_types: dict[str, type] = {
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

    def __init__(
        self,
        timeseries: str,
        detector: DetectorModel,
        corruption_type: CorruptionType | None = None,
        corruption_params: dict[str, Any] | None = None,
        data_perfectness: DataPerfectness = DataPerfectness.PERFECT,
        evaluation_methods: list[EvaluationMethod] | None = None,
        options: dict[Any, Any] | None = None,
    ):
        self.timeseries = timeseries
        self.detector = detector
        self.corruption_type = corruption_type
        self.corruption_params = corruption_params
        self.data_perfectness = data_perfectness
        self.evaluation_methods = evaluation_methods
        self.options = options

        self._should_compute = not self._exists_in_postgres()

        self.timeseries_df = None
        self.decision_scores_ = None

    @classmethod
    def short_name(cls):
        return "EXP"

    def _run(self) -> None:
        """
        Fit `self.detector` on `self.timeseries` (corrupted first, if `self.data_error`
        is set) and write the experiment's corruption parameters to Postgres.
        """
        from time import perf_counter

        import pandas as pd

        from tsadquality.data.clients.PostgresClient import PostgresClient
        from tsadquality.reproducibility import ReproducibleOperations

        LOG.info(f"Entering the _run() function of experiment {self!s}")

        # Read the input timeseries
        df = pd.read_csv(self._timeseries_path())

        # Corrupt the timeseries if a corruption type is specified, and record the corruption parameters
        corruption_params = None
        if self.corruption_type:
            corruptor = self._apply_corruption(df)
            df = corruptor.get_corrupted_df()
            corruption_params = self.corruption_params

        self.timeseries_df = df

        values = df[self._value_col].to_numpy(dtype=float)

        window = self._window_length()
        values = self._z_score(values)

        detector_instance = ReproducibleOperations.get_detector(
            self.detector, window=window, **(self.options or {})
        )

        start = perf_counter()
        detector_instance.fit(values)
        execution_time = perf_counter() - start
        self.decision_scores_ = detector_instance.decision_scores_

        LOG.info(
            f"Fitted {self.detector!s} on {self.timeseries} in {execution_time:.3f}s for experiment {self!s}"
        )

        PostgresClient.write_experiment(
            experiment_id=str(self),
            dataset_name=self.timeseries,
            random_seed=str(ReproducibleOperations.get_current_random_seed()),
            data_perfectness=str(self.data_perfectness),
            corruption_type=str(self.corruption_type) if self.corruption_type else None,
            detector=str(self.detector),
            execution_time=execution_time,
            corruption_params=corruption_params,
        )
        LOG.info(f"Successfully wrote the metadata of experiment {self!s} to Postgres.")

        # Evaluate the detector's decision scores against the ground-truth labels
        # with every requested evaluation method, and record each score.
        y_true = df[self._label_col].to_numpy()
        for evaluation_method in self.evaluation_methods or []:
            eval_start = perf_counter()
            score = evaluation_method.get_class().evaluate(
                y_true=y_true, decision_scores=self.decision_scores_
            )
            eval_execution_time = perf_counter() - eval_start

            PostgresClient.write_evaluation(
                experiment_id=str(self),
                dataset_name=self.timeseries,
                random_seed=str(ReproducibleOperations.get_current_random_seed()),
                data_perfectness=str(self.data_perfectness),
                corruption_type=str(self.corruption_type) if self.corruption_type else None,
                detector=str(self.detector),
                metric=str(evaluation_method),
                score=score,
                execution_time=eval_execution_time,
            )
            LOG.info(
                f"Wrote evaluation '{evaluation_method!s}'={score} for experiment {self!s} to Postgres."
            )

    def _timeseries_path(self) -> Path:
        return _PROJECT_ROOT / "data" / "TSB-AD-U" / f"{self.timeseries}.csv"

    def _window_length(self) -> int:
        from tsadquality.data.clients.PostgresClient import PostgresClient

        escaped_timeseries = self.timeseries.replace("'", "''")
        result = PostgresClient.query(
            f"SELECT window_length FROM window_lengths WHERE dataset = '{escaped_timeseries}'"
        )
        if result.empty:
            raise ValueError(
                f"No precomputed window length found for timeseries '{self.timeseries}'. "
                "Run `python -m tsadquality.utils.window_table` first."
            )
        return int(result.iloc[0]["window_length"])

    @staticmethod
    def _z_score(values):
        import numpy as np

        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std == 0 or np.isnan(std):
            std = 1.0
        return (values - mean) / std

    def _apply_corruption(self, df):
        corruptor = self.corruption_type.get_class()(
            df,
            value_col=self._value_col,
            label_col=self._label_col,
        )

        corruptor.inject(**(self.corruption_params or {}))
        return corruptor

    # IMPORTANT: Keep this method aligned with the from_str() method!
    def _get_experiment_id_parts(self):
        from tsadquality.reproducibility.ReproducibleOperations import (
            ReproducibleOperations,
        )

        corruption_params = self.corruption_params or {}

        return [
            str(
                self.short_name()
            ),  # Experiment shortname, e.g., 'EXP'
            str(
                self.timeseries
            ),  # Timeseries name, e.g., '001_NAB_id_1_Facility_tr_1007_1st_2014'
            str(ReproducibleOperations.get_current_random_seed()),  # Random seed
            str(
                self.data_perfectness
            ),  # Data perfectness level, e.g., 'PERF' for perfect
            str(self.corruption_type)
            if self.corruption_type
            else self._NULL,  # Corruption type, e.g., 'SPK' for spikes
            *(
                str(corruption_params[field])
                if field in corruption_params
                else self._NULL
                for field in self._corruption_param_fields
            ),  # One fixed-position slot per corruption param column, e.g. 'NULL' when not applicable
            str(self.detector),  # Detector type, e.g., 'IsolationForest'
        ]

    # IMPORTANT: Keep this method aligned with the _get_experiment_id_parts() method!
    @classmethod
    def from_str(cls, experiment_id: str) -> tuple[Self, int]:
        experiment_id_parts = experiment_id.split(cls._delimiter)
        timeseries = experiment_id_parts[1]
        random_seed = int(experiment_id_parts[2])
        data_perfectness = DataPerfectness(experiment_id_parts[3])
        corruption_type = (
            None
            if experiment_id_parts[4] == cls._NULL
            else CorruptionType(experiment_id_parts[4])
        )

        corruption_param_parts = experiment_id_parts[
            5 : 5 + len(cls._corruption_param_fields)
        ]
        corruption_params = {
            field: cls._corruption_param_types[field](value)
            for field, value in zip(
                cls._corruption_param_fields, corruption_param_parts
            )
            if value != cls._NULL
        } or None

        detector = DetectorModel(experiment_id_parts[5 + len(cls._corruption_param_fields)])

        return cls(
            timeseries=timeseries,
            detector=detector,
            corruption_type=corruption_type,
            corruption_params=corruption_params,
            data_perfectness=data_perfectness,
        ), random_seed

    def __str__(self):
        experiment_id_parts = self._get_experiment_id_parts()
        return self._delimiter.join(experiment_id_parts)

    def perfect_counterpart(self) -> Self:
        from copy import deepcopy

        perfect_experiment = deepcopy(self)
        perfect_experiment.data_perfectness = DataPerfectness.PERFECT
        perfect_experiment.corruption_type = None
        perfect_experiment.corruption_params = None
        return perfect_experiment

    def run(self, force: bool = False) -> Self:
        if not self._should_compute and not force:
            from tsadquality.data.clients.PostgresClient import PostgresClient

            LOG.info(
                f"Running experiment {self!s} will be skipped because it already exists in Postgres."
            )
            PostgresClient.write_skipped_computation(
                computation_id=str(self), reason="Already exists in Postgres."
            )
            return self

        self._run()
        return self

    def _exists_in_postgres(self) -> bool:
        from tsadquality.data.clients.PostgresClient import PostgresClient

        # skip experiments that have already been executed and evaluated before
        if PostgresClient.experiment_exists(str(self)) and PostgresClient.experiment_exists(
            str(self),
            experiments_table_name="evaluations",
        ):
            return True

        return False
