from pathlib import Path
from typing import Any, Self

from tsadquality.enums.corruption import (
    CORRUPTION_PARAM_FIELDS,
    CORRUPTION_PARAM_TYPES,
    CORRUPTION_TARGETS,
)
from tsadquality.enums.data import CorruptionType, DataPerfectness
from tsadquality.enums.detectors import DetectorModel
from tsadquality.enums.metrics import METRIC_COLUMNS
from tsadquality.logging.logging import get_logger

LOG = get_logger(__file__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Experiment:
    _delimiter: str = "#"
    _NULL: str = "NULL"
    _value_col: str = "Data"
    _label_col: str = "Label"

    def __init__(
        self,
        timeseries: str,
        detector: DetectorModel,
        corruption_type: CorruptionType | None = None,
        corruption_params: dict[str, Any] | None = None,
        data_perfectness: DataPerfectness = DataPerfectness.PERFECT,
        options: dict[Any, Any] | None = None,
    ):
        self.timeseries = timeseries
        self.detector = detector
        self.corruption_type = corruption_type
        self.corruption_params = corruption_params
        self.data_perfectness = data_perfectness
        self.options = options

        self._should_compute = not self._exists_in_postgres()

        self.timeseries_df = None
        self.decision_scores_ = None
        self.model_ = None

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
        periodicity = self._periodicity()
        # values = self._z_score(values) # This is done internally by TSB-AD's detectors, so we don't need to do it here.

        start = perf_counter()
        self.decision_scores_, self.model_ = ReproducibleOperations.run_detector(
            self.detector, values, window=window, periodicity=periodicity, **(self.options or {})
        )
        execution_time = perf_counter() - start

        LOG.info(
            f"Fitted {self.detector!s} on {self.timeseries} in {execution_time:.3f}s for experiment {self!s}"
        )

        if self.data_perfectness == DataPerfectness.PERFECT:
            from tsadquality.data.clients.MinioClient import MinioClient

            MinioClient.write_decision_scores(
                detector=str(self.detector),
                random_seed=str(ReproducibleOperations.get_current_random_seed()),
                dataset_name=self.timeseries,
                scores=self.decision_scores_,
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

        # Evaluate the detector's decision scores against the ground-truth labels.
        # TSB-AD's get_metrics() computes AUC-PR/ROC, VUS-PR/ROC and the
        # threshold-dependent F1 variants together from shared intermediate work.
        # The threshold-dependent variants are scored against a real, deployable
        # threshold (mean + 2*std of the min-max-normalized scores) rather than
        # get_metrics()'s default oracle/best-F1 threshold; we also keep that
        # threshold's own precision as `precision_2sigma`.
        y_true = df[self._label_col].to_numpy()
        eval_start = perf_counter()

        import numpy as np
        from sklearn.metrics import precision_score
        from TSB_AD.evaluation.metrics import get_metrics

        from tsadquality.evaluators.internal_metrics import compute_internal_metrics

        self.decision_scores_ = self._min_max(self.decision_scores_)
        pred = self.decision_scores_ > (
            self.decision_scores_.mean() + 2 * self.decision_scores_.std()
        )

        raw_metrics = get_metrics(
            self.decision_scores_, y_true, slidingWindow=window, pred=pred
        )
        metrics = {
            METRIC_COLUMNS[name]: float(value) for name, value in raw_metrics.items()
        }

        metrics["precision_2sigma"] = float(
            precision_score(y_true, pred.astype(int), zero_division=np.nan)
        )
        metrics.update(
            compute_internal_metrics(self.detector, self.model_, self.decision_scores_, y_true)
        )

        eval_execution_time = perf_counter() - eval_start

        PostgresClient.write_evaluation(
            experiment_id=str(self),
            dataset_name=self.timeseries,
            random_seed=str(ReproducibleOperations.get_current_random_seed()),
            data_perfectness=str(self.data_perfectness),
            corruption_type=str(self.corruption_type) if self.corruption_type else None,
            detector=str(self.detector),
            metrics=metrics,
            execution_time=eval_execution_time,
        )
        LOG.info(f"Wrote evaluation {metrics} for experiment {self!s} to Postgres.")

    @staticmethod
    def _min_max(scores):
        from sklearn.preprocessing import MinMaxScaler

        return MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()

    def _timeseries_path(self) -> Path:
        return _PROJECT_ROOT / "data" / "TSB-AD-U" / f"{self.timeseries}.csv"

    def _ts_metadata(self) -> "pd.Series":
        from tsadquality.data.clients.PostgresClient import PostgresClient

        escaped_timeseries = self.timeseries.replace("'", "''")
        result = PostgresClient.query(
            f"SELECT window_length, periodicity FROM ts_metadata WHERE dataset = '{escaped_timeseries}'"
        )
        if result.empty:
            raise ValueError(
                f"No precomputed metadata found for timeseries '{self.timeseries}'. "
                "Run `python -m tsadquality.utils.ts_metadata_table` first."
            )
        return result.iloc[0]

    def _window_length(self) -> int:
        return int(self._ts_metadata()["window_length"])

    def _periodicity(self) -> int:
        return 1

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
            corruption_target=str(CORRUPTION_TARGETS[self.corruption_type]),
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
                for field in CORRUPTION_PARAM_FIELDS
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
            5 : 5 + len(CORRUPTION_PARAM_FIELDS)
        ]
        corruption_params = {
            field: CORRUPTION_PARAM_TYPES[field](value)
            for field, value in zip(
                CORRUPTION_PARAM_FIELDS, corruption_param_parts
            )
            if value != cls._NULL
        } or None

        detector = DetectorModel(experiment_id_parts[5 + len(CORRUPTION_PARAM_FIELDS)])

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
            LOG.info(
                f"Running experiment {self!s} will be skipped because it already exists in Postgres."
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
