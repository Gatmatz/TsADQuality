class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class _RandomSeedOperations:
    _random_seed: int | float = None


class ReproducibleOperations(_RandomSeedOperations, metaclass=Singleton):
    @classmethod
    def _ensure_reproducibility(self) -> None:
        import numpy as np

        from tsadquality.reproducibility import ReproducibilityError

        if self._random_seed:
            np.random.seed(self._random_seed)
            return
        raise ReproducibilityError(
            "The reproducibility of the requested operation that involves randomness cannot be ensured. \
                Make sure to call the set_random_seed(some_seed) function at least once in your program."
        )

    @classmethod
    def seed_everything(self) -> None:
        """
        Set random seeds for reproducibility across all libraries.

        Args:
            seed: Random seed value
        """

        import os
        import random

        import numpy as np
        import torch

        if self._random_seed is not None:
            os.environ["PL_GLOBAL_SEED"] = str(self._random_seed)
            random.seed(self._random_seed)
            np.random.seed(self._random_seed)
            torch.manual_seed(self._random_seed)
            torch.cuda.manual_seed(self._random_seed)
            torch.cuda.manual_seed_all(self._random_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    @classmethod
    def set_random_seed(cls, random_seed: float):
        cls._random_seed = random_seed

    @classmethod
    def get_current_random_seed(cls) -> int:
        return cls._random_seed

    @classmethod
    def sample_from(
        cls,
        elements: list,
        how_many: float,
        at_least: int = 1,
        sampling_with_replacement: bool = False,
    ) -> list:
        """Samples `how_many` items from `elements`. Internally uses numpy. Reproducibility is ensured as long as you stick to
        functions of this class throughout the application for all operations that require randomness.

        Args:
            elements (List): the elements to sample from
            how_many (int | float): the desired number of items to sample
            at_least (int, optional): number of items to be sampled at least. Overrides `how_many` in case `how_many` < `at_least`.
            Defaults to 1.
            sampling_with_replacement (bool, optional): Whether replacement should be applied during sampling. Defaults to False.

        Returns:
            List: the sampled items
        """
        if not elements:
            return []

        if len(elements) <= how_many:
            return elements

        import numpy as np

        cls._ensure_reproducibility()
        return np.random.choice(
            a=elements,
            size=max(int(how_many), at_least),
            replace=sampling_with_replacement,
        )

    @classmethod
    def uniform(cls, low: float, high: float, size: float | None = None):
        """Wraps a call to `np.random.uniform` for reproducibility purposes. For more info
        see https://numpy.org/devdocs/reference/random/generated/numpy.random.uniform.html.
        """
        import numpy as np

        cls._ensure_reproducibility()
        return np.random.uniform(low=low, high=high, size=size)

    @classmethod
    def normal(cls, loc: float, scale: float, size: int | tuple[int]):
        """Wraps a call to `np.random.normal` for reproducibility purposes. For more info
        see https://numpy.org/devdocs/reference/random/generated/numpy.random.normal.html.
        """
        import numpy as np

        cls._ensure_reproducibility()
        return np.random.normal(loc=loc, scale=scale, size=size)

    @classmethod
    def choice(cls, a, size: int | tuple[int] | None = None, replace: bool = True, p=None):
        """Wraps a call to `np.random.choice` for reproducibility purposes. For more info
        see https://numpy.org/devdocs/reference/random/generated/numpy.random.choice.html.
        """
        import numpy as np

        cls._ensure_reproducibility()
        return np.random.choice(a=a, size=size, replace=replace, p=p)

    @classmethod
    def derangement(cls, x):
        """Wraps a call to `np.random.permutation` for reproducibility purposes. For more info
        see https://numpy.org/devdocs/reference/random/generated/numpy.random.permutation.html.
        The function returns a **derangement**, i.e., there is no element that remains in its
        original position. This is performed by generating consecutive permutations with the same
        random seed, until the first derangement is found and returned.
        """
        if len(x) == 1:
            return x

        import numpy as np

        cls._ensure_reproducibility()

        while True:
            permutation = np.random.permutation(x=x)

            permutation_is_a_derangement = True
            for original_element, permuted_element in zip(x, permutation):
                if original_element == permuted_element:
                    permutation_is_a_derangement = False
                    break

            if permutation_is_a_derangement:
                return permutation

    @classmethod
    def shuffle_reindex_dataframe(cls, df):
        """Shuffles a dataframe and resets its index.

        Args:
            df (pd.DataFrame): the pandas dataframe to shuffle.

        Returns:
            pd.DataFrame: the shuffled pandas dataframe.
        """
        # pandas uses numpy's random seed internally: https://stackoverflow.com/a/52375474
        cls._ensure_reproducibility()
        return df.sample(frac=1, replace=False).reset_index(drop=True)

    @classmethod
    def get_lof(
        cls,
        window: int = 100,
        n_neighbors: int = 20,
        metric: str = "minkowski",
        n_jobs: int = 1,
    ):
        """Builds a `LOFDetector`. LOF has no random component of its own, but
        reproducibility is still enforced here for consistency with the rest of
        this class.
        """
        cls._ensure_reproducibility()

        from tsadquality.detectors import LOFDetector

        return LOFDetector(
            window=window, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs
        )

    @classmethod
    def get_isolation_forest(
        cls, window: int = 100, n_estimators: int = 100, max_features: float = 1, n_jobs: int = 1
    ):
        """Builds an `IsolationForestDetector`. TSB-AD's `run_IForest` wrapper
        pins `random_state=0` internally (it doesn't expose an override), so
        reproducibility here is enforced only for consistency with the rest of
        this class.
        """
        cls._ensure_reproducibility()

        from tsadquality.detectors import IsolationForestDetector

        return IsolationForestDetector(
            window=window,
            n_estimators=n_estimators,
            max_features=max_features,
            n_jobs=n_jobs,
        )

    @classmethod
    def get_matrix_profile(cls, periodicity: int = 1, n_jobs: int = 1):
        """Builds a `MatrixProfileDetector`. STUMPY's matrix profile computation is
        deterministic, but reproducibility is still enforced here for consistency
        with the rest of this class.
        """
        cls._ensure_reproducibility()

        from tsadquality.detectors import MatrixProfileDetector

        return MatrixProfileDetector(periodicity=periodicity, n_jobs=n_jobs)

    @classmethod
    def get_autoencoder(cls, window: int = 100, hidden_neurons: list | None = None, n_jobs: int = 1):
        """Builds an `AutoEncoderDetector`, seeding numpy/random/torch (via
        `seed_everything`) since TSB-AD's AutoEncoder trains with torch.
        """
        cls._ensure_reproducibility()
        cls.seed_everything()

        from tsadquality.detectors import AutoEncoderDetector

        return AutoEncoderDetector(window=window, hidden_neurons=hidden_neurons, n_jobs=n_jobs)

    @classmethod
    def get_detector(
        cls, detector_model, window: int = 100, periodicity: int = 1, **options
    ):
        """Builds a detector for `detector_model` (a `DetectorModel` member),
        dispatching to this class's own `get_*` factory so the seed set via
        `set_random_seed` reaches every detector's randomness the same way it
        reaches the corruption injectors. `window` and `periodicity` come from
        `ts_metadata` (see `Experiment._window_length()`/`_periodicity()`);
        `MatrixProfileDetector` is the only one that consumes `periodicity`
        instead of `window`.
        """
        from tsadquality.enums.detectors import DetectorModel

        match detector_model:
            case DetectorModel.LOF:
                return cls.get_lof(window=window, **options)
            case DetectorModel.ISO:
                return cls.get_isolation_forest(window=window, **options)
            case DetectorModel.MP:
                return cls.get_matrix_profile(periodicity=periodicity, **options)
            case DetectorModel.AutoEncoder:
                return cls.get_autoencoder(window=window, **options)
