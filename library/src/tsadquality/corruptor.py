"""A small API for corrupting time series: configure corruptions, then corrupt DataFrames."""

import numpy as np
import pandas as pd

from tsadquality.corruption import CORRUPTIONS, MECHANISMS, TARGETS

_NO_LABELS = "__tsadquality_labels__"


class Corruptor:
    """Applies a sequence of corruptions to a time series held in a pandas DataFrame.

    Example:
        corruptor = Corruptor(seed=42)
        corruptor.add("noise", snr_db=10)
        corruptor.add("spikes", fraction=0.01, multiplier=5)
        corrupted = corruptor.corrupt(df, value_col="value", label_col="label")

    Corruptions run in the order they were added, each on the previous one's output.
    With a seed, `corrupt()` is deterministic: the same input gives the same output on
    every call. numpy's global random state is never touched.
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self._steps: list[tuple[str, dict, str, int]] = []

    def add(self, kind: str, *, target: str = "global", window: int = 50, **params) -> "Corruptor":
        """Adds a corruption step and returns the Corruptor, so calls can be chained.

        Args:
            kind: one of "noise", "spikes", "stuck", "point_swap", "segment_swap",
                "permutation", "point_missing", "burst_missing", "gilbert_elliott".
            target: which points may be corrupted, relative to the labelled anomalies:
                "global" (default, all points), "only_normal", "overlapping_anomaly",
                "near_anomaly", "before_anomaly", "after_anomaly" or "far_from_anomaly".
                Anything but "global" needs `label_col` in `corrupt()`.
            window: how many points around an anomaly count as "near" it, for the
                anomaly-relative targets.
            **params: the corruption's settings, e.g. `snr_db=10` for "noise".
        """
        if kind not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption {kind!r}. Choose one of: {', '.join(CORRUPTIONS)}.")
        _, required, optional = CORRUPTIONS[kind]
        missing = [name for name in required if name not in params]
        unknown = [name for name in params if name not in required + optional]
        if missing or unknown:
            expected = ", ".join(required + tuple(f"{name} (optional)" for name in optional))
            problems = []
            if missing:
                problems.append(f"missing {', '.join(missing)}")
            if unknown:
                problems.append(f"unexpected {', '.join(unknown)}")
            raise TypeError(f"{kind!r}: {'; '.join(problems)}. It takes: {expected}.")
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}. Choose one of: {', '.join(TARGETS)}.")
        if "mechanism" in params and params["mechanism"] not in MECHANISMS:
            raise ValueError(f"Unknown mechanism {params['mechanism']!r}. Choose one of: {', '.join(MECHANISMS)}.")

        self._steps.append((kind, dict(params), target, window))
        return self

    def corrupt(
        self,
        df: pd.DataFrame,
        value_col: str = "value",
        label_col: str | None = None,
        return_mask: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
        """Returns a corrupted copy of `df`; `df` itself is left unchanged.

        Only `value_col` is modified. Labels (if any) are never corrupted, and the
        result keeps `df`'s index and other columns. With `return_mask=True`, also
        returns a boolean Series marking every point any step corrupted.
        """
        if not self._steps:
            raise ValueError("No corruptions added; call add() first.")
        if value_col not in df.columns:
            raise KeyError(f"Column {value_col!r} not found in the DataFrame.")
        if label_col is not None and label_col not in df.columns:
            raise KeyError(f"Column {label_col!r} not found in the DataFrame.")
        if label_col is None and any(target != "global" for _, _, target, _ in self._steps):
            raise ValueError("Anomaly-relative targets need label_col.")

        rng = np.random.default_rng(self.seed)
        work = df.reset_index(drop=True)
        labels = label_col
        if labels is None:
            labels = _NO_LABELS
            work = work.assign(**{_NO_LABELS: 0})

        mask = pd.Series(False, index=work.index)
        for kind, params, target, window in self._steps:
            injector_class = CORRUPTIONS[kind][0]
            injector = injector_class(
                work, value_col=value_col, label_col=labels, rng=rng,
                corruption_target=target, window_size=window,
            )
            injector.inject(**params)
            work = injector.get_corrupted_df()
            mask |= injector.get_corruption_mask()

        if label_col is None:
            work = work.drop(columns=_NO_LABELS)
        work.index = df.index
        mask.index = df.index

        return (work, mask) if return_mask else work

    def __repr__(self) -> str:
        steps = []
        for kind, params, target, window in self._steps:
            settings = [f"{name}={value!r}" for name, value in params.items()]
            if target != "global":
                settings += [f"target={target!r}", f"window={window}"]
            steps.append(f"{kind}({', '.join(settings)})")
        return f"Corruptor(seed={self.seed!r}, steps=[{', '.join(steps)}])"
