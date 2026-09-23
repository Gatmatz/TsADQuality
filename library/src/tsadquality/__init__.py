"""Controlled, reproducible corruption of time series for anomaly-detection research."""

from importlib.metadata import PackageNotFoundError, version

from tsadquality.corruptor import Corruptor

try:
    __version__ = version("tsadquality")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["Corruptor", "__version__"]
