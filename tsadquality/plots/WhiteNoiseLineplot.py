"""Metric-vs-SNR lineplot for the white-noise (`NOISE_SNR`) corruption.

One solid line per detector shows the chosen metric across SNR (dB) levels;
a dotted flat line per detector shows that same detector's uncorrupted
(`PERFECT`) baseline for comparison, in the same color as its solid line.
"""

from typing import TYPE_CHECKING

from tsadquality.enums.data import CorruptionType, DataPerfectness
from tsadquality.enums.detectors import DETECTORS, DetectorModel
from tsadquality.enums.metrics import METRIC_COLUMNS
from tsadquality.enums.plots import (
    DEFAULT_METRIC,
    DETECTOR_COLORS,
    DETECTOR_DISPLAY_NAMES,
    DETECTOR_MARKERS,
    FONT_FAMILY,
)

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd


class WhiteNoiseLineplot:
    def __init__(
        self,
        metric: str = DEFAULT_METRIC,
        detectors: list[DetectorModel] | None = None,
    ):
        if metric not in METRIC_COLUMNS:
            raise ValueError(f"Unknown metric '{metric}'. Must be one of {sorted(METRIC_COLUMNS)}.")
        self.metric = metric
        self.metric_column = METRIC_COLUMNS[metric]
        self.detectors = detectors or list(DETECTORS)

    def _corrupted_series(self, detector: DetectorModel) -> "pd.Series":
        from tsadquality.plots.fetch import fetch_evaluations

        df = fetch_evaluations(
            detector=detector,
            corruption_type=CorruptionType.NOISE_SNR,
            data_perfectness=DataPerfectness.IMPERFECT,
            with_corruption_params=True,
        )
        return df.groupby("x_snr_db")[self.metric_column].mean().sort_index()

    def _baseline_value(self, detector: DetectorModel) -> float | None:
        from tsadquality.plots.fetch import fetch_evaluations

        df = fetch_evaluations(detector=detector, data_perfectness=DataPerfectness.PERFECT)
        if df.empty:
            return None
        return float(df[self.metric_column].mean())

    def plot(self, ax: "matplotlib.axes.Axes | None" = None) -> "matplotlib.axes.Axes":
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = FONT_FAMILY

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        for detector in self.detectors:
            color = DETECTOR_COLORS[detector]
            marker = DETECTOR_MARKERS[detector]

            series = self._corrupted_series(detector)
            if not series.empty:
                ax.plot(
                    series.index,
                    series.to_numpy(),
                    color=color,
                    marker=marker,
                    linewidth=2,
                    markersize=6,
                    label=DETECTOR_DISPLAY_NAMES[detector],
                    zorder=3,
                )

            baseline = self._baseline_value(detector)
            if baseline is not None:
                ax.axhline(
                    baseline,
                    color=color,
                    linestyle=(0, (3, 3)),
                    linewidth=4,
                    dash_capstyle="round",
                    alpha=0.6,
                    zorder=1,
                )

        ax.set_xlim(40, -20)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(self.metric)
        ax.set_title(f"{self.metric} vs. White Noise (SNR)")
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=True, title="Detector (dotted = baseline)")
        return ax


def main() -> None:
    import argparse

    from tsadquality.plots.output import SUPPORTED_FORMATS, save_figure

    parser = argparse.ArgumentParser(
        description="Plot a metric vs. SNR (white noise) per detector."
    )
    parser.add_argument(
        "--metric", default=DEFAULT_METRIC, choices=sorted(METRIC_COLUMNS)
    )
    parser.add_argument("--format", default="png", choices=SUPPORTED_FORMATS)
    parser.add_argument(
        "--filename",
        default=None,
        help="Output filename stem, saved under results/plots/. Defaults to 'white_noise_<metric>'.",
    )
    args = parser.parse_args()

    plot = WhiteNoiseLineplot(metric=args.metric)
    ax = plot.plot()
    fig = ax.figure
    fig.tight_layout()

    filename = args.filename or f"white_noise_{plot.metric_column}"
    output_path = save_figure(fig, filename, format=args.format)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
