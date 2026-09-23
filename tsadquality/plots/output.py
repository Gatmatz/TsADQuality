"""Shared save-to-disk logic for the `plots` package."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SUPPORTED_FORMATS = ("png", "eps")


def save_figure(fig: "matplotlib.figure.Figure", filename: str, format: str = "png") -> Path:
    """Saves `fig` as `tsadquality/plots/results/{filename}.{format}`, creating the directory if needed."""
    if format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{format}'. Must be one of {SUPPORTED_FORMATS}.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{filename}.{format}"
    fig.savefig(output_path, format=format, dpi=300, bbox_inches="tight")
    return output_path
