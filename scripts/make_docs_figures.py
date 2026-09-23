"""Regenerates the figures used by the project page in docs/ (GitHub Pages).

Applies each corruption type in the sweep, through tsadquality's own injectors and
with one of the swept settings, to an excerpt of a real TSB-AD-U series, and saves
them as small multiples to docs/static/images/corruptions.svg.

Usage (from the project root):
    PYTHONPATH=. uv run python scripts/make_docs_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from tsadquality.enums.data import CorruptionType
from tsadquality.reproducibility import ReproducibleOperations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "339_UCR_id_37_Sensor_tr_2750_1st_5400"
EXCERPT = slice(5050, 5950)
OUTPUT = PROJECT_ROOT / "docs" / "static" / "images" / "corruptions.svg"

CLEAN_COLOR = "#2a78d6"
CORRUPTED_COLOR = "#eb6834"
ANOMALY_COLOR = "#d9d9d9"
INK = "#3b3b3b"
MUTED = "#8a8a8a"

# (title, corruption type, settings) — every setting is one value from the sweep.
PANELS = [
    ("White noise · SNR 10 dB", CorruptionType.NOISE_SNR, {"snr_db": 10}),
    ("Spikes · 1% of points, 5σ", CorruptionType.SPIKE, {"fraction": 0.01, "multiplier": 5}),
    ("Stuck sensor · 20%, 1 block", CorruptionType.STUCK, {"fraction": 0.2, "stuck_blocks": 1}),
    ("Point swap · 40% of points", CorruptionType.POINT_SWAP, {"fraction": 0.4}),
    ("Segment swap · 30%, 1 swap", CorruptionType.SEGMENT_SWAP, {"fraction": 0.3, "num_swaps": 1}),
    ("Permutation · 8 segments", CorruptionType.PERMUTATION_SWAP, {"num_permutations": 8}),
]


def main() -> None:
    ReproducibleOperations.set_random_seed(47382)

    df = pd.read_csv(PROJECT_ROOT / "data" / "TSB-AD-U" / f"{DATASET}.csv")
    excerpt = df.iloc[EXCERPT].reset_index(drop=True)
    x = np.arange(len(excerpt))
    clean = excerpt["Data"].to_numpy(dtype=float)
    anomaly = np.flatnonzero(excerpt["Label"].to_numpy() == 1)

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.4), sharex=True)
    for ax, (title, corruption_type, settings) in zip(axes.ravel(), PANELS):
        corruptor = corruption_type.get_class()(excerpt, value_col="Data", label_col="Label")
        corruptor.inject(**settings)
        corrupted = corruptor.get_corrupted_df()["Data"].to_numpy(dtype=float)

        ax.axvspan(anomaly.min(), anomaly.max() + 1, color=ANOMALY_COLOR, lw=0, zorder=0)
        ax.plot(x, corrupted, color=CORRUPTED_COLOR, lw=1.3, zorder=2)
        ax.plot(x, clean, color=CLEAN_COLOR, lw=1.0, zorder=3)

        ax.set_title(title, loc="left", fontsize=13, color=INK, pad=6)
        ax.set_yticks([])
        ax.tick_params(axis="x", colors=MUTED, labelsize=10, length=3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#cfcfcf")

    for ax in axes[1]:
        ax.set_xlabel("time step (excerpt)", color=MUTED, fontsize=11)

    fig.legend(
        handles=[
            plt.Line2D([], [], color=CLEAN_COLOR, lw=2, label="Clean series"),
            plt.Line2D([], [], color=CORRUPTED_COLOR, lw=2, label="Corrupted series"),
            Patch(color=ANOMALY_COLOR, label="Labelled anomaly"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=12,
        labelcolor=INK,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, format="svg", bbox_inches="tight", metadata={"Date": None})
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
