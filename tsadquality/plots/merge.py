"""Composes multiple `tsadquality.plots` plot objects into one figure of subplots.

Any plot class following the `plots/CLAUDE.md` convention (a `plot(ax=None)` method
that draws onto a given axes and returns it) can be placed in a grid cell here, so
plots don't need to know anything about being merged.

Example:
    from tsadquality.plots.merge import merge_plots
    from tsadquality.plots.WhiteNoiseLineplot import WhiteNoiseLineplot

    grid = [
        [WhiteNoiseLineplot(metric="AUC-ROC"), WhiteNoiseLineplot(metric="VUS-ROC")],
        [WhiteNoiseLineplot(metric="Standard-F1"), None],  # None leaves a cell blank
    ]
    fig = merge_plots(grid)
"""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


class Plottable(Protocol):
    def plot(self, ax: "matplotlib.axes.Axes") -> "matplotlib.axes.Axes": ...


def merge_plots(
    grid: list[list["Plottable | None"]],
    figsize: tuple[float, float] | None = None,
) -> "matplotlib.figure.Figure":
    """Lays out `grid` (a list of rows, each a list of plot objects or `None`) as a
    single figure of subplots, one cell per grid position. Rows may have different
    lengths; missing/`None` cells are left blank. Each non-`None` cell's `plot(ax=...)`
    is called with that cell's axes.
    """
    import matplotlib.pyplot as plt

    n_rows = len(grid)
    n_cols = max((len(row) for row in grid), default=0)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("`grid` must have at least one row and one column.")

    if figsize is None:
        figsize = (n_cols * 6, n_rows * 4.5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for row_index, row in enumerate(grid):
        for col_index in range(n_cols):
            ax = axes[row_index][col_index]
            plot_obj = row[col_index] if col_index < len(row) else None
            if plot_obj is None:
                ax.axis("off")
                continue
            plot_obj.plot(ax=ax)

    fig.tight_layout()
    return fig


def main() -> None:
    """Example CLI: merges a small template grid of `WhiteNoiseLineplot`s.

    This is meant as a starting template, not a general-purpose CLI (there's no
    reasonable string syntax for "arbitrary plot classes in arbitrary cells"):
    copy the `grid` construction into a script/notebook and swap in whatever plot
    instances/layout you need.
    """
    import argparse

    from tsadquality.plots.output import SUPPORTED_FORMATS, save_figure
    from tsadquality.plots.WhiteNoiseLineplot import WhiteNoiseLineplot

    parser = argparse.ArgumentParser(description="Merge plots into one figure of subplots.")
    parser.add_argument("--format", default="png", choices=SUPPORTED_FORMATS)
    parser.add_argument("--filename", default="merged_example")
    args = parser.parse_args()

    grid = [
        [WhiteNoiseLineplot(metric="AUC-ROC"), WhiteNoiseLineplot(metric="AUC-PR"), WhiteNoiseLineplot(metric="VUS-ROC")],
    ]
    fig = merge_plots(grid)

    output_path = save_figure(fig, args.filename, format=args.format)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
