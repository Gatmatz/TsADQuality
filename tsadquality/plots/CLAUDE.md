# tsadquality/plots

Plotting package for this project's results. General rules for any new plot added here.

## Data access
- Never query Postgres directly from a plot module. Go through `fetch.py`
  (`fetch_experiments`/`fetch_evaluations`), which only issues `SELECT`/`WITH`
  statements via `PostgresClient.query()`. Plots must never write to the database.
- If a new query shape is needed, add it to `fetch.py` rather than building raw SQL
  inline in a plot class.

## Styling
- Import shared styling constants from `tsadquality.enums.plots`
  (`DEFAULT_METRIC`, `DETECTOR_COLORS`, `DETECTOR_MARKERS`, `DETECTOR_DISPLAY_NAMES`,
  `FONT_FAMILY`) instead of redefining per-detector colors/markers/defaults/fonts/labels
  in each plot module. If a new enum-keyed style constant is needed for multiple plots
  (e.g. per-corruption-type colors), add it to `tsadquality/enums/plots.py` so every
  plot can share it.
- Use `DETECTOR_DISPLAY_NAMES[detector]` for any label/legend/title text shown to a
  reader, never `str(detector)`/`DetectorModel`'s own value (e.g. "MatrixProfile") —
  those are identifiers, not display text.
- Set `plt.rcParams["font.family"] = FONT_FAMILY` in every plot's `plot()` method
  so all figures share the same typeface.
- Colors must stay colorblind-safe, and avoid green for anything that isn't
  meant to read as "good"/"correct" (it biases the reader).
- Metric choices must be validated against `tsadquality.enums.metrics.METRIC_COLUMNS`,
  the single source of truth mapping TSB-AD metric names to `evaluations` table columns.

## Structure of a plot class
- One class per plot type (e.g. `WhiteNoiseLineplot`), named for what it plots, with:
  - `__init__` that validates/stores the metric and detector selection.
  - Private `_fetch_*`/`_corrupted_series`/`_baseline_value`-style helpers that call
    into `fetch.py`.
  - A `plot(ax=None)` method that draws onto a given (or new) `matplotlib.axes.Axes`
    and returns it, so plots can be composed into subplots by callers.
  - A `main()` function with `argparse` (`--metric`, `--format`, `--filename`) plus
    `if __name__ == "__main__": main()`, so every plot is runnable standalone via
    `uv run python -m tsadquality.plots.<ModuleName>`.

## Output
- Save figures through `output.save_figure(fig, filename, format=...)`, never with a
  raw `fig.savefig(...)` call in a plot module. This keeps every plot writing to
  `tsadquality/plots/results/` and supports the same PNG/EPS format choice.
- Don't hardcode `"png"` — use `output.SUPPORTED_FORMATS` for `argparse` choices.

## Composability / merging into subplots
- Every plot's `plot(ax=None)` method must draw onto the given axes and return it
  (never create its own figure when `ax` is passed), so it can be dropped into a
  grid cell without modification.
- To combine multiple plots into one figure, use `tsadquality.plots.merge.merge_plots`
  rather than hand-rolling `plt.subplots` calls elsewhere: it takes a 2D `grid` of
  plot instances (rows of `Plottable | None`, `None` = blank cell) and calls each
  cell's `plot(ax=...)` for you. See `merge.py`'s module docstring for an example.

## Exports
- Every new public plot class/function must be added to `tsadquality/plots/__init__.py`
  and to its `__all__` (kept alphabetically sorted, per Ruff).

## Module naming
- Plot classes currently use PascalCase filenames (e.g. `WhiteNoiseLineplot.py`) to
  match the class they contain 1:1, even though this trips Ruff's module-naming
  lint rule. Keep this convention for new plot files unless told otherwise.
