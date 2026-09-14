# Corruption experiments — TSB-AD

Every data-quality experiment of the paper, evaluated with the official TSB-AD pipeline
(`run_Unsupervise_AD` / `run_Semisupervise_AD`, `find_length_rank`, `get_metrics`) on the
350-series TSB-AD-U eval list. The thesis generation (TSB-UAD) lives in
[`../corruption_tsbuad/`](../corruption_tsbuad/) and is frozen.

## How the code is split

| Where | What it decides |
|---|---|
| `src/ts_corruptor/` | **what** is done to a series: `injectors.py` (registry), `swap.py`, `missing.py`, `localized.py` |
| `src/tsbad/` | **how** a corrupted series is evaluated: models, metrics, checkpoints, process pool, summary |
| this folder | one file per experiment: its grid, its corruption, its extra columns |

Each runner (except `propagation.py`) is a `Spec` passed to `tsbad.harness.run_sweep`.

## Experiments

File name = results directory without the `_tsbad` suffix.

| Runner | Results (`results/experiments/…`) | TSB-UAD original | Family | Run so far (full 350) |
|---|---|---|---|---|
| `white_noise_snr.py` | `white_noise_snr_tsbad/` | `run_whitenoise_snr.py` | full | IForest, MatrixProfile, Sub_PCA |
| `spikes.py` | `spikes_tsbad/` | `run_spikes.py` | full | — |
| `spikes_normal_only.py` | `spikes_normal_only_tsbad/` | `run_spikes_normal_only.py` | full | IForest, MatrixProfile |
| `freeze.py` | `freeze_tsbad/` | `run_freeze.py` | full | IForest |
| `swap_point.py` | `swap_point_tsbad/` | `run_swap_point.py` | full | IForest |
| `swap_segment.py` | `swap_segment_tsbad/` | `run_swap_segment.py` | full | IForest |
| `swap_permutation.py` | `swap_permutation_tsbad/` | `run_swap_permutation.py` | full | IForest |
| `missing_true_impact.py` | `missing_true_impact_tsbad/` | `run_missing_true_impact.py` | survivors, always | IForest |
| `gilbert_elliott_true_impact.py` | `gilbert_elliott_true_impact_tsbad/` | `run_gilbert_elliott_true_impact.py` | survivors, always | IForest |
| `missing_mnar.py` | `missing_mnar_tsbad/` | `run_missing_mnar.py` | survivors, always | — |
| `missing_mnar_burst.py` | `missing_mnar_burst_tsbad/` | `run_missing_mnar_burst.py` | survivors, always | — |
| `compound_corruptions.py` | `compound_corruptions_tsbad/` | `run_compound_corruptions.py` | survivors, if_dropped | — |
| `propagation.py` | `propagation_tsbad/` | `run_propagation.py` | own worker (no metrics row) | — |

"Run so far" is a snapshot from 2026-09-14; the checkpoints are the source of truth.

`compound_corruptions.py` runs only the combinations. Its clean anchor and singles are imported
from `freeze` (clean), `white_noise_snr`, `missing_true_impact`, `spikes_normal_only`, `freeze` and
`gilbert_elliott_true_impact` (see `SINGLE_SOURCES`), restricted to the files the compounds ran
on, so run those for the same models first; the run prints which terms it could not find.
`--compute-singles` generates them in-run instead. Spikes land on normal points only (in singles
and compounds): with global placement a spike on an anomaly hides most of the damage — see
deviation 5 in the module docstring. Interaction and Shapley are written per metric
(`--interaction-metrics`, default `AUC_ROC VUS_PR`) to `*_<metric>.csv`; the column names stay
`baseline_auc`, `auc_A`, … for every metric, and the `metric` column says which one it is.
Each interaction/Shapley row averages its terms over the series all of them have (`n_series`;
`n_series_excluded` counts the ones left out), so it can be recomputed with `--summary-only` at
any point — after more combinations or singles are run — without mixing different series.
Next to the original columns, each row carries a 95% bootstrap interval over series
(`interaction_type_ci` only labels a row synergistic/sub-additive when it excludes zero) and a
saturation flag (`predicted_below_chance`, `frac_series_below_chance`): see deviation 7.

TSB-UAD experiments with **no** TSB-AD version yet: `run_noise_position.py`,
`run_anomaly_aware_corruption.py`, `run_gradual_drift.py`, `run_propagation_multiscale.py`,
`run_missing_masking.py`. See [`../corruption_tsbuad/README.md`](../corruption_tsbuad/README.md)
for the ones that are covered without a separate port.

## Running

From the repository root, with the project venv:

```
python src/experiments/corruption_tsbad/freeze.py --test
python src/experiments/corruption_tsbad/freeze.py --models IForest --workers 4
python src/experiments/corruption_tsbad/freeze.py --models MatrixProfile --workers 4
```

Flags every `run_sweep` runner has:

| Flag | Meaning |
|---|---|
| `--test` | 3 files (compound: 2) and a reduced grid |
| `--models` | TSB-AD model names; default `IForest` |
| `--workers` | process pool size; default 4 |
| `--no-clean` | skip the clean anchor condition |
| `--window-mode clean\|native` | `clean` (default) fixes periodicity-model windows from the clean signal |
| `--files-csv`, `--max-files` | a different file list, or the first N files |
| `--results-dir` | write somewhere other than `results/experiments/<name>/` |
| `--summary-only` | rebuild `summary.csv` (and compound's interaction/Shapley) from checkpoints |

Runs resume: rows already in `checkpoint_<MODEL>.csv` (or a legacy `checkpoint.csv`) are
skipped, keyed on `file|condition|seed|model`. `summary.csv` always covers every model on disk.
Condition names are part of that key — never rename them.

## Evaluation families

- **full** — the detector sees the whole corrupted series; metrics on every point.
- **survivors** — the corruption deletes points (NaN rows). The detector sees only the
  survivors; lost anomalies are scored as missed with `score.min()` (TSB-AD scores are raw, a
  literal 0 would rank them near the top), lost normal points are excluded, and the model window
  is clamped to `max(min(w, n_kept // 4), 10)`.
  - `always` — on every condition, the clean anchor included (missing / MNAR / Gilbert-Elliott).
  - `if_dropped` — only where points were actually lost (compound).

Known consequence, kept on purpose: with `always`, the floor of 10 also applies to the clean
anchor, so for window models (MatrixProfile, POLY, Sub_PCA, KShapeAD, KMeansAD_U) the clean row
of a missing-family experiment can differ from the clean row of a full-family one on series
whose window is below 10 or above n/4. IForest is unaffected. The TSB-UAD originals applied the
floor everywhere.

## Adding an experiment

A new runner is a grid, a corruption and a `Spec`. For example, linear drift:

```python
"""Linear drift robustness — TSB-AD."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import ts_corruptor.injectors  # noqa: E402
from ts_corruptor.core import TSCorruptor  # noqa: E402
from tsbad.harness import Spec, condition, run_sweep  # noqa: E402

DRIFT_FACTORS = [0.5, 1.0, 2.0, 5.0]


def build_conditions(args):
    factors = [1.0] if args.test else DRIFT_FACTORS
    grid = ([] if args.no_clean else [None]) + factors
    return [condition("clean" if d is None else f"drift_{d}", {'drift_factor': d}) for d in grid]


def corrupt(ctx, p):
    if p['drift_factor'] is None:
        return ctx.clean_data, {}
    corruptor = TSCorruptor(ctx.df.copy(), value_col=ctx.value_col, label_col=ctx.label_col,
                            seed=ctx.seed)
    ts_corruptor.injectors.inject_drift(corruptor, drift_factor=p['drift_factor'])
    return corruptor.get_corrupted_df().iloc[:, 0:-1].values.astype(float), {}


SPEC = Spec(
    name='drift_tsbad',
    title='Linear drift — TSB-AD',
    family='full',
    build_conditions=build_conditions,
    corrupt=corrupt,
    summary_keys=('condition', 'drift_factor', 'model'),
)

if __name__ == "__main__":
    run_sweep(SPEC)
```

Rules the harness relies on:

- Hooks are module-level functions, never lambdas — jobs are pickled to worker processes.
- `import tsbad.env` comes before anything that imports numpy.
- `corrupt` returns the `(N, features)` float array the detector sees plus a dict of extra row
  columns; in the survivors family, deleted points are NaN rows. Raise `tsbad.harness.Skip` for
  a file the condition cannot be evaluated on.
- The corruption draws from its own RNG seeded with `ctx.seed`; the detector is seeded per job
  by the harness (`tsbad.models.seed_job`).

## How this layout was verified

Before the move onto the shared harness (tag `pre-corruption-refactor`), every original runner
was run with `--test` (IForest + POLY, and KMeansAD_U for the SNR seed key). Each migrated runner
reproduced its checkpoints, summary and analysis tables exactly, and where results already
existed the test rows matched them too. See the `refactor(corruption)` commits for per-runner
details.
