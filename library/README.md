# tsadquality

Controlled, reproducible corruption of time series, for studying how data-quality
problems affect anomaly detectors. Configure one or more corruptions, then apply
them to any pandas DataFrame.

```bash
pip install tsadquality
```

## Usage

```python
import pandas as pd
from tsadquality.corruptor import Corruptor

df = pd.read_csv("series.csv")          # e.g. columns "value" and "label"

corruptor = Corruptor(seed=42)
corruptor.add("noise", snr_db=10)
corruptor.add("spikes", fraction=0.01, multiplier=5)

corrupted = corruptor.corrupt(df, value_col="value", label_col="label")
```

- Corruptions run in the order they were added, each on the previous one's output.
- `corrupt()` returns a corrupted copy: your DataFrame, its index, the labels and
  every other column are left unchanged.
- With a `seed`, the same input always gives the same output. numpy's global random
  state is never touched.
- `corrupt(..., return_mask=True)` also returns a boolean Series marking the
  corrupted points.
- `label_col` is optional unless you use an anomaly-relative `target` (below).

## Corruptions

| Name | What it does | Settings |
|---|---|---|
| `noise` | Adds Gaussian noise at a target signal-to-noise ratio | `snr_db` |
| `spikes` | Moves a fraction of points by ±`multiplier`·σ of the series | `fraction`, `multiplier` |
| `stuck` | Freezes the signal at its current value, in `stuck_blocks` blocks covering `fraction` of the series | `fraction`, `stuck_blocks` |
| `point_swap` | Swaps pairs of single points, covering `fraction` of the series | `fraction` |
| `segment_swap` | Swaps `num_swaps` pairs of segments covering `fraction` of the series | `fraction`, `num_swaps` |
| `permutation` | Cuts the series into `num_permutations` equal segments and reorders them so none stays in place | `num_permutations` |
| `point_missing` | Sets a fraction of points to NaN | `fraction`, `mechanism` (optional) |
| `burst_missing` | Sets `num_bursts` contiguous blocks covering `fraction` of the series to NaN | `fraction`, `num_bursts`, `mechanism` (optional) |
| `gilbert_elliott` | NaN gaps from a two-state Markov chain: `a` = chance of starting a gap, `b` = chance of ending it | `a`, `b` |

`mechanism` chooses which points go missing: `"MCAR"` (default, uniformly at random),
`"MNAR_extreme"` (more likely at extreme values in either tail) or `"MNAR_high"`
(more likely at high values).

## Choosing where to corrupt

Every `add()` also takes `target`, which restricts the corruption to points relative
to the labelled anomalies (this needs `label_col`):

```python
corruptor.add("noise", snr_db=0, target="near_anomaly", window=50)
```

`target` is one of `"global"` (default, all points), `"only_normal"`,
`"overlapping_anomaly"`, `"near_anomaly"`, `"before_anomaly"`, `"after_anomaly"`
or `"far_from_anomaly"`; `window` is how many points around an anomaly count as near it.

## Project

Part of the TsADQuality benchmark: <https://gatmatz.github.io/TsADQuality/>

Released under the [MIT License](LICENSE).
