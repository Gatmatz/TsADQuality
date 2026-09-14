# Segment Swap: `swap_segment.py` vs `run_swap_segment.py`

What changed when the segment-swap experiment was ported from TSB-UAD to TSB-AD, why,
and what it means for results already produced with the old script.

Date: 2026-09-03. Scripts: `run_swap_segment.py` (old, TSB-UAD) →
`swap_segment.py` (new, TSB-AD).

---

## 1. What did NOT change

The experiment itself is unchanged — same question, same grid, same formula:

```
FRACTIONS   = [0.01, 0.05, 0.10, 0.20, 0.30]
NUM_SWAPS   = [1, 3, 5, 10, 20]
swap_length = max(1, int(fraction * n / (2 * num_swaps)))
N_SEEDS     = 1
```

Total swapped fraction is held constant while granularity varies: small `num_swaps` means
few long segments, large `num_swaps` means many short ones. Labels are never touched —
only values move, so the marginal distribution of the series is preserved exactly (the
corruption is a permutation of the original samples).

> **Key question:** for the same amount of swap corruption, is it worse to have a few
> large segment swaps or many small ones?

---

## 2. Port-level differences (shared with the other two TSB-AD ports)

These are the same changes already made in `swap_permutation.py` and
`swap_point.py`, listed here for completeness.

| | Old (TSB-UAD) | New (TSB-AD) |
| --- | --- | --- |
| Data | 141-series subset (`final_subset.csv`) | full 350-series `TSB-AD-U-Eva.csv` |
| Detectors | IForest, PCA, MP, LOF, AE (hand-instantiated) | official `run_Unsupervise_AD` / `run_Semisupervise_AD` |
| Preprocessing | manual `StandardScaler` + `Window` + `MinMaxScaler` | handled inside the official wrappers |
| Metric window | `find_length(clean_scaled)` | `find_length_rank(clean_data, rank=1)` |
| Metrics | `TSB_UAD.vus.metrics.get_metrics` (11 cols) | `TSB_AD.evaluation.metrics.get_metrics` (9 cols) |
| Clean anchor | none | yes — `condition = "clean"` |
| Checkpoints | one shared `checkpoint.csv` | one per model, `checkpoint_<model>.csv` |
| Scheduling | file order | longest-file-first |
| Seeding | injector seed only | + deterministic per-job md5 seed for stochastic detectors |

Because the benchmark, the detectors and the metric implementations all differ, **numbers
are not comparable between old and new runs**. What is comparable is the *shape* of the
degradation curves.

### Clean anchor

The new script runs one extra condition per file with no corruption at all, through the
identical code path. Verified bit-identical to a literal transcription of the official
`TSB-AD/benchmark_exp/Run_Detector_U.py` (IForest, 3 files, all 9 metrics):

```
657_YAHOO_id_107_WebService...   max|Δ| = 0.00e+00   IDENTICAL
644_YAHOO_id_94_WebService...    max|Δ| = 0.00e+00   IDENTICAL
558_YAHOO_id_8_WebService...     max|Δ| = 0.00e+00   IDENTICAL
```

So the x=0 point of every degradation curve is the genuine TSB-AD baseline.

---

## 3. The substantive difference: how segments are placed

This is the change that matters, and the reason the two scripts are not merely
"the same experiment on different data".

### 3.1 What the old injector does

`ts_corruptor.injectors.inject_swap` picks one random start at a time and checks whether
the segment fits:

```python
if (idx1 + swap_length <= n) and (idx2 + swap_length <= n):
    ...perform the swap...
# else: nothing. No retry, no warning, RNG already consumed.
```

Two ways corruption is silently lost:

1. **Bounds skip.** If a start lands within `swap_length` of the series end, the whole
   pair is dropped. The probability is `L/n = fraction / (2 * num_swaps)` per attempt —
   so with `num_swaps = 1` and `fraction = 0.30` there is a **15% chance the series is
   left completely untouched**, because there is only one attempt.
2. **Cross-pair overlap.** After a swap, only the exact used indices are pruned from the
   candidate array. A later segment can still *start* just before an earlier one and
   overlap it. Overlapping points are counted once, so the delivered fraction drops.

The shortfall grows with `L/n`:

| swap_length (n=20 000, fraction=0.20) | shortfall |
| --- | --- |
| 1 (point swap) | **0.0%** |
| 10 / 50 / 100 | 4.2% / 4.6% / 5.6% |
| 500 | 7.2% |
| 1 000 | 12.5% |

### 3.2 How much was actually lost in the published run

Delivery depends only on `n`, `L` and the seed — not on the data values — so the original
run can be replayed exactly from the series lengths alone. Replayed over the real
141-file subset at seed 0 (`N_SEEDS = 1` in the original run), **delivered / requested, %**:

| fraction \ num_swaps | 1 | 3 | 5 | 10 | 20 |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 91.5 | 100.0 | 100.0 | 99.9 | 100.0 |
| 0.05 | 85.8 | 96.0 | 97.9 | 98.7 | 99.0 |
| 0.10 | 82.3 | 92.5 | 96.3 | 96.8 | 98.2 |
| 0.20 | 86.5 | 91.0 | 90.4 | 93.9 | 94.8 |
| 0.30 | **71.6** | 79.4 | 88.5 | 89.0 | 93.0 |

Worse, the failures are not spread evenly. Files that received **zero** corruption, out of
141 (every other column is 0):

| fraction | 0.01 | 0.05 | 0.10 | 0.20 | 0.30 |
| --- | --- | --- | --- | --- | --- |
| `num_swaps = 1` | 12 | 20 | 25 | 19 | **40** |

At `fraction = 0.30, num_swaps = 1`, **28% of the subset passed through clean** and
contributed its baseline score to the mean.

**The shortfall is largest exactly where `num_swaps` is smallest** — i.e. it varies along
the axis the experiment exists to measure.

### 3.3 What the new implementation does

`_swap_segments_fast` places `2 * num_swaps` **strictly disjoint** segments of length `L`,
drawn uniformly, using the standard bijection between "k disjoint blocks of length L in n
slots" and "k distinct starts in n − k(L−1) slots":

```python
free   = n - k * (L - 1)
starts = np.sort(rng.choice(free, size=k, replace=False)) + np.arange(k) * (L - 1)
```

Draw the starts without replacement, sort them, then push start *i* right by `i*(L-1)`.
The largest start is then at most `n - L`, so no segment can run off the end and none can
overlap — by construction, not by rejection. Every condition therefore delivers exactly
`2 * num_swaps * L` points.

Rows are swapped whole, so multivariate channels stay aligned.

### 3.4 Verification

| Check | Result |
| --- | --- |
| Exact point count, 25 conditions × 5 seeds, n = 1k / 20k / 900k | exact in every cell |
| Overlaps or out-of-bounds placements, 300 random trials | 0 |
| Value multiset preserved (corruption is a permutation) | yes |
| Same seed → identical output; different seed → different | yes |
| Multivariate channel alignment | preserved |
| Cost: all 25 conditions on the 900 000-point series | 0.44 s |

`ts_corruptor/injectors.py` was **deliberately left untouched** so previously produced
TSB-UAD results remain reproducible. `--injector reference` restores the original path
for cross-checking.

---

## 4. Consequence for the old results

Recomputing `results/experiments/swap_segment/` over **only the files that were actually
corrupted**. The drop-out probability is `fraction / (2 * num_swaps)`, identical for every
file, so the survivors are an unbiased random subset.

**IForest reverses.** AUC-ROC, gap = (num_swaps=20) − (num_swaps=1):

| fraction | reported `ns=1` | corrected `ns=1` | reported gap | corrected gap |
| --- | --- | --- | --- | --- |
| 0.10 | 0.6828 | 0.6559 | +0.0003 | **+0.0271** |
| 0.20 | 0.6574 | 0.6228 | −0.0047 | **+0.0299** |
| 0.30 | 0.6547 | **0.5936** | −0.0042 | **+0.0569** |

The published table reads "granularity barely matters for IForest". Corrected, it reads
"few large segments are clearly worse". Sign flip, and ~14× the magnitude.

**AE, LOF, MP keep their direction** (many small swaps worse) but the `fraction = 0.30`
gaps collapse toward zero — e.g. AE AUC-ROC: −0.0080 → −0.0005.

Note this correction only removes the *zero-corruption* contamination. It does not fix the
partial under-delivery (71.6% on average at `fraction=0.30, num_swaps=1`), so the true
`num_swaps = 1` effect is likely **stronger** than the corrected column shows.

**Caveat.** The replay matched the `swap_length` recorded in the run for 120 of 141 files
(85%); `data_len` in `final_subset.csv` does not always equal the length the run actually
loaded. The direction is robust; the exact decimals are approximate.

---

## 5. Known limitation of the new script: integer flooring of L

`L = max(1, int(fraction * n / (2 * num_swaps)))` truncates. On a 1 440-point series with
`fraction=0.30, num_swaps=20`, `L` becomes 10 instead of 10.8, so 0.2778 of the series is
swapped rather than 0.30.

Measured across all 350 TSB-AD-U files (`n`: min 1 000, median 18 227, max 900 000),
delivered / nominal:

| fraction | median | worst single file |
| --- | --- | --- |
| 0.01 | 96.4 – 99.7% | 50.5% |
| 0.05 | 97.9 – 99.9% | 52.0% |
| 0.10 | 99.2 – 100% | 74.6% |
| 0.20 | 99.7 – 100% | 91.1% |
| 0.30 | 99.6 – 100% | 92.6% |

Unlike the old bug this is **deterministic, seed-independent, predictable, inherent to the
experiment's own formula, and recorded**: every result row carries `fraction_actual` and
every summary row carries `mean_fraction_actual`.

**Recommendation:** plot against `mean_fraction_actual`, not the nominal `fraction`. The
limitation then disappears entirely — the x-axis shows what was actually applied.

---

## 6. Scope: other scripts using the same injector

| Status | Scripts |
| --- | --- |
| **Unaffected** (`swap_length = 1`, shortfall exactly 0%) | `run_swap_point.py`, `run_corruption_overlap.py`, `run_cleaning_recovery.py`, `run_moment_robustness.py`; permutation never calls `inject_swap` |
| **Mildly affected** (fixed L = 10/50/100 → 4–6%) | `run_swap.py` |
| **Seriously affected** (L derived, reaches a large share of n) | `run_swap_segment.py`, `run_swap_ratio.py`, `run_internal_analysis.py`, `run_iforest_pathlength_analysis.py`, `run_segment_vulnerability.py`, and the segment branch of `run_chronos_robustness.py` |

The same replay check can be run cheaply on the "seriously affected" scripts — it needs no
detector, only the series lengths.

---

## 7. Running the new script

```bash
# smoke test: 3 files, 4 conditions + clean anchor
python src/experiments/corruption_tsbad/swap_segment.py --test

# full run
python src/experiments/corruption_tsbad/swap_segment.py --models IForest --workers 4
```

350 files × 26 conditions (25 + clean) = **9 100 jobs per model**. Checkpoint/resume is per
model, so a run can be interrupted and continued.

Output: `results/experiments/swap_segment_tsbad/checkpoint_<model>.csv` and `summary.csv`
(with `mean_swap_length` and `mean_fraction_actual` per condition).

Flags:

| Flag | Meaning |
| --- | --- |
| `--window-mode clean` (default) | periodicity-derived windows fixed from the clean signal, as in the original experiment |
| `--window-mode native` | pure TSB-AD wrapper, window re-estimated on the corrupted signal |
| `--injector fast` (default) | strictly disjoint segments, exact fraction |
| `--injector reference` | original `ts_corruptor.inject_swap`, for cross-checking only |
| `--no-clean` | skip the clean anchor |
