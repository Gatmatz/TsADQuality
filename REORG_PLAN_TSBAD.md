# Phase 6 — TSB-AD era cleanup (PLANNED, not executed)

Written 2026-08-22. Everything below is a **proposal**; nothing has been moved
or deleted yet. No commits will be made — the working tree is left dirty for
review.

Context: Phases 0–3 cleaned the *thesis-era* tree. Since then the project moved
from TSB-UAD to the **TSB-AD** benchmark, and a second layer of scratch files,
result dumps and misplaced outputs accumulated at the root.

## 6.0 — Fix the `.gitignore` bugs FIRST

These are not cosmetic: source code is currently being silently excluded from
git, and 11 MB of results is currently *not* excluded.

| Problem | Evidence | Fix |
|---|---|---|
| `outputs/*` hides **4 source scripts** | `git ls-files outputs` = 0 tracked; `git check-ignore` points at `.gitignore:11` | Move the scripts out (6.8), then keep the ignore |
| `eval_*/` **not** ignored | `git check-ignore eval_350` = no match; 2100 files, 11 MB would be committed | Covered by moving them under `results/` (6.3) |
| 34 untracked `.py` outside vendored libs | `git status --porcelain -uall` | After the moves, review and `git add` the real source |

Do this step first so nothing written later gets lost again.

## 6.1 — Delete (dead / duplicate)

| Path | Why safe |
|---|---|
| `tt.py` | **0 bytes** |
| `old_basic_metricor.py` | **0 bytes** |
| `src/experiments/corrupt_tsb-ad/` | its only file `run_whiteNoise_Snr_tsb_ad.py` is **0 bytes**; superseded by `corruption/run_whitenoise_snr_tsbad.py` (407 lines) |
| `results/*.csv` (33 files) + `results/results_checkpoint.csv` | abandoned partial run — **bit-identical md5** to `eval_thre_50/` (verified on sample); checkpoint holds only 60 rows |
| `test.py` | 9-line Optuna tutorial stub, unrelated to the thesis |
| `__pycache__/` (root, `src/`, `src/analysis/`, `src/ts_corruptor/`, `src/experiments/baselines/`) | regenerable |

**Not deleted** — `scratch_avg.py` (12 lines) is kept: it computes the
full-dataset TSB-UAD AUC-PR/ROC means for IForest/MP, a number that may appear
in the thesis text. It moves to the investigation folder instead.

## 6.2 — Root scratch scripts to `src/analysis/tsb_ad_investigation/`

These nine scripts *are* the audit trail for the **VUS version-drift finding**
(measured 0.34 vs leaderboard 0.30). That evidence belongs in the paper as a
footnote, so the code should be kept and named, not deleted.

```
prove_vus_bug.py          exhaustive_test.py       investigate_perfect.py
verify_github_code.py     compare_results.py       debug_tsb.py
scores.py                 scratch_avg.py
old_basic_metrics.py      new_basic_metrics.py     <- the two VUS versions being diffed
```

Also: `download_tsb_ad_data.py` to `src/data_prep/`

**Path anchors that break on the move** (must be patched in the same step):

| Script | Line | Current | Needs |
|---|---|---|---|
| `debug_tsb.py` | 6 | `dirname(abspath(__file__))` | 3 extra `dirname()` levels |
| `verify_github_code.py` | 7 | `dirname(abspath(__file__))` | 3 extra `dirname()` levels |
| `scores.py` | 6 | `dirname(abspath(__file__))` | 3 extra `dirname()` levels |
| `scratch_avg.py` | 2, 8 | hardcoded `C:/Users/gkost/...` | replace with repo-relative |

The rest use CWD-relative paths (`pd.read_csv('TSB-AD/...')`) and keep working
as long as they are launched from the repo root — unchanged behaviour.

## 6.3 — `eval_*/` to `results/baselines_tsbad/`

8 directories, 2100 files, ~11 MB, currently at the repo root.

```
eval_350/  eval_350_benchmark/  eval_350_raw/  eval_subset_350/
eval_subset_easy/  eval_subset_medium/  eval_subset_hard/  eval_thre_50/
        -> results/baselines_tsbad/<same name>/
```

Verified these are **real, distinct results**, not duplicates: all five
`results_final.csv` have different md5 and 700 rows each (350 files x 2 models).
They land under the already-ignored `results/`.

Add `results/baselines_tsbad/README.md` recording what distinguishes each
variant (`_raw` vs `_benchmark` vs `_thre_50` ...) — currently that lives only in
the directory names.

## 6.4 — `src/results/` to `results/baselines_tsbuad_legacy/`

Results must not live inside the source tree. Six subdirs
(`run_AE`, `run_IForest_Replicated`, `run_IForest_Wrapper`, `run_Sub_LOF`,
`run_Sub_LOF_AE`, `run_Sub_LOF_v2`).

Check afterwards: `scores.py` points at `results/baselines/run_Sub_PCA_350` —
confirm which of `results/baselines/` and `src/results/baselines/` it means.

## 6.5 — Subset definitions out of the vendored library, to `data/subsets/` (**tracked**)

These are **reproducibility inputs**, and they were written into the wrong
vendored library (they are TSB-**AD** subsets sitting in `TSB-UAD/result/`):

```
representative_subset_350.csv        representative_subset_350_resolved.csv
representative_subset_350_filenames.txt
representative_sample.csv            representative_sample_v2.csv
subset_easy.csv   subset_medium.csv   subset_hard.csv
        -> data/subsets/
```

`.gitignore` already has `!data/subsets/*`, so they become tracked — which is
what we want: the 200/350 subset is an input every corruption run depends on.

Then patch the writer `src/analysis/select_representative_subset_vuspr.py` and
any reader to point at `data/subsets/`.

## 6.6 — Split `src/experiments/baselines/` by benchmark

16 scripts, two benchmarks, no separation. Classified by imports:

```
baselines/tsb_ad/      Baseline_TSB_AD.py  Tutorial_IForest.py
                       Run_{AutoEncoder,AutoEncoder_Unsupervised,IForest_200,
                            KMeansAD,KShapeAD,MMPAD,POLY,StreamVAE,Sub_PCA}_*.py
                       baseline_ts_ad{,_benchmark,_raw}.py
                       baseline_tsb_uad_subset.py   (TSB_AD=3, despite the name)
baselines/tsb_uad/     mass_experiment_tsb_full_zscore.py   (the only pure TSB_UAD one)
```

Note `baseline_tsb_uad_subset.py` is **misnamed** — it imports TSB_AD. Rename to
`baseline_tsbad_subset.py` when moving.

Carry over the known-good/known-bad status from the earlier audit in a
`baselines/tsb_ad/README.md`, including the still-open item:

> **`Run_MMPAD_350.py` — UNFIXED.** HP is `{'periodicity':1,'backend':'gpu'}`,
> so `n_neighbor` falls back to 1 instead of the official 5. Should be
> `{**Optimal_Uni_algo_HP_dict['MMPAD'], 'backend':'gpu'}`.

## 6.7 — Thesis documents to `docs/thesis_drafts/`

7 files (`kostilis4408.docx`, `timeSeries_*.docx`, `timeSeries_v36_final.pdf`)
plus `changelog_20260516_1236.md`. Already gitignored via `*.docx` / `*.pdf`;
this is purely to empty the root.

## 6.8 — `outputs/` scripts to `src/analysis/docx_tooling/`

`create_clarity_sections_docx.py`, `export_and_render_clarity_sections.py`,
`export_clarity_sections_html.py`, `plot_feedback_v14/make_corrected_plots.py`
are **source code inside a gitignored directory** — see 6.0. Move them into
`src/` so they get tracked; `outputs/` then holds only generated artefacts.

## Net effect

| | before | after |
|---|---|---|
| root entries | 43 | ~16 |
| root scratch `.py` | 14 | 0 |
| result dirs at root | 8 | 0 |
| source files hidden by `.gitignore` | 4 | 0 |
| results inside `src/` | 6 dirs | 0 |

Nothing is renamed inside `src/ts_corruptor/`, `src/experiments/corruption/`,
`results/experiments/`, `figures/`, or the vendored `TSB-AD/` and `TSB-UAD/`
trees — so no experiment script changes behaviour.

## Execution order

1. 6.0 `.gitignore`, 6.1 deletes, 6.7 docs (zero risk, no code touched)
2. 6.3, 6.4 result dirs (data only)
3. 6.2, 6.8 script moves **+ path-anchor patches** (the one step that can break)
4. 6.5 subsets + patch writer/readers
5. 6.6 baselines split + READMEs
6. Smoke test: `python -c "import src.data_loader"`, then re-run
   `Tutorial_IForest.py` on 2-3 files and confirm VUS-PR is unchanged

## Phase 7 — Reproducibility layer (still TODO, unchanged from Phase 4)

`requirements.txt` (freeze of `venv/`), `README.md` with a "how to reproduce"
section, and a `MANIFEST.md` mapping every experiment to its script and results
dir.
