# Reorganization Plan — Thesis → Paper

Goal: clean structure + reproducibility, so all experiments can be re-run
deterministically for the paper.

Safety net: branch `reorg-cleanup`, restore point commit `faef59c`
(`git reset --hard faef59c` to undo everything).

---

## Phase 0 — Git safety net ✅ DONE
- Created branch `reorg-cleanup`.
- Snapshot commit `faef59c` of all code/scripts/figures/tables (343 files).
- Excluded heavy regenerable outputs (`results/`, `outputs/`) and garbage.

## Phase 1 — Delete real garbage ✅ DONE
- Removed 16 `~$*.docx` Word lock files, Edge cache
  (`edge_tmp_profile_section442/`), `win32com_gen_py/`, all `__pycache__/`,
  scratch dirs (`.tmp.drivedownload`, `.tmp_missing_plots`, `.tmp_prop_plots`,
  `.tmp_swap_imgs`).
- All 9 real `.docx` and all source code left intact.
- Left intentionally: `.tmp.driveupload/` (908 MB, Drive-managed),
  `~WRL3051.tmp` (locked by Word — delete after closing Word).

## Phase 2 — Tidy root scripts (TODO)
Move with `git mv` (nothing deleted):
- 20 root `plot_*_141.py` (final paper figure generators) → `src/experiments/plotting/paper/`
- analysis utils → `src/analysis/`:
  `compare_baseline_vs_benchmark.py`, `check_robust_subset_representativeness.py`,
  `block_audit_timeseries.py`, `compute_window_slide_stats_141.py`,
  `make_compound_v20_shapley_priority.py`
- `_winstats.py`, `_winstats2.py` → check if scratch; else `src/utils/`
- OPEN QUESTION: consolidate figures into one `figures/` or keep current dirs?

## Phase 3 — Separate deliverable figures (TODO)
Currently 4 image dirs (`regenerated_plots/`, `thesis_figures_updated/`,
`outputs/`, loose root `.png`). Decide on a single `figures/` layout per experiment.

## Phase 4 — Reproducibility layer (TODO — the key for the paper)
- `configs/` or `experiments.yaml`: seeds, 141-subset, per-experiment params in one place.
- `run_all.py` (or Makefile): single deterministic entry point
  (corruption → detection → metrics → figures).
- `requirements.txt` / environment freeze + `README.md` "how to reproduce".
- Fixed seeds everywhere.

### Phase 4b — Framework consolidation (DEFERRED — agreed to do later)
Finding: experiments are HALF integrated into the `ts_corruptor` framework.
- Shared backbone (data_loader, TSB_UAD models + vus.metrics) is uniform everywhere. ✅
- **Group A** (mature error types) go through `TSCorruptor` + `ts_corruptor.injectors`:
  freeze, spikes, swap, whitenoise_snr, missing_true_impact/masking/burst_length,
  gilbert_elliott_true_impact, combined_noise_missing, compound_corruptions.
- **Group B** (newer types) define corruption INLINE inside each script
  (not via injectors) — reproducibility risk (logic duplicated across
  experiment vs recovery scripts, seeds/params hardcoded):
  - `run_missing_mnar`, `run_missing_mnar_burst` (inline MNAR masking)
  - `run_propagation`, `run_propagation_multiscale`,
    `run_propagation_fixed_scaler_multiseed` (`inject_localized_*`)
  - `run_gradual_drift` (`apply_mean_drift_*`, `apply_variance_drift_*`)
  - `run_swap_permutation` (`permute_segments`)
  - `run_anomaly_aware_corruption`, `run_recovery_mnar(_burst)`

**TODO later:** lift Group B inline logic into `ts_corruptor.injectors`
(`inject_mnar`, `inject_propagated_*`, `inject_drift`, `inject_permutation`)
so every corruption type is defined in one place → identical logic in
experiment & recovery, deterministic seeds, trivial `run_all.py`.
Before refactor: diff Group B inline logic against any existing injectors to
check divergence.

## Phase 5 — Clean `.gitignore` + initial commit of organized structure (TODO)
