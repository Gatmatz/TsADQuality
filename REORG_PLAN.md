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

## Phase 2 — Tidy root scripts ✅ DONE
- 20 root `plot_*_141.py` + their PNGs → `figures/` (commit 5d81234),
  patched for location-independent paths.
- 9 analysis/docx-tooling scripts (5 root + 2 `scripts/` + `_winstats*` pair)
  → `src/analysis/` (commit be06cd2). Path anchors patched where the move
  broke them; CWD-relative scripts still run from repo root.

## Phase 3 — Consolidate figures ✅ DONE
- `regenerated_plots/` deleted (60 PNGs, regenerable via
  `src/experiments/plotting/`, recoverable from faef59c) — commit 6864172.
- `thesis_figures_updated/` → `figures/thesis/`; loose root PNG → `figures/`
  (commit 7ad0ee9).
- `.gitignore` now ignores `results/` + `outputs/` wholesale, EXCEPT
  `results/tables/robust_subset_TSB.csv` (141-subset definition, tracked).

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
