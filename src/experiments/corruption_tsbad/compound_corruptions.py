"""
Compound Corruptions — Multiple Simultaneous Data Quality Issues, TSB-AD version.

Port of run_compound_corruptions.py to the TSB-AD benchmark, built on the same
skeleton as missing_true_impact.py / freeze.py.

Question: when several data quality problems hit the same series at once, is the
damage the SUM of the individual damages, or more (synergistic) / less
(sub-additive)? Answering it needs three things measured under identical
conditions: the clean anchor, each corruption alone, and the combination.

Combinations (identical to the original):
  1. noise + missing          (noisy sensor + transmission loss)
  2. noise + spikes           (sensor degradation + transient outliers)
  3. spikes + missing         (outliers + data gaps)
  4. missing + freeze         (data loss + stuck sensor)
  5. noise + spikes + missing (worst realistic — the triple)
  6. noise + ge_missing       (noise + bursty loss via Gilbert-Elliott)

Application order (physically motivated, byte-identical to the original):
      freeze -> noise -> spikes -> missing
A sensor sticks first, noise rides on whatever it outputs, spikes are transient
events on top, and transmission loss is the last thing that happens to the bytes.

Evaluation — chosen per condition by whether NaNs actually reached the series:
  * NaNs present -> "true impact", no imputation. Drop the NaN points, map the
    scores back onto the original timeline, give lost ANOMALY points the minimum
    score (they count as false negatives), and EXCLUDE lost normal points.
  * No NaNs -> standard evaluation on the full corrupted series.
The branch is decided from the realised nan_mask, not from the declared condition,
so a `missing` level that happens to drop nothing still takes the correct path.

Detection and metrics use the OFFICIAL TSB-AD pipeline (run_Unsupervise_AD /
run_Semisupervise_AD + find_length_rank + TSB_AD get_metrics) on the 350-series
TSB-AD-U eval list. The `clean` anchor runs that pipeline with no corruption at
all, so it reduces exactly to the official TSB-AD baseline.

DELIBERATE DEVIATIONS from the TSB-UAD original — all documented, none accidental:

  1. Singles and the clean anchor are IMPORTED from the single-corruption runs, as in the
     original, but only from TSB-AD runs that share this runner's pipeline: same
     injector and parameters, same corruption seed and target, same evaluation path
     (see SINGLE_SOURCES). The original imported its spike singles from
     spikes_no_zscore while its compounds were z-scored, so those interaction terms
     mixed two preprocessings; on TSB-AD every runner uses the same official pipeline.
     Imported rows are restricted to the files each model's compounds were run on, so
     every term of the interaction is a mean over the same series, and are written to
     imported_singles.csv with their source. --compute-singles generates them in this
     run instead (identical for deterministic models; for stochastic ones, e.g.
     KMeansAD_U, the per-job seed depends on the condition name). Condition NAMES are
     unchanged (`clean`, `<type>_<sev>_only`, `a+b`), so tables group exactly as before.
  2. No max(window, 10) floor on the metric window, and no MinMax on the scores —
     the same two deviations as every other TSB-AD port here, so the clean anchor
     reproduces the official baseline bit for bit. Rank-based metrics are
     invariant to the dropped MinMax.
  3. Lost anomalies get score.min(), not a literal 0.0. TSB-AD returns
     UNNORMALIZED scores (IForest sits around [-0.06, +0.03] with most points
     below zero), so a hardcoded 0.0 would rank a destroyed anomaly in the top
     ~10% and INFLATE the metrics as more anomalies are lost. score.min() carries
     the original's semantics — tied last with the least anomalous point.
  4. Interaction and Shapley analysis iterate the severity levels actually
     declared in SEVERITY instead of a hardcoded ['low','med','high'], so the
     noise 'extreme' (0 dB) arm is analysed too. This only ADDS rows; every row
     the original would have produced is still produced identically.

Usage:
    python src/experiments/corruption_tsbad/compound_corruptions.py --test
    python src/experiments/corruption_tsbad/compound_corruptions.py --models IForest --workers 4
    # the singles come from white_noise_snr, missing_true_impact, spikes, freeze and
    # gilbert_elliott_true_impact: run those first for the same models and files
    # the grid is large — start with one pair to size the run:
    python src/experiments/corruption_tsbad/compound_corruptions.py \
        --models IForest --combinations noise_missing
    # or run on a smaller validated file list:
    python src/experiments/corruption_tsbad/compound_corruptions.py \
        --models IForest --files-csv results/tables/representative_subset_tsb_ad_vuspr_n200.csv
"""

import itertools
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tsbad.env  # noqa: E402,F401  thread caps — must come before numpy

import pandas as pd  # noqa: E402

import ts_corruptor.injectors  # noqa: E402
from ts_corruptor.core import TSCorruptor  # noqa: E402
from tsbad import paths  # noqa: E402
from tsbad.checkpoint import load_all_for_summary  # noqa: E402
from tsbad.harness import Spec, condition, run_sweep  # noqa: E402

# ==========================================
# CONFIG
# ==========================================
N_SEEDS = 1

# The original passes no corruption_target, i.e. the default. Corruption may therefore land on
# anomaly points — deliberately: destroying anomalies is part of what this experiment measures.
CORRUPTION_TARGET = 'global'

# Severity levels per corruption type — byte-identical to the original.
# Noise carries a fourth 'extreme' level (0 dB) that the others do not.
SEVERITY = {
    'noise':   {'low': {'snr_db': 20},
                'med': {'snr_db': 10},
                'high': {'snr_db': 5},
                'extreme': {'snr_db': 0}},
    'missing': {'low': {'fraction': 0.05},
                'med': {'fraction': 0.10},
                'high': {'fraction': 0.20}},
    'spikes':  {'low': {'fraction': 0.05, 'multiplier': 3, 'sequential': False, 'sequence_length': 1},
                'med': {'fraction': 0.10, 'multiplier': 3, 'sequential': False, 'sequence_length': 1},
                'high': {'fraction': 0.10, 'multiplier': 10, 'sequential': False, 'sequence_length': 1}},
    'freeze':  {'low': {'num_stucks': 3, 'freeze_fraction': 0.05},
                'med': {'num_stucks': 5, 'freeze_fraction': 0.10},
                'high': {'num_stucks': 5, 'freeze_fraction': 0.20}},
    'ge_missing': {'low': {'alpha': 0.01, 'beta': 0.25},    # ~3.8% missing, burst ~4
                   'med': {'alpha': 0.01, 'beta': 0.10},    # ~9.1% missing, burst ~10
                   'high': {'alpha': 0.05, 'beta': 0.10}},  # ~33% missing, burst ~10
}

COMBINATIONS = [
    ('noise_missing',        ['noise', 'missing']),
    ('noise_spikes',         ['noise', 'spikes']),
    ('spikes_missing',       ['spikes', 'missing']),
    ('missing_freeze',       ['missing', 'freeze']),
    ('noise_spikes_missing', ['noise', 'spikes', 'missing']),
    ('noise_ge_missing',     ['noise', 'ge_missing']),
]

# Physical order: sensor sticks -> noise added -> spikes occur -> data lost
APPLICATION_ORDER = {'freeze': 0, 'noise': 1, 'spikes': 2, 'missing': 3, 'ge_missing': 3}

# Corruption types that introduce NaNs (i.e. trigger the true-impact evaluation branch)
MISSING_TYPES = {'missing', 'ge_missing'}

# Where each single and the clean anchor are imported from. Every source applies the same
# injector with the same parameters, corruption seed and target, and takes the same evaluation
# path as the compound runner does for that single (full series without NaNs, true impact with).
# Condition names are those the source runners write.
SINGLE_SOURCES = {
    'noise':      ('white_noise_snr_tsbad',
                   lambda p: f"snr_{p['snr_db']}dB"),
    'missing':    ('missing_true_impact_tsbad',
                   lambda p: f"point_frac_{p['fraction']}"),
    'spikes':     ('spikes_tsbad',
                   lambda p: f"frac_{p['fraction']}_mult_{float(p['multiplier'])}_point"),
    'freeze':     ('freeze_tsbad',
                   lambda p: f"frac_{p['freeze_fraction']}_ns_{p['num_stucks']}"),
    'ge_missing': ('gilbert_elliott_true_impact_tsbad',
                   lambda p: f"a_{p['alpha']}_b_{p['beta']}"),
}
# Full-series experiments whose clean anchor matches this runner's (no NaNs -> full evaluation,
# no window clamp). The first one that has the (file, model) row is used.
CLEAN_SOURCES = ('freeze_tsbad', 'spikes_tsbad', 'swap_point_tsbad', 'swap_segment_tsbad',
                 'swap_permutation_tsbad', 'spikes_normal_only_tsbad', 'white_noise_snr_tsbad')


# ==========================================
# CONDITION BUILDER
# ==========================================

def _tag(ctype, sev):
    """Canonical severity tag, e.g. 'noise_high'. Used inside every condition name."""
    return f"{ctype}_{sev}"


def build_grid(combos=None, include_clean=True, include_singles=True):
    """Build clean anchor + singles + compounds.

    Unlike the original — which built compounds only and imported the rest from other
    experiments' checkpoints — every term of the interaction equation is produced here, under
    one window policy and one evaluation policy. Names are unchanged, so downstream tables and
    the interaction/Shapley code group exactly as before.
    """
    selected = COMBINATIONS if combos is None else [c for c in COMBINATIONS if c[0] in combos]
    conditions = []

    if include_clean:
        conditions.append({
            'name': 'clean',
            'combination_name': 'baseline',
            'condition_type': 'baseline',
            'corruptions': [],
            'has_missing': False,
            'single_tag': None,
        })

    # Singles: every (type, severity) pair that appears anywhere in the selected combinations.
    # Needed for the drop terms — a compound is only interpretable against its own parts.
    if include_singles:
        seen = set()
        for _, ctypes in selected:
            for ctype in ctypes:
                for sev in SEVERITY[ctype]:
                    if (ctype, sev) in seen:
                        continue
                    seen.add((ctype, sev))
                    conditions.append({
                        'name': f"{_tag(ctype, sev)}_only",
                        'combination_name': 'single',
                        'condition_type': 'single',
                        'corruptions': [{'type': ctype, 'severity': sev,
                                         'params': SEVERITY[ctype][sev].copy()}],
                        'has_missing': ctype in MISSING_TYPES,
                        'single_tag': _tag(ctype, sev),
                    })

    # Compounds: the full severity cross-product per combination
    for combo_name, ctypes in selected:
        per_type_levels = [list(SEVERITY[ct].keys()) for ct in ctypes]
        for sev_combo in itertools.product(*per_type_levels):
            corr_list, name_parts, has_missing = [], [], False
            for ctype, sev in zip(ctypes, sev_combo):
                corr_list.append({'type': ctype, 'severity': sev,
                                  'params': SEVERITY[ctype][sev].copy()})
                name_parts.append(_tag(ctype, sev))
                if ctype in MISSING_TYPES:
                    has_missing = True
            conditions.append({
                'name': '+'.join(name_parts),
                'combination_name': combo_name,
                'condition_type': 'compound',
                'corruptions': corr_list,
                'has_missing': has_missing,
                'single_tag': None,
            })

    return conditions


# ==========================================
# CORRUPTION APPLICATION
# ==========================================

def apply_corruptions(corruptor, condition, series_length):
    """Apply every corruption of a condition, in canonical physical order.

    Order is by APPLICATION_ORDER, not by list position, so `noise+missing` and `missing+noise`
    would corrupt identically. Each injector mutates the same corruptor, i.e. later injectors see
    the output of the earlier ones — that stacking IS the compound effect being measured.
    """
    for corr in sorted(condition['corruptions'], key=lambda c: APPLICATION_ORDER[c['type']]):
        ctype = corr['type']
        params = corr['params'].copy()

        if ctype == 'noise':
            ts_corruptor.injectors.inject_white_noise_snr(corruptor, **params)
        elif ctype == 'missing':
            ts_corruptor.injectors.inject_point_missing(corruptor, **params)
        elif ctype == 'spikes':
            ts_corruptor.injectors.inject_spikes(corruptor, **params)
        elif ctype == 'freeze':
            # Same dynamic block length as run_freeze.py / freeze.py: the requested
            # fraction of the series split evenly across num_stucks frozen blocks.
            freeze_frac = params.pop('freeze_fraction')
            num_stucks = params['num_stucks']
            stuck_length = max(1, int(freeze_frac * series_length / num_stucks))
            ts_corruptor.injectors.inject_sensor_stuck(
                corruptor, num_stucks=num_stucks, stuck_length=stuck_length)
        elif ctype == 'ge_missing':
            ts_corruptor.injectors.inject_gilbert_elliott(
                corruptor, p_good_to_bad=params['alpha'],
                p_bad_to_good=params['beta'], noise_type='missing')
        else:
            raise ValueError(f'unknown corruption type: {ctype}')


def _extract_params(condition):
    """Flatten a condition's corruption parameters into result columns."""
    out = {'noise_snr_db': None, 'missing_fraction': 0.0,
           'spikes_fraction': 0.0, 'spikes_multiplier': 0.0,
           'freeze_num_stucks': 0, 'freeze_fraction': 0.0,
           'ge_alpha': None, 'ge_beta': None}
    for corr in condition['corruptions']:
        p, ctype = corr['params'], corr['type']
        if ctype == 'noise':
            out['noise_snr_db'] = p.get('snr_db')
        elif ctype == 'missing':
            out['missing_fraction'] = p.get('fraction', 0.0)
        elif ctype == 'spikes':
            out['spikes_fraction'] = p.get('fraction', 0.0)
            out['spikes_multiplier'] = p.get('multiplier', 0.0)
        elif ctype == 'freeze':
            out['freeze_num_stucks'] = p.get('num_stucks', 0)
            out['freeze_fraction'] = p.get('freeze_fraction', 0.0)
        elif ctype == 'ge_missing':
            out['ge_alpha'] = p.get('alpha')
            out['ge_beta'] = p.get('beta')
    return out


# ==========================================
# INTERACTION ANALYSIS
# ==========================================

def _classify(interaction):
    """Damage beyond the additive prediction is synergistic; below it, sub-additive.

    The +-0.01 AUC dead zone is the original's, kept so the three labels stay comparable.
    """
    if interaction > 0.01:
        return 'synergistic'
    if interaction < -0.01:
        return 'sub-additive'
    return 'additive'


def compute_interaction_analysis(df_results, output_dir, metric='AUC_ROC'):
    """Is compound damage additive, synergistic, or sub-additive?

    For each severity cross of each combination:
        drop(X)      = AUC(clean) - AUC(X alone)
        predicted    = sum of the individual drops
        interaction  = drop(compound) - predicted
    A positive interaction means the corruptions hurt each other's detectability more than
    their separate damages would suggest.
    """
    if metric not in df_results.columns:
        print(f"[Interaction] metric {metric} not in results — skipping.")
        return
    ok = df_results[df_results['error'].isnull()]
    if ok.empty:
        return

    # Mean over files, per (condition, model)
    mean_auc = ok.groupby(['condition', 'model'])[metric].mean().to_dict()
    models = sorted(ok['model'].dropna().unique())
    baseline = {m: mean_auc[('clean', m)] for m in models if ('clean', m) in mean_auc}

    if not baseline:
        print("[Interaction] no clean baseline found — skipping.")
        return

    rows = []
    for combo_name, ctypes in COMBINATIONS:
        # Severity levels come from SEVERITY, so the noise 'extreme' arm is covered too.
        for sev_combo in itertools.product(*[list(SEVERITY[ct].keys()) for ct in ctypes]):
            tags = [_tag(ct, sv) for ct, sv in zip(ctypes, sev_combo)]
            compound_name = '+'.join(tags)

            for model in models:
                bl = baseline.get(model)
                auc_singles = [mean_auc.get((f"{t}_only", model)) for t in tags]
                auc_compound = mean_auc.get((compound_name, model))
                if bl is None or auc_compound is None or any(a is None for a in auc_singles):
                    continue

                drops = [bl - a for a in auc_singles]
                drop_compound = bl - auc_compound
                predicted = sum(drops)
                interaction = drop_compound - predicted
                pct = (interaction / predicted * 100) if abs(predicted) > 1e-6 else 0.0

                row = {'combination_name': combo_name,
                       'condition': compound_name,
                       'n_corruptions': len(tags),
                       'model': model,
                       'baseline_auc': round(bl, 4),
                       'auc_compound': round(auc_compound, 4),
                       'drop_compound': round(drop_compound, 4),
                       'predicted_additive': round(predicted, 4),
                       'interaction': round(interaction, 4),
                       'interaction_pct': round(pct, 1),
                       'interaction_type': _classify(interaction)}
                for i, (t, a, d) in enumerate(zip(tags, auc_singles, drops)):
                    letter = chr(ord('A') + i)
                    row[f'severity_{letter}'] = t
                    row[f'auc_{letter}'] = round(a, 4)
                    row[f'drop_{letter}'] = round(d, 4)
                rows.append(row)

    if not rows:
        print("[Interaction] no complete compound/single/baseline triples found — skipping.")
        return

    df_inter = pd.DataFrame(rows)
    df_inter.to_csv(os.path.join(output_dir, "interaction_analysis.csv"), index=False)
    print(f"Interaction analysis: {len(df_inter)} rows -> interaction_analysis.csv")

    summary = df_inter.groupby(['combination_name', 'model']).agg(
        mean_interaction=('interaction', 'mean'),
        mean_interaction_pct=('interaction_pct', 'mean'),
        n_synergistic=('interaction_type', lambda x: (x == 'synergistic').sum()),
        n_additive=('interaction_type', lambda x: (x == 'additive').sum()),
        n_subadditive=('interaction_type', lambda x: (x == 'sub-additive').sum()),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "interaction_summary.csv"), index=False)

    print(f"\n=== Interaction summary ({metric}) ===")
    print(f"{'Combination':<24} {'Model':<14} {'mean int':>10} {'syn':>5} {'add':>5} {'sub':>5}")
    print("-" * 68)
    for _, r in summary.sort_values(['model', 'mean_interaction'], ascending=[True, False]).iterrows():
        print(f"{r['combination_name']:<24} {r['model']:<14} {r['mean_interaction']:>+10.4f} "
              f"{int(r['n_synergistic']):>5} {int(r['n_additive']):>5} {int(r['n_subadditive']):>5}")


# ==========================================
# SHAPLEY VALUE ABLATION
# ==========================================

def compute_shapley_analysis(df_results, output_dir, metric='AUC_ROC'):
    """Shapley values for the triple (noise + spikes + missing).

    Each corruption's average marginal contribution to total damage across all coalitions,
    answering the practical question: which problem should you fix FIRST for the biggest
    recovery? Needs the baseline, all 3 singles, all 3 pairs and the triple — every one of
    which this script produces, so no cross-experiment lookup can go stale.
    """
    if metric not in df_results.columns:
        return
    ok = df_results[df_results['error'].isnull()]
    if ok.empty:
        return

    mean_auc = ok.groupby(['condition', 'model'])[metric].mean().to_dict()
    models = sorted(ok['model'].dropna().unique())
    baseline = {m: mean_auc[('clean', m)] for m in models if ('clean', m) in mean_auc}
    if not baseline:
        print("[Shapley] no clean baseline found — skipping.")
        return

    players = ['noise', 'spikes', 'missing']
    n_players = len(players)
    rows = []

    for sev_combo in itertools.product(*[list(SEVERITY[p].keys()) for p in players]):
        sevs = dict(zip(players, sev_combo))
        tag = {p: _tag(p, sevs[p]) for p in players}

        # Coalition -> condition name. Pair names are spelled in the order the types appear in
        # COMBINATIONS, which is exactly how build_conditions() joins them.
        conditions = {
            frozenset():                    'clean',
            frozenset(['noise']):           f"{tag['noise']}_only",
            frozenset(['spikes']):          f"{tag['spikes']}_only",
            frozenset(['missing']):         f"{tag['missing']}_only",
            frozenset(['noise', 'spikes']):   f"{tag['noise']}+{tag['spikes']}",
            frozenset(['noise', 'missing']):  f"{tag['noise']}+{tag['missing']}",
            frozenset(['spikes', 'missing']): f"{tag['spikes']}+{tag['missing']}",
            frozenset(players):             f"{tag['noise']}+{tag['spikes']}+{tag['missing']}",
        }

        for model in baseline:
            aucs, skip = {}, False
            for coalition, cond_name in conditions.items():
                auc = mean_auc.get((cond_name, model))
                if auc is None:
                    skip = True
                    break
                aucs[coalition] = auc
            if skip:
                continue

            # Damage function v(S) = AUC(clean) - AUC(S)
            v = {s: baseline[model] - a for s, a in aucs.items()}
            total_damage = v[frozenset(players)]

            for corr in players:
                others = [c for c in players if c != corr]
                shapley = 0.0
                for size in range(len(others) + 1):
                    for subset in itertools.combinations(others, size):
                        s = frozenset(subset)
                        marginal = v[s | {corr}] - v[s]
                        weight = (math.factorial(len(s))
                                  * math.factorial(n_players - len(s) - 1)
                                  / math.factorial(n_players))
                        shapley += weight * marginal

                # Recovery: what you gain by removing just this corruption from the triple
                recovery = aucs[frozenset(players) - {corr}] - aucs[frozenset(players)]

                rows.append({
                    'noise_severity': sevs['noise'],
                    'spikes_severity': sevs['spikes'],
                    'missing_severity': sevs['missing'],
                    'model': model,
                    'corruption': corr,
                    'corruption_severity': sevs[corr],
                    'shapley_value': round(shapley, 4),
                    'recovery_if_fixed': round(recovery, 4),
                    'total_damage': round(total_damage, 4),
                    'shapley_pct': (round(shapley / total_damage * 100, 1)
                                    if total_damage > 0.001 else 0.0),
                })

    if not rows:
        print("[Shapley] no complete coalitions found — skipping.")
        return

    df_sh = pd.DataFrame(rows)
    df_sh.to_csv(os.path.join(output_dir, "shapley_ablation.csv"), index=False)
    print(f"Shapley ablation: {len(df_sh)} rows -> shapley_ablation.csv")

    summary = df_sh.groupby(['corruption', 'model']).agg(
        mean_shapley=('shapley_value', 'mean'),
        mean_recovery=('recovery_if_fixed', 'mean'),
        mean_pct=('shapley_pct', 'mean'),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "shapley_summary.csv"), index=False)

    print(f"\n=== Shapley summary ({metric}) — mean contribution to damage ===")
    print(f"{'Corruption':<12} {'Model':<14} {'Shapley':>9} {'Recovery':>10} {'% damage':>10}")
    print("-" * 60)
    for _, r in summary.sort_values(['model', 'mean_shapley'], ascending=[True, False]).iterrows():
        print(f"{r['corruption']:<12} {r['model']:<14} {r['mean_shapley']:>9.4f} "
              f"{r['mean_recovery']:>+10.4f} {r['mean_pct']:>9.1f}%")

# ==========================================
# IMPORTED SINGLES
# ==========================================

def import_singles(df_compounds, source_root):
    """Clean anchor + single-corruption rows from the single-corruption runs.

    Restricted, per model, to the files that model's compounds were run on. Returns the rows
    relabelled with this runner's condition names, plus a `source_experiment` /
    `source_condition` provenance pair.
    """
    ok = df_compounds[df_compounds['error'].isnull()]
    files = {m: set(g['file']) for m, g in ok.groupby('model')}
    cache = {}

    def source(experiment):
        if experiment not in cache:
            d = load_all_for_summary(os.path.join(source_root, experiment))
            cache[experiment] = d[d['error'].isnull()] if not d.empty else d
        return cache[experiment]

    frames = []
    for model, model_files in files.items():
        # clean anchor: first full-series source that has the (file, model) row
        needed = set(model_files)
        for experiment in CLEAN_SOURCES:
            d = source(experiment)
            if d.empty or not needed:
                continue
            hit = d[(d['condition'] == 'clean') & (d['model'] == model) & d['file'].isin(needed)]
            hit = hit.drop_duplicates('file')
            if len(hit):
                frames.append(hit.assign(condition='clean', combination_name='baseline',
                                         condition_type='baseline', source_experiment=experiment,
                                         source_condition='clean'))
                needed -= set(hit['file'])

        for ctype, levels in SEVERITY.items():
            experiment, name_of = SINGLE_SOURCES[ctype]
            d = source(experiment)
            for sev, params in levels.items():
                src_condition = name_of(params)
                if d.empty:
                    continue
                hit = d[(d['condition'] == src_condition) & (d['model'] == model)
                        & d['file'].isin(model_files)].drop_duplicates('file')
                if len(hit):
                    frames.append(hit.assign(condition=f"{_tag(ctype, sev)}_only",
                                             combination_name='single', condition_type='single',
                                             source_experiment=experiment,
                                             source_condition=src_condition))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def report_coverage(imported, df_compounds):
    """Print, per model, how many of the compound files each imported term covers."""
    ok = df_compounds[df_compounds['error'].isnull()]
    terms = ['clean'] + [f"{_tag(ct, sv)}_only" for ct, lv in SEVERITY.items() for sv in lv]
    for model, g in ok.groupby('model'):
        n_files = g['file'].nunique()
        have = (imported[imported['model'] == model].groupby('condition')['file'].nunique()
                if not imported.empty else pd.Series(dtype=int))
        missing = [t for t in terms if have.get(t, 0) == 0]
        partial = [f"{t} ({have[t]}/{n_files})" for t in terms if 0 < have.get(t, 0) < n_files]
        print(f"[Singles] {model}: {len(terms) - len(missing)}/{len(terms)} terms imported for "
              f"{n_files} compound files")
        if missing:
            print(f"          not found (compounds that need them are skipped): {missing}")
        if partial:
            print(f"          partial file coverage: {partial}")


# ==========================================
# HARNESS HOOKS
# ==========================================

# --test keeps two files and this handful of conditions: one of each single type plus a
# compound with and without deleted points
TEST_CONDITIONS = {'clean', 'noise_low_only', 'missing_low_only', 'freeze_low_only',
                   'noise_low+missing_low', 'noise_high+missing_high', 'missing_low+freeze_low'}


def add_args(ap):
    ap.add_argument('--combinations', nargs='+', default=None,
                    choices=[c[0] for c in COMBINATIONS],
                    help='Run only these combinations (default: all six)')
    ap.add_argument('--compute-singles', action='store_true',
                    help='Run the clean anchor and the single-corruption arms here instead of '
                         'importing them from the single-corruption experiments')
    ap.add_argument('--singles-root', default=None,
                    help='Directory holding the <experiment>_tsbad folders the singles are '
                         'imported from (default: results/experiments)')
    ap.add_argument('--interaction-metric', default='AUC_ROC',
                    help='Metric the interaction/Shapley analysis is computed on '
                         '(default AUC_ROC, as in the original; VUS_PR is TSB-AD\'s headline)')


def build_conditions(args):
    grid = build_grid(combos=args.combinations,
                      include_clean=args.compute_singles and not args.no_clean,
                      include_singles=args.compute_singles)
    if args.test:
        grid = [c for c in grid if c['name'] in TEST_CONDITIONS]
    return [condition(c['name'],
                      {'combination_name': c['combination_name'],
                       'condition_type': c['condition_type'],
                       **_extract_params(c),
                       'has_missing': c['has_missing']},
                      params=c, n_seeds=N_SEEDS)
            for c in grid]


def before_run(args, conditions):
    types = [c['row']['condition_type'] for c in conditions]
    print("  Application order: freeze -> noise -> spikes -> missing")
    print("  NaNs present -> true impact (lost anomaly = min score, lost normal excluded)")
    print("  No NaNs      -> standard evaluation on the full corrupted series")
    print(f"Conditions: {len(conditions)}  (baseline: {types.count('baseline')}, "
          f"singles: {types.count('single')}, compounds: {types.count('compound')})")
    print(f"Combinations: {args.combinations or [c[0] for c in COMBINATIONS]}")
    print(f"corruption_target: {CORRUPTION_TARGET} | interaction metric: {args.interaction_metric}")
    if args.compute_singles:
        print("Singles and clean anchor: computed in this run")
    else:
        print(f"Singles and clean anchor: imported from "
              f"{args.singles_root or paths.RESULTS_ROOT} ({', '.join(sorted({s for s, _ in SINGLE_SOURCES.values()}))})")
    print("  Large grid? --combinations / --files-csv / --max-files cut it down, and the run "
          "is resumable.\n")


def corrupt(ctx, cond):
    """Every corruption of this condition, in physical order (nothing for the clean anchor)."""
    if not cond['corruptions']:
        return ctx.clean_data, {}
    corruptor = TSCorruptor(ctx.df.copy(), value_col=ctx.value_col, label_col=ctx.label_col,
                            seed=ctx.seed, corruption_target=CORRUPTION_TARGET)
    apply_corruptions(corruptor, cond, ctx.n)
    return corruptor.get_corrupted_df().iloc[:, 0:-1].values.astype(float), {}


def post_summary(df_all, results_dir, args):
    if not args.compute_singles:
        in_run = df_all['condition_type'].isin(['baseline', 'single'])
        if in_run.any():
            print(f"[Singles] ignoring {int(in_run.sum())} in-run clean/single rows "
                  f"(pass --compute-singles to analyse those instead)")
        compounds = df_all[~in_run]
        source_root = str(paths.resolve(args.singles_root)) if args.singles_root else str(paths.RESULTS_ROOT)
        imported = import_singles(compounds, source_root)
        report_coverage(imported, compounds)
        if not imported.empty:
            imported.to_csv(os.path.join(results_dir, "imported_singles.csv"), index=False)
        df_all = pd.concat([compounds, imported], ignore_index=True)
    compute_interaction_analysis(df_all, results_dir, args.interaction_metric)
    compute_shapley_analysis(df_all, results_dir, args.interaction_metric)


SPEC = Spec(
    name='compound_corruptions_tsbad',
    title='Compound corruptions — TSB-AD',
    family='survivors',
    # the survivor path (drop, clamp, true-impact remap) only where points were actually lost;
    # conditions without NaNs are evaluated on the full corrupted series
    survivor_rules='if_dropped',
    build_conditions=build_conditions,
    corrupt=corrupt,
    add_args=add_args,
    before_run=before_run,
    post_summary=post_summary,
    summary_keys=('condition', 'combination_name', 'condition_type', 'model'),
    summary_means=(('actual_missing_rate', 4), ('n_lost_anomalies', 4), ('n_kept', 4)),
    test_help='Smoke test: 2 files, tiny grid',
    test_files=2,
    progress_desc='compound',
)

if __name__ == "__main__":
    run_sweep(SPEC)
