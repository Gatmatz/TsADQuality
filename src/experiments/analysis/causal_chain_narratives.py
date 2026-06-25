"""
Causal Chain Narratives — WHY each corruption affects each model

Synthesizes existing analysis results into structured causal chains:
  Step 1: CORRUPTION → what changes in the data
  Step 2: INTERNAL MECHANISM → what changes inside the algorithm
  Step 3: ERROR TYPE → FP increase, FN increase, or both
  Step 4: PROPAGATION → local or global impact on scores
  Step 5: METRIC DROP → final AUC/precision/recall numbers

Sources (no new experiments needed):
  - internal_analysis/aggregate_summary.csv
  - fp_fn_decomposition/fp_fn_by_corruption.csv
  - propagation/cross_analysis/*
  - unified/ranking_inversion.csv
  - unified/cri_table.csv
  - segment_vulnerability/bin_detection_drop.csv

Output:
  results/analysis/causal_chains/
    - causal_chain_full.csv         (structured per corruption x model)
    - causal_chain_summary.csv      (one-line per combination)
    - causal_chain_narratives.txt   (readable text for thesis)
    - mechanism_heatmap.png         (visual summary)

Usage:
    python causal_chain_narratives.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "analysis", "causal_chains")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# DATA LOADING
# ==========================================

def load_internal_analysis():
    """Load aggregate internal metrics per corruption x model."""
    path = os.path.join(PROJECT_ROOT, "results", "experiments", "internal_analysis", "aggregate_summary.csv")
    if not os.path.exists(path):
        print(f"  [WARN] Internal analysis not found: {path}")
        return None
    df = pd.read_csv(path)
    return df


def load_fp_fn():
    """Load FP/FN decomposition per corruption x model."""
    path = os.path.join(PROJECT_ROOT, "results", "experiments", "fp_fn_decomposition", "fp_fn_by_corruption.csv")
    if not os.path.exists(path):
        print(f"  [WARN] FP/FN decomposition not found: {path}")
        return None
    return pd.read_csv(path)


def load_propagation():
    """Load propagation profiles and classification."""
    profiles_path = os.path.join(PROJECT_ROOT, "results", "experiments", "propagation",
                                  "cross_analysis", "zone_decay_profiles.csv")
    class_path = os.path.join(PROJECT_ROOT, "results", "experiments", "propagation",
                               "cross_analysis", "model_classification.csv")
    agg_path = os.path.join(PROJECT_ROOT, "results", "experiments", "propagation",
                             "cross_analysis", "aggregate_comparison.csv")

    profiles = pd.read_csv(profiles_path) if os.path.exists(profiles_path) else None
    classification = pd.read_csv(class_path) if os.path.exists(class_path) else None
    aggregate = pd.read_csv(agg_path) if os.path.exists(agg_path) else None
    return profiles, classification, aggregate


def load_ranking_inversion():
    """Load ranking inversion data."""
    path = os.path.join(PROJECT_ROOT, "results", "analysis", "unified", "ranking_inversion.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_cri():
    """Load CRI (Corruption Robustness Index) table."""
    path = os.path.join(PROJECT_ROOT, "results", "analysis", "unified", "cri_table.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_causal_chain_data():
    """Load the existing causal_chain.csv from unified analysis."""
    path = os.path.join(PROJECT_ROOT, "results", "analysis", "unified", "causal_chain.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_segment_vulnerability():
    """Load segment vulnerability summary."""
    path = os.path.join(PROJECT_ROOT, "results", "experiments", "segment_vulnerability", "bin_detection_drop.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


# ==========================================
# INTERNAL MECHANISM EXTRACTION
# ==========================================

def extract_iforest_mechanism(internal_df, corruption_prefix):
    """Extract IForest-specific mechanism: separation gap change."""
    rows = internal_df[
        (internal_df['model'] == 'IForest') &
        (internal_df['corruption'].str.startswith(corruption_prefix))
    ]
    if rows.empty:
        return None

    # Get the moderate and severe conditions
    results = []
    for _, row in rows.iterrows():
        gap_clean = row.get('separation_gap_clean_mean', np.nan)
        gap_corrupted = row.get('separation_gap_corrupted_mean', np.nan)
        gap_change = row.get('separation_gap_change_pct_mean', np.nan)

        results.append({
            'condition': row['corruption'],
            'gap_clean': gap_clean,
            'gap_corrupted': gap_corrupted,
            'gap_change_pct': gap_change,
        })

    return pd.DataFrame(results)


def extract_lof_mechanism(internal_df, corruption_prefix):
    """Extract LOF-specific mechanism: k-distance gap change."""
    rows = internal_df[
        (internal_df['model'] == 'LOF') &
        (internal_df['corruption'].str.startswith(corruption_prefix))
    ]
    if rows.empty:
        return None

    results = []
    for _, row in rows.iterrows():
        results.append({
            'condition': row['corruption'],
            'kdist_gap_clean': row.get('kdist_gap_clean_mean', np.nan),
            'kdist_gap_corrupted': row.get('kdist_gap_corrupted_mean', np.nan),
            'kdist_gap_change_pct': row.get('kdist_gap_change_pct_mean', np.nan),
        })
    return pd.DataFrame(results)


def extract_mp_mechanism(internal_df, corruption_prefix):
    """Extract MP-specific mechanism: nn distance gap + neighbor change."""
    rows = internal_df[
        (internal_df['model'] == 'MP') &
        (internal_df['corruption'].str.startswith(corruption_prefix))
    ]
    if rows.empty:
        return None

    results = []
    for _, row in rows.iterrows():
        results.append({
            'condition': row['corruption'],
            'nn_dist_gap_clean': row.get('nn_dist_gap_clean_mean', np.nan),
            'nn_dist_gap_corrupted': row.get('nn_dist_gap_corrupted_mean', np.nan),
            'nn_dist_gap_change_pct': row.get('nn_dist_gap_change_pct_mean', np.nan),
            'pct_nn_changed': row.get('pct_nn_changed_mean', np.nan),
        })
    return pd.DataFrame(results)


def extract_ae_mechanism(internal_df, corruption_prefix):
    """Extract AE-specific mechanism: error ratio change."""
    rows = internal_df[
        (internal_df['model'] == 'AE') &
        (internal_df['corruption'].str.startswith(corruption_prefix))
    ]
    if rows.empty:
        return None

    results = []
    for _, row in rows.iterrows():
        results.append({
            'condition': row['corruption'],
            'error_ratio_clean': row.get('error_ratio_clean_mean', np.nan),
            'error_ratio_corrupted': row.get('error_ratio_corrupted_mean', np.nan),
            'error_ratio_change_pct': row.get('error_ratio_change_pct_mean', np.nan),
        })
    return pd.DataFrame(results)


# ==========================================
# NARRATIVE GENERATION
# ==========================================

# Corruption type descriptions
CORRUPTION_DESCRIPTIONS = {
    'noise': {
        'name': 'White Noise',
        'data_change': 'Additive Gaussian noise across the entire series, controlled by SNR (dB). '
                       'Lower SNR = more noise relative to signal.',
        'what_changes': 'All values are perturbed; the distribution broadens; local patterns are obscured.',
    },
    'spikes': {
        'name': 'Point Outliers (Spikes)',
        'data_change': 'Random points are replaced with extreme values (multiplier x std). '
                       'Creates sudden, large deviations.',
        'what_changes': 'Isolated extreme values appear in normal regions. Value distribution gets heavy tails.',
    },
    'missing': {
        'name': 'Missing Values (MCAR)',
        'data_change': 'Random points or bursts are removed and filled (forward-fill / zero / interpolation). '
                       'Creates flat segments or discontinuities.',
        'what_changes': 'Local structure is destroyed in affected regions. Flat segments mimic freeze patterns.',
    },
    'freeze': {
        'name': 'Sensor Freeze (Stuck-at)',
        'data_change': 'Contiguous segments are replaced with a constant (last observed value). '
                       'Simulates sensor malfunction.',
        'what_changes': 'Variance drops to zero in frozen regions. Creates artificial flat segments.',
    },
    'swap': {
        'name': 'Temporal Swap',
        'data_change': 'Segments of the series are swapped in position. '
                       'Values are preserved but temporal order is disrupted.',
        'what_changes': 'Temporal dependencies are broken but the VALUE distribution is unchanged.',
    },
}

# Model mechanism descriptions
MODEL_MECHANISMS = {
    'IForest': {
        'full_name': 'Isolation Forest',
        'principle': 'isolates anomalies via random splits on feature values (windowed)',
        'key_metric': 'separation_gap (path_length_anomaly - path_length_normal)',
        'what_matters': 'VALUE distribution within windows — extreme or rare values get shorter paths',
        'invariant_to': 'temporal order (processes windows independently)',
    },
    'LOF': {
        'full_name': 'Local Outlier Factor',
        'principle': 'compares local density around each point to its k-neighbors',
        'key_metric': 'kdist_gap (k-distance_anomaly - k-distance_normal)',
        'what_matters': 'local density contrast — anomalies should be in sparser regions',
        'invariant_to': 'global distribution shifts (compares locally)',
    },
    'MP': {
        'full_name': 'Matrix Profile',
        'principle': 'finds the nearest-neighbor distance for each subsequence (shape matching)',
        'key_metric': 'nn_dist_gap (nn_distance_anomaly - nn_distance_normal)',
        'what_matters': 'subsequence SHAPE — anomalous shapes should have no good match',
        'invariant_to': 'amplitude scaling (z-normalized matching)',
    },
    'AE': {
        'full_name': 'Autoencoder',
        'principle': 'learns to reconstruct normal patterns; anomalies have high reconstruction error',
        'key_metric': 'error_ratio (recon_error_anomaly / recon_error_normal)',
        'what_matters': 'learned normal pattern — anomalies deviate from what was learned',
        'invariant_to': 'nothing specific (sensitive to distribution shifts)',
    },
}


def get_corruption_category(corruption_name):
    """Map internal corruption names to categories."""
    if corruption_name.startswith('noise') or corruption_name.startswith('snr'):
        return 'noise'
    elif corruption_name.startswith('spikes'):
        return 'spikes'
    elif corruption_name.startswith('mcar') or corruption_name.startswith('missing'):
        return 'missing'
    elif corruption_name.startswith('mnar'):
        return 'missing'  # group with missing
    elif corruption_name.startswith('freeze'):
        return 'freeze'
    elif corruption_name.startswith('swap'):
        return 'swap'
    return None


def classify_error_dominance(prec_drop_pct, recall_drop_pct):
    """Classify whether FP or FN dominates."""
    if abs(prec_drop_pct) < 5 and abs(recall_drop_pct) < 5:
        return 'negligible'
    ratio = prec_drop_pct / max(recall_drop_pct, 0.01)
    if ratio > 1.5:
        return 'FP-dominant (precision drops more → corruption mimics anomalies)'
    elif ratio < 0.67:
        return 'FN-dominant (recall drops more → real anomalies are masked)'
    else:
        return 'balanced (both FP and FN increase)'


def build_narrative(corruption_type, model, fp_fn_row, prop_row, prop_class,
                    internal_summary, ranking_row, cri_value, seg_vuln_summary):
    """Build a single causal chain narrative for one corruption x model."""

    corr_info = CORRUPTION_DESCRIPTIONS.get(corruption_type, {})
    model_info = MODEL_MECHANISMS.get(model, {})

    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  {corr_info.get('name', corruption_type).upper()} → {model_info.get('full_name', model)}")
    lines.append(f"{'='*70}")
    lines.append("")

    # Step 1: Data Change
    lines.append("STEP 1 — DATA CHANGE:")
    lines.append(f"  {corr_info.get('data_change', 'N/A')}")
    lines.append(f"  Effect: {corr_info.get('what_changes', 'N/A')}")
    lines.append("")

    # Step 2: Internal Mechanism
    lines.append(f"STEP 2 — INTERNAL MECHANISM ({model}):")
    lines.append(f"  Algorithm principle: {model_info.get('principle', 'N/A')}")
    lines.append(f"  Key metric: {model_info.get('key_metric', 'N/A')}")

    if internal_summary is not None:
        lines.append(f"  Observed change:")
        for col in internal_summary.columns:
            if col == 'condition':
                continue
            val = internal_summary[col].iloc[-1] if len(internal_summary) > 0 else np.nan
            if not np.isnan(val) and 'change_pct' in col:
                # Cap display at reasonable values (some metrics have huge % due to near-zero baselines)
                display_val = min(abs(val), 999.9)
                direction = "decreased" if val < 0 else "increased"
                if abs(val) > 1000:
                    lines.append(f"    - {col}: {direction} dramatically (baseline near zero)")
                else:
                    lines.append(f"    - {col}: {direction} by {display_val:.1f}%")
            elif not np.isnan(val) and 'pct_nn_changed' in col:
                lines.append(f"    - Nearest neighbors changed: {val:.1f}%")

    # WHY this matters for this specific corruption x model
    why = generate_why_explanation(corruption_type, model, internal_summary)
    if why:
        lines.append(f"  WHY: {why}")
    lines.append("")

    # Step 3: Error Type
    lines.append("STEP 3 — ERROR MANIFESTATION (FP vs FN):")
    if fp_fn_row is not None:
        prec_drop = fp_fn_row.get('mean_precision_drop_pct', 0)
        recall_drop = fp_fn_row.get('mean_recall_drop_pct', 0)
        auc_drop = fp_fn_row.get('mean_auc_drop', 0)
        dominance = classify_error_dominance(prec_drop, recall_drop)

        lines.append(f"  Precision drop: {prec_drop:.1f}% (more FP = corruption mistaken for anomaly)")
        lines.append(f"  Recall drop:    {recall_drop:.1f}% (more FN = real anomalies missed)")
        lines.append(f"  Mean AUC drop:  {auc_drop:.4f}")
        lines.append(f"  Dominant error: {dominance}")
    else:
        lines.append("  [No FP/FN data available]")
    lines.append("")

    # Step 4: Propagation
    lines.append("STEP 4 — PROPAGATION PATTERN:")
    if prop_row is not None and not prop_row.empty:
        row = prop_row.iloc[0] if isinstance(prop_row, pd.DataFrame) else prop_row
        decay = row.get('decay_type', 'unknown')
        decay_ratio = row.get('decay_ratio', np.nan)
        corruption_zone = row.get('mean_corruption', np.nan)
        far_zone = row.get('mean_far', np.nan)

        lines.append(f"  Decay type: {decay}")
        lines.append(f"  Decay ratio: {decay_ratio:.3f}" if not np.isnan(decay_ratio) else "")
        lines.append(f"  Score change in corruption zone: {corruption_zone:.3f}" if not np.isnan(corruption_zone) else "")
        lines.append(f"  Score change far from corruption: {far_zone:.3f}" if not np.isnan(far_zone) else "")

        if decay == 'global (flat)':
            lines.append("  → Corruption effect spreads across the ENTIRE series scores")
        elif decay == 'local (sharp)':
            lines.append("  → Corruption effect stays LOCALIZED near the corrupted region")
        else:
            lines.append("  → Moderate spread of corruption effect")
    elif prop_class is not None:
        sensitivity = prop_class.get('sensitivity_type', 'unknown')
        lines.append(f"  Model sensitivity type: {sensitivity}")
    else:
        lines.append("  [No propagation data available]")
    lines.append("")

    # Step 5: Metric Impact
    lines.append("STEP 5 — METRIC IMPACT:")
    if cri_value is not None:
        lines.append(f"  CRI (Corruption Robustness Index): {cri_value:.4f}")
        if cri_value > 0.9:
            lines.append("  → Virtually no impact")
        elif cri_value > 0.7:
            lines.append("  → Mild degradation")
        elif cri_value > 0.5:
            lines.append("  → Moderate degradation")
        else:
            lines.append("  → Severe degradation")

    if ranking_row is not None:
        inverted = ranking_row.get('inverted', False)
        if inverted:
            clean_rank = ranking_row.get('clean_ranking', '')
            corrupted_rank = ranking_row.get('corrupted_ranking', '')
            lines.append(f"  RANKING INVERSION: Yes!")
            lines.append(f"    Clean:     {clean_rank}")
            lines.append(f"    Corrupted: {corrupted_rank}")
        else:
            lines.append(f"  Ranking preserved (no inversion)")
    lines.append("")

    # Step 6: Most vulnerable anomalies
    if seg_vuln_summary is not None:
        lines.append("MOST VULNERABLE ANOMALY TYPES:")
        lines.append(f"  {seg_vuln_summary}")
        lines.append("")

    # Synthesis
    lines.append("SYNTHESIS (one-paragraph explanation):")
    synthesis = generate_synthesis(corruption_type, model, fp_fn_row, prop_row,
                                   internal_summary, cri_value, ranking_row)
    lines.append(f"  {synthesis}")
    lines.append("")

    return "\n".join(lines)


def generate_why_explanation(corruption_type, model, internal_summary):
    """Generate the WHY explanation connecting corruption mechanism to model internals."""

    explanations = {
        ('noise', 'IForest'): (
            "Noise inflates values within feature windows. IForest splits on value ranges, "
            "so noisy values create new split candidates that compete with true anomalies. "
            "The separation gap between anomaly and normal path lengths shrinks because "
            "normal points under noise also get shorter paths (look more anomalous)."
        ),
        ('noise', 'LOF'): (
            "Noise perturbs local neighborhoods uniformly. Since LOF compares local densities, "
            "the density contrast between anomalies and normal points decreases — "
            "all points become somewhat unusual, diluting the anomaly signal."
        ),
        ('noise', 'MP'): (
            "Noise destroys subsequence shapes. MP relies on finding similar subsequences "
            "(nearest neighbors); under noise, ALL subsequences become dissimilar. "
            "The NN distance increases everywhere, erasing the gap between anomalous and normal. "
            "This is why MP suffers the MOST under noise — its core assumption (shape matching) breaks."
        ),
        ('noise', 'AE'): (
            "Noise shifts the input distribution away from what the AE learned. "
            "Reconstruction error increases for ALL points (not just anomalies), "
            "reducing the error ratio between anomalous and normal regions."
        ),
        ('spikes', 'IForest'): (
            "Spikes create extreme values that are trivially easy to isolate (very short paths). "
            "The isolation budget is 'stolen' by spikes — they get the shortest paths, "
            "pushing real anomalies higher in the path distribution. "
            "At high severity, spike paths become SHORTER than anomaly paths (gap inverts)."
        ),
        ('spikes', 'LOF'): (
            "Spikes create isolated points in extremely sparse regions. LOF assigns them "
            "very high outlier scores. Real anomalies that were previously the most sparse "
            "are now relatively less sparse compared to spikes, reducing their LOF scores."
        ),
        ('spikes', 'MP'): (
            "Spikes create unique subsequence shapes (sharp jumps) that have no match anywhere. "
            "These get the highest NN distances, dominating the anomaly scores. "
            "Real anomalous patterns now have relatively lower NN distances — "
            "their uniqueness is drowned out by spike-induced uniqueness."
        ),
        ('spikes', 'AE'): (
            "Spikes are extreme values the AE has never seen, causing very high reconstruction "
            "error at spike locations. The error ratio shifts: spike regions compete with "
            "anomaly regions for the 'most abnormal' ranking."
        ),
        ('missing', 'IForest'): (
            "Forward-filled missing values create low-variance flat segments. "
            "IForest may or may not isolate these easily depending on context. "
            "The main effect is removing informative values — some anomaly-adjacent "
            "points are replaced, reducing the anomaly's footprint in feature windows."
        ),
        ('missing', 'LOF'): (
            "Missing values (filled) create clusters of identical/similar points. "
            "These artificially dense regions alter the local density landscape. "
            "If missing values overlap with anomaly neighborhoods, the anomaly's "
            "relative sparseness changes."
        ),
        ('missing', 'MP'): (
            "Filled missing values create artificial flat subsequences. "
            "If these flat patterns appear multiple times, they become 'normal' "
            "(low NN distance), but they also alter the profile of adjacent windows "
            "that partially overlap with the filled region."
        ),
        ('missing', 'AE'): (
            "Missing values (filled with constants or interpolation) create patterns "
            "that differ from what the AE learned. Reconstruction error increases "
            "at filled locations. If many regions are filled, the baseline error rises, "
            "making it harder to distinguish true anomalies."
        ),
        ('freeze', 'IForest'): (
            "Frozen segments have zero variance — all values are identical. "
            "In feature windows, this creates points that may be easy or hard to isolate "
            "depending on whether the stuck value is common or rare. "
            "Large freeze blocks (few, long) are worse because they create "
            "fewer but more disruptive artifacts."
        ),
        ('freeze', 'LOF'): (
            "Frozen segments create dense clusters of identical points. "
            "LOF treats these as very 'normal' (high density). This shifts the "
            "density baseline, potentially making some anomalies look less anomalous "
            "in comparison."
        ),
        ('freeze', 'MP'): (
            "Frozen segments are perfectly flat subsequences. If the series doesn't "
            "naturally contain flat segments, these become unique shapes with high NN "
            "distances, acting like artificial anomalies. If flat patterns exist, "
            "they may match each other (low distance) and not cause issues."
        ),
        ('freeze', 'AE'): (
            "The AE may or may not have learned constant-value patterns. "
            "If the stuck value is within the normal range, reconstruction error stays low. "
            "The main issue is that freeze segments replace potentially anomalous regions, "
            "removing detectable anomaly signal."
        ),
        ('swap', 'IForest'): (
            "Swap rearranges temporal order but preserves ALL values exactly. "
            "IForest operates on feature windows derived from values — "
            "since the same values exist (just reordered), the isolation splits "
            "see a very similar value distribution. This is why IForest is nearly immune to swap."
        ),
        ('swap', 'LOF'): (
            "Swap changes which points are neighbors of which. "
            "Since LOF compares local densities, rearranging the order creates "
            "new local neighborhoods, but the GLOBAL density structure is similar. "
            "Moderate impact because some anomaly-specific local contexts are disrupted."
        ),
        ('swap', 'MP'): (
            "Swap breaks temporal patterns that MP relies on for shape matching. "
            "However, since the values are preserved, the disruption is moderate — "
            "swapped segments still contain valid subsequences, just in different positions. "
            "MP is more robust than expected because the shapes themselves survive."
        ),
        ('swap', 'AE'): (
            "Swap creates sequences the AE hasn't seen in that specific order. "
            "If the AE learned temporal dependencies, swapped regions will have higher "
            "reconstruction error. But if the AE mainly learned value distributions, "
            "the impact is minimal."
        ),
    }

    return explanations.get((corruption_type, model), None)


def generate_synthesis(corruption_type, model, fp_fn_row, prop_row,
                       internal_summary, cri_value, ranking_row):
    """Generate a one-paragraph synthesis of the causal chain."""

    model_info = MODEL_MECHANISMS.get(model, {})
    corr_info = CORRUPTION_DESCRIPTIONS.get(corruption_type, {})

    # Get key numbers
    prec_drop = fp_fn_row.get('mean_precision_drop_pct', 0) if fp_fn_row is not None else 0
    recall_drop = fp_fn_row.get('mean_recall_drop_pct', 0) if fp_fn_row is not None else 0
    auc_drop = fp_fn_row.get('mean_auc_drop', 0) if fp_fn_row is not None else 0
    cri_str = f"{cri_value:.2f}" if cri_value is not None else "N/A"

    # Determine severity
    if cri_value is not None:
        if cri_value > 0.9:
            severity = "negligible"
        elif cri_value > 0.7:
            severity = "mild"
        elif cri_value > 0.5:
            severity = "moderate"
        else:
            severity = "severe"
    else:
        severity = "unknown"

    # Determine dominant error
    if prec_drop > recall_drop * 1.5:
        error_story = (f"primarily through increased false positives "
                       f"(precision drops {prec_drop:.0f}%), meaning the corruption is "
                       f"mistaken for anomalous behavior")
    elif recall_drop > prec_drop * 1.5:
        error_story = (f"primarily through increased false negatives "
                       f"(recall drops {recall_drop:.0f}%), meaning real anomalies are masked "
                       f"by the corruption")
    else:
        error_story = (f"through both increased false positives (precision -{prec_drop:.0f}%) "
                       f"and false negatives (recall -{recall_drop:.0f}%)")

    # Propagation story
    if prop_row is not None and not prop_row.empty:
        row = prop_row.iloc[0] if isinstance(prop_row, pd.DataFrame) else prop_row
        decay = row.get('decay_type', '')
        if 'global' in str(decay):
            prop_story = "The impact spreads globally across all anomaly scores"
        elif 'local' in str(decay):
            prop_story = "The impact remains localized near the corrupted region"
        else:
            prop_story = "The impact shows moderate spread beyond the corrupted region"
    else:
        prop_story = "Propagation pattern unknown"

    # Ranking inversion
    ranking_story = ""
    if ranking_row is not None and ranking_row.get('inverted', False):
        clean_best = ranking_row.get('clean_rank1', '?')
        corrupted_best = ranking_row.get('corrupted_rank1', '?')
        if clean_best != corrupted_best:
            ranking_story = (f" Notably, this causes a ranking inversion: under clean data "
                             f"{clean_best} is best, but under corruption {corrupted_best} takes over.")

    # WHY explanation (the key insight)
    why = generate_why_explanation(corruption_type, model, internal_summary)
    why_sentence = f" Mechanistically: {why.rstrip('.')}." if why else ""

    # Build paragraph
    paragraph = (
        f"{corr_info.get('name', corruption_type)} has {severity} impact on "
        f"{model_info.get('full_name', model)} (CRI={cri_str}). "
        f"{model} {model_info.get('principle', 'N/A')}. "
        f"Under this corruption, {corr_info.get('what_changes', 'the data is modified').lower().rstrip('.')}. "
        f"The degradation manifests {error_story}. "
        f"{prop_story}, with a mean AUC drop of {abs(auc_drop):.3f}."
        f"{why_sentence}{ranking_story}"
    )

    return paragraph


def get_worst_vulnerable_bins(seg_vuln_df, corruption_type, model):
    """Get the most vulnerable anomaly bins for a corruption x model."""
    if seg_vuln_df is None:
        return None

    filtered = seg_vuln_df[
        (seg_vuln_df['corruption_type'] == corruption_type) &
        (seg_vuln_df['model'] == model)
    ]
    if filtered.empty:
        # Try matching on condition prefix
        filtered = seg_vuln_df[
            (seg_vuln_df['condition'].str.contains(corruption_type, case=False, na=False)) &
            (seg_vuln_df['model'] == model)
        ]

    if filtered.empty:
        return None

    # Get worst bins (highest detection drop)
    worst = filtered.nlargest(3, 'detection_drop_pct')
    bins = []
    for _, row in worst.iterrows():
        bins.append(f"{row.get('combined_bin', '?')} (drop: {row['detection_drop_pct']:.0f}%)")

    return "Most vulnerable: " + ", ".join(bins) if bins else None


# ==========================================
# MAIN
# ==========================================

def main():
    print("Loading analysis data...")
    internal_df = load_internal_analysis()
    fp_fn_df = load_fp_fn()
    prop_profiles, prop_class, prop_aggregate = load_propagation()
    ranking_df = load_ranking_inversion()
    cri_df = load_cri()
    seg_vuln_df = load_segment_vulnerability()

    print(f"  Internal analysis: {'loaded' if internal_df is not None else 'MISSING'}")
    print(f"  FP/FN decomposition: {'loaded' if fp_fn_df is not None else 'MISSING'}")
    print(f"  Propagation profiles: {'loaded' if prop_profiles is not None else 'MISSING'}")
    print(f"  Ranking inversions: {'loaded' if ranking_df is not None else 'MISSING'}")
    print(f"  CRI table: {'loaded' if cri_df is not None else 'MISSING'}")
    print(f"  Segment vulnerability: {'loaded' if seg_vuln_df is not None else 'MISSING'}")

    corruptions = ['noise', 'spikes', 'missing', 'freeze', 'swap']
    models = ['IForest', 'LOF', 'MP', 'AE']

    # Internal analysis corruption prefixes
    INTERNAL_PREFIXES = {
        'noise': 'noise_snr',
        'spikes': 'spikes_f',
        'missing': 'mcar_',
        'freeze': None,  # not in internal analysis
        'swap': 'swap_',
    }

    # FP/FN corruption name mapping
    FPFN_NAMES = {
        ('noise', 'IForest'): ('noise', 'IForest'),
        ('noise', 'LOF'): ('noise', 'LOF'),
        ('noise', 'MP'): ('noise', 'MP'),
        ('noise', 'AE'): ('noise', 'AE'),
        ('spikes', 'IForest'): ('spikes', 'IForest'),
        ('spikes', 'LOF'): ('spikes', 'LOF'),
        ('spikes', 'MP'): ('spikes', 'MP'),
        ('missing', 'IForest'): ('missing_burst', 'IForest'),
        ('missing', 'LOF'): ('missing_burst', 'LOF'),
        ('missing', 'MP'): ('missing_burst', 'MP'),
        ('freeze', 'IForest'): ('freeze', 'IForest'),
        ('freeze', 'LOF'): ('freeze', 'LOF'),
        ('freeze', 'MP'): ('freeze', 'MP'),
        ('swap', 'IForest'): ('swap', 'IForest'),
    }

    all_narratives = []
    summary_rows = []

    for corruption in corruptions:
        for model in models:
            print(f"\n--- {corruption} x {model} ---")

            # 1. Internal mechanism
            prefix = INTERNAL_PREFIXES.get(corruption)
            internal_summary = None
            if prefix and internal_df is not None:
                if model == 'IForest':
                    internal_summary = extract_iforest_mechanism(internal_df, prefix)
                elif model == 'LOF':
                    internal_summary = extract_lof_mechanism(internal_df, prefix)
                elif model == 'MP':
                    internal_summary = extract_mp_mechanism(internal_df, prefix)
                elif model == 'AE':
                    internal_summary = extract_ae_mechanism(internal_df, prefix)

            # 2. FP/FN
            fp_fn_row = None
            if fp_fn_df is not None:
                fpfn_key = FPFN_NAMES.get((corruption, model))
                if fpfn_key:
                    match = fp_fn_df[
                        (fp_fn_df['corruption'] == fpfn_key[0]) &
                        (fp_fn_df['model'] == fpfn_key[1])
                    ]
                    if not match.empty:
                        fp_fn_row = match.iloc[0].to_dict()

            # 3. Propagation
            prop_row = None
            if prop_profiles is not None:
                match = prop_profiles[
                    (prop_profiles['model'] == model) &
                    (prop_profiles['corruption_type'] == corruption)
                ]
                if not match.empty:
                    prop_row = match

            prop_class_row = None
            if prop_class is not None:
                match = prop_class[prop_class['model'] == model]
                if not match.empty:
                    prop_class_row = match.iloc[0].to_dict()

            # 4. Ranking
            ranking_row = None
            if ranking_df is not None:
                match = ranking_df[ranking_df['corruption'] == corruption]
                if not match.empty:
                    # Get the moderate condition
                    ranking_row = match.iloc[len(match) // 2].to_dict()

            # 5. CRI
            cri_value = None
            if cri_df is not None:
                match = cri_df[
                    (cri_df['model'] == model) &
                    (cri_df['corruption'] == corruption)
                ]
                if not match.empty:
                    cri_value = match.iloc[0]['CRI']

            # 6. Segment vulnerability
            seg_summary = get_worst_vulnerable_bins(seg_vuln_df, corruption, model)

            # Build narrative
            narrative = build_narrative(
                corruption, model, fp_fn_row, prop_row, prop_class_row,
                internal_summary, ranking_row, cri_value, seg_summary
            )
            all_narratives.append(narrative)

            # Summary row
            summary_rows.append({
                'corruption': corruption,
                'model': model,
                'CRI': cri_value,
                'precision_drop_pct': fp_fn_row.get('mean_precision_drop_pct', None) if fp_fn_row else None,
                'recall_drop_pct': fp_fn_row.get('mean_recall_drop_pct', None) if fp_fn_row else None,
                'mean_auc_drop': fp_fn_row.get('mean_auc_drop', None) if fp_fn_row else None,
                'propagation': (prop_row.iloc[0]['decay_type'] if prop_row is not None and not prop_row.empty
                                else (prop_class_row.get('sensitivity_type', None) if prop_class_row else None)),
                'ranking_inverted': ranking_row.get('inverted', None) if ranking_row else None,
                'most_vulnerable': seg_summary,
            })

    # Save narratives
    narrative_path = os.path.join(OUTPUT_DIR, "causal_chain_narratives.txt")
    with open(narrative_path, 'w', encoding='utf-8') as f:
        f.write("CAUSAL CHAIN NARRATIVES\n")
        f.write("=" * 70 + "\n")
        f.write("Thesis: Impact of Data Quality Issues on Time-Series Anomaly Detection\n")
        f.write("Generated from existing analysis results — no new experiments.\n")
        f.write("=" * 70 + "\n\n")
        for narrative in all_narratives:
            f.write(narrative + "\n\n")
    print(f"\nSaved: {narrative_path}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "causal_chain_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Generate heatmap
    print("\nGenerating mechanism heatmap...")
    generate_mechanism_heatmap(summary_df)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


def generate_mechanism_heatmap(summary_df):
    """Create a visual heatmap of CRI values + error type annotations."""
    corruptions = ['noise', 'spikes', 'missing', 'freeze', 'swap']
    models = ['IForest', 'LOF', 'MP', 'AE']

    # CRI matrix
    cri_matrix = np.full((len(models), len(corruptions)), np.nan)
    prec_matrix = np.full((len(models), len(corruptions)), np.nan)
    recall_matrix = np.full((len(models), len(corruptions)), np.nan)

    for i, model in enumerate(models):
        for j, corruption in enumerate(corruptions):
            row = summary_df[
                (summary_df['model'] == model) &
                (summary_df['corruption'] == corruption)
            ]
            if not row.empty:
                cri_val = row.iloc[0]['CRI']
                if pd.notna(cri_val):
                    cri_matrix[i, j] = cri_val
                prec_val = row.iloc[0].get('precision_drop_pct')
                if pd.notna(prec_val):
                    prec_matrix[i, j] = prec_val
                recall_val = row.iloc[0].get('recall_drop_pct')
                if pd.notna(recall_val):
                    recall_matrix[i, j] = recall_val

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: CRI heatmap
    ax = axes[0]
    im = ax.imshow(cri_matrix, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=0.75)
    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels([c.capitalize() for c in corruptions], fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_title('CRI (Robustness Index)', fontsize=12, fontweight='bold')
    for i in range(len(models)):
        for j in range(len(corruptions)):
            val = cri_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Plot 2: Precision drop
    ax = axes[1]
    im2 = ax.imshow(prec_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=90)
    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels([c.capitalize() for c in corruptions], fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_title('Precision Drop % (→ FP increase)', fontsize=12, fontweight='bold')
    for i in range(len(models)):
        for j in range(len(corruptions)):
            val = prec_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 50 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=color)
    fig.colorbar(im2, ax=ax, shrink=0.8)

    # Plot 3: Recall drop
    ax = axes[2]
    im3 = ax.imshow(recall_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels([c.capitalize() for c in corruptions], fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_title('Recall Drop % (→ FN increase)', fontsize=12, fontweight='bold')
    for i in range(len(models)):
        for j in range(len(corruptions)):
            val = recall_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 50 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=color)
    fig.colorbar(im3, ax=ax, shrink=0.8)

    fig.suptitle('Causal Chain Summary: Robustness & Error Decomposition',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'mechanism_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: mechanism_heatmap.png")


if __name__ == '__main__':
    main()
