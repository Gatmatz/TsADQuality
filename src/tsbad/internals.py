"""Internal-model measurements of the thesis, taken from the TSB-AD detectors.

The thesis (src/experiments/analysis/run_internal_analysis.py) measured what changes INSIDE each
detector when the data are corrupted: anomaly-vs-normal separation of IForest path lengths, LOF
k-distances, Matrix Profile nearest-neighbour distances (and how many neighbours change),
PCA components / explained variance / scores, AutoEncoder reconstruction errors.

This module takes the same measurements from the TSB-AD models, fitted exactly as the corruption
harness fits them (same wrapper settings, same clean-signal window, same clamp), so the
internals belong to the detector whose performance is reported.

`map_window_labels` and the `compute_*_metrics` functions are copied verbatim from the thesis
script — they define the measurements and must not drift.

Phase 1 models: IForest, MatrixProfile, Sub_PCA.
"""
import inspect
import math

import numpy as np

from .models import model_window

PHASE1_MODELS = ('IForest', 'MatrixProfile', 'Sub_PCA')


# ==========================================
# THESIS MEASUREMENTS (verbatim from src/experiments/analysis/run_internal_analysis.py)
# ==========================================

def map_window_labels(labels, window):
    """Map point-level labels to window-level. Window is anomaly if ANY point in it is anomaly."""
    n_windows = len(labels) - window + 1
    # Use cumsum for efficiency
    cum = np.cumsum(np.concatenate(([0], labels)))
    window_sums = cum[window:] - cum[:n_windows]
    return window_sums > 0


def compute_iforest_metrics(clean, corrupted):
    # gap = anomaly - normal: positive means anomalies score higher (good separation)
    gap_clean = (clean['path_anomaly'] or 0) - (clean['path_normal'] or 0)
    gap_corr = (corrupted['path_anomaly'] or 0) - (corrupted['path_normal'] or 0)
    change = ((gap_corr - gap_clean) / abs(gap_clean) * 100) if abs(gap_clean) > 1e-10 else np.nan
    return {
        'path_anomaly_clean': clean['path_anomaly'],
        'path_anomaly_corrupted': corrupted['path_anomaly'],
        'path_normal_clean': clean['path_normal'],
        'path_normal_corrupted': corrupted['path_normal'],
        'separation_gap_clean': round(gap_clean, 6),
        'separation_gap_corrupted': round(gap_corr, 6),
        'separation_gap_change_pct': round(change, 2) if not np.isnan(change) else np.nan,
    }


def compute_lof_metrics(clean, corrupted):
    gap_clean = (clean['kdist_anomaly'] or 0) - (clean['kdist_normal'] or 0)
    gap_corr = (corrupted['kdist_anomaly'] or 0) - (corrupted['kdist_normal'] or 0)
    change = ((gap_corr - gap_clean) / abs(gap_clean) * 100) if abs(gap_clean) > 1e-10 else np.nan
    return {
        'kdist_anomaly_clean': clean['kdist_anomaly'],
        'kdist_anomaly_corrupted': corrupted['kdist_anomaly'],
        'kdist_normal_clean': clean['kdist_normal'],
        'kdist_normal_corrupted': corrupted['kdist_normal'],
        'kdist_gap_clean': round(gap_clean, 6),
        'kdist_gap_corrupted': round(gap_corr, 6),
        'kdist_gap_change_pct': round(change, 2) if not np.isnan(change) else np.nan,
    }


def compute_mp_metrics(clean, corrupted, can_compare_nn=True):
    gap_clean = (clean['nn_dist_anomaly'] or 0) - (clean['nn_dist_normal'] or 0)
    gap_corr = (corrupted['nn_dist_anomaly'] or 0) - (corrupted['nn_dist_normal'] or 0)
    change = ((gap_corr - gap_clean) / abs(gap_clean) * 100) if abs(gap_clean) > 1e-10 else np.nan

    pct_changed = np.nan
    pct_changed_anom = np.nan
    if can_compare_nn and 'nn_indices' in clean and 'nn_indices' in corrupted:
        idx_c = clean['nn_indices']
        idx_r = corrupted['nn_indices']
        min_len = min(len(idx_c), len(idx_r))
        if min_len > 0:
            pct_changed = float(np.mean(idx_c[:min_len] != idx_r[:min_len]) * 100)

    return {
        'nn_dist_anomaly_clean': clean['nn_dist_anomaly'],
        'nn_dist_anomaly_corrupted': corrupted['nn_dist_anomaly'],
        'nn_dist_normal_clean': clean['nn_dist_normal'],
        'nn_dist_normal_corrupted': corrupted['nn_dist_normal'],
        'nn_dist_gap_clean': round(gap_clean, 6),
        'nn_dist_gap_corrupted': round(gap_corr, 6),
        'nn_dist_gap_change_pct': round(change, 2) if not np.isnan(change) else np.nan,
        'pct_nn_changed': round(pct_changed, 2) if not np.isnan(pct_changed) else np.nan,
    }


def compute_ae_metrics(clean_errors, corrupted_errors, clean_labels, corrupted_labels):
    anom_c = clean_labels == 1
    norm_c = clean_labels == 0
    anom_r = corrupted_labels == 1
    norm_r = corrupted_labels == 0

    re_anom_clean = float(np.mean(clean_errors[anom_c])) if anom_c.any() else np.nan
    re_norm_clean = float(np.mean(clean_errors[norm_c])) if norm_c.any() else np.nan
    re_anom_corr = float(np.mean(corrupted_errors[anom_r])) if anom_r.any() else np.nan
    re_norm_corr = float(np.mean(corrupted_errors[norm_r])) if norm_r.any() else np.nan

    ratio_clean = re_anom_clean / (re_norm_clean + 1e-10) if not np.isnan(re_anom_clean) else np.nan
    ratio_corr = re_anom_corr / (re_norm_corr + 1e-10) if not np.isnan(re_anom_corr) else np.nan
    change = ((ratio_corr - ratio_clean) / abs(ratio_clean) * 100) if (
        not np.isnan(ratio_clean) and abs(ratio_clean) > 1e-10) else np.nan

    return {
        'recon_error_anomaly_clean': round(re_anom_clean, 6) if not np.isnan(re_anom_clean) else np.nan,
        'recon_error_anomaly_corrupted': round(re_anom_corr, 6) if not np.isnan(re_anom_corr) else np.nan,
        'recon_error_normal_clean': round(re_norm_clean, 6) if not np.isnan(re_norm_clean) else np.nan,
        'recon_error_normal_corrupted': round(re_norm_corr, 6) if not np.isnan(re_norm_corr) else np.nan,
        'error_ratio_clean': round(ratio_clean, 4) if not np.isnan(ratio_clean) else np.nan,
        'error_ratio_corrupted': round(ratio_corr, 4) if not np.isnan(ratio_corr) else np.nan,
        'error_ratio_change_pct': round(change, 2) if not np.isnan(change) else np.nan,
    }


def compute_pca_metrics(clean, corrupted):
    # Cosine similarity for top principal components (use abs to handle sign flips)
    n_pc = min(3, len(clean['components']), len(corrupted['components']))
    cosines = {}
    for i in range(n_pc):
        v1 = clean['components'][i]
        v2 = corrupted['components'][i]
        min_d = min(len(v1), len(v2))
        v1, v2 = v1[:min_d], v2[:min_d]
        cos_sim = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
        cosines[f'pc{i+1}_cosine'] = round(float(cos_sim), 6)
    for i in range(n_pc, 3):
        cosines[f'pc{i+1}_cosine'] = np.nan

    ev_clean = float(np.sum(clean['explained_variance_ratio'][:3]))
    ev_corr = float(np.sum(corrupted['explained_variance_ratio'][:3]))

    gap_clean = (clean['score_anomaly'] or 0) - (clean['score_normal'] or 0)
    gap_corr = (corrupted['score_anomaly'] or 0) - (corrupted['score_normal'] or 0)
    change = ((gap_corr - gap_clean) / abs(gap_clean) * 100) if abs(gap_clean) > 1e-10 else np.nan

    return {
        **cosines,
        'explained_var_top3_clean': round(ev_clean, 6),
        'explained_var_top3_corrupted': round(ev_corr, 6),
        'score_anomaly_clean': clean['score_anomaly'],
        'score_anomaly_corrupted': corrupted['score_anomaly'],
        'score_normal_clean': clean['score_normal'],
        'score_normal_corrupted': corrupted['score_normal'],
        'score_gap_change_pct': round(change, 2) if not np.isnan(change) else np.nan,
    }


# ==========================================
# TSB-AD EXTRACTORS
# ==========================================

def _wrapper_kwargs(wrapper, hp):
    """The keyword arguments a TSB-AD run_<model> wrapper ends up using: its defaults, then HP."""
    kwargs = {name: p.default for name, p in inspect.signature(wrapper).parameters.items()
              if p.default is not inspect.Parameter.empty}
    kwargs.update(hp)
    return kwargs


def _unpad(scores, n_samples, window):
    """Undo the edge padding TSB-AD applies to window-level decision scores."""
    if len(scores) != n_samples or window <= 1:
        return scores
    head, tail = math.ceil((window - 1) / 2), (window - 1) // 2
    return scores[head:n_samples - tail]


def _split(values, window_labels):
    """Mean of `values` over anomalous and over normal windows, as the thesis extractors do."""
    anom_mask = window_labels[:len(values)]
    norm_mask = ~anom_mask
    return (float(np.mean(values[anom_mask])) if anom_mask.any() else np.nan,
            float(np.mean(values[norm_mask])) if norm_mask.any() else np.nan)


def _iforest(data, labels, hp):
    """run_IForest (fixed slidingWindow, no clean-window injection) + the thesis path-length proxy."""
    from TSB_AD import model_wrapper as mw
    from TSB_AD.models.IForest import IForest
    from TSB_AD.models.feature import Window
    from TSB_AD.utils.utility import zscore

    kw = _wrapper_kwargs(mw.run_IForest, hp)
    clf = IForest(slidingWindow=kw['slidingWindow'], n_estimators=kw['n_estimators'],
                  max_features=kw['max_features'], n_jobs=kw['n_jobs'])
    clf.fit(data)
    scores = clf.decision_scores_.ravel()
    w = kw['slidingWindow']

    # Rebuild the matrix IForest.fit trained on (sliding windows, then its internal z-score)
    X = Window(window=w).convert(data)
    if clf.normalize:
        X = zscore(X, axis=0, ddof=0) if data.shape[1] == 1 else zscore(X, axis=1, ddof=1)
    raw_scores = -clf.detector_.score_samples(X)   # thesis: average path length proxy
    # the rebuilt matrix must be the one the model saw: its window scores are raw + offset_
    if not np.allclose(raw_scores + clf.detector_.offset_, _unpad(scores, len(data), w)):
        raise RuntimeError('IForest internals: rebuilt window matrix does not match the fitted model')

    path_anomaly, path_normal = _split(raw_scores, map_window_labels(labels, w))
    return scores, {'window': w, 'path_anomaly': path_anomaly, 'path_normal': path_normal}


def _matrix_profile(data, clean_data, labels, hp, clamp):
    """MatrixProfile with the clean-signal window (as the harness) + NN distances and indices."""
    from TSB_AD.models.MatrixProfile import MatrixProfile

    w, _ = model_window('MatrixProfile', clean_data, hp, len(data), clamp)
    clf = MatrixProfile(window=w)
    clf.fit(data)
    nn_distances = clf.profile[:, 0].astype(float)
    nn_indices = clf.profile[:, 1].astype(int)
    nn_dist_anomaly, nn_dist_normal = _split(nn_distances, map_window_labels(labels, w))
    return clf.decision_scores_.ravel(), {'window': w, 'nn_dist_anomaly': nn_dist_anomaly,
                                          'nn_dist_normal': nn_dist_normal, 'nn_indices': nn_indices}


def _sub_pca(data, clean_data, labels, hp, clamp):
    """Sub_PCA with the clean-signal window (as the harness) + components, variance, window scores."""
    from TSB_AD.models.PCA import PCA

    w, _ = model_window('Sub_PCA', clean_data, hp, len(data), clamp)
    clf = PCA(slidingWindow=w, n_components=hp.get('n_components'))
    clf.fit(data)
    scores = clf.decision_scores_.ravel()
    score_anomaly, score_normal = _split(_unpad(scores, len(data), w), map_window_labels(labels, w))
    return scores, {'window': w,
                    'components': clf.detector_.components_.copy(),
                    'explained_variance_ratio': clf.detector_.explained_variance_ratio_.copy(),
                    'score_anomaly': score_anomaly, 'score_normal': score_normal}


def measure(model_name, data, clean_data, labels, hp, clamp=False):
    """Fit `model_name` on `data` as the harness would and return (scores, internals).

    `data` is what the detector sees (the survivors when points were deleted), `labels` the
    matching point labels, `clean_data` the full clean series the window is estimated on, and
    `clamp` the harness clamp flag. `scores` equals what run_model_full / run_model_survivors
    return for the same inputs and seed.
    """
    if model_name == 'IForest':
        return _iforest(data, labels, hp)
    if model_name == 'MatrixProfile':
        return _matrix_profile(data, clean_data, labels, hp, clamp)
    if model_name == 'Sub_PCA':
        return _sub_pca(data, clean_data, labels, hp, clamp)
    raise NotImplementedError(f'internal measurements for {model_name} are not implemented yet')


def compare(model_name, clean, corrupted, can_compare_nn=True):
    """The thesis metric row for one condition: clean internals vs corrupted internals."""
    if model_name == 'IForest':
        return compute_iforest_metrics(clean, corrupted)
    if model_name == 'MatrixProfile':
        return compute_mp_metrics(clean, corrupted, can_compare_nn=can_compare_nn)
    if model_name == 'Sub_PCA':
        return compute_pca_metrics(clean, corrupted)
    raise NotImplementedError(f'internal measurements for {model_name} are not implemented yet')
