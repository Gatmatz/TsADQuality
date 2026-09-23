"""Internal algorithmic metrics: WHY each detector is robust or vulnerable to a corruption.

Ported from src/experiments/analysis/run_internal_analysis.py, reading the internals
from the model already fitted by `Experiment._run()` (via
`ReproducibleOperations.run_detector`) instead of refitting a separate one. Each
experiment gets one value for its detector; compare an IMPERFECT experiment's value
with its PERFECT counterpart's to see what the corruption changed inside the model:

  - IForest:       separation_gap = mean -score_samples on anomaly - normal points
  - LOF:           kdist_gap      = mean k-distance on anomaly - normal points
  - MatrixProfile: nn_dist_gap    = mean nearest-neighbour distance on anomaly - normal points
  - AutoEncoder:   error_ratio    = mean min-max-scaled reconstruction error on anomaly / normal points

Per-window internals are assigned to each window's centre point and split by the
point-level labels: the same alignment TSB-AD uses for decision scores, so these
line up with the AUC/VUS/F1 computed from them.
"""

import numpy as np

from tsadquality.enums.detectors import DetectorModel


def _to_points(window_values, n, window):
    """Places each window's value at its centre point. TSB-AD centres window i on
    point i + ceil((window - 1) / 2) (IForest/LOF) or i + window // 2 (MatrixProfile),
    which are equal for every window length. Edge points that no window is centred on
    stay NaN, instead of repeating the first/last value as TSB-AD's padding does.
    """
    points = np.full(n, np.nan)
    front = window // 2
    points[front:front + len(window_values)] = window_values
    return points


def _class_means(point_values, labels):
    """Mean of `point_values` on anomaly points and on normal points, ignoring NaN."""
    valid = ~np.isnan(point_values)
    anom = valid & (labels == 1)
    norm = valid & (labels == 0)
    anomaly = float(np.mean(point_values[anom])) if anom.any() else np.nan
    normal = float(np.mean(point_values[norm])) if norm.any() else np.nan
    return anomaly, normal


def _gap(point_values, labels):
    anomaly, normal = _class_means(point_values, labels)
    return round(anomaly - normal, 6)


def iforest_separation_gap(model, labels):
    window = model.slidingWindow
    # TSB-AD stores decision_scores_ = -(score_samples - offset_) per window, centre-padded
    # to the series length; strip the padding and the offset to recover -score_samples.
    front = window // 2
    raw_scores = model.decision_scores_[front:front + len(labels) - window + 1] - model.detector_.offset_
    return _gap(_to_points(raw_scores, len(labels), window), labels)


def lof_kdist_gap(model, labels):
    # k-distance: distance to k-th nearest neighbor
    kdist = model.detector_._distances_fit_X_[:, -1]
    return _gap(_to_points(kdist, len(labels), model.slidingWindow), labels)


def mp_nn_dist_gap(model, labels):
    nn_distances = model.profile[:, 0].astype(float)
    return _gap(_to_points(nn_distances, len(labels), model.window), labels)


def ae_error_ratio(scaled_scores, labels):
    anomaly, normal = _class_means(scaled_scores, labels)
    ratio = anomaly / (normal + 1e-10)
    return round(ratio, 4) if not np.isnan(ratio) else np.nan


def compute_internal_metrics(detector, model, scaled_scores, labels) -> dict:
    """The internal metric of a fitted `detector` (a `DetectorModel` member).

    `model` is the fitted TSB-AD model, `scaled_scores` its min-max-scaled decision
    scores (only the AutoEncoder uses them), `labels` the point-level ground truth.
    """
    labels = np.asarray(labels).astype(int)
    match detector:
        case DetectorModel.ISO:
            return {'separation_gap': iforest_separation_gap(model, labels)}
        case DetectorModel.LOF:
            return {'kdist_gap': lof_kdist_gap(model, labels)}
        case DetectorModel.MP:
            return {'nn_dist_gap': mp_nn_dist_gap(model, labels)}
        case DetectorModel.AutoEncoder:
            return {'error_ratio': ae_error_ratio(np.asarray(scaled_scores, dtype=float), labels)}
    return {}
