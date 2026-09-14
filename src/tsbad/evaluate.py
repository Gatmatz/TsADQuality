"""Scoring helpers: metric columns, score length, and the true-impact remap for lost points."""
import numpy as np

METRIC_COLS = ['AUC_PR', 'AUC_ROC', 'VUS_PR', 'VUS_ROC', 'Standard_F1',
               'PA_F1', 'Event_based_F1', 'R_based_F1', 'Affiliation_F']


def metric_fields(metrics):
    """Every metric TSB-AD get_metrics returns, with '-' in the key turned into '_'."""
    return {k.replace('-', '_'): v for k, v in metrics.items()}


def fit_length(score, n_kept):
    """Defensive: TSB-AD wrappers return full-length scores, but pad/truncate if one does not."""
    score = np.asarray(score).ravel()
    if len(score) > n_kept:
        return score[:n_kept]
    if len(score) < n_kept:
        return np.pad(score, (0, n_kept - len(score)), mode='edge')
    return score


def true_impact(score, n, nan_mask, labels):
    """Map survivor scores back onto the original timeline for evaluation.

    Lost anomalies are scored as missed; lost normal points are excluded.

    The TSB-UAD original wrote a literal 0.0 for lost anomalies, valid there because it
    MinMax-scaled scores to [0, 1] first, making 0.0 the true minimum. TSB-AD wrappers return
    RAW scores: IForest lives in roughly [-0.06, +0.03] with ~90% of points BELOW zero, so a
    literal 0.0 would rank a destroyed anomaly in the top ~10% — inverting the intended meaning
    and inflating the metrics as more anomalies are lost. Use the series minimum instead: same
    semantics as the original (tied last with the least anomalous point), scale-free, and rank
    metrics are invariant to the difference.

    Returns (eval_scores, eval_labels, eval_mask).
    """
    masked_anomaly = nan_mask & (labels == 1)
    masked_normal = nan_mask & (labels == 0)
    lost_anomaly_score = float(score.min())
    full_score = np.full(n, np.nan)
    full_score[~nan_mask] = score
    full_score[masked_anomaly] = lost_anomaly_score   # destroyed anomaly = missed
    eval_mask = ~masked_normal            # normals destroyed by the loss = excluded
    return full_score[eval_mask], labels[eval_mask], eval_mask
