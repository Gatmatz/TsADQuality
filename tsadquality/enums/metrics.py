"""Shared metric definitions, so both `Experiment` and `tsadquality.plots` agree on
what metrics exist and which `evaluations` table column each one is stored under.
"""

# Maps TSB_AD.evaluation.metrics.get_metrics() keys to `evaluations` table columns.
METRIC_COLUMNS: dict[str, str] = {
    "AUC-PR": "auc_pr",
    "AUC-ROC": "auc_roc",
    "VUS-PR": "vus_pr",
    "VUS-ROC": "vus_roc",
    "Standard-F1": "standard_f1",
    "PA-F1": "pa_f1",
    "Event-based-F1": "event_based_f1",
    "R-based-F1": "r_based_f1",
    "Affiliation-F": "affiliation_f",
}
