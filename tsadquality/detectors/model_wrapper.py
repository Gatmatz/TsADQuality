"""Copy of the parts of TSB-AD 1.5's `TSB_AD.model_wrapper` used by this package.

Each `run_*` is copied line for line from upstream, except that it returns
`(scores, clf)` so the fitted TSB-AD model is available for internal-metric
evaluation. Upstream's dispatchers catch every exception with a bare `except`
and return an error *string* instead of scores; these copies let exceptions
propagate instead.
"""


def run_Unsupervise_AD(model_name, data, **kwargs):
    function_name = f'run_{model_name}'
    if function_name not in globals():
        raise ValueError(f"Model function '{function_name}' is not defined.")
    return globals()[function_name](data, **kwargs)


def run_Semisupervise_AD(model_name, data_train, data_test, **kwargs):
    function_name = f'run_{model_name}'
    if function_name not in globals():
        raise ValueError(f"Model function '{function_name}' is not defined.")
    return globals()[function_name](data_train, data_test, **kwargs)


def run_IForest(data, slidingWindow=100, n_estimators=100, max_features=1, n_jobs=1):
    from TSB_AD.models.IForest import IForest
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel(), clf


def run_LOF(data, slidingWindow=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from TSB_AD.models.LOF import LOF
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel(), clf


def run_MatrixProfile(data, periodicity=1, n_jobs=1):
    from TSB_AD.models.MatrixProfile import MatrixProfile
    from TSB_AD.utils.slidingWindows import find_length_rank
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = MatrixProfile(window=slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel(), clf


def run_AutoEncoder(data_train, data_test, window_size=100, hidden_neurons=[64, 32], n_jobs=1):
    from TSB_AD.models.AE import AutoEncoder
    clf = AutoEncoder(slidingWindow=window_size, hidden_neurons=hidden_neurons, batch_size=128, epochs=50)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel(), clf
