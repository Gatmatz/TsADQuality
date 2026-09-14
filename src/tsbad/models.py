"""TSB-AD detectors, run the way every corruption experiment runs them.

Two entry points, one per evaluation family:

- `run_model_full`: the detector sees the whole (corrupted) series.
- `run_model_survivors`: points were deleted; the detector sees only the survivors, the
  semi-supervised train split is remapped onto them, and the model window can be clamped.

Both inject the window estimated on the CLEAN signal for periodicity-derived models when
window_mode='clean' (the default), so corruption cannot move it.
"""
import hashlib
import random

import numpy as np

SEMISUP = {'AutoEncoder', 'AutoEncoder_2', 'StreamVAE', 'CNN', 'LSTMAD'}

# Models whose window is derived from the data via find_length_rank(periodicity). Under
# corruption the official wrapper would re-estimate it on the CORRUPTED signal; the original
# experiment instead fixed the window from the CLEAN signal. In window_mode='clean' we
# replicate each wrapper body verbatim but inject the clean window. (IForest is absent on
# purpose: it uses a fixed slidingWindow=100, so corruption cannot move it.)
WINDOW_MODELS = {'MatrixProfile', 'POLY', 'Sub_PCA', 'KShapeAD', 'KMeansAD_U'}


def get_hp(model_name):
    from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
    # AutoEncoder_2 shares AutoEncoder's HP
    key = 'AutoEncoder' if model_name in ('AutoEncoder', 'AutoEncoder_2') else model_name
    return dict(Optimal_Uni_algo_HP_dict.get(key, {}))


def seed_job(key):
    """Deterministic per-job seeding.

    Some TSB-AD models are stochastic (e.g. KMeansAD builds sklearn KMeans without
    random_state), so without this the same job would give different scores on every run.
    md5, not Python's hash(): that one is salted per process.
    """
    js = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % (2**31 - 1)
    np.random.seed(js)
    random.seed(js)


def model_window(model_name, clean_data, hp, n_kept, clamp):
    """Window for periodicity-derived models, estimated on the CLEAN signal.

    Each model uses ITS OWN official periodicity (e.g. KMeansAD_U uses rank=2). With `clamp`,
    the window is limited to the length actually reaching the detector — a window valid for
    the full series can exceed the survivor count once points are deleted. Same clamp as the
    TSB-UAD original (`min(w, n_kept // 4)`, floor 10); callers report it per row so a firing
    clamp stays visible.
    """
    from TSB_AD.utils.slidingWindows import find_length_rank
    w = int(find_length_rank(clean_data, rank=hp.get('periodicity', 1)))
    if not clamp:
        return w, False
    w_clamped = max(min(w, n_kept // 4), 10)
    return w_clamped, (w_clamped != w)


def _window_model_scores(model_name, data, hp, w):
    if model_name == 'MatrixProfile':
        from TSB_AD.models.MatrixProfile import MatrixProfile
        clf = MatrixProfile(window=w); clf.fit(data)
        return clf.decision_scores_.ravel()
    if model_name == 'POLY':
        from TSB_AD.models.POLY import POLY
        clf = POLY(power=hp.get('power', 3), window=w); clf.fit(data)
        return clf.decision_scores_.ravel()
    if model_name == 'Sub_PCA':
        from TSB_AD.models.PCA import PCA
        clf = PCA(slidingWindow=w, n_components=hp.get('n_components')); clf.fit(data)
        return clf.decision_scores_.ravel()
    if model_name == 'KShapeAD':
        from TSB_AD.models.SAND import SAND
        clf = SAND(pattern_length=w, subsequence_length=4 * w)
        clf.fit(data.squeeze(), overlaping_rate=int(1.5 * w))
        return clf.decision_scores_.ravel()
    if model_name == 'KMeansAD_U':
        from TSB_AD.models.KMeansAD import KMeansAD
        clf = KMeansAD(k=hp.get('n_clusters', 20), window_size=w, stride=1, n_jobs=1)
        return clf.fit_predict(data).ravel()
    raise ValueError(f'unhandled window model: {model_name}')


def run_model_full(model_name, data, clean_data, hp, window_mode, file_name):
    """Run a TSB-AD detector on the whole series. Returns the scores.

    window_mode='native' -> pure official wrapper (re-estimates the window on the given data).
    window_mode='clean'  -> same model/HP, but periodicity-derived models use the window
                            estimated from the clean signal (matches the original experiment).
    On the clean condition both modes coincide exactly.
    """
    from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD

    if model_name in SEMISUP:
        train_index = int(file_name.split('.')[0].split('_')[-3])
        return run_Semisupervise_AD(model_name, data[:train_index, :], data, **hp)

    if window_mode == 'native' or model_name not in WINDOW_MODELS:
        return run_Unsupervise_AD(model_name, data, **hp)

    w, _ = model_window(model_name, clean_data, hp, len(data), clamp=False)
    return _window_model_scores(model_name, data, hp, w)


def run_model_survivors(model_name, data, clean_data, hp, window_mode, file_name, nan_mask, clamp):
    """Run a TSB-AD detector on the points that survived deletion.

    `data` is the array the detector sees — the shortened one when points were dropped.
    `nan_mask` is over the ORIGINAL timeline and is needed to remap the semi-supervised
    train split. Same window_mode semantics as `run_model_full`.

    Returns (scores, model_window, window_clamped).
    """
    from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD

    n_kept = len(data)
    if model_name in SEMISUP:
        # train_index counts points on the ORIGINAL timeline; after deletion the split moves.
        # Map it to the number of survivors before the original cut point.
        train_index = int(file_name.split('.')[0].split('_')[-3])
        train_kept = int((~nan_mask[:train_index]).sum())
        if train_kept < 10:
            raise ValueError(f'train split collapsed after data loss: {train_kept} points')
        return run_Semisupervise_AD(model_name, data[:train_kept, :], data, **hp), None, False

    if window_mode == 'native' or model_name not in WINDOW_MODELS:
        return run_Unsupervise_AD(model_name, data, **hp), None, False

    w, clamped = model_window(model_name, clean_data, hp, n_kept, clamp)
    return _window_model_scores(model_name, data, hp, w), w, clamped
