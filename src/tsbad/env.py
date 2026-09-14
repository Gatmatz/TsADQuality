"""Process setup for batch TSB-AD evaluation.

Runners import this first, right after putting src/ on the path: the thread caps below only
take effect if they are set before numpy/numba/stumpy are imported.
"""
import os

# Pin every numeric backend to one thread BEFORE numpy/numba/stumpy are imported.
# We already parallelise across files with ProcessPoolExecutor; without this each worker
# also grabs every core (stumpy.stump, used by MatrixProfile, is numba-parallel) and the
# resulting oversubscription slows the run badly. Must stay above the numpy import to take effect.
for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings


def silence_warnings():
    """Mute the expected, harmless noise from batch evaluation.

    sklearn raises UndefinedMetricWarning whenever a threshold yields no predicted
    positives (precision = 0/0) — routine once scores flatten, and it only touches the
    threshold-dependent F1s, never AUC/VUS. Called at import time so the spawned worker
    processes (which re-import the runner, and with it this module) inherit the filters too.
    """
    import numpy as np
    from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # numpy moved RankWarning in 2.0; POLY's polyfit raises it on flat/noisy segments
    rank_warning = getattr(getattr(np, 'exceptions', None), 'RankWarning', None) \
        or getattr(np, 'RankWarning', None)
    if rank_warning is not None:
        warnings.filterwarnings("ignore", category=rank_warning)


silence_warnings()
