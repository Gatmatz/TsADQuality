CREATE TABLE IF NOT EXISTS ts_metadata (
    dataset TEXT PRIMARY KEY,
    -- TSB_AD.utils.slidingWindows.find_length(data)
    window_length INTEGER NOT NULL,
    -- TSB_AD.utils.slidingWindows.find_length_rank(data, rank=1)
    periodicity INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    dataset_name TEXT,
    random_seed TEXT,
    data_perfectness TEXT,
    corruption_type TEXT,
    detector TEXT,
    -- Flattened corruption_params: which columns are populated depends on
    -- `corruption_type` (e.g., a SPK row uses fraction/multiplier, a GE row
    -- uses a/b); the rest stay NULL. Keep this in sync with CORRUPTION_SETTINGS
    -- in tsadquality/environment/corruption.py.
    snr_db DOUBLE PRECISION,
    fraction DOUBLE PRECISION,
    multiplier DOUBLE PRECISION,
    num_swaps INTEGER,
    num_permutations INTEGER,
    stuck_blocks INTEGER,
    num_bursts INTEGER,
    mechanism TEXT,
    a DOUBLE PRECISION,
    b DOUBLE PRECISION,
    execution_time DOUBLE PRECISION,
    execution_profile TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluations (
    experiment_id TEXT PRIMARY KEY,
    dataset_name TEXT,
    detector TEXT,
    random_seed TEXT,
    data_perfectness TEXT,
    corruption_type TEXT,
    -- From TSB_AD.evaluation.metrics.get_metrics(), computed together in one call.
    -- The threshold-dependent metrics (standard_f1 through affiliation_f) use a
    -- real mu +/- 3*sigma threshold on the min-max-normalized decision scores
    -- (see Experiment._run()), not get_metrics()'s default oracle/best-F1 threshold.
    auc_pr DOUBLE PRECISION,
    auc_roc DOUBLE PRECISION,
    vus_pr DOUBLE PRECISION,
    vus_roc DOUBLE PRECISION,
    standard_f1 DOUBLE PRECISION,
    pa_f1 DOUBLE PRECISION,
    event_based_f1 DOUBLE PRECISION,
    r_based_f1 DOUBLE PRECISION,
    affiliation_f DOUBLE PRECISION,
    -- Precision of that same 3-sigma threshold; not part of get_metrics().
    precision_3sigma DOUBLE PRECISION,
    execution_time DOUBLE PRECISION,
    execution_profile TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS skipped_computations (
    id SERIAL PRIMARY KEY,
    computation_id TEXT NOT NULL,
    reason TEXT,
    execution_profile TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS errors (
    id SERIAL PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    file_path TEXT,
    error_message TEXT,
    execution_profile TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
