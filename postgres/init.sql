CREATE TABLE IF NOT EXISTS window_lengths (
    dataset TEXT PRIMARY KEY,
    window_length INTEGER NOT NULL
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
    experiment_id TEXT NOT NULL,
    dataset_name TEXT,
    detector TEXT,
    random_seed TEXT,
    data_perfectness TEXT,
    corruption_type TEXT,
    metric TEXT NOT NULL,
    score DOUBLE PRECISION,
    execution_time DOUBLE PRECISION,
    execution_profile TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (experiment_id, metric)
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
