-- Finds/deletes rows in `experiments`/`evaluations` whose counterpart row is
-- missing in the other table. This happens when a run is killed mid-`Experiment._run()`
-- (see tsadquality/experiment/Experiment.py): `experiments` is written before the
-- detector is evaluated and `evaluations` is written, so a kill in between leaves an
-- orphaned `experiments` row that `Experiment._exists_in_postgres()` will never treat
-- as complete, but that also can't be recomputed until it's removed (the primary-key
-- INSERT would otherwise fail).

SELECT e.experiment_id
FROM experiments e
LEFT JOIN evaluations v ON v.experiment_id = e.experiment_id
WHERE v.experiment_id IS NULL;

DELETE FROM experiments e
WHERE NOT EXISTS (
    SELECT 1 FROM evaluations v WHERE v.experiment_id = e.experiment_id
);

-- ── evaluations with no matching experiments row ────────────────────────────
-- (shouldn't normally happen, since `evaluations` is only ever written after
-- `experiments` for the same experiment_id; included for symmetry/cleanup.)

SELECT v.experiment_id
FROM evaluations v
LEFT JOIN experiments e ON e.experiment_id = v.experiment_id
WHERE e.experiment_id IS NULL;

DELETE FROM evaluations v
WHERE NOT EXISTS (
    SELECT 1 FROM experiments e WHERE e.experiment_id = v.experiment_id
);
