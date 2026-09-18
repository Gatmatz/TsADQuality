#!/usr/bin/env bash
# Launches N shards of `tsadquality.utils.run_experiment` in parallel, each covering
# a disjoint slice of the experiment sweep (see --shard-index/--shard-count in
# tsadquality/utils/run_experiment.py), streaming their output to the terminal.
#
# Usage:
#   scripts/run_parallel_experiments.sh [N]
#
# N defaults to the number of available CPU cores.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOCKFILE="/tmp/run_parallel_experiments.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    echo "Another run_parallel_experiments.sh is already running (lock: $LOCKFILE, held by pid $(cat "$LOCKFILE" 2>/dev/null))." >&2
    exit 1
fi
echo $$ >&200

N="${1:-$(nproc)}"

RUNNER=(python -m tsadquality.utils.run_experiment)
if command -v uv >/dev/null 2>&1; then
    RUNNER=(uv run python -m tsadquality.utils.run_experiment)
fi

echo "Launching $N shard(s) with: ${RUNNER[*]}"

pids=()
for ((i = 0; i < N; i++)); do
    "${RUNNER[@]}" --shard-index "$i" --shard-count "$N" 2>&1 | sed -u "s/^/[shard $i] /" &
    pids+=("$!")
    echo "  shard $i -> pid ${pids[-1]}"
done

cleanup() {
    echo "Stopping shards..."
    kill "${pids[@]}" 2>/dev/null || true
}
trap cleanup INT TERM

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done

exit "$status"
