#!/usr/bin/env bash
# Launches N shards of `tsadquality.utils.run_experiment` in parallel, each covering
# a disjoint slice of the experiment sweep (see --shard-index/--shard-count in
# tsadquality/utils/run_experiment.py), and tails their logs.
#
# Usage:
#   scripts/run_parallel_experiments.sh [N]
#
# N defaults to the number of available CPU cores.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

N="${1:-$(nproc)}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

RUNNER=(python -m tsadquality.utils.run_experiment)
if command -v uv >/dev/null 2>&1; then
    RUNNER=(uv run python -m tsadquality.utils.run_experiment)
fi

echo "Launching $N shard(s) with: ${RUNNER[*]}"

pids=()
for ((i = 0; i < N; i++)); do
    log_file="$LOG_DIR/shard_${i}.log"
    "${RUNNER[@]}" --shard-index "$i" --shard-count "$N" > "$log_file" 2>&1 &
    pids+=("$!")
    echo "  shard $i -> pid ${pids[-1]}, log $log_file"
done

cleanup() {
    echo "Stopping shards..."
    kill "${pids[@]}" 2>/dev/null || true
}
trap cleanup INT TERM

tail -f "$LOG_DIR"/shard_*.log &
tail_pid=$!

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done

kill "$tail_pid" 2>/dev/null || true
exit "$status"
