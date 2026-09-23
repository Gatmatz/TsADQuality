#!/usr/bin/env bash
# Kills every running `tsadquality.utils.run_experiment` shard process owned by the
# current user. Run this before relaunching the sweep (e.g. via
# scripts/run_parallel_experiments.sh), so you don't end up with two processes
# running the same --shard-index concurrently.
#
# Usage:
#   scripts/kill_experiment_shards.sh

set -euo pipefail

PATTERN="tsadquality.utils.run_experiment"
USER_NAME="$(whoami)"

pids="$(pgrep -u "$USER_NAME" -f "$PATTERN" || true)"

if [ -z "$pids" ]; then
    echo "No running $PATTERN processes found for user $USER_NAME."
    exit 0
fi

echo "Killing the following $PATTERN processes:"
# shellcheck disable=SC2086
ps -o pid,etime,args -p $pids

pkill -u "$USER_NAME" -f "$PATTERN"

sleep 2
remaining="$(pgrep -u "$USER_NAME" -f "$PATTERN" || true)"
if [ -n "$remaining" ]; then
    echo "Some processes are still alive after SIGTERM, force-killing:" >&2
    # shellcheck disable=SC2086
    kill -9 $remaining
fi

echo "Done."
