#!/usr/bin/env bash
# Backs up the `evaluations`, `experiments` and `ts_metadata` tables (the
# results tsadquality/plots and Experiment._exists_in_postgres() depend on)
# from the Postgres container defined in docker-compose.yml, via `pg_dump`
# run inside the container so no local `pg_dump`/`psql` install is needed.
#
# Usage:
#   scripts/backup_postgres_tables.sh [output_dir]
#
# output_dir defaults to backups/ under the project root. Each run writes a
# timestamped, gzip-compressed SQL dump; nothing is ever deleted or overwritten.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-"$PROJECT_ROOT/backups"}"
TABLES=(evaluations experiments ts_metadata)

# shellcheck disable=SC1091
set -a
source "$PROJECT_ROOT/.env"
set +a

mkdir -p "$OUTPUT_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_path="$OUTPUT_DIR/tsadquality_${timestamp}.sql.gz"

table_args=()
for table in "${TABLES[@]}"; do
    table_args+=(-t "$table")
done

echo "Backing up tables (${TABLES[*]}) to $output_path ..."

docker compose -f "$PROJECT_ROOT/docker-compose.yml" exec -T postgres \
    pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" "${table_args[@]}" \
    | gzip > "$output_path"

echo "Done: $output_path"
