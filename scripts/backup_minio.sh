#!/usr/bin/env bash
# Backs up the MinIO bucket ($MINIO_BUCKET: the PERFECT experiments' decision
# scores) from the MinIO service in docker-compose.yml, as plain objects (e.g.
# perfect/<detector>/<seed>/<dataset>.npy) in a timestamped .tar.gz. Uses the
# `mc` client bundled in the compose file's MinIO image, so no local install is
# needed. Run it on the machine hosting MinIO.
#
# Usage:
#   scripts/backup_minio.sh [output_dir]
#
# output_dir defaults to backups/ under the project root. Each run writes a new
# archive; nothing is ever deleted or overwritten. To restore, extract it and
# mirror the bucket folder back, e.g. `mc mirror <extracted>/<bucket> <alias>/<bucket>`.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-"$PROJECT_ROOT/backups"}"

# shellcheck disable=SC1091
set -a
source "$PROJECT_ROOT/.env"
set +a

MINIO_IMAGE="$(docker compose -f "$PROJECT_ROOT/docker-compose.yml" config --images | grep minio)"

mkdir -p "$OUTPUT_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_path="$OUTPUT_DIR/minio_${MINIO_BUCKET}_${timestamp}.tar.gz"
staging_dir="$(mktemp -d "$OUTPUT_DIR/.minio_staging.XXXXXX")"
trap 'rm -rf "$staging_dir"' EXIT

echo "Backing up MinIO bucket '$MINIO_BUCKET' to $output_path ..."

# Credentials are passed as env vars (not in the command line) so they don't show up in `ps`.
docker run --rm --network host \
    --user "$(id -u):$(id -g)" \
    -e MC_CONFIG_DIR=/tmp/.mc \
    -e MINIO_ROOT_USER -e MINIO_ROOT_PASSWORD \
    -v "$staging_dir":/backup \
    --entrypoint sh "$MINIO_IMAGE" -c \
    "mc alias set src http://127.0.0.1:${MINIO_MAPPED_PORT} \"\$MINIO_ROOT_USER\" \"\$MINIO_ROOT_PASSWORD\" >/dev/null \
     && mc mirror --quiet src/${MINIO_BUCKET} /backup/${MINIO_BUCKET} >/dev/null"

tar czf "$output_path" -C "$staging_dir" "$MINIO_BUCKET"

echo "Done: $output_path ($(du -h "$output_path" | cut -f1), $(tar tzf "$output_path" | grep -c '\.npy$') objects)"
