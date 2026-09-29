#!/usr/bin/env bash
# Back up wildlife_data.db and uploads/ before storage-related changes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BACKUP_DIR="backups/pre_storage_cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
DB="${DB_PATH:-wildlife_data.db}"
if [[ -f "$DB" ]]; then
  cp "$DB" "$BACKUP_DIR/"
fi
if [[ -d uploads ]]; then
  cp -r uploads "$BACKUP_DIR/"
fi
echo "Backup written to: $BACKUP_DIR"
du -sh "$BACKUP_DIR"
echo "File count: $(find "$BACKUP_DIR" -type f | wc -l)"
