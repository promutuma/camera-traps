#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -f "$SCRIPT_DIR/dev.sh" ]]; then
    echo "[ERROR] dev.sh not found. Please re-run ./install.sh."
    exit 1
fi
exec bash "$SCRIPT_DIR/dev.sh"
