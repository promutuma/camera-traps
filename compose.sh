#!/usr/bin/env bash
# Wrapper so `docker compose` always uses BuildKit + pip/npm cache mounts (Dockerfile).
# Usage: ./compose.sh up --build   (same flags as docker compose)
set -euo pipefail

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
# Skip SBOM/provenance attestation — saves minutes on large ML image exports.
export BUILDX_NO_DEFAULT_ATTESTATIONS=1

exec docker compose "$@"
