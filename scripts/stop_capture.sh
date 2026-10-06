#!/usr/bin/env bash
# E-stop: stop every recording on every remote and download what was captured.
# Wrapper around collect.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Stopping all recordings..."
exec "${SCRIPT_DIR}/collect.sh" stop all "$@"
