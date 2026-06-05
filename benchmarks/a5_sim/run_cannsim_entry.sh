#!/usr/bin/env bash
# Entry point for cannsim record (standalone executable script).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
exec python3 "${SCRIPT_DIR}/gdn_sim.py" "$@"
