#!/usr/bin/env bash
# Run inside the existing successful main-experiment Python environment.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_ROOT="${1:-$HOME/tievnas2/YinCH}"
REF_ROOT="${2:-$SEARCH_ROOT/GENTLE-consistant}"
DIAG_ROOT="${3:-$SEARCH_ROOT/shift-diagnostics-v3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ ! -d "$SEARCH_ROOT" || ! -d "$REF_ROOT" ]]; then
    printf '%s\n' 'Historical paths do not exist. Pass actual search root, repo root, and diagnostic root as arguments.' >&2
    exit 2
fi
AUDIT_OUT="$DIAG_ROOT/audits/A-$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$PYTHON_BIN" "$SCRIPT_DIR/audit.py" \
    --repo-root "$REF_ROOT" \
    --search-root "$SEARCH_ROOT" \
    --search-root "$SEARCH_ROOT/shared_protocol/shared-main-v1" \
    --hash-checkpoints \
    --out "$AUDIT_OUT"
printf '\n%s\n' "Resource audit: $AUDIT_OUT" 'Stage A still requires verified assignments, adapters and real-environment smoke.'
