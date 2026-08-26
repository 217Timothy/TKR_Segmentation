#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -x "$PACKAGE_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$PACKAGE_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
export PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-help}" in
  infer)
    shift
    exec "$PYTHON_BIN" -m tkr_inference.cli "$@"
    ;;
  verify)
    shift
    exec "$PYTHON_BIN" "$PACKAGE_DIR/verify_package.py" "$@"
    ;;
  help|-h|--help)
    echo "Usage:"
    echo "  ./run.sh verify"
    echo "  ./run.sh infer --input IMAGE_OR_DIR --output NEW_OUTPUT_DIR [--device auto]"
    ;;
  *)
    echo "Unknown command: $1" >&2
    exit 2
    ;;
esac
