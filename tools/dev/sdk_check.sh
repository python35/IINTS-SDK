#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-quick}"

cd "$ROOT_DIR"

usage() {
  cat <<'USAGE'
Usage: tools/dev/sdk_check.sh [quick|edge|docs|full|release]

Modes:
  quick    Fast local confidence check for day-to-day changes.
  edge     Edge/Pi/UNO Q focused tests.
  docs     Documentation and manual build checks.
  full     Full local pre-push check.
  release  Full check plus release-version audit.

Environment:
  IINTS_RELEASE_VERSION=1.5.6  Required for release mode.
USAGE
}

run_quick() {
  echo "[IINTS] quick: architecture boundaries"
  python3 tools/ci/check_architecture_boundaries.py

  echo "[IINTS] quick: targeted tests"
  python3 -m pytest \
    tests/core/test_numeric_guards.py \
    tests/live_patient/test_long_study.py \
    tests/live_patient/test_edge_ops.py \
    tests/test_cli_edge_runtime.py \
    -q

  echo "[IINTS] quick: lint and types"
  flake8 src/iints tests tools
  mypy src/iints/
}

run_edge() {
  echo "[IINTS] edge: Pi/UNO Q and long-study tests"
  python3 -m pytest \
    tests/live_patient/test_long_study.py \
    tests/live_patient/test_edge_ops.py \
    tests/test_cli_edge_runtime.py \
    -q
}

run_docs() {
  echo "[IINTS] docs: generated API reference"
  python3 tools/docs/generate_api_reference.py

  echo "[IINTS] docs: MkDocs strict build"
  mkdocs build --strict

  echo "[IINTS] docs: technical manual PDF"
  bash tools/docs/build_manuals.sh
}

run_full() {
  echo "[IINTS] full: complete test suite"
  python3 -m pytest tests/ -q

  echo "[IINTS] full: lint"
  flake8 .

  echo "[IINTS] full: type check"
  mypy src/iints/

  echo "[IINTS] full: docs"
  python3 tools/docs/generate_api_reference.py
  mkdocs build --strict

  echo "[IINTS] full: package build"
  python3 -m build

  echo "[IINTS] full: manual PDF"
  bash tools/docs/build_manuals.sh
}

case "$MODE" in
  quick)
    run_quick
    ;;
  edge)
    run_edge
    ;;
  docs)
    run_docs
    ;;
  full)
    run_full
    ;;
  release)
    if [[ -z "${IINTS_RELEASE_VERSION:-}" ]]; then
      echo "Set IINTS_RELEASE_VERSION first, for example: IINTS_RELEASE_VERSION=1.5.6 tools/dev/sdk_check.sh release" >&2
      exit 2
    fi
    run_full
    bash tools/dev/release_audit.sh "$IINTS_RELEASE_VERSION"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

echo "[IINTS] $MODE check passed."
