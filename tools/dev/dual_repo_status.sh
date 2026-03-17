#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SDK_REPO="${ROOT_DIR}"
MDMP_REPO="${ROOT_DIR}/local/mdmp-private"

if [[ ! -d "${MDMP_REPO}/.git" ]]; then
  echo "MDMP repo not found at ${MDMP_REPO}"
  exit 1
fi

echo "== SDK repo =="
(
  cd "${SDK_REPO}"
  git rev-parse --abbrev-ref HEAD
  git status --short
)

echo

echo "== MDMP repo =="
(
  cd "${MDMP_REPO}"
  git rev-parse --abbrev-ref HEAD
  git status --short
)
