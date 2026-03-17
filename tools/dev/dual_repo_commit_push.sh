#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SDK_REPO="${ROOT_DIR}"
MDMP_REPO="${ROOT_DIR}/local/mdmp-private"

SDK_MSG=""
MDMP_MSG=""
SDK_BRANCH="main"
MDMP_BRANCH="main"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sdk-msg)
      SDK_MSG="$2"
      shift 2
      ;;
    --mdmp-msg)
      MDMP_MSG="$2"
      shift 2
      ;;
    --sdk-branch)
      SDK_BRANCH="$2"
      shift 2
      ;;
    --mdmp-branch)
      MDMP_BRANCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${SDK_MSG}" || -z "${MDMP_MSG}" ]]; then
  echo "Usage: $0 --sdk-msg \"...\" --mdmp-msg \"...\" [--sdk-branch main] [--mdmp-branch main]"
  exit 1
fi

if [[ ! -d "${MDMP_REPO}/.git" ]]; then
  echo "MDMP repo not found at ${MDMP_REPO}"
  exit 1
fi

commit_and_push() {
  local repo_dir="$1"
  local message="$2"
  local branch="$3"
  local label="$4"

  echo "== ${label} =="
  (
    cd "${repo_dir}"
    if [[ -n "$(git status --porcelain)" ]]; then
      git add -A
      git commit -m "${message}"
    else
      echo "No local changes to commit."
    fi
    git push origin "${branch}"
  )
}

commit_and_push "${SDK_REPO}" "${SDK_MSG}" "${SDK_BRANCH}" "SDK"
commit_and_push "${MDMP_REPO}" "${MDMP_MSG}" "${MDMP_BRANCH}" "MDMP"

echo "Done."
