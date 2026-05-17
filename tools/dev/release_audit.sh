#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_VERSION="${1:-}"

if [[ -z "$RAW_VERSION" ]]; then
  echo "Usage: tools/dev/release_audit.sh <version>" >&2
  echo "Example: tools/dev/release_audit.sh 1.5.5" >&2
  exit 2
fi

VERSION="${RAW_VERSION#v}"
TAG="v$VERSION"

cd "$ROOT_DIR"

require_contains() {
  local file="$1"
  local pattern="$2"
  local message="$3"
  if ! grep -Fq "$pattern" "$file"; then
    echo "[IINTS] release audit failed: $message" >&2
    echo "  file: $file" >&2
    echo "  expected: $pattern" >&2
    exit 1
  fi
}

require_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "[IINTS] release audit failed: missing $file" >&2
    exit 1
  fi
}

require_file "docs/releases/$TAG.md"

require_contains "pyproject.toml" "version = \"$VERSION\"" "pyproject version is not $VERSION"
require_contains "src/iints/__init__.py" "__version__ = \"$VERSION\"" "fallback __version__ is not $VERSION"
require_contains "docs/releases/$TAG.md" "# $TAG" "release note header is not $TAG"
require_contains "docs/releases/INDEX.md" "[$TAG]($TAG.md)" "release archive does not link $TAG"
require_contains "mkdocs.yml" "Latest ($TAG): releases/$TAG.md" "MkDocs latest release nav is not $TAG"
require_contains "docs/UPDATING.md" "==$VERSION" "updating guide does not show the pinned version"
require_contains "docs/manuals/IINTS-AF_SDK_Manual.md" "Version $VERSION" "manual markdown version is not $VERSION"
require_contains "docs/manuals/pandoc.yaml" "Version $VERSION" "manual PDF metadata version is not $VERSION"

if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "[IINTS] release audit: local tag $TAG exists."
else
  echo "[IINTS] release audit: local tag $TAG does not exist yet."
fi

echo "[IINTS] release audit passed for $TAG."
