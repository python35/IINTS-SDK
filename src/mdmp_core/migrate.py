from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple
import copy
import json


CURRENT_SPEC_VERSION = "1.0"
KNOWN_VERSIONS = ("0.2", "0.3", "1.0")

MigrationFn = Callable[[dict], dict]
MigrationRegistry: Dict[Tuple[str, str], MigrationFn] = {}


@dataclass(frozen=True)
class MigrationResult:
    source: str
    destination: str
    before_version: str
    after_version: str
    changed: bool


def migration(from_v: str, to_v: str):
    """Decorator for registering a migration step."""

    def decorator(fn: MigrationFn) -> MigrationFn:
        MigrationRegistry[(from_v, to_v)] = fn
        return fn

    return decorator


@migration("0.2", "0.3")
def migrate_0_2_to_0_3(artifact: dict) -> dict:
    a = copy.deepcopy(artifact)
    a["spec_version"] = "0.3"
    # 0.2 sometimes used issuer instead of signed_by
    if "issuer" in a and "signed_by" not in a:
        a["signed_by"] = a.pop("issuer")
    # 0.2 artifacts without signature fields should be explicit
    if "signature" not in a and "signature_status" not in a:
        a["signature_status"] = "unsigned"
    return a


@migration("0.3", "1.0")
def migrate_0_3_to_1_0(artifact: dict) -> dict:
    a = copy.deepcopy(artifact)
    a["spec_version"] = "1.0"
    # 0.3 sometimes used dataset_grade instead of grade
    if "dataset_grade" in a and "grade" not in a:
        a["grade"] = a.pop("dataset_grade")
    # 1.0 requires explicit object type for portability
    if "mdmp_object" not in a:
        a["mdmp_object"] = "dataset_card"
    return a


def _normalize_version(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return "0.2"
    return text


def find_migration_path(from_v: str, to_v: str) -> list[tuple[str, str]]:
    chain = list(KNOWN_VERSIONS)
    try:
        start = chain.index(from_v)
        end = chain.index(to_v)
    except ValueError:
        return []

    if start >= end:
        return []

    path: list[tuple[str, str]] = []
    for idx in range(start, end):
        edge = (chain[idx], chain[idx + 1])
        if edge not in MigrationRegistry:
            return []
        path.append(edge)
    return path


def migrate(artifact: dict, target_version: str = CURRENT_SPEC_VERSION) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be a JSON object")

    current = _normalize_version(str(artifact.get("spec_version", "0.2")))
    target = _normalize_version(target_version)

    if current == target:
        return copy.deepcopy(artifact)

    path = find_migration_path(current, target)
    if not path:
        raise ValueError(
            f"No migration path from {current} to {target}. "
            f"Known versions: {', '.join(KNOWN_VERSIONS)}"
        )

    result = copy.deepcopy(artifact)
    for from_v, to_v in path:
        fn = MigrationRegistry[(from_v, to_v)]
        result = fn(result)
    return result


def detect_version(artifact: dict) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be a JSON object")

    current = _normalize_version(str(artifact.get("spec_version", "0.2")))
    path = find_migration_path(current, CURRENT_SPEC_VERSION)
    return {
        "current_version": current,
        "target_version": CURRENT_SPEC_VERSION,
        "migration_available": len(path) > 0,
        "migration_path": [f"{a}->{b}" for a, b in path],
        "up_to_date": current == CURRENT_SPEC_VERSION,
    }


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact file must contain a JSON object")
    return payload


def save_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def migrate_file(
    source: str | Path,
    *,
    target_version: str = CURRENT_SPEC_VERSION,
    destination: str | Path | None = None,
    in_place: bool = False,
    backup: bool = False,
) -> MigrationResult:
    src = Path(source)
    artifact = load_json(src)
    before_version = _normalize_version(str(artifact.get("spec_version", "0.2")))
    migrated = migrate(artifact, target_version=target_version)
    after_version = _normalize_version(str(migrated.get("spec_version", before_version)))
    changed = migrated != artifact

    if destination is not None and in_place:
        raise ValueError("use either destination or in_place, not both")

    if in_place:
        if backup:
            backup_path = src.with_suffix(src.suffix + ".bak")
            backup_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        save_json(src, migrated)
        dest_path = src
    else:
        dest_path = Path(destination) if destination is not None else src
        save_json(dest_path, migrated)

    return MigrationResult(
        source=str(src),
        destination=str(dest_path),
        before_version=before_version,
        after_version=after_version,
        changed=changed,
    )


def iter_json_files(root: str | Path) -> Iterable[Path]:
    base = Path(root)
    for path in sorted(base.rglob("*.json")):
        if path.is_file():
            yield path
