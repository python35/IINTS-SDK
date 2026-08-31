from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import hashlib
import platform
import random
import sys
import uuid
import subprocess
import os
from importlib import metadata as importlib_metadata
from typing import Mapping, cast
from pathlib import Path
from typing import Any, Dict, Optional, Union

from iints.core.formula_registry import FORMULAS, FORMULA_REGISTRY_VERSION

try:
    from importlib.metadata import version as pkg_version
except Exception:  # pragma: no cover - stdlib fallback
    pkg_version = None  # type: ignore[assignment]

RUN_METADATA_FORMAT_VERSION = "1.0"
RUN_MANIFEST_FORMAT_VERSION = "1.0"
RESULTS_CSV_FORMAT_VERSION = "1.0"


def resolve_seed(seed: Optional[int]) -> int:
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    return int(seed)


def generate_run_id(seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:6]
    return f"{timestamp}-{seed}-{token}"


def resolve_output_dir(output_dir: Optional[Union[str, Path]], run_id: str) -> Path:
    if output_dir is None:
        output_path = Path.cwd() / "results" / run_id
    else:
        output_path = Path(output_dir).expanduser()
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        else:
            output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _serialize_payload(payload: Any) -> Any:
    if is_dataclass(payload) and not isinstance(payload, type):
        return asdict(payload)
    if isinstance(payload, Path):
        return str(payload)
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    safe_payload = {key: _serialize_payload(value) for key, value in payload.items()}
    path.write_text(json.dumps(safe_payload, indent=2, sort_keys=True))


def get_sdk_version(package_name: str = "iints-sdk-python35") -> str:
    if pkg_version is None:
        return "unknown"
    try:
        return pkg_version(package_name)
    except Exception:
        return "unknown"


def _git_provenance() -> Dict[str, Any]:
    candidates = [Path.cwd(), Path(__file__).resolve().parents[3]]
    for candidate in candidates:
        try:
            root_result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=candidate,
                capture_output=True,
                text=True,
                check=False,
            )
            if root_result.returncode != 0:
                continue
            repository_root = Path(root_result.stdout.strip())
            sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=False,
            )
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=False,
            )
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=False,
            )
            return {
                "available": sha_result.returncode == 0,
                "sha": sha_result.stdout.strip() if sha_result.returncode == 0 else "unknown",
                "branch": branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown",
                "dirty": bool(status_result.stdout.strip()) if status_result.returncode == 0 else None,
                "repository_path_recorded": False,
            }
        except (OSError, ValueError):
            continue
    return {
        "available": False,
        "sha": "unknown",
        "branch": "unknown",
        "dirty": None,
        "repository_path_recorded": False,
    }


def _formula_provenance() -> Dict[str, Any]:
    evidence_counts: Dict[str, int] = {}
    for formula in FORMULAS:
        evidence_counts[formula.evidence_class] = evidence_counts.get(formula.evidence_class, 0) + 1
    return {
        "registry_version": FORMULA_REGISTRY_VERSION,
        "formula_count": len(FORMULAS),
        "evidence_class_counts": evidence_counts,
        "ai_numeric_authority": False,
    }


def _configured_evidence_files(config: Dict[str, Any], output_dir: Path) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    seen: set[Path] = set()

    def visit(value: Any, key_path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, (*key_path, str(key)))
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, (*key_path, str(index)))
            return
        if not key_path or not isinstance(value, (str, Path)):
            return
        evidence_key = key_path[-1].lower()
        if not any(token in evidence_key for token in ("manifest", "checkpoint", "model_path", "calibration")):
            return
        candidate = Path(value).expanduser()
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            return
        if resolved in seen or not resolved.is_file():
            return
        seen.add(resolved)
        try:
            display_path = resolved.relative_to(output_dir.resolve()).as_posix()
            scope = "run_relative"
        except ValueError:
            display_path = resolved.name
            scope = "external_path_redacted"
        records.append(
            {
                "config_key": ".".join(key_path),
                "path": display_path,
                "path_scope": scope,
                "sha256": compute_sha256(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )

    visit(config, ())
    return records


def build_run_metadata(run_id: str, seed: int, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    dependencies = []
    try:
        for dist in importlib_metadata.distributions():
            metadata = cast(Mapping[str, str], dist.metadata)
            name = metadata.get("Name") or metadata.get("Summary") or metadata.get("name")
            if name:
                dependencies.append({"name": name, "version": dist.version})
        dependencies = sorted(dependencies, key=lambda item: item["name"].lower())
    except Exception:
        dependencies = []

    git = _git_provenance()

    return {
        "schema_version": RUN_METADATA_FORMAT_VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "output_dir": ".",
        "output_path_policy": "run-relative; host filesystem path intentionally omitted",
        "sdk_version": get_sdk_version(),
        "format_versions": {
            "run_metadata": RUN_METADATA_FORMAT_VERSION,
            "run_manifest": RUN_MANIFEST_FORMAT_VERSION,
            "results_csv": RESULTS_CSV_FORMAT_VERSION,
        },
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        # Keep git_sha for consumers of the v1 schema and provide the richer,
        # privacy-preserving source-control record alongside it.
        "git_sha": git["sha"],
        "source_control": git,
        "formula_registry": _formula_provenance(),
        "parameter_provenance": {
            "patient_parameters": "resolved configuration embedded in config",
            "scenario_parameters": "resolved configuration embedded in config",
            "configured_evidence_files": _configured_evidence_files(config, output_dir),
        },
        "dependencies": dependencies,
        "config": config,
    }


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(output_dir: Path, files: Dict[str, Path]) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "schema_version": RUN_MANIFEST_FORMAT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": ".",
        "path_policy": "paths are relative to the run root; external host paths are redacted",
        "files": {},
    }
    for label, path in files.items():
        resolved = path.expanduser().resolve()
        try:
            manifest_path = resolved.relative_to(output_dir.expanduser().resolve()).as_posix()
            path_scope = "run_relative"
        except ValueError:
            manifest_path = resolved.name
            path_scope = "external_path_redacted"
        entry: Dict[str, Any] = {"path": manifest_path, "path_scope": path_scope}
        if resolved.is_file():
            entry["sha256"] = compute_sha256(resolved)
            entry["size_bytes"] = resolved.stat().st_size
        elif resolved.exists():
            entry["invalid_type"] = "not_a_regular_file"
        else:
            entry["missing"] = True
        manifest["files"][label] = entry
    return manifest


def maybe_sign_manifest(manifest_path: Path) -> Optional[Path]:
    """Optionally sign a manifest if IINTS_ATTEST_KEY is set."""
    key_path = os.getenv("IINTS_ATTEST_KEY")
    if not key_path:
        return None
    sig_path = manifest_path.with_suffix(".sig")
    try:
        result = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", key_path, "-out", str(sig_path), str(manifest_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return sig_path
    except Exception:
        return None
