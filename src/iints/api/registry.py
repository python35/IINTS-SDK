from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import List, Optional, Mapping, Sequence, Iterable, Any, cast
import importlib
import importlib.util

try:
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore

from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmMetadata
from iints.core.algorithms.discovery import discover_algorithms

PLUGIN_REGISTRY_SCHEMA_VERSION = "1.0"
IINTS_PLUGIN_HOME_ENV = "IINTS_PLUGIN_HOME"
SUPPORTED_LOCAL_PLUGIN_KINDS = {"algorithm", "patient_model", "data_source", "validator"}


@dataclass
class AlgorithmListing:
    name: str
    class_path: str
    source: str
    metadata: Optional[AlgorithmMetadata]
    status: str = "available"
    error: Optional[str] = None
    plugin_path: Optional[str] = None


@dataclass
class LocalPluginRecord:
    kind: str
    name: str
    installed_path: str
    source_path: str
    module_stem: str
    class_name: Optional[str] = None
    registered_at_utc: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LocalPluginRecord":
        return cls(
            kind=str(payload.get("kind", "")),
            name=str(payload.get("name", "")),
            installed_path=str(payload.get("installed_path", "")),
            source_path=str(payload.get("source_path", "")),
            module_stem=str(payload.get("module_stem", "")),
            class_name=str(payload["class_name"]) if payload.get("class_name") else None,
            registered_at_utc=str(payload.get("registered_at_utc", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "installed_path": self.installed_path,
            "source_path": self.source_path,
            "module_stem": self.module_stem,
            "class_name": self.class_name,
            "registered_at_utc": self.registered_at_utc,
        }


def get_plugin_home() -> Path:
    """Return the writable local plugin home used by CLI installs."""
    override = os.getenv(IINTS_PLUGIN_HOME_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".iints" / "plugins").resolve()


def get_plugin_registry_path() -> Path:
    return get_plugin_home() / "registry.json"


def _empty_registry() -> dict[str, Any]:
    return {"schema_version": PLUGIN_REGISTRY_SCHEMA_VERSION, "plugins": []}


def _read_local_plugin_registry() -> dict[str, Any]:
    path = get_plugin_registry_path()
    if not path.exists():
        return _empty_registry()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Plugin registry must be a JSON object: {path}")
    plugins = payload.get("plugins", [])
    if not isinstance(plugins, list):
        raise ValueError(f"Plugin registry 'plugins' must be a list: {path}")
    return {
        "schema_version": str(payload.get("schema_version") or PLUGIN_REGISTRY_SCHEMA_VERSION),
        "plugins": plugins,
    }


def _write_local_plugin_registry(payload: Mapping[str, Any]) -> Path:
    path = get_plugin_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = {
        "schema_version": PLUGIN_REGISTRY_SCHEMA_VERSION,
        "plugins": list(payload.get("plugins", [])),
    }
    path.write_text(json.dumps(safe_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _slugify_plugin_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip().lower()).strip("-._")
    return slug or "plugin"


def _kind_directory(kind: str) -> str:
    return {
        "algorithm": "algorithms",
        "patient_model": "patient_models",
        "data_source": "data_sources",
        "validator": "validators",
    }[kind]


def _module_name_for_path(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    return f"_iints_local_plugin_{path.stem}_{digest}_{time.monotonic_ns()}"


def _load_algorithm_class_from_path(path: Path) -> type[InsulinAlgorithm]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Plugin file not found: {resolved}")
    module_name = _module_name_for_path(resolved)
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load plugin module spec: {resolved}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__.setdefault("iints", importlib.import_module("iints"))
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    for _, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, InsulinAlgorithm) and obj is not InsulinAlgorithm:
            return obj
    raise ImportError(f"No InsulinAlgorithm subclass found in plugin file: {resolved}")


def _store_local_plugin_record(record: LocalPluginRecord) -> Path:
    payload = _read_local_plugin_registry()
    records = [
        item
        for item in payload.get("plugins", [])
        if not (
            isinstance(item, dict)
            and str(item.get("kind")) == record.kind
            and str(item.get("name")).lower() == record.name.lower()
        )
    ]
    records.append(record.to_dict())
    payload["plugins"] = records
    return _write_local_plugin_registry(payload)


def list_local_plugin_records(kind: str | None = None) -> list[LocalPluginRecord]:
    payload = _read_local_plugin_registry()
    records: list[LocalPluginRecord] = []
    for item in payload.get("plugins", []):
        if not isinstance(item, dict):
            continue
        record = LocalPluginRecord.from_dict(item)
        if kind is None or record.kind == kind:
            records.append(record)
    return sorted(records, key=lambda record: (record.kind, record.name.lower()))


def install_file_plugin(kind: str, source_path: str | Path, name: str | None = None) -> LocalPluginRecord:
    """Install a local extension file into the user plugin home."""
    normalized_kind = kind.strip().lower().replace("-", "_")
    if normalized_kind not in SUPPORTED_LOCAL_PLUGIN_KINDS:
        supported = ", ".join(sorted(SUPPORTED_LOCAL_PLUGIN_KINDS))
        raise ValueError(f"Unsupported plugin kind '{kind}'. Supported kinds: {supported}")

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Plugin file not found: {source}")
    if source.suffix != ".py":
        raise ValueError("Local plugins must be Python .py files.")

    display_name = (name or source.stem).strip()
    if not display_name:
        raise ValueError("Plugin name cannot be empty.")
    slug = _slugify_plugin_name(display_name)
    target_dir = get_plugin_home() / _kind_directory(normalized_kind)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{slug}.py"
    if source != target:
        shutil.copy2(source, target)

    record = LocalPluginRecord(
        kind=normalized_kind,
        name=display_name,
        installed_path=str(target),
        source_path=str(source),
        module_stem=target.stem,
        class_name=None,
        registered_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    _store_local_plugin_record(record)
    return record


def install_algorithm_plugin(source_path: str | Path, name: str | None = None) -> LocalPluginRecord:
    """Validate and install a local InsulinAlgorithm plugin file."""
    source = Path(source_path).expanduser().resolve()
    algorithm_class = _load_algorithm_class_from_path(source)
    instance = algorithm_class()
    metadata = instance.get_algorithm_metadata()
    display_name = (name or metadata.name or source.stem).strip()
    record = install_file_plugin("algorithm", source, display_name)
    record.class_name = algorithm_class.__name__
    _store_local_plugin_record(record)
    return record


def uninstall_local_plugin(name: str, kind: str | None = None, *, remove_file: bool = False) -> bool:
    payload = _read_local_plugin_registry()
    normalized_kind = kind.strip().lower().replace("-", "_") if kind else None
    kept: list[dict[str, Any]] = []
    removed = False
    for item in payload.get("plugins", []):
        if not isinstance(item, dict):
            continue
        record = LocalPluginRecord.from_dict(item)
        matches_name = record.name.lower() == name.lower()
        matches_kind = normalized_kind is None or record.kind == normalized_kind
        if matches_name and matches_kind:
            removed = True
            if remove_file:
                try:
                    Path(record.installed_path).unlink(missing_ok=True)
                except Exception:
                    pass
            continue
        kept.append(record.to_dict())
    if removed:
        payload["plugins"] = kept
        _write_local_plugin_registry(payload)
    return removed


def _load_entry_point(ep) -> AlgorithmListing:
    try:
        obj = ep.load()
        if isinstance(obj, type) and issubclass(obj, InsulinAlgorithm):
            instance = obj()
            meta = instance.get_algorithm_metadata()
            return AlgorithmListing(
                name=meta.name,
                class_path=f"{obj.__module__}.{obj.__name__}",
                source=f"entry_point:{ep.name}",
                metadata=meta,
            )
        return AlgorithmListing(
            name=ep.name,
            class_path=f"{ep.module}:{ep.attr}",
            source=f"entry_point:{ep.name}",
            metadata=None,
            status="invalid",
            error="Entry point does not resolve to an InsulinAlgorithm",
        )
    except Exception as exc:
        return AlgorithmListing(
            name=ep.name,
            class_path=f"{ep.module}:{ep.attr}",
            source=f"entry_point:{ep.name}",
            metadata=None,
            status="unavailable",
            error=str(exc),
        )


def list_algorithm_plugins() -> List[AlgorithmListing]:
    listings: List[AlgorithmListing] = []

    # Built-in discovery
    try:
        discovered = discover_algorithms()
        for name, cls in discovered.items():
            try:
                instance = cls()
                meta = instance.get_algorithm_metadata()
                listings.append(
                    AlgorithmListing(
                        name=meta.name,
                        class_path=f"{cls.__module__}.{cls.__name__}",
                        source="builtin",
                        metadata=meta,
                    )
                )
            except Exception as exc:
                listings.append(
                    AlgorithmListing(
                        name=name,
                        class_path=f"{cls.__module__}.{cls.__name__}",
                        source="builtin",
                        metadata=None,
                        status="unavailable",
                        error=str(exc),
                    )
                )
    except Exception:
        pass

    # Entry points
    try:
        eps = importlib_metadata.entry_points()
        group: Iterable[Any]
        if hasattr(eps, "select"):
            group = list(eps.select(group="iints.algorithms"))
        else:
            eps_mapping = cast(Mapping[str, Sequence[object]], eps)
            group = eps_mapping.get("iints.algorithms", ())
        for ep in group:
            listings.append(_load_entry_point(ep))
    except Exception:
        pass

    # Local user-installed plugins
    try:
        for record in list_local_plugin_records("algorithm"):
            plugin_path = Path(record.installed_path)
            try:
                algorithm_class = _load_algorithm_class_from_path(plugin_path)
                instance = algorithm_class()
                meta = instance.get_algorithm_metadata()
                listings.append(
                    AlgorithmListing(
                        name=record.name or meta.name,
                        class_path=f"{plugin_path}:{algorithm_class.__name__}",
                        source="local",
                        metadata=meta,
                        plugin_path=str(plugin_path),
                    )
                )
            except Exception as exc:
                listings.append(
                    AlgorithmListing(
                        name=record.name,
                        class_path=f"{plugin_path}:{record.class_name or ''}",
                        source="local",
                        metadata=None,
                        status="unavailable",
                        error=str(exc),
                        plugin_path=str(plugin_path),
                    )
                )
    except Exception as exc:
        listings.append(
            AlgorithmListing(
                name="local plugin registry",
                class_path=str(get_plugin_registry_path()),
                source="local",
                metadata=None,
                status="unavailable",
                error=str(exc),
            )
        )

    return listings
