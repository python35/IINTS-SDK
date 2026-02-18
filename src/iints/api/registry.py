from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Mapping, Sequence, cast
import importlib

try:
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore

from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmMetadata
from iints.core.algorithms.discovery import discover_algorithms


@dataclass
class AlgorithmListing:
    name: str
    class_path: str
    source: str
    metadata: Optional[AlgorithmMetadata]
    status: str = "available"
    error: Optional[str] = None


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
        if hasattr(eps, "select"):
            group = eps.select(group="iints.algorithms")
        else:
            eps_mapping = cast(Mapping[str, Sequence[object]], eps)
            group = eps_mapping.get("iints.algorithms", ())
        for ep in group:
            listings.append(_load_entry_point(ep))
    except Exception:
        pass

    return listings
