from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Python 3.9+
    from importlib.resources import files
except Exception:  # pragma: no cover
    from importlib import resources as files  # type: ignore


class DatasetRegistryError(RuntimeError):
    pass


class DatasetFetchError(RuntimeError):
    pass


def _registry_path() -> Path:
    try:
        return files("iints.data").joinpath("datasets.json")  # type: ignore[attr-defined]
    except Exception as exc:
        raise DatasetRegistryError(f"Unable to locate datasets.json: {exc}") from exc


def load_dataset_registry() -> List[Dict[str, Any]]:
    registry_path = _registry_path()
    return json.loads(registry_path.read_text())


def get_dataset(dataset_id: str) -> Dict[str, Any]:
    for entry in load_dataset_registry():
        if entry.get("id") == dataset_id:
            return entry
    raise DatasetRegistryError(f"Unknown dataset '{dataset_id}'. Run 'iints data list' to see options.")


def list_dataset_ids() -> List[str]:
    return [entry.get("id") for entry in load_dataset_registry()]


def _download_file(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as exc:
        raise DatasetFetchError(f"Failed to download {url}: {exc}") from exc
    return output_path


def _maybe_extract_zip(path: Path, output_dir: Path) -> None:
    if path.suffix.lower() != ".zip":
        return
    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
    except Exception as exc:
        raise DatasetFetchError(f"Failed to extract {path.name}: {exc}") from exc


def fetch_dataset(dataset_id: str, output_dir: Path, extract: bool = True) -> List[Path]:
    dataset = get_dataset(dataset_id)
    urls = dataset.get("download_urls") or []
    if not urls:
        raise DatasetFetchError(
            "This dataset requires manual download or approval. Use 'iints data info' for instructions."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    for url in urls:
        filename = url.split("/")[-1]
        target = output_dir / filename
        downloaded.append(_download_file(url, target))
        if extract:
            _maybe_extract_zip(target, output_dir)
    return downloaded
