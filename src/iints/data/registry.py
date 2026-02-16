from __future__ import annotations

import json
import urllib.request
import zipfile
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from importlib.abc import Traversable

try:  # Python 3.9+
    from importlib.resources import files
except Exception:  # pragma: no cover
    from importlib import resources as files  # type: ignore


class DatasetRegistryError(RuntimeError):
    pass


class DatasetFetchError(RuntimeError):
    pass


def _registry_path() -> Traversable:
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
    ids: List[str] = []
    for entry in load_dataset_registry():
        dataset_id = entry.get("id")
        if isinstance(dataset_id, str) and dataset_id:
            ids.append(dataset_id)
    return ids


def _download_file(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as exc:
        raise DatasetFetchError(f"Failed to download {url}: {exc}") from exc
    return output_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_expected_hash(dataset: Dict[str, Any], index: int = 0) -> Optional[str]:
    expected = dataset.get("sha256")
    if isinstance(expected, list):
        if index < len(expected):
            return expected[index] or None
        return None
    if isinstance(expected, str):
        return expected or None
    return None


def _maybe_extract_zip(path: Path, output_dir: Path) -> None:
    if path.suffix.lower() != ".zip":
        return
    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
    except Exception as exc:
        raise DatasetFetchError(f"Failed to extract {path.name}: {exc}") from exc


def fetch_dataset(
    dataset_id: str,
    output_dir: Path,
    extract: bool = True,
    verify: bool = True,
) -> List[Path]:
    dataset = get_dataset(dataset_id)
    urls = dataset.get("download_urls") or []
    access = dataset.get("access", "manual")

    if access == "bundled":
        bundled_path = dataset.get("bundled_path")
        if not bundled_path:
            raise DatasetFetchError("Bundled dataset missing bundled_path entry.")
        try:
            source_path = files("iints.data").joinpath(bundled_path)  # type: ignore[attr-defined]
        except Exception as exc:
            raise DatasetFetchError(f"Unable to locate bundled dataset: {exc}") from exc
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / Path(bundled_path).name
        # Traversable may not be a real filesystem path; stream bytes instead.
        with source_path.open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        expected = _get_expected_hash(dataset, 0)
        if verify and expected:
            actual = _sha256(target)
            if actual != expected:
                raise DatasetFetchError(f"SHA-256 mismatch for {target.name}. Expected {expected}, got {actual}.")
        return [target]

    if not urls:
        raise DatasetFetchError(
            "This dataset requires manual download or approval. Use 'iints data info' for instructions."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    for idx, url in enumerate(urls):
        filename = url.split("/")[-1]
        target = output_dir / filename
        downloaded.append(_download_file(url, target))
        expected = _get_expected_hash(dataset, idx)
        if verify and expected:
            actual = _sha256(target)
            if actual != expected:
                raise DatasetFetchError(f"SHA-256 mismatch for {target.name}. Expected {expected}, got {actual}.")
        elif verify and expected is None:
            # If no hash is provided, at least emit the computed hash for user reference.
            actual = _sha256(target)
            checksum_path = output_dir / "SHA256SUMS.txt"
            with checksum_path.open("a") as handle:
                handle.write(f"{actual}  {target.name}\n")
        if extract:
            _maybe_extract_zip(target, output_dir)
    return downloaded
