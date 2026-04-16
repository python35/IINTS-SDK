from pathlib import Path
import zipfile

import pytest

from iints.data.registry import DatasetFetchError, _maybe_extract_zip, _validate_download_url, fetch_dataset


def test_registry_rejects_non_https_remote_urls() -> None:
    with pytest.raises(DatasetFetchError, match="must use https"):
        _validate_download_url("http://example.com/dataset.zip")


def test_registry_allows_https_and_loopback_http() -> None:
    assert _validate_download_url("https://example.com/dataset.zip") == "https://example.com/dataset.zip"
    assert _validate_download_url("http://127.0.0.1:8000/dataset.zip") == "http://127.0.0.1:8000/dataset.zip"


def test_registry_rejects_zip_slip_members(tmp_path: Path) -> None:
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")

    with pytest.raises(DatasetFetchError, match="unsafe ZIP member"):
        _maybe_extract_zip(archive, tmp_path / "extract")


def test_registry_blocks_misleading_verify_without_pinned_hash(monkeypatch, tmp_path: Path) -> None:
    dataset = {
        "id": "public-demo",
        "access": "public-download",
        "download_urls": ["https://example.com/dataset.zip"],
        "sha256": None,
    }

    def _fake_download(url: str, output_path: Path) -> Path:
        output_path.write_text("demo", encoding="utf-8")
        return output_path

    monkeypatch.setattr("iints.data.registry.get_dataset", lambda dataset_id: dataset)
    monkeypatch.setattr("iints.data.registry._download_file", _fake_download)

    with pytest.raises(DatasetFetchError, match="does not publish a pinned SHA-256"):
        fetch_dataset("public-demo", output_dir=tmp_path / "downloads", verify=True)
