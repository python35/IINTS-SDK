import pytest

from iints.data.registry import DatasetFetchError, _validate_download_url


def test_registry_rejects_non_https_remote_urls() -> None:
    with pytest.raises(DatasetFetchError, match="must use https"):
        _validate_download_url("http://example.com/dataset.zip")


def test_registry_allows_https_and_loopback_http() -> None:
    assert _validate_download_url("https://example.com/dataset.zip") == "https://example.com/dataset.zip"
    assert _validate_download_url("http://127.0.0.1:8000/dataset.zip") == "http://127.0.0.1:8000/dataset.zip"
