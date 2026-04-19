from __future__ import annotations

import pytest

from iints.data.nightscout import NightscoutConfig
from iints.data.tidepool import TidepoolClient


def test_nightscout_config_rejects_unsafe_urls() -> None:
    with pytest.raises(ValueError):
        NightscoutConfig(url="file:///etc/passwd")

    with pytest.raises(ValueError):
        NightscoutConfig(url="http://example.com")

    with pytest.raises(ValueError):
        NightscoutConfig(url="https://user:pass@example.com")


def test_cloud_clients_allow_https_and_local_http() -> None:
    config = NightscoutConfig(url="https://demo.nightscout.example")
    assert config.url == "https://demo.nightscout.example"

    local_config = NightscoutConfig(url="http://localhost:1337")
    assert local_config.url == "http://localhost:1337"

    client = TidepoolClient(base_url="https://api.tidepool.org/")
    assert client.base_url == "https://api.tidepool.org"

    with pytest.raises(ValueError):
        TidepoolClient(base_url="http://tidepool.example")
