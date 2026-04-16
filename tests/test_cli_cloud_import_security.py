from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data.importer import ImportResult


runner = CliRunner()


def test_import_nightscout_reads_secrets_from_env(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, str | None] = {}

    def _fake_import(config, scenario_name: str):
        seen["api_secret"] = config.api_secret
        seen["token"] = config.token
        return ImportResult(
            dataframe=pd.DataFrame(
                [{"timestamp": 0.0, "glucose": 123.0, "carbs": 0.0, "insulin": 0.0}]
            ),
            scenario={"name": scenario_name},
        )

    monkeypatch.setenv("IINTS_NS_SECRET", "secret-from-env")
    monkeypatch.setenv("IINTS_NS_TOKEN", "token-from-env")
    monkeypatch.setattr("iints.cli.cli.import_nightscout", _fake_import)

    result = runner.invoke(
        app,
        [
            "import-nightscout",
            "--url",
            "https://example.com",
            "--output-dir",
            str(tmp_path / "nightscout"),
            "--api-secret-env",
            "IINTS_NS_SECRET",
            "--token-env",
            "IINTS_NS_TOKEN",
        ],
    )

    assert result.exit_code == 0
    assert seen == {
        "api_secret": "secret-from-env",
        "token": "token-from-env",
    }
    assert (tmp_path / "nightscout" / "scenario.json").is_file()
    assert (tmp_path / "nightscout" / "cgm_standard.csv").is_file()


def test_import_nightscout_rejects_multiple_secret_sources(tmp_path: Path) -> None:
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("secret", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "import-nightscout",
            "--url",
            "https://example.com",
            "--api-secret",
            "inline-secret",
            "--api-secret-file",
            str(secret_file),
        ],
    )

    assert result.exit_code != 0
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "Choose only one source for api_secret" in combined_output


def test_import_tidepool_reads_token_from_file(monkeypatch, tmp_path: Path) -> None:
    token_file = tmp_path / "token.txt"
    token_file.write_text("file-token", encoding="utf-8")

    seen: dict[str, str | None] = {}

    class _FakeTidepoolClient:
        def __init__(self, base_url: str, token: str | None) -> None:
            seen["base_url"] = base_url
            seen["token"] = token

        def _headers(self):
            return {"Authorization": f"Bearer {seen['token']}"}

    monkeypatch.setattr("iints.cli.cli.TidepoolClient", _FakeTidepoolClient)

    result = runner.invoke(
        app,
        [
            "import-tidepool",
            "--base-url",
            "https://api.tidepool.example",
            "--token-file",
            str(token_file),
        ],
    )

    assert result.exit_code == 0
    assert seen["base_url"] == "https://api.tidepool.example"
    assert seen["token"] == "file-token"


def test_import_tidepool_warns_on_plain_token(monkeypatch) -> None:
    class _FakeTidepoolClient:
        def __init__(self, base_url: str, token: str | None) -> None:
            self.base_url = base_url
            self.token = token

        def _headers(self):
            return {"Authorization": f"Bearer {self.token}"}

    monkeypatch.setattr("iints.cli.cli.TidepoolClient", _FakeTidepoolClient)

    result = runner.invoke(
        app,
        [
            "import-tidepool",
            "--token",
            "plain-token",
        ],
    )

    assert result.exit_code == 0
    assert "passing token directly on the command line can leak" in result.stdout
