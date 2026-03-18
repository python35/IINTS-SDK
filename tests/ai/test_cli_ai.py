from __future__ import annotations

import json

from typer.testing import CliRunner

from iints.ai.assistant import AIResponse
from iints.ai.mdmp_guard import GuardResult
from iints.cli.cli import app


runner = CliRunner()


def test_ai_explain_command_writes_output(tmp_path, monkeypatch) -> None:
    input_json = tmp_path / "step.json"
    cert_json = tmp_path / "report.signed.mdmp"
    output_md = tmp_path / "explanation.md"
    input_json.write_text(json.dumps({"glucose": 140, "decision": {"insulin": 0.4}}), encoding="utf-8")
    cert_json.write_text(json.dumps({"signature": "demo"}), encoding="utf-8")

    class _FakeAssistant:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def explain_decision(self, payload):
            return AIResponse(
                task="explain_decision",
                text="The controller delivered a small corrective dose.",
                backend="fake",
                model="ministral-3:8b",
                certification=GuardResult(
                    cert_path=str(cert_json),
                    grade="research_grade",
                    issued_by="MDMP-Authority-v1",
                    verification_mode="bundled_root",
                    key_id="mdmp_pub_v1",
                    raw_result={"valid": True},
                ),
            )

    monkeypatch.setattr("iints.ai.cli.IINTSAssistant", _FakeAssistant)

    result = runner.invoke(
        app,
        [
            "ai",
            "explain",
            str(input_json),
            "--mdmp-cert",
            str(cert_json),
            "--output",
            str(output_md),
        ],
    )

    assert result.exit_code == 0
    assert output_md.is_file()
    assert "small corrective dose" in output_md.read_text(encoding="utf-8")


def test_ai_command_rejects_public_key_and_trust_store_together(tmp_path) -> None:
    input_json = tmp_path / "step.json"
    cert_json = tmp_path / "report.signed.mdmp"
    trust_store = tmp_path / "trust.json"
    public_key = tmp_path / "pub.pem"
    input_json.write_text(json.dumps({"glucose": 120}), encoding="utf-8")
    cert_json.write_text(json.dumps({"signature": "demo"}), encoding="utf-8")
    trust_store.write_text("{}", encoding="utf-8")
    public_key.write_text("demo", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "ai",
            "trends",
            str(input_json),
            "--mdmp-cert",
            str(cert_json),
            "--public-key",
            str(public_key),
            "--trust-store",
            str(trust_store),
        ],
    )

    assert result.exit_code != 0
    assert "either --public-key or --trust-store" in result.stdout


def test_ai_local_check_reports_ready(monkeypatch) -> None:
    class _FakeBackend:
        def __init__(self, *args, **kwargs) -> None:
            self.base_url = "http://127.0.0.1:11434"

        def available(self) -> bool:
            return True

        def healthcheck(self) -> dict[str, object]:
            return {
                "available": True,
                "base_url": self.base_url,
                "requested_model": "ministral",
                "resolved_model": "ministral-3:8b",
                "installed_models": ["ministral-3:8b"],
                "ready": True,
                "pull_command": None,
                "timeout_seconds": 120.0,
                "server_version": "0.13.1",
                "version_ok": True,
            }

    monkeypatch.setattr("iints.ai.cli.OllamaBackend", _FakeBackend)

    result = runner.invoke(app, ["ai", "local-check", "--model", "ministral"])

    assert result.exit_code == 0
    assert "Local Ollama backend is ready" in result.stdout
    assert "ministral-3:8b" in result.stdout
    assert "120.0" in result.stdout
    assert "0.13.1" in result.stdout


def test_ai_local_check_fails_when_model_missing(monkeypatch) -> None:
    class _FakeBackend:
        def __init__(self, *args, **kwargs) -> None:
            self.base_url = "http://127.0.0.1:11434"

        def available(self) -> bool:
            return True

        def healthcheck(self) -> dict[str, object]:
            return {
                "available": True,
                "base_url": self.base_url,
                "requested_model": "ministral",
                "resolved_model": None,
                "installed_models": ["llama3.2:latest"],
                "ready": False,
                "pull_command": "ollama pull ministral",
                "timeout_seconds": 120.0,
                "server_version": "0.13.1",
                "version_ok": True,
            }

    monkeypatch.setattr("iints.ai.cli.OllamaBackend", _FakeBackend)

    result = runner.invoke(app, ["ai", "local-check", "--model", "ministral"])

    assert result.exit_code == 1
    assert "requested model is missing" in result.stdout.lower()
    assert "ollama pull ministral" in result.stdout


def test_ai_models_command_lists_profiles() -> None:
    result = runner.invoke(app, ["ai", "models"])

    assert result.exit_code == 0
    assert "ministral-3:3b" in result.stdout
    assert "ministral-3:8b" in result.stdout
    assert "ministral-3:14b" in result.stdout
