import hashlib
import json
from pathlib import Path

import pytest

from iints_desktop.engine import DesktopRunResult, _new_run_directory
from iints_desktop.run_bundle import load_run_bundle, prepare_run_bundle


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    root = tmp_path / "run"
    root.mkdir()
    csv = root / "results.csv"
    csv.write_text("time_minutes,glucose_actual_mgdl\n0,100\n5,115\n10,130\n15,120\n")
    pdf = root / "report.pdf"
    pdf.write_bytes(b"fixture report")
    return DesktopRunResult("fixture", "Meal test", "quickstart_meal", 42, root, csv, pdf, None, "Completed")


def test_repeated_experiments_never_overwrite_previous_data(tmp_path):
    first = _new_run_directory(tmp_path, "meal")
    (first / "results.csv").write_text("original")
    second = _new_run_directory(tmp_path, "meal")
    assert first != second
    assert (first / "results.csv").read_text() == "original"


def test_run_preparation_and_reopen_preserve_artifacts_and_hashes(run):
    pytest.importorskip("mdmp_core.crypto")
    prepared = prepare_run_bundle(run)
    assert prepared["status"] == "ready", prepared["steps"]
    reopened = load_run_bundle(run.results_csv)
    assert reopened["mdmp"]["quick"] is False
    assert reopened["mdmp"]["row_count"] == 4
    assert reopened["academic_bundle"]["artifact_count"] > 0
    assert all(Path(item["path"]).is_file() for item in reopened["artifacts"])
    assert not list(run.output_dir.rglob("*private*"))
    crate = json.loads(Path(reopened["academic_bundle"]["ro_crate_metadata"]).read_text())
    for entity in crate["@graph"]:
        if entity.get("@type") == "File" and "sha256" in entity:
            file = run.output_dir / entity["@id"]
            assert hashlib.sha256(file.read_bytes()).hexdigest() == entity["sha256"]
    assert not any(entity.get("@id") == "desktop_run_bundle.json" for entity in crate["@graph"])


def test_optional_exporter_failure_is_visible_but_keeps_simulation(run, monkeypatch):
    import iints_desktop.mdmp as mdmp

    def unavailable(*args, **kwargs):
        raise RuntimeError("Test signing backend unavailable")

    monkeypatch.setattr(mdmp, "create_desktop_mdmp_certificate", unavailable)
    prepared = prepare_run_bundle(run)
    assert prepared["status"] == "ready_with_warnings"
    assert any(step["key"] == "mdmp" and step["status"] == "warning" for step in prepared["steps"])
    assert prepared["academic_bundle"]
    assert run.results_csv.is_file()
    assert load_run_bundle(run.results_csv)["status"] == "ready_with_warnings"


def test_no_preparation_and_cancel_never_start_ollama_or_signing(run, monkeypatch):
    import iints_desktop.local_ai as local_ai
    import iints_desktop.mdmp as mdmp

    def unexpected(*args, **kwargs):
        pytest.fail("Disabled stage executed")

    monkeypatch.setattr(mdmp, "create_desktop_mdmp_certificate", unexpected)
    monkeypatch.setattr(local_ai, "check_local_ai", unexpected)
    prepared = prepare_run_bundle(run, auto_prepare=False)
    assert all(step["status"] == "skipped" for step in prepared["steps"] if step["key"] in {"mdmp", "ai"})
    prepared = prepare_run_bundle(run, ai_review=True, cancelled=lambda: True)
    assert all(step["status"] == "skipped" for step in prepared["steps"])


def test_optional_ai_uses_the_selected_run_and_is_saved(run, monkeypatch):
    from iints_desktop.local_ai import LocalAIAnswer, LocalAIStatus
    import iints_desktop.local_ai as local_ai

    monkeypatch.setattr(local_ai, "check_local_ai", lambda **kwargs: LocalAIStatus(True, "Ready", "test-model"))

    def ask(**kwargs):
        assert kwargs["result_csv"] == run.results_csv
        assert kwargs["model"] == "test-model"
        return LocalAIAnswer(answer="Review replicate runs.", model="test-model", context_used=True)

    monkeypatch.setattr(local_ai, "ask_local_ai", ask)
    prepared = prepare_run_bundle(run, auto_prepare=False, ai_review=True, ai_model="test-model")
    assert prepared["ai_review"]["answer"] == "Review replicate runs."
    assert (run.output_dir / "local_ai_review.md").is_file()
    assert load_run_bundle(run.results_csv)["ai_review"]["model"] == "test-model"


def test_reopen_rejects_unrelated_csv_and_external_artifact_paths(run, tmp_path):
    prepare_run_bundle(run, auto_prepare=False)
    manifest = run.output_dir / "desktop_run_bundle.json"
    data = json.loads(manifest.read_text())
    other = tmp_path / "unrelated.txt"
    other.write_text("not a run artifact")
    data["artifacts"].append({"label": "external", "path": str(other)})
    manifest.write_text(json.dumps(data))
    assert not any(item["label"] == "external" for item in load_run_bundle(run.results_csv)["artifacts"])
    data["results_csv"] = str(other)
    manifest.write_text(json.dumps(data))
    assert load_run_bundle(run.results_csv) is None


def test_changed_csv_invalidates_cached_certificate_and_review(run):
    prepare_run_bundle(run, auto_prepare=False)
    assert load_run_bundle(run.results_csv) is not None
    run.results_csv.write_text("time_minutes,glucose_actual_mgdl\n0,300\n5,320\n")
    assert load_run_bundle(run.results_csv) is None


def test_reopen_sanitizes_nested_paths_and_malformed_lists(run, tmp_path):
    prepare_run_bundle(run, auto_prepare=False)
    external = tmp_path / "outside.pdf"
    external.write_text("not part of the run")
    link = run.output_dir / "linked.pdf"
    link.symlink_to(external)
    manifest = run.output_dir / "desktop_run_bundle.json"
    payload = json.loads(manifest.read_text())
    payload.update(artifacts=None, steps={}, report_pdf=str(link), output_dir=str(tmp_path))
    payload["mdmp"] = {"certificate_path": str(external), "private_key_path": str(external)}
    payload["academic_bundle"] = {"audit_json": str(link), "run_dir": str(tmp_path)}
    manifest.write_text(json.dumps(payload))
    reopened = load_run_bundle(run.results_csv)
    assert reopened["output_dir"] == str(run.output_dir.resolve())
    assert reopened["artifacts"] == reopened["steps"] == []
    assert reopened["report_pdf"] is None
    assert reopened["mdmp"]["certificate_path"] is None
    assert "private_key_path" not in reopened["mdmp"]
    assert reopened["academic_bundle"]["audit_json"] is None
