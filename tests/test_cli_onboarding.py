from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def test_run_dry_run_supports_builtin_preset(tmp_path) -> None:
    output_dir = tmp_path / "demo_run"

    result = runner.invoke(
        app,
        ["run", "--preset", "baseline_t1d", "--dry-run", "--output-dir", str(output_dir)],
    )

    assert result.exit_code == 0
    assert "Dry Run Plan" in result.stdout
    assert "Clinical Baseline" in result.stdout
    assert "baseline_t1d" in result.stdout
    assert not output_dir.exists()


def test_run_missing_algorithm_offers_suggestion(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    algorithms_dir = tmp_path / "algorithms"
    algorithms_dir.mkdir()
    (algorithms_dir / "example_algorithm.py").write_text("class Placeholder: pass\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "run",
            "--algo",
            "algorithms/example_algoritm.py",
            "--dry-run",
        ],
    )

    assert result.exit_code == 1
    assert "Algorithm file not found" in result.stdout
    assert "Did you mean" in result.stdout
    assert "example_algorithm.py" in result.stdout


def test_demo_writes_editable_algorithm_after_run(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("iints.cli.cli.run", _fake_run)

    output_dir = tmp_path / "demo"
    result = runner.invoke(app, ["demo", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert captured["preset"] == "quickstart_meal"
    assert captured["algo"] is None
    assert (output_dir / "demo_assets" / "example_algorithm.py").is_file()
    assert "What To Do Next" in result.stdout


def test_guide_dispatches_to_demo(monkeypatch) -> None:
    called = {"demo": 0}

    def _fake_demo(**kwargs):
        called["demo"] += 1

    monkeypatch.setattr("iints.cli.cli.demo", _fake_demo)

    result = runner.invoke(app, ["guide"], input="1\n")

    assert result.exit_code == 0
    assert called["demo"] == 1
    assert "IINTS Guide" in result.stdout


def test_doctor_suggest_prints_actionable_next_steps(monkeypatch) -> None:
    monkeypatch.setattr("iints.cli.cli._module_available", lambda name: False if name == "pandas" else True)
    monkeypatch.setattr("iints.cli.cli._load_presets", lambda: [{"name": "baseline_t1d"}])
    monkeypatch.setattr("iints.cli.cli.load_validation_profiles", lambda path=None: {"starter": object()})

    result = runner.invoke(app, ["doctor", "--suggest"])

    assert result.exit_code == 1
    assert "Suggested Next Steps" in result.stdout
    assert "Install the main SDK stack" in result.stdout


def test_start_prints_beginner_plan(tmp_path) -> None:
    output_dir = tmp_path / "demo"

    result = runner.invoke(app, ["start", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert "Recommended First Step" in result.stdout
    assert "iints demo" in result.stdout
    assert "demo_assets" in result.stdout
    assert not output_dir.exists()


def test_start_supports_goal_aliases() -> None:
    result = runner.invoke(app, ["start", "--goal", "pi"])

    assert result.exit_code == 0
    assert "Goal: edge" in result.stdout
    assert "iints edge doctor" in result.stdout
    assert "iints edge quickstart" in result.stdout
    assert "./start_edge_easy.sh" in result.stdout


def test_start_run_dispatches_to_demo(monkeypatch, tmp_path) -> None:
    called: dict[str, object] = {}

    def _fake_demo(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("iints.cli.cli.demo", _fake_demo)

    output_dir = tmp_path / "starter_demo"
    result = runner.invoke(app, ["start", "--goal", "demo", "--run", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert called["output_dir"] == output_dir


def test_start_rejects_unknown_goal() -> None:
    result = runner.invoke(app, ["start", "--goal", "unknown"])

    assert result.exit_code == 1
    assert "Unknown goal" in result.stdout
