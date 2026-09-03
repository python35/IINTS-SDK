"""Tests for the jury demonstration seeder.

The seeder exists so that no panel of the desktop app stands empty during a
demonstration. Two properties therefore matter more than the artifacts
themselves: it must write the portfolio where the panel actually looks, and it
must report a step it could not complete instead of failing silently or
aborting the rest of the seeding.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iints_desktop import jury_demo
from iints_desktop.engine import get_desktop_preset

FRONTEND_JS = Path("apps/iints-tauri/frontend/main.js")
TAURI_MAIN_RS = Path("apps/iints-tauri/src-tauri/src/main.rs")


def test_jury_preset_is_registered_and_resolvable() -> None:
    preset = get_desktop_preset(jury_demo.JURY_PRESET_KEY)

    assert preset.preset_name == "realistic_reference_day"
    assert preset.talk_track
    # A jury preset that oversells the run is worse than no preset, so the
    # description has to keep the simulation caveat with the claim.
    assert "not a clinical validation run" in preset.description


def test_supporting_presets_exist() -> None:
    for key in jury_demo.SUPPORTING_PRESET_KEYS:
        assert get_desktop_preset(key).preset_name


@pytest.mark.skipif(not FRONTEND_JS.exists(), reason="Tauri frontend not present")
def test_portfolio_default_matches_the_path_the_panel_requests() -> None:
    """The panel asks the bridge for a relative path; the default must match it.

    The Scientific Portfolio button invokes the bridge with a relative output
    directory, and the shell runs the bridge from the user's home folder. If
    this constant and that string ever drift apart, seeding writes a complete
    portfolio into a folder the panel never reads, and the panel is empty on
    stage with no error anywhere.
    """

    source = FRONTEND_JS.read_text(encoding="utf-8")
    requested = jury_demo.PORTFOLIO_PANEL_SUBPATH.as_posix()

    assert f'outputDir: "{requested}"' in source
    assert (
        jury_demo.build_jury_demo.__doc__ is not None
    )  # the behaviour below is documented


def test_default_portfolio_dir_resolves_under_the_home_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jury_demo.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        jury_demo, "run_demo_preset", _raising_run_demo_preset, raising=True
    )

    result = jury_demo.build_jury_demo(
        output_dir=tmp_path / "runs", include_portfolio=False
    )

    assert result.portfolio_dir == tmp_path / jury_demo.PORTFOLIO_PANEL_SUBPATH


def _raising_run_demo_preset(**_kwargs: object):
    raise RuntimeError("simulated engine failure")


def test_seeding_records_failures_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken step must not take the other panels down with it."""

    monkeypatch.setattr(
        jury_demo, "run_demo_preset", _raising_run_demo_preset, raising=True
    )

    result = jury_demo.build_jury_demo(
        output_dir=tmp_path, include_portfolio=False, portfolio_dir=tmp_path / "pf"
    )

    failed_names = {step.name for step in result.failed_steps}
    assert any("jury-walkthrough" in name for name in failed_names)
    assert "simulated engine failure" in "\n".join(
        step.detail for step in result.failed_steps
    )
    # Later steps still ran, and the documents were still written.
    assert result.walkthrough_path is not None and result.walkthrough_path.exists()
    assert result.manifest_path is not None and result.manifest_path.exists()


def test_walkthrough_discloses_gaps_rather_than_hiding_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        jury_demo, "run_demo_preset", _raising_run_demo_preset, raising=True
    )

    result = jury_demo.build_jury_demo(
        output_dir=tmp_path, include_portfolio=False, portfolio_dir=tmp_path / "pf"
    )
    assert result.walkthrough_path is not None
    text = result.walkthrough_path.read_text(encoding="utf-8")

    assert "NOT READY" in text
    assert "simulated engine failure" in text
    assert "Research use only" in text
    for command in jury_demo.UNWIRED_BRIDGE_COMMANDS:
        assert command in text


def test_manifest_is_valid_json_and_marks_research_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        jury_demo, "run_demo_preset", _raising_run_demo_preset, raising=True
    )

    result = jury_demo.build_jury_demo(
        output_dir=tmp_path, include_portfolio=False, portfolio_dir=tmp_path / "pf"
    )
    assert result.manifest_path is not None
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert payload["research_only"] is True
    assert payload["medical_device"] is False
    assert payload["jury_preset_key"] == jury_demo.JURY_PRESET_KEY
    assert payload["failed_step_count"] >= 1


@pytest.mark.skipif(
    not (FRONTEND_JS.exists() and TAURI_MAIN_RS.exists()),
    reason="Tauri app not present",
)
def test_unwired_commands_are_still_unwired() -> None:
    """Keep the disclosed gap list truthful.

    Each name below exists as a bridge command in the Rust layer but is never
    invoked by the frontend, which is why the walkthrough tells the operator to
    open those artifacts as files. When a button is finally added for one of
    them, this test fails on purpose so the claim gets corrected instead of
    quietly becoming wrong.
    """

    rust = TAURI_MAIN_RS.read_text(encoding="utf-8")
    frontend = FRONTEND_JS.read_text(encoding="utf-8")

    for command in jury_demo.UNWIRED_BRIDGE_COMMANDS:
        assert f"fn {command}(" in rust, f"{command} is no longer a bridge command"
        assert (
            f'invoke("{command}"' not in frontend
        ), f"{command} now has a frontend caller; update UNWIRED_BRIDGE_COMMANDS"
