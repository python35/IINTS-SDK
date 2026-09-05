from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class DesktopPreset:
    """One curated desktop workflow mapped onto a normal SDK preset."""

    key: str
    title: str
    preset_name: str
    audience: str
    description: str
    expected_output: str
    talk_track: tuple[str, ...]


@dataclass(frozen=True)
class DesktopRunResult:
    """Small, UI-friendly summary of a completed SDK run."""

    run_id: str
    workflow_title: str
    preset_name: str
    seed: int
    output_dir: Path
    results_csv: Path | None
    report_pdf: Path | None
    config_path: Path | None
    summary: str


@dataclass(frozen=True)
class DesktopRunHistoryEntry:
    """Persisted summary row for the desktop run-history view."""

    timestamp_utc: str
    workflow_title: str
    preset_name: str
    seed: int | None
    run_id: str
    output_dir: str
    results_csv: str | None
    report_pdf: str | None


@dataclass(frozen=True)
class DesktopEnvironment:
    """Small status object for the app header."""

    sdk_version: str
    qt_available: bool


DEFAULT_DESKTOP_PRESET_KEY = "doctor-safety"
RUN_HISTORY_FILENAME = ".iints-desktop-history.jsonl"

DESKTOP_PRESETS: tuple[DesktopPreset, ...] = (
    DesktopPreset(
        key="doctor-safety",
        title="Doctor safety discussion",
        preset_name="hypo_prone_night",
        audience="Clinical feedback",
        description=(
            "An overnight hypoglycemia challenge for discussing safety checks, "
            "uncertainty, and failure modes with clinical reviewers. The generated "
            "trace must be reviewed before it is used as an example."
        ),
        expected_output="CSV data, safety/audit artifacts, and a clinical PDF report.",
        talk_track=(
            "Start with the clinical question: how can we discuss algorithm risk without touching a real patient?",
            "Explain that this is a simulated night-risk scenario, not a treatment tool.",
            "Show the generated report and ask which missing clinical variables matter most.",
            "End by asking what would make the scenario medically more believable.",
        ),
    ),
    DesktopPreset(
        key="eucys-experiment",
        title="EUCYS experiment run",
        preset_name="stress_test_meal",
        audience="Science jury",
        description=(
            "A clear experiment-style stress test: meal disturbance, glucose rise, "
            "algorithm response, and research-only safety interpretation."
        ),
        expected_output="A reproducible run folder for poster figures and report screenshots.",
        talk_track=(
            "State the research question before showing outputs.",
            "Frame the run as normal simulation plus a stressor, not as a magic AI demo.",
            "Point to reproducibility: fixed preset, fixed seed, generated artifacts.",
            "Explain what still needs validation against real CGM datasets.",
        ),
    ),
    DesktopPreset(
        key="booth-demo",
        title="Booth meal-response demo",
        preset_name="quickstart_meal",
        audience="General audience",
        description=(
            "A shorter meal-stress scenario for explaining the digital patient idea "
            "without asking the audience to read logs or code. Transient glucose "
            "above the target range is expected and should not be presented as a "
            "well-controlled reference day."
        ),
        expected_output="Fast run artifacts that are easy to show live.",
        talk_track=(
            "Open with: this is a virtual patient used for safe explanation.",
            "Avoid formulas unless asked; focus on glucose changing after an event.",
            "Show outputs as evidence that the SDK records what happened.",
            "Repeat that it is education/research only.",
        ),
    ),
    DesktopPreset(
        key="pizza-late-absorption",
        title="Pizza / delayed absorption",
        preset_name="pizza_paradox",
        audience="Research discussion",
        description=(
            "A late-absorption scenario for explaining why glucose prediction is "
            "harder than simply reacting to the current CGM value."
        ),
        expected_output="Longer trace with delayed meal dynamics and report artifacts.",
        talk_track=(
            "Use this to explain why trend prediction is hard in diabetes.",
            "Point out that delayed absorption can make a later rise look surprising.",
            "Connect the scenario to AI forecasting and uncertainty.",
            "Ask reviewers what real-world meal patterns should be added next.",
        ),
    ),
    DesktopPreset(
        key="baseline-reference",
        title="Reference baseline day",
        preset_name="baseline_t1d",
        audience="SDK verification",
        description=(
            "A full-day baseline pipeline run. It checks whether the SDK installation, "
            "reporting pipeline, and artifact generation work; it is not a validated "
            "clinical reference trace."
        ),
        expected_output="Full-day CSV output, audit artifacts, and a PDF report.",
        talk_track=(
            "Use this as an installation and reporting smoke test.",
            "Check that CSV, report, and audit artifacts are generated.",
            "Do not oversell it as a clinical validation run.",
            "Compare later scenarios against this baseline.",
        ),
    ),
    DesktopPreset(
        key="jury-walkthrough",
        title="Jury walkthrough (full day)",
        preset_name="realistic_reference_day",
        audience="Science jury",
        description=(
            "The run used as the spoken example during a jury demonstration: a full "
            "24-hour day at a five-minute step, which is the 288 controller decisions "
            "a closed loop makes per day. The preset is evaluated against the bundled "
            "free-living reference envelope, so the trace can be discussed next to "
            "recorded data instead of on its own. It remains a simulation and is not "
            "a clinical validation run."
        ),
        expected_output=(
            "Full-day CSV, compartment timeline, audit artifacts, and a PDF report; "
            "the seeded demo folder additionally holds the portfolio and safety "
            "artifacts used by the other panels."
        ),
        talk_track=(
            "Open with the number: 288 insulin decisions per day, each one a chance to harm.",
            "Show the full-day trace first, then say which parts are model and which are reference data.",
            "Move to reproducibility: fixed preset, fixed seed, signed manifest, same numbers on their machine.",
            "Name the limits yourself before the jury does: simulated patient, no clinical validation, research use only.",
        ),
    ),
)


def list_desktop_presets() -> list[DesktopPreset]:
    """Return the curated workflows shown by the native app."""

    return list(DESKTOP_PRESETS)


def get_desktop_preset(key: str) -> DesktopPreset:
    """Resolve a desktop workflow key with a clear error for the GUI."""

    for preset in DESKTOP_PRESETS:
        if preset.key == key:
            return preset
    available = ", ".join(preset.key for preset in DESKTOP_PRESETS)
    raise ValueError(f"Unknown desktop workflow '{key}'. Available workflows: {available}")


def run_demo_preset(
    *,
    output_dir: str | Path,
    desktop_preset_key: str = DEFAULT_DESKTOP_PRESET_KEY,
    preset_name: str | None = None,
    seed: int = 42,
    step_callback: Callable[[int, int, float], None] | None = None,
) -> DesktopRunResult:
    """Run one deterministic demo preset through the normal SDK engine.

    The desktop app deliberately calls the same public SDK runner as CLI users.
    This keeps the GUI friendly without creating a second simulation engine.
    """

    base_output = Path(output_dir).expanduser().resolve()
    mpl_cache = base_output / ".cache" / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    # Keep imports lazy so the desktop window opens quickly and optional
    # reporting/scientific stacks load only when a run actually starts.
    from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
    from iints.highlevel import run_full
    from iints.presets import get_preset

    desktop_preset = get_desktop_preset(desktop_preset_key)
    resolved_preset_name = preset_name or desktop_preset.preset_name
    preset = get_preset(resolved_preset_name)
    folder_name = _safe_slug(f"{desktop_preset.key}-{resolved_preset_name}")
    target = base_output / folder_name
    outputs: dict[str, Any] = run_full(
        algorithm=ClinicalBaselineAlgorithm(),
        scenario=preset["scenario"],
        patient_config=preset["patient_config"],
        duration_minutes=int(preset["duration_minutes"]),
        time_step=int(preset["time_step_minutes"]),
        seed=seed,
        output_dir=target,
        step_callback=step_callback,
    )

    results_csv = _optional_path(outputs.get("results_csv"))
    report_pdf = _optional_path(outputs.get("report_pdf"))
    config_path = _optional_path(outputs.get("config_path"))
    run_id = str(outputs.get("run_id", "unknown-run"))
    threshold_lines: list[str] = []
    if results_csv is not None and results_csv.exists():
        from iints_desktop.results import screen_results_csv

        screen = screen_results_csv(results_csv)
        selected_metrics = (
            "Duration",
            "Mean glucose",
            "Min glucose",
            "Max glucose",
            "Glucose CV",
            "Time in 70-180",
            "Time below 70",
            "Time above 180",
            "Safety-triggered samples",
        )
        threshold_lines = [f"Threshold screen: {screen.label}"]
        threshold_lines.extend(
            f"{key}: {screen.metrics[key]}"
            for key in selected_metrics
            if key in screen.metrics
        )
        threshold_lines.extend(f"Review flag: {flag}" for flag in screen.flags)
        threshold_lines.append(
            "Threshold flags describe this simulated trace; they do not establish physiological or clinical validity."
        )

    summary = "\n".join(
        line
        for line in [
            f"Workflow: {desktop_preset.title}",
            f"SDK preset: {resolved_preset_name}",
            f"Seed: {seed}",
            f"Run completed: {run_id}",
            f"Output folder: {target}",
            f"Results CSV: {results_csv}" if results_csv else "Results CSV: not generated",
            f"Clinical report: {report_pdf}" if report_pdf else "Clinical report: not generated",
            *threshold_lines,
            "Research only: not a medical device and not for treatment decisions.",
        ]
    )
    result = DesktopRunResult(
        run_id=run_id,
        workflow_title=desktop_preset.title,
        preset_name=resolved_preset_name,
        seed=seed,
        output_dir=target,
        results_csv=results_csv,
        report_pdf=report_pdf,
        config_path=config_path,
        summary=summary,
    )
    try:
        append_run_history(base_output, result)
    except OSError:
        # History is useful but should never make a scientific run fail.
        pass
    return result


def run_custom_preset(
    *,
    output_dir: str | Path,
    custom_preset: dict[str, Any],
    seed: int = 42,
    step_callback: Callable[[int, int, float], None] | None = None,
) -> DesktopRunResult:
    """Run a dynamically constructed scenario from the UI."""
    base_output = Path(output_dir).expanduser().resolve()
    mpl_cache = base_output / ".cache" / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
    from iints.highlevel import run_full

    resolved_preset_name = custom_preset.get("name", "custom_scenario")
    folder_name = _safe_slug(f"custom-{resolved_preset_name}-{seed}")
    target = base_output / folder_name
    outputs: dict[str, Any] = run_full(
        algorithm=ClinicalBaselineAlgorithm(),
        scenario=custom_preset.get("scenario", {}),
        patient_config=custom_preset.get("patient_config", {}),
        duration_minutes=int(custom_preset.get("duration_minutes", 1440)),
        time_step=int(custom_preset.get("time_step_minutes", 5)),
        seed=seed,
        output_dir=target,
        step_callback=step_callback,
    )

    results_csv = _optional_path(outputs.get("results_csv"))
    report_pdf = _optional_path(outputs.get("report_pdf"))
    config_path = _optional_path(outputs.get("config_path"))
    run_id = str(outputs.get("run_id", "unknown-run"))
    summary = "\n".join(
        line
        for line in [
            "Workflow: Custom Scenario Builder",
            f"SDK preset: {resolved_preset_name}",
            f"Seed: {seed}",
            f"Run completed: {run_id}",
            f"Output folder: {target}",
            f"Results CSV: {results_csv}" if results_csv else "Results CSV: not generated",
            f"Clinical report: {report_pdf}" if report_pdf else "Clinical report: not generated",
            "Research only: not a medical device and not for treatment decisions.",
        ]
    )
    result = DesktopRunResult(
        run_id=run_id,
        workflow_title="Custom Scenario",
        preset_name=resolved_preset_name,
        seed=seed,
        output_dir=target,
        results_csv=results_csv,
        report_pdf=report_pdf,
        config_path=config_path,
        summary=summary,
    )
    try:
        append_run_history(base_output, result)
    except OSError:
        pass
    return result


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _safe_slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "iints-desktop-run"


def append_run_history(base_output_dir: str | Path, result: DesktopRunResult) -> Path:
    """Append one run to the desktop history JSONL file."""

    base = Path(base_output_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    history_path = base / RUN_HISTORY_FILENAME
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workflow_title": result.workflow_title,
        "preset_name": result.preset_name,
        "seed": result.seed,
        "run_id": result.run_id,
        "output_dir": str(result.output_dir),
        "results_csv": str(result.results_csv) if result.results_csv else None,
        "report_pdf": str(result.report_pdf) if result.report_pdf else None,
    }
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return history_path


def read_run_history(base_output_dir: str | Path, *, limit: int = 25) -> list[DesktopRunHistoryEntry]:
    """Read the newest desktop run-history entries first."""

    history_path = Path(base_output_dir).expanduser().resolve() / RUN_HISTORY_FILENAME
    if not history_path.exists():
        return []

    entries: list[DesktopRunHistoryEntry] = []
    lines = history_path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            entries.append(
                DesktopRunHistoryEntry(
                    timestamp_utc=str(payload.get("timestamp_utc", "")),
                    workflow_title=str(payload.get("workflow_title", "")),
                    preset_name=str(payload.get("preset_name", "")),
                    seed=_optional_int(payload.get("seed")),
                    run_id=str(payload.get("run_id", "")),
                    output_dir=str(payload.get("output_dir", "")),
                    results_csv=payload.get("results_csv"),
                    report_pdf=payload.get("report_pdf"),
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if len(entries) >= limit:
            break
    return entries


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def get_desktop_environment(*, qt_available: bool) -> DesktopEnvironment:
    """Return lightweight runtime details for the desktop shell."""

    try:
        sdk_version = metadata.version("iints-sdk-python35")
    except metadata.PackageNotFoundError:
        sdk_version = "editable"
    return DesktopEnvironment(sdk_version=sdk_version, qt_available=qt_available)
