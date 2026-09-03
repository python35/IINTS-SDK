"""Seed one folder with a worked example for every panel of the desktop app.

Why this module exists: the desktop shell spreads its evidence over ten panels,
and each panel reads artifacts produced by a different SDK entry point. Before a
demonstration the operator would otherwise have to remember every one of those
actions, and any panel they forget stands empty in front of the audience. This
module performs them once, into the folder the app is configured with, and then
writes down what succeeded, what needs an optional engine that is not installed,
and what the app cannot display yet even though the artifact exists.

Nothing here is a second simulation engine. Every step calls the same public
entry point the corresponding panel calls, so a seeded folder contains exactly
what the panel would have produced on its own.

Research use only. The seeded artifacts describe a simulated patient; they are
not clinical evidence and not output of a medical device.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from iints_desktop.engine import (
    DesktopRunResult,
    get_desktop_preset,
    list_desktop_presets,
    run_demo_preset,
)

#: Preset spoken through during the demonstration.
JURY_PRESET_KEY = "jury-walkthrough"

#: Extra runs so the history and results panels show more than a single row,
#: and so the operator can contrast a full day against an overnight low and a
#: short meal window without leaving the app.
SUPPORTING_PRESET_KEYS: tuple[str, ...] = ("doctor-safety", "booth-demo")

WALKTHROUGH_FILENAME = "JURY_WALKTHROUGH.md"
MANIFEST_FILENAME = "jury_demo_manifest.json"

#: The Scientific Portfolio panel calls the bridge with the *relative* path
#: "results/scientific_portfolio", and the Tauri shell runs the bridge with the
#: user's home directory as its working directory. Seeding the portfolio
#: anywhere else leaves that panel empty even though the artifacts exist, so the
#: default target is derived the same way the panel derives it.
PORTFOLIO_PANEL_SUBPATH = Path("results") / "scientific_portfolio"

#: Bridge commands that exist in the Rust layer but that no frontend control
#: calls yet. Their artifacts are still seeded, because a file that can be
#: opened from disk is worth more on stage than a capability that is invisible.
UNWIRED_BRIDGE_COMMANDS: tuple[str, ...] = (
    "generate_scientific_visualizations",
    "run_fda_safety_benchmark",
    "load_cgmacros_cohort",
)

RESEARCH_ONLY_NOTICE = (
    "Research use only. Every number in this folder comes from a simulated "
    "patient. It is not clinical evidence, not validated against a specific "
    "person, and not output of a medical device."
)


@dataclass
class JuryDemoStep:
    """One seeding action and its honest outcome."""

    name: str
    panel: str
    status: str  # "ok", "skipped", or "failed"
    detail: str
    artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "panel": self.panel,
            "status": self.status,
            "detail": self.detail,
            "artifacts": list(self.artifacts),
        }


@dataclass
class JuryDemoResult:
    """Everything the seeding run produced, plus what it could not produce."""

    output_dir: Path
    portfolio_dir: Path
    steps: list[JuryDemoStep] = field(default_factory=list)
    manifest_path: Path | None = None
    walkthrough_path: Path | None = None

    @property
    def failed_steps(self) -> list[JuryDemoStep]:
        return [step for step in self.steps if step.status == "failed"]

    @property
    def skipped_steps(self) -> list[JuryDemoStep]:
        return [step for step in self.steps if step.status == "skipped"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "portfolio_dir": str(self.portfolio_dir),
            "steps": [step.to_dict() for step in self.steps],
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "walkthrough_path": (
                str(self.walkthrough_path) if self.walkthrough_path else None
            ),
            "failed_step_count": len(self.failed_steps),
            "skipped_step_count": len(self.skipped_steps),
            "unwired_bridge_commands": list(UNWIRED_BRIDGE_COMMANDS),
            "research_only": True,
            "medical_device": False,
        }


def _existing(*candidates: Any) -> tuple[str, ...]:
    """Return the string form of the candidate paths that are really on disk."""

    found: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(str(candidate))
        if path.exists():
            found.append(str(path))
    return tuple(found)


def build_jury_demo(
    *,
    output_dir: str | Path,
    seed: int = 42,
    creator_name: str | None = None,
    creator_orcid: str | None = None,
    license_id: str = "NOASSERTION",
    portfolio_dir: str | Path | None = None,
    include_portfolio: bool = True,
    progress: Callable[[str], None] | None = None,
) -> JuryDemoResult:
    """Fill ``output_dir`` with one worked example per desktop panel.

    A failing step never aborts the others. A demonstration folder that is
    nine-tenths complete and says so is more useful than an empty folder and a
    traceback, and the optional research engines are genuinely absent on many
    machines.

    ``creator_name`` and ``creator_orcid`` are written into the academic bundle
    exactly as supplied and are left out when they are ``None``; this module
    will not invent authorship metadata.
    """

    base = Path(output_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    # Matplotlib needs a writable cache before any reporting import happens;
    # this mirrors what the normal desktop runner does.
    mpl_cache = base / ".cache" / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    resolved_portfolio = (
        Path(portfolio_dir).expanduser().resolve()
        if portfolio_dir is not None
        else Path.home() / PORTFOLIO_PANEL_SUBPATH
    )

    result = JuryDemoResult(output_dir=base, portfolio_dir=resolved_portfolio)

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    def record(
        name: str,
        panel: str,
        action: Callable[[], tuple[str, tuple[str, ...]]],
    ) -> None:
        announce(f"{name}...")
        try:
            detail, artifacts = action()
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - reported, never raised onward
            detail = f"{type(exc).__name__}: {exc}"
            artifacts = ()
            status = "failed"
        result.steps.append(
            JuryDemoStep(
                name=name, panel=panel, status=status, detail=detail, artifacts=artifacts
            )
        )
        announce(f"{name}: {status}")

    runs: dict[str, DesktopRunResult] = {}

    def run_preset(key: str) -> tuple[str, tuple[str, ...]]:
        preset = get_desktop_preset(key)
        run = run_demo_preset(
            output_dir=base, desktop_preset_key=key, seed=seed
        )
        runs[key] = run
        return (
            f"{preset.title} ({preset.preset_name}, seed {seed}) completed as {run.run_id}.",
            _existing(run.results_csv, run.report_pdf, run.config_path),
        )

    for key in (JURY_PRESET_KEY, *SUPPORTING_PRESET_KEYS):
        record(f"Run preset '{key}'", "Run protocols", lambda k=key: run_preset(k))

    main_run = runs.get(JURY_PRESET_KEY) or next(iter(runs.values()), None)

    def check_history() -> tuple[str, tuple[str, ...]]:
        from iints_desktop.engine import RUN_HISTORY_FILENAME, read_run_history

        entries = read_run_history(base, limit=25)
        if not entries:
            raise RuntimeError(
                "No run history was written, so the overview panel would be empty."
            )
        titles = ", ".join(entry.workflow_title for entry in entries[:5])
        return (
            f"{len(entries)} run(s) in the desktop history: {titles}.",
            _existing(base / RUN_HISTORY_FILENAME),
        )

    record("Register runs in history", "Overview", check_history)

    def check_results() -> tuple[str, tuple[str, ...]]:
        if main_run is None or main_run.results_csv is None:
            raise RuntimeError("No results CSV was produced, so nothing can be shown.")
        from iints_desktop.results import load_compartment_timeline, screen_results_csv

        screen = screen_results_csv(main_run.results_csv)
        timeline = load_compartment_timeline(main_run.results_csv, max_points=400)
        if not timeline.available:
            raise RuntimeError(
                f"Compartment timeline unavailable: {timeline.reason}. "
                "The compartment view would stay blank."
            )
        return (
            f"Threshold screen '{screen.label}'; compartment timeline has "
            f"{len(timeline.times)} points across {len(timeline.compartments)} compartments.",
            _existing(main_run.results_csv),
        )

    record("Check results and compartment view", "Inspect results", check_results)

    def certify() -> tuple[str, tuple[str, ...]]:
        if main_run is None or main_run.results_csv is None:
            raise RuntimeError("A results CSV is required before it can be certified.")
        from iints_desktop.mdmp import create_desktop_mdmp_certificate

        cert = create_desktop_mdmp_certificate(
            Path(main_run.results_csv), quick=True, quick_rows=500
        )
        return (
            f"Grade {cert.grade} over {cert.row_count} rows "
            f"(compliance score {cert.compliance_score}); quick verification.",
            _existing(cert.certificate_path, cert.report_path, cert.public_key_path),
        )

    record("Sign an MDMP certificate", "Reproducibility", certify)

    def bundle() -> tuple[str, tuple[str, ...]]:
        if main_run is None:
            raise RuntimeError("A completed run is required before it can be bundled.")
        from iints.research.academic_bundle import build_academic_bundle

        preset = get_desktop_preset(JURY_PRESET_KEY)
        bundle_result = build_academic_bundle(
            Path(main_run.output_dir),
            title=f"IINTS jury demonstration - {preset.title}",
            description=(
                f"Simulated full-day closed-loop run from SDK preset "
                f"'{main_run.preset_name}' at seed {main_run.seed}. {RESEARCH_ONLY_NOTICE}"
            ),
            creator_name=creator_name,
            creator_orcid=creator_orcid,
            license_id=license_id,
        )
        # The result carries its artifacts under descriptive field names
        # (ro_crate_metadata, audit_json, ...) rather than a path suffix, so
        # every string-like field is offered and _existing keeps the real ones.
        candidates = [
            value
            for value in vars(bundle_result).values()
            if isinstance(value, (str, Path))
        ]
        return (
            f"Bundle readiness {bundle_result.readiness_status} "
            f"({bundle_result.readiness_score_pct}%) over "
            f"{bundle_result.artifact_count} artifact(s).",
            _existing(*candidates),
        )

    record("Export an academic bundle", "Reproducibility", bundle)

    if include_portfolio:

        def portfolio() -> tuple[str, tuple[str, ...]]:
            from iints.research.eucys_playbook_generator import (
                generate_complete_eucys_jury_portfolio,
            )

            resolved_portfolio.mkdir(parents=True, exist_ok=True)
            built = generate_complete_eucys_jury_portfolio(output_dir=resolved_portfolio)
            payload = built.to_dict()
            figure_count = payload.get("total_figures", "unknown")
            return (
                f"{figure_count} portfolio figure(s) written where the panel looks: "
                f"{resolved_portfolio}.",
                _existing(
                    payload.get("index_html_path"), payload.get("manifest_json_path")
                ),
            )

        record("Generate the scientific portfolio", "Scientific Portfolio", portfolio)

        def visualizations() -> tuple[str, tuple[str, ...]]:
            from iints.research.visualizer import (
                generate_all_scientific_visualizations,
            )

            target = base / "scientific_visualizations"
            target.mkdir(parents=True, exist_ok=True)
            artifacts = generate_all_scientific_visualizations(output_dir=target)
            payload = artifacts.to_dict()
            return (
                "Figures written. No frontend control calls "
                "'generate_scientific_visualizations' yet, so open these as files.",
                _existing(*payload.values()),
            )

        record(
            "Render visualizer figures", "Foundation AI & Visualizer", visualizations
        )

        def safety_benchmark() -> tuple[str, tuple[str, ...]]:
            from iints.safety.openfda_safety import run_fda_safety_benchmark

            target = base / "fda_safety_benchmark"
            target.mkdir(parents=True, exist_ok=True)
            report = run_fda_safety_benchmark(output_dir=target)
            payload = report.to_dict()
            return (
                "Hazard detection rate "
                f"{payload.get('hazard_detection_rate_pct')}% over "
                f"{payload.get('total_cases_evaluated')} simulated cases. No frontend "
                "control calls 'run_fda_safety_benchmark' yet, so open the report as a file.",
                _existing(
                    payload.get("report_json_path"), payload.get("report_md_path")
                ),
            )

        record("Run the safety benchmark", "Foundation AI & Visualizer", safety_benchmark)

    def evidence() -> tuple[str, tuple[str, ...]]:
        from dataclasses import asdict

        from iints_desktop.evidence_connectors import list_evidence_connectors

        connectors = [asdict(connector) for connector in list_evidence_connectors()]
        target = base / "evidence_connectors.json"
        target.write_text(
            json.dumps(connectors, indent=2, default=str), encoding="utf-8"
        )
        return (
            f"{len(connectors)} evidence connector(s) described offline; "
            "live queries need network access.",
            _existing(target),
        )

    record("Snapshot the evidence sources", "Evidence sources", evidence)

    def environment() -> tuple[str, tuple[str, ...]]:
        notes: dict[str, Any] = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        }
        try:
            from iints.research.mechanistic_models import roadrunner_status

            notes["mechanistic_engine"] = {
                **roadrunner_status(),
                "inspection_available": True,
            }
        except Exception as exc:  # noqa: BLE001 - a probe, not a requirement
            notes["mechanistic_engine"] = f"probe failed: {type(exc).__name__}: {exc}"
        try:
            from iints_desktop.local_ai import check_local_ai

            ai_status = check_local_ai()
            notes["local_ai"] = {
                "available": ai_status.available,
                "message": ai_status.message,
                "resolved_model": ai_status.resolved_model,
            }
        except Exception as exc:  # noqa: BLE001 - a probe, not a requirement
            notes["local_ai"] = f"probe failed: {type(exc).__name__}: {exc}"

        target = base / "demo_environment.json"
        target.write_text(json.dumps(notes, indent=2, default=str), encoding="utf-8")
        ai = notes.get("local_ai")
        ai_line = (
            "local model reachable"
            if isinstance(ai, dict) and ai.get("available")
            else "no local model reachable"
        )
        return (f"Environment probed: {ai_line}.", _existing(target))

    record("Probe optional engines", "Local AI review / Research tools", environment)

    def write_documents() -> tuple[str, tuple[str, ...]]:
        manifest = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "jury_preset_key": JURY_PRESET_KEY,
            "runs": {
                key: {
                    "run_id": run.run_id,
                    "preset_name": run.preset_name,
                    "output_dir": str(run.output_dir),
                    "results_csv": str(run.results_csv) if run.results_csv else None,
                    "report_pdf": str(run.report_pdf) if run.report_pdf else None,
                }
                for key, run in runs.items()
            },
            "notice": RESEARCH_ONLY_NOTICE,
            **result.to_dict(),
        }
        manifest_path = base / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        result.manifest_path = manifest_path

        walkthrough_path = base / WALKTHROUGH_FILENAME
        walkthrough_path.write_text(
            _build_walkthrough(result, runs=runs, seed=seed), encoding="utf-8"
        )
        result.walkthrough_path = walkthrough_path
        return ("Manifest and spoken walkthrough written.", _existing(
            manifest_path, walkthrough_path
        ))

    record("Write the walkthrough", "Overview", write_documents)

    return result


def _status_mark(status: str) -> str:
    return {"ok": "ready", "skipped": "skipped", "failed": "NOT READY"}.get(
        status, status
    )


def _build_walkthrough(
    result: JuryDemoResult,
    *,
    runs: dict[str, DesktopRunResult],
    seed: int,
) -> str:
    """Write the spoken order of the demonstration, with honest gaps."""

    preset = get_desktop_preset(JURY_PRESET_KEY)
    lines: list[str] = []
    lines.append("# Jury walkthrough")
    lines.append("")
    lines.append(RESEARCH_ONLY_NOTICE)
    lines.append("")
    lines.append(
        f"Seeded {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} into "
        f"`{result.output_dir}` at seed {seed}."
    )
    lines.append("")
    lines.append("## Before the room fills")
    lines.append("")
    lines.append(
        f"1. Open Settings and set the output folder to `{result.output_dir}`, "
        "otherwise every panel looks in a different place than this folder."
    )
    lines.append(
        "2. Open Overview and confirm the run history lists the seeded runs."
    )
    lines.append(
        "3. Read the status table below and decide which panels you will open. "
        "Do not open a panel marked NOT READY in front of the jury."
    )
    lines.append("")
    lines.append("## What is seeded")
    lines.append("")
    lines.append("| Panel | Step | Status |")
    lines.append("| --- | --- | --- |")
    for step in result.steps:
        lines.append(f"| {step.panel} | {step.name} | {_status_mark(step.status)} |")
    lines.append("")

    lines.append("## Spoken order")
    lines.append("")
    lines.append("### 1. Run protocols - the claim")
    lines.append("")
    lines.append(
        f"Select **{preset.title}**. The run is already seeded, so you can talk over "
        "the existing artifacts instead of waiting for a simulation on stage."
    )
    lines.append("")
    for line in preset.talk_track:
        lines.append(f"- {line}")
    lines.append("")

    main_run = runs.get(JURY_PRESET_KEY)
    lines.append("### 2. Inspect results - the evidence")
    lines.append("")
    if main_run is not None and main_run.results_csv is not None:
        lines.append(f"Load `{main_run.results_csv}`.")
    else:
        lines.append(
            "The full-day run did not complete during seeding; use one of the other "
            "runs in the history instead."
        )
    lines.append("")
    lines.append(
        "- Show the glucose trace first, then the compartment view, and say plainly "
        "which curves are model state and which are the reference envelope."
    )
    lines.append(
        "- If a jury member asks how close this is to a real patient, the honest "
        "answer is that the trace is simulated and only the envelope comes from "
        "recorded free-living data."
    )
    lines.append("")

    lines.append("### 3. Reproducibility - the strongest panel")
    lines.append("")
    lines.append(
        "This is the part a science jury tends to reward: the same preset and the "
        f"same seed ({seed}) reproduce the same numbers, and the certificate plus the "
        "citable bundle let a reviewer check that claim without your laptop."
    )
    lines.append("")

    lines.append("### 4. Scientific Portfolio - the written argument")
    lines.append("")
    lines.append(
        f"The portfolio is seeded in `{result.portfolio_dir}`, which is where the "
        "panel looks: the button asks the bridge for the relative path "
        "`results/scientific_portfolio`, and the bridge runs with your home folder "
        "as its working directory. Seeding it elsewhere leaves the panel empty."
    )
    lines.append("")

    lines.append("### 5. Evidence sources and Research tools - scope, honestly")
    lines.append("")
    lines.append(
        "The connector list loads offline; a live query needs network access, so do "
        "not promise one on a conference network you have not tested."
    )
    lines.append(
        "Model *inspection* works without extra software. Executing SBML, COPASI or "
        "OpenCOR models needs optional engines - check `demo_environment.json` in "
        "this folder for what this machine actually has, and demonstrate inspection "
        "only if the engines are missing."
    )
    lines.append("")

    lines.append("## Known gaps to say out loud rather than hide")
    lines.append("")
    lines.append(
        "- The Local AI review panel needs a local model server running. If "
        "`demo_environment.json` reports no reachable model, skip the panel or start "
        "the model before the session."
    )
    lines.append(
        "- These capabilities exist in the app's bridge but no button calls them yet, "
        "so their artifacts must be opened as files: "
        + ", ".join(f"`{name}`" for name in UNWIRED_BRIDGE_COMMANDS)
        + "."
    )
    failed = result.failed_steps
    if failed:
        lines.append("- Seeding steps that did not complete:")
        for step in failed:
            lines.append(f"  - {step.name} ({step.panel}): {step.detail}")
    else:
        lines.append("- Every seeding step completed.")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    for step in result.steps:
        if not step.artifacts:
            continue
        lines.append(f"**{step.name}**")
        for artifact in step.artifacts:
            lines.append(f"- `{artifact}`")
        lines.append("")

    lines.append("## Panels in the app")
    lines.append("")
    lines.append(
        "For reference, the shell exposes: "
        + ", ".join(sorted({preset.audience for preset in list_desktop_presets()}))
        + " audiences across the preset catalogue."
    )
    lines.append("")
    return "\n".join(lines)
