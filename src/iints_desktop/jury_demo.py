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
from functools import partial
from pathlib import Path
from typing import Any, Callable

from iints_desktop.engine import (
    DesktopRunResult,
    get_desktop_preset,
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


def _sensor_error_pairs(
    run: DesktopRunResult | None,
) -> tuple[Any, Any] | None:
    """Return ``(model truth, value handed to the algorithm)`` from a run.

    This is the pairing an error grid was designed for: a reference value
    against the measurement a controller actually acted on. Here both come from
    the simulator, so the grid characterises the sensor error model of the
    simulation - it is not an evaluation of a physical CGM. If the two columns
    are identical the grid would be a meaningless diagonal, so nothing is
    returned and the figure stays honestly ungenerated.
    """

    if run is None or run.results_csv is None:
        return None
    try:
        import pandas as pd

        frame = pd.read_csv(run.results_csv)
        reference = frame["glucose_actual_mgdl"]
        measured = frame["glucose_to_algo_mgdl"]
        usable = reference.notna() & measured.notna()
        if usable.sum() < 10:
            return None
        reference, measured = reference[usable], measured[usable]
        if bool((reference == measured).all()):
            return None
        return (reference.to_numpy(), measured.to_numpy())
    except Exception:  # noqa: BLE001 - an optional figure, never a hard failure
        return None


def _glycemic_summary(results_csv: str | Path | None) -> dict[str, float] | None:
    """Measure the consensus glycemic metrics of a run.

    The operator has to know these before standing in front of an audience: a
    jury that knows the Battelino (2019) targets will read them off the trace
    whether or not they are mentioned.
    """

    if results_csv is None:
        return None
    try:
        import pandas as pd

        glucose = pd.read_csv(results_csv)["glucose_actual_mgdl"].dropna()
        if glucose.empty:
            return None
        return {
            "mean_mgdl": round(float(glucose.mean()), 1),
            "cv_pct": round(float(glucose.std() / glucose.mean() * 100), 1),
            "time_in_range_70_180_pct": round(
                float(glucose.between(70, 180).mean() * 100), 1
            ),
            "time_below_70_pct": round(float((glucose < 70).mean() * 100), 1),
            "time_below_54_pct": round(float((glucose < 54).mean() * 100), 1),
            "time_above_180_pct": round(float((glucose > 180).mean() * 100), 1),
            "time_above_250_pct": round(float((glucose > 250).mean() * 100), 1),
            "min_mgdl": round(float(glucose.min()), 1),
            "max_mgdl": round(float(glucose.max()), 1),
        }
    except Exception:  # noqa: BLE001 - reporting aid, never a hard failure
        return None


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
        record(f"Run preset '{key}'", "Run protocols", partial(run_preset, key))

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

    # Evidence the portfolio can consume has to exist before the portfolio is
    # generated, so the benchmark and the paired-trace assembly run first.
    if include_portfolio:
        portfolio_evidence: dict[str, Any] = {}

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
            portfolio_evidence["benchmark_dir"] = target
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

        def pair_safety_trace() -> tuple[str, tuple[str, ...]]:
            """Join one case's two measured traces into the shape the figure needs.

            The benchmark writes the comparator and the supervised run of each
            case to separate files; the portfolio figure wants both glucose
            columns in one table. The case is chosen by sorted identifier
            rather than by which one looks best, and the identifier is stated
            wherever the figure is described.
            """

            import pandas as pd

            benchmark_dir = portfolio_evidence.get("benchmark_dir")
            if benchmark_dir is None:
                raise RuntimeError(
                    "The safety benchmark did not complete, so there is no trace to pair."
                )
            cases = sorted(
                path.name.removesuffix("_unmitigated.csv")
                for path in Path(benchmark_dir).glob("*_unmitigated.csv")
            )
            if not cases:
                raise RuntimeError("The benchmark wrote no comparator traces.")
            case_id = cases[0]
            unmitigated = pd.read_csv(Path(benchmark_dir) / f"{case_id}_unmitigated.csv")
            supervised = pd.read_csv(Path(benchmark_dir) / f"{case_id}_supervised.csv")
            paired = (
                unmitigated[["time_minutes", "glucose_actual_mgdl"]]
                .rename(columns={"glucose_actual_mgdl": "unsupervised_glucose_mgdl"})
                .merge(
                    supervised[["time_minutes", "glucose_actual_mgdl"]].rename(
                        columns={"glucose_actual_mgdl": "supervised_glucose_mgdl"}
                    ),
                    on="time_minutes",
                    how="inner",
                )
                .sort_values("time_minutes")
            )
            if len(paired) < 2:
                raise RuntimeError(
                    f"Only {len(paired)} paired row(s) for {case_id}; the figure needs at least two."
                )
            target = Path(benchmark_dir) / f"{case_id}_paired_trace.csv"
            paired.to_csv(target, index=False)
            portfolio_evidence["safety_trace"] = target
            portfolio_evidence["safety_case_id"] = case_id
            return (
                f"Paired comparator and supervisor trace for {case_id} "
                f"({len(paired)} rows), both measured in this seeding run.",
                _existing(target),
            )

        record(
            "Pair a safety benchmark trace",
            "Scientific Portfolio",
            pair_safety_trace,
        )

        def portfolio() -> tuple[str, tuple[str, ...]]:
            from iints.research.eucys_playbook_generator import (
                generate_complete_eucys_jury_portfolio,
            )

            resolved_portfolio.mkdir(parents=True, exist_ok=True)
            built = generate_complete_eucys_jury_portfolio(
                output_dir=resolved_portfolio,
                ega_pairs=_sensor_error_pairs(main_run),
                safety_trace=portfolio_evidence.get("safety_trace"),
            )
            payload = built.to_dict()
            rendered = [
                figure
                for figure in payload.get("figures", [])
                if "Not generated" not in (figure.get("subtitle") or "")
            ]
            return (
                f"{len(rendered)} of {payload.get('total_figures')} portfolio figures "
                f"rendered where the panel looks ({resolved_portfolio}); the rest state "
                "which evidence they still need.",
                _existing(
                    resolved_portfolio / "index.html",
                    resolved_portfolio / "eucys_portfolio_manifest.json",
                ),
            )

        record("Generate the scientific portfolio", "Scientific Portfolio", portfolio)

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
                    "glycemic_summary": _glycemic_summary(run.results_csv),
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


def main(argv: list[str] | None = None) -> int:
    """Seed a demonstration folder from the command line.

    Exit status is 0 when every step completed and 1 when any step failed, so
    the operator finds out at the terminal rather than in front of an audience.
    """

    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m iints_desktop.jury_demo",
        description=(
            "Fill one folder with a worked example for every panel of the IINTS "
            "desktop app, then write a walkthrough that names what is ready and "
            "what is not. Research use only; the artifacts are simulated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "iints-jury-demo"),
        help="Folder to seed; set the app's output folder to this path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--portfolio-dir",
        default=None,
        help=(
            "Where to write the Scientific Portfolio. Defaults to "
            f"~/{PORTFOLIO_PANEL_SUBPATH.as_posix()}, which is the path that panel reads."
        ),
    )
    parser.add_argument(
        "--skip-portfolio",
        action="store_true",
        help="Skip the slower figure and benchmark generation.",
    )
    parser.add_argument(
        "--creator-name",
        default=None,
        help="Written into the academic bundle as supplied; omitted when absent.",
    )
    parser.add_argument("--creator-orcid", default=None)
    args = parser.parse_args(argv)

    result = build_jury_demo(
        output_dir=args.output_dir,
        seed=args.seed,
        portfolio_dir=args.portfolio_dir,
        include_portfolio=not args.skip_portfolio,
        creator_name=args.creator_name,
        creator_orcid=args.creator_orcid,
        progress=lambda message: print(f"  {message}", flush=True),
    )

    print()
    for step in result.steps:
        print(f"{_status_mark(step.status):<10} {step.panel:<34} {step.name}")
        if step.status != "ok":
            print(f"{'':<10} -> {step.detail}")
    print()
    print(f"Walkthrough: {result.walkthrough_path}")
    print(f"Set the app output folder to: {result.output_dir}")
    if result.failed_steps:
        print(f"{len(result.failed_steps)} step(s) failed; read the walkthrough before presenting.")
        return 1
    return 0


def _rendered_portfolio_figures(portfolio_dir: Path) -> list[dict[str, Any]]:
    """Read back which portfolio figures actually rendered.

    The manifest counts every planned figure, including the ones it honestly
    declined to draw, so the walkthrough reads the subtitles rather than the
    total and reports only what an operator can really open.
    """

    manifest = Path(portfolio_dir) / "eucys_portfolio_manifest.json"
    if not manifest.exists():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [
        figure
        for figure in payload.get("figures", [])
        if "Not generated" not in (figure.get("subtitle") or "")
    ]


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

    summary = _glycemic_summary(main_run.results_csv if main_run else None)
    if summary is not None:
        lines.append(
            "Measured on the seeded trace, so you are not guessing when the jury asks:"
        )
        lines.append("")
        lines.append(
            f"- Time in range 70-180 mg/dL: **{summary['time_in_range_70_180_pct']}%** "
            f"(mean {summary['mean_mgdl']} mg/dL, CV {summary['cv_pct']}%)"
        )
        lines.append(
            f"- Below 70 mg/dL: {summary['time_below_70_pct']}% "
            f"(below 54: {summary['time_below_54_pct']}%)"
        )
        lines.append(
            f"- Above 180 mg/dL: {summary['time_above_180_pct']}% "
            f"(above 250: {summary['time_above_250_pct']}%)"
        )
        lines.append(
            f"- Range spanned: {summary['min_mgdl']} to {summary['max_mgdl']} mg/dL"
        )
        lines.append("")
        if summary["time_in_range_70_180_pct"] < 70.0:
            lines.append(
                "> A jury member who knows the consensus targets (Battelino et al., "
                "2019) will notice this run sits below 70% time in range. That is the "
                "**intended** behaviour of this profile, and saying so first is the "
                "strong answer. `reference_free_living_t1d` was calibrated against "
                "aggregate OhioT1DM statistics (12 subjects, 188,980 CGM rows, mean "
                "glucose 159.6 mg/dL, time in range 63.8%), so the reference day "
                "reproduces the control a real free-living cohort actually achieves "
                "rather than an idealised one. A simulator that returned 90% here "
                "would be the suspicious result. What this run demonstrates is the "
                "pipeline on an empirically anchored day - it is not a controller "
                "performance claim."
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

    rendered = _rendered_portfolio_figures(result.portfolio_dir)
    if rendered:
        lines.append(
            f"{len(rendered)} figure(s) render from the evidence this seeding produced; "
            "the others state on their own card which experiment they still need, which "
            "is a better answer to a jury than a placeholder."
        )
        lines.append("")
        for figure in rendered:
            lines.append(f"- **{figure['figure_id']}** {figure['title']}")
        lines.append("")
        if any(figure["figure_id"] == "FIG-05" for figure in rendered):
            lines.append(
                "> Describe the error grid precisely. It pairs the simulator's own "
                "glucose against the value handed to the controller, so it "
                "characterises the **sensor error model of the simulation**. A high "
                "Zone A percentage here is a property of that model, not a clinical "
                "accuracy result, and claiming otherwise is the one thing a jury will "
                "not forgive."
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

    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
