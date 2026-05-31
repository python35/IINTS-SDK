"""FPGA safety-core workflow helpers for bench-only hardware research."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FPGA_CONFIRMATION = "I understand this is bench-only FPGA research and not for treatment"
FPGA_DEFAULT_BAUDRATE = 115200
FPGA_READY_BANNER = "IINTS FPGA Safety Core ready"


DEFAULT_FPGA_SAFETY_CONTRACT: dict[str, Any] = {
    "schema_version": "1.0",
    "target": "fpga_safety_core",
    "scope": "education_simulation_preclinical_research_only",
    "medical_device": False,
    "insulin_delivery_enabled": False,
    "allowed_outputs": ["NORMAL", "WARNING", "CRITICAL", "SENSOR_ERROR", "CHECK_REQUIRED"],
    "requires_sdk_comparison": True,
    "requires_run_manifest": True,
    "notes": [
        "The FPGA core is a deterministic safety/risk demonstrator, not a dosing controller.",
        "All outputs must be compared against the SDK software reference before demo use.",
        "Do not connect FPGA outputs to real insulin delivery hardware.",
    ],
}


DEFAULT_FPGA_EVENTS: list[dict[str, Any]] = [
    {
        "minute": 0,
        "glucose_mgdl": 118.0,
        "trend_mgdl_per_min": 0.0,
        "sensor_status": "OK",
        "meal_event": False,
        "insulin_event": False,
    },
    {
        "minute": 5,
        "glucose_mgdl": 142.0,
        "trend_mgdl_per_min": 1.1,
        "sensor_status": "OK",
        "meal_event": True,
        "insulin_event": False,
    },
    {
        "minute": 10,
        "glucose_mgdl": 86.0,
        "trend_mgdl_per_min": -1.8,
        "sensor_status": "OK",
        "meal_event": False,
        "insulin_event": True,
    },
    {
        "minute": 15,
        "glucose_mgdl": 64.0,
        "trend_mgdl_per_min": -1.2,
        "sensor_status": "OK",
        "meal_event": False,
        "insulin_event": True,
    },
    {
        "minute": 20,
        "glucose_mgdl": 151.0,
        "trend_mgdl_per_min": 0.4,
        "sensor_status": "ERROR",
        "meal_event": False,
        "insulin_event": False,
    },
]


FPGA_NIGHT_HYPO_RISK_EVENTS: list[dict[str, Any]] = [
    {
        "minute": 0,
        "glucose_mgdl": 112.0,
        "trend_mgdl_per_min": -0.2,
        "sensor_status": "OK",
        "meal_event": False,
        "insulin_event": False,
    },
    {
        "minute": 30,
        "glucose_mgdl": 101.0,
        "trend_mgdl_per_min": -0.5,
        "sensor_status": "OK",
        "meal_event": False,
        "insulin_event": False,
    },
    {
        "minute": 60,
        "glucose_mgdl": 91.0,
        "trend_mgdl_per_min": -0.9,
        "sensor_status": "OK",
        "meal_event": False,
        "insulin_event": True,
    },
    {
        "minute": 90,
        "glucose_mgdl": 82.0,
        "trend_mgdl_per_min": -1.4,
        "sensor_status": "OK",
        "meal_event": False,
        "insulin_event": True,
    },
    {
        "minute": 120,
        "glucose_mgdl": 66.0,
        "trend_mgdl_per_min": -1.7,
        "sensor_status": "OK",
        "meal_event": False,
        "insulin_event": False,
    },
]


@dataclass(frozen=True)
class FPGARunSummary:
    """Paths and pass/fail metadata from one FPGA safety-core comparison run."""

    output_dir: Path
    events_path: Path
    results_csv: Path
    comparison_json: Path
    report_md: Path
    manifest_json: Path
    mismatch_count: int
    max_latency_ms: float
    transport: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "events_path": str(self.events_path),
            "results_csv": str(self.results_csv),
            "comparison_json": str(self.comparison_json),
            "report_md": str(self.report_md),
            "manifest_json": str(self.manifest_json),
            "mismatch_count": self.mismatch_count,
            "max_latency_ms": self.max_latency_ms,
            "transport": self.transport,
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_fpga_event(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize one SDK event into the tiny FPGA safety-core schema."""

    return {
        "minute": int(float(event.get("minute", event.get("time_minutes", 0)) or 0)),
        "glucose_mgdl": float(event.get("glucose_mgdl", event.get("glucose", 120.0)) or 120.0),
        "trend_mgdl_per_min": float(
            event.get("trend_mgdl_per_min", event.get("trend", event.get("glucose_trend", 0.0))) or 0.0
        ),
        "sensor_status": str(event.get("sensor_status", "OK") or "OK").upper(),
        "meal_event": _as_bool(event.get("meal_event", event.get("carbs", 0.0))),
        "insulin_event": _as_bool(event.get("insulin_event", event.get("insulin", 0.0))),
    }


def evaluate_fpga_safety_reference(event: dict[str, Any]) -> dict[str, Any]:
    """Deterministic software reference for the first IINTS FPGA safety core."""

    normalized = normalize_fpga_event(event)
    glucose = float(normalized["glucose_mgdl"])
    trend = float(normalized["trend_mgdl_per_min"])
    sensor_status = str(normalized["sensor_status"]).upper()
    reasons: list[str] = []
    risk_label = "NORMAL"
    risk_score = 0
    check_required = False

    if sensor_status not in {"OK", "VALID"}:
        return {
            **normalized,
            "risk_label": "SENSOR_ERROR",
            "risk_score": 3,
            "check_required": True,
            "reasons": ["sensor status is not valid"],
        }

    if glucose < 54.0:
        risk_label = "CRITICAL"
        risk_score = 3
        check_required = True
        reasons.append("severe hypoglycemia threshold")
    elif glucose < 70.0:
        risk_label = "CRITICAL"
        risk_score = 3
        check_required = True
        reasons.append("hypoglycemia threshold")
    elif glucose < 90.0:
        risk_label = "WARNING"
        risk_score = 2
        check_required = True
        reasons.append("near-low glucose")
    elif glucose > 250.0:
        risk_label = "WARNING"
        risk_score = 2
        check_required = True
        reasons.append("very high glucose")

    if trend <= -2.0 and glucose < 120.0:
        risk_label = "CRITICAL" if glucose < 90.0 else "WARNING"
        risk_score = max(risk_score, 3 if glucose < 90.0 else 2)
        check_required = True
        reasons.append("fast downward glucose trend")
    elif trend <= -1.0 and glucose < 110.0:
        risk_label = "WARNING" if risk_label == "NORMAL" else risk_label
        risk_score = max(risk_score, 2)
        check_required = True
        reasons.append("downward trend near lower range")

    if normalized["insulin_event"] and glucose < 110.0:
        risk_label = "WARNING" if risk_label == "NORMAL" else risk_label
        risk_score = max(risk_score, 2)
        check_required = True
        reasons.append("insulin event while glucose is low or falling")

    if not reasons:
        reasons.append("inside deterministic demo guard band")

    return {
        **normalized,
        "risk_label": risk_label,
        "risk_score": risk_score,
        "check_required": check_required,
        "reasons": reasons,
    }


class MockFPGATransport:
    """Reference transport that behaves like an FPGA safety core without hardware."""

    transport_name = "mock"

    def evaluate(self, event: dict[str, Any]) -> tuple[dict[str, Any], float]:
        start = time.perf_counter()
        response = evaluate_fpga_safety_reference(event)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return response, latency_ms


class SerialFPGATransport:
    """JSON-lines serial transport for a future FPGA board or MCU bridge."""

    transport_name = "serial"

    def __init__(self, port: str, *, baudrate: int = FPGA_DEFAULT_BAUDRATE, timeout_seconds: float = 1.5) -> None:
        try:
            import serial  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional host package
            raise RuntimeError("pyserial is required for FPGA serial mode. Install the SDK with edge/full extras.") from exc
        self._serial_module = serial
        self.port = port
        self.baudrate = baudrate
        self.timeout_seconds = timeout_seconds

    def evaluate(self, event: dict[str, Any]) -> tuple[dict[str, Any], float]:  # pragma: no cover - hardware path
        payload = json.dumps(normalize_fpga_event(event), sort_keys=True).encode("utf-8") + b"\n"
        start = time.perf_counter()
        with self._serial_module.Serial(self.port, self.baudrate, timeout=self.timeout_seconds) as handle:
            handle.write(payload)
            raw = handle.readline().decode("utf-8", errors="replace").strip()
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not raw:
            raise RuntimeError("FPGA serial core returned an empty response.")
        response = json.loads(raw)
        if not isinstance(response, dict):
            raise RuntimeError("FPGA serial core returned a non-object JSON response.")
        return response, latency_ms


def load_fpga_events(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(event) for event in DEFAULT_FPGA_EVENTS]
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"FPGA event file not found: {source}")
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("events", [])
        if not isinstance(payload, list):
            raise ValueError("FPGA JSON input must be a list or an object with an events list.")
        return [normalize_fpga_event(dict(item)) for item in payload if isinstance(item, dict)]
    rows: list[dict[str, Any]] = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(normalize_fpga_event(dict(row)))
    return rows


def _write_events_jsonl(path: Path, events: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(normalize_fpga_event(event), sort_keys=True) + "\n")


def _write_events_csv(path: Path, events: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "minute",
        "glucose_mgdl",
        "trend_mgdl_per_min",
        "sensor_status",
        "meal_event",
        "insulin_event",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            normalized = normalize_fpga_event(event)
            writer.writerow({key: normalized.get(key, "") for key in fieldnames})


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "minute",
        "glucose_mgdl",
        "trend_mgdl_per_min",
        "sensor_status",
        "meal_event",
        "insulin_event",
        "software_risk_label",
        "fpga_risk_label",
        "software_risk_score",
        "fpga_risk_score",
        "match",
        "latency_ms",
        "software_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_fpga_report(path: Path, *, comparison: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# IINTS FPGA Mode Report",
        "",
        "Scope: education, simulation, and pre-clinical research demos only.",
        "",
        "This report compares the SDK software reference with a hardware-style FPGA safety-core output.",
        "It is not a medical device report and must not be used for treatment decisions.",
        "",
        "## Summary",
        "",
        f"- Transport: `{comparison['transport']}`",
        f"- Scenario: `{comparison['scenario_name']}`",
        f"- Events: {comparison['event_count']}",
        f"- Mismatches: {comparison['mismatch_count']}",
        f"- Max latency: {comparison['max_latency_ms']:.3f} ms",
        f"- Passed: {comparison['passed']}",
        "",
        "## Scenario Used",
        "",
        f"This run used `{comparison['scenario_name']}` as the FPGA safety-core verification scenario.",
        "The bundled golden scenario is `night_hypo_risk`, which demonstrates falling overnight glucose risk.",
        "",
        "## Software Reference Logic",
        "",
        "The SDK software reference normalizes each event and classifies it with deterministic guard bands:",
        "",
        "- invalid sensor status -> `SENSOR_ERROR`",
        "- glucose below 70 mg/dL -> `CRITICAL`",
        "- glucose below 90 mg/dL or above 250 mg/dL -> `WARNING`",
        "- fast downward trend near the lower range -> `WARNING` or `CRITICAL`",
        "- insulin marker while glucose is low/falling -> `WARNING`",
        "",
        "## FPGA / Mock FPGA Logic",
        "",
        "Mock mode mirrors the same deterministic safety-core contract without hardware.",
        "Serial mode expects a future FPGA bridge to return one JSON object per input event.",
        "The SDK compares `risk_label`, `risk_score`, and `check_required` for every event.",
        "",
        "## Result Table",
        "",
        "| Minute | Glucose | Trend | Software | FPGA | Match | Reasons |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {minute} | {glucose_mgdl:.1f} | {trend_mgdl_per_min:.2f} | {software_risk_label} | "
            "{fpga_risk_label} | {match} | {software_reasons} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `match=True` means the FPGA-style result matched the SDK software reference for the deterministic safety core.",
            "- Any mismatch should be treated as a hardware verification finding, not as a clinical finding.",
            "- Keep the FPGA output isolated from any real actuator.",
            "",
            "## Limitations",
            "",
            "- This is not a physiological validation study.",
            "- The safety core is deliberately small and explainable.",
            "- FPGA mode does not dose insulin and must not control a pump.",
            "- Real hardware testing still needs timing, transport, reset, fault-injection, and watchdog validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fpga_safety_simulation(
    *,
    output_dir: str | Path,
    events_path: str | Path | None = None,
    transport: str = "mock",
    port: str | None = None,
    baudrate: int = FPGA_DEFAULT_BAUDRATE,
    timeout_seconds: float = 1.5,
    scenario_name: str | None = None,
) -> FPGARunSummary:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    events = load_fpga_events(events_path)
    resolved_scenario_name = scenario_name or ("default_fpga_demo" if events_path is None else Path(events_path).stem)
    if transport == "mock":
        transport_impl: Any = MockFPGATransport()
    elif transport == "serial":
        if not port:
            raise ValueError("--port is required for FPGA serial transport.")
        transport_impl = SerialFPGATransport(port, baudrate=baudrate, timeout_seconds=timeout_seconds)
    else:
        raise ValueError("FPGA transport must be 'mock' or 'serial'.")

    rows: list[dict[str, Any]] = []
    mismatch_count = 0
    max_latency_ms = 0.0
    for event in events:
        normalized = normalize_fpga_event(event)
        software = evaluate_fpga_safety_reference(normalized)
        hardware, latency_ms = transport_impl.evaluate(normalized)
        max_latency_ms = max(max_latency_ms, latency_ms)
        match = (
            str(software.get("risk_label")) == str(hardware.get("risk_label"))
            and int(software.get("risk_score", -1)) == int(hardware.get("risk_score", -2))
            and bool(software.get("check_required")) == bool(hardware.get("check_required"))
        )
        if not match:
            mismatch_count += 1
        rows.append(
            {
                **normalized,
                "software_risk_label": software["risk_label"],
                "fpga_risk_label": hardware.get("risk_label", "UNKNOWN"),
                "software_risk_score": software["risk_score"],
                "fpga_risk_score": hardware.get("risk_score", ""),
                "match": match,
                "latency_ms": round(latency_ms, 6),
                "software_reasons": "; ".join(str(item) for item in software["reasons"]),
            }
        )

    events_output = root / "fpga_input_events.jsonl"
    friendly_events_csv = root / "events.csv"
    results_csv = root / "fpga_results.csv"
    friendly_results_json = root / "results.json"
    comparison_json = root / "fpga_comparison.json"
    report_md = root / "fpga_report.md"
    friendly_report_md = root / "report.md"
    manifest_json = root / "fpga_run_manifest.json"
    friendly_manifest_json = root / "manifest.json"
    _write_events_jsonl(events_output, events)
    _write_events_csv(friendly_events_csv, events)
    _write_results_csv(results_csv, rows)
    comparison = {
        "schema_version": "1.0",
        "transport": transport,
        "scenario_name": resolved_scenario_name,
        "software_reference": "evaluate_fpga_safety_reference",
        "fpga_logic": "mock_reference_core" if transport == "mock" else "serial_json_lines_core",
        "event_count": len(rows),
        "mismatch_count": mismatch_count,
        "max_latency_ms": max_latency_ms,
        "passed": mismatch_count == 0,
    }
    _write_json(comparison_json, comparison)
    _write_json(friendly_results_json, {"comparison": comparison, "rows": rows})
    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": DEFAULT_FPGA_SAFETY_CONTRACT["scope"],
        "transport": transport,
        "scenario_name": resolved_scenario_name,
        "contract": DEFAULT_FPGA_SAFETY_CONTRACT,
        "artifacts": {
            "events": events_output.name,
            "events_csv": friendly_events_csv.name,
            "results_csv": results_csv.name,
            "results_json": friendly_results_json.name,
            "comparison": comparison_json.name,
            "report": report_md.name,
            "friendly_report": friendly_report_md.name,
        },
    }
    _write_json(manifest_json, manifest)
    _write_json(friendly_manifest_json, manifest)
    write_fpga_report(report_md, comparison=comparison, rows=rows)
    friendly_report_md.write_text(report_md.read_text(encoding="utf-8"), encoding="utf-8")
    return FPGARunSummary(
        output_dir=root,
        events_path=events_output,
        results_csv=results_csv,
        comparison_json=comparison_json,
        report_md=report_md,
        manifest_json=manifest_json,
        mismatch_count=mismatch_count,
        max_latency_ms=max_latency_ms,
        transport=transport,
    )


def create_fpga_lab(output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    rtl_dir = root / "rtl"
    scenarios_dir = root / "scenarios"
    reports_dir = root / "reports"
    scripts_dir = root / "scripts"
    for directory in (rtl_dir, scenarios_dir, reports_dir, scripts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    contract_path = root / "fpga_safety_contract.json"
    _write_json(contract_path, DEFAULT_FPGA_SAFETY_CONTRACT)

    scenario_path = scenarios_dir / "fpga_demo_events.json"
    _write_json(scenario_path, {"events": DEFAULT_FPGA_EVENTS})

    night_hypo_path = scenarios_dir / "night_hypo_risk.json"
    _write_json(
        night_hypo_path,
        {
            "name": "night_hypo_risk",
            "description": "Golden FPGA demo scenario: falling overnight glucose with a low-risk transition.",
            "events": FPGA_NIGHT_HYPO_RISK_EVENTS,
        },
    )

    rtl_path = rtl_dir / "iints_fpga_safety_core.v"
    rtl_path.write_text(
        """// IINTS FPGA Safety Core - educational deterministic risk classifier.
// Scope: simulation/pre-clinical research demos only. Not a medical device.
module iints_fpga_safety_core(
    input  wire [15:0] glucose_mgdl,
    input  wire signed [15:0] trend_mgdl_per_min_x100,
    input  wire sensor_ok,
    input  wire meal_event,
    input  wire insulin_event,
    output reg  [1:0] risk_score,
    output reg        check_required
);
always @(*) begin
    risk_score = 2'd0;
    check_required = 1'b0;
    if (!sensor_ok) begin
        risk_score = 2'd3;
        check_required = 1'b1;
    end else if (glucose_mgdl < 16'd70) begin
        risk_score = 2'd3;
        check_required = 1'b1;
    end else if (glucose_mgdl < 16'd90 || glucose_mgdl > 16'd250) begin
        risk_score = 2'd2;
        check_required = 1'b1;
    end else if (insulin_event && glucose_mgdl < 16'd110) begin
        risk_score = 2'd2;
        check_required = 1'b1;
    end else if (trend_mgdl_per_min_x100 <= -16'sd100 && glucose_mgdl < 16'd110) begin
        risk_score = 2'd2;
        check_required = 1'b1;
    end
end
endmodule
""",
        encoding="utf-8",
    )

    protocol_path = root / "fpga_protocol.json"
    _write_json(
        protocol_path,
        {
            "transport": "json_lines_uart",
            "baudrate": FPGA_DEFAULT_BAUDRATE,
            "input_fields": list(normalize_fpga_event(DEFAULT_FPGA_EVENTS[0]).keys()),
            "output_fields": ["risk_label", "risk_score", "check_required", "reasons"],
            "ready_banner": FPGA_READY_BANNER,
        },
    )

    readme_path = root / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# IINTS FPGA Lab",
                "",
                "This workspace is for transparent medical-device logic on reconfigurable hardware.",
                "It is educational and pre-clinical only. It is not a medical device and not for treatment decisions.",
                "",
                "## Flow",
                "",
                "1. IINTS creates digital-patient CGM/events.",
                "2. The SDK sends those events to a software reference, mock FPGA, or serial FPGA bridge.",
                "3. The FPGA-style output is compared against the SDK reference.",
                "4. IINTS writes CSV, JSON, manifest, and report artifacts.",
                "",
                "## First Demo",
                "",
                "```bash",
                "iints fpga simulate --events scenarios/night_hypo_risk.json --output-dir reports/night_hypo_mock_run",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    story_path = root / "FPGA_STORY.md"
    story_path.write_text(
        "\n".join(
            [
                "# IINTS FPGA Story",
                "",
                "## Why FPGA Mode Exists",
                "",
                "IINTS already simulates digital patients and algorithm behavior in software.",
                "FPGA mode adds a hardware-logic layer so deterministic safety rules can be tested as if they were implemented on reconfigurable hardware.",
                "",
                "## What It Demonstrates",
                "",
                "- how glucose/events from a simulator can be sent to a hardware-style safety core",
                "- how a software reference and FPGA-style output can be compared event by event",
                "- how mismatch counts, latency, and manifests can become research evidence",
                "- how medical-device logic can be discussed transparently before any clinical or actuator use",
                "",
                "## What It Does Not Do",
                "",
                "- it does not dose insulin",
                "- it does not control a pump",
                "- it does not certify clinical safety",
                "- it is not a medical device",
                "",
                "## Future Silicon / AI Path",
                "",
                "The intended roadmap is incremental: mock FPGA mode, stable serial protocol, real UART loopback, real safety core, timing comparison, feature extractor, tiny neural network accelerator, then a future silicon/ASIC story.",
                "",
                "The first real hardware milestone should stay simple: SDK sends one JSON-line event, the bridge returns one risk state, and the SDK logs, compares, and reports the result.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    demo_script = scripts_dir / "run_mock_fpga_demo.sh"
    demo_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd \"$(dirname \"$0\")/..\"",
                "iints fpga simulate --events scenarios/night_hypo_risk.json --output-dir reports/night_hypo_mock_run",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    demo_script.chmod(0o755)

    return {
        "output_dir": str(root),
        "contract": str(contract_path),
        "scenario": str(scenario_path),
        "night_hypo_scenario": str(night_hypo_path),
        "rtl": str(rtl_path),
        "protocol": str(protocol_path),
        "readme": str(readme_path),
        "story": str(story_path),
        "demo_script": str(demo_script),
    }


def fpga_environment_report() -> dict[str, Any]:
    try:
        import serial  # type: ignore  # noqa: F401

        pyserial_available = True
    except ImportError:
        pyserial_available = False
    return {
        "mock_transport_ready": True,
        "serial_transport_ready": pyserial_available,
        "default_baudrate": FPGA_DEFAULT_BAUDRATE,
        "scope": DEFAULT_FPGA_SAFETY_CONTRACT["scope"],
        "medical_device": False,
    }
