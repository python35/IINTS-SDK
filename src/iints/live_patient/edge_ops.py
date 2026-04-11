from __future__ import annotations

import json
import shutil
import stat
import zipfile
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

from .runtime import PatientRuntimeConfig, get_runtime_scenario_profile, is_process_alive, load_runtime_status
from .service_export import write_service_artifacts
from .uno_q import export_uno_q_bridge


def _sdk_version() -> str:
    try:
        return version("iints-sdk-python35")
    except PackageNotFoundError:  # pragma: no cover - source tree fallback
        return "source"


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _format_iso_timestamp(epoch_seconds: float | None) -> str | None:
    if epoch_seconds is None:
        return None
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _review_preview(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                return text[:180]
    except Exception:
        pass
    return "Review file present."


def _certification_summary(bundle_dir: Path) -> dict[str, Any]:
    for candidate in (
        bundle_dir / "certification.json",
        bundle_dir / "audit" / "certification.json",
    ):
        payload = _read_json_if_exists(candidate)
        if payload is None:
            continue
        grade = payload.get("mdmp_grade") or payload.get("grade") or "draft"
        certified = bool(payload.get("certified_for_medical_research", False))
        return {
            "exists": True,
            "path": str(candidate),
            "grade": grade,
            "certified_for_medical_research": certified,
            "compliance_score": payload.get("compliance_score"),
            "label": f"{grade} ({'certified' if certified else 'not certified'})",
        }
    return {
        "exists": False,
        "path": None,
        "grade": None,
        "certified_for_medical_research": False,
        "compliance_score": None,
        "label": "Not certified yet",
    }


def _review_summary(bundle_dir: Path) -> dict[str, Any]:
    review_path = bundle_dir / "ai" / "realism_review.md"
    if not review_path.is_file():
        return {
            "exists": False,
            "path": None,
            "updated_at_utc": None,
            "label": "No realism review yet",
            "preview": None,
        }
    stat_info = review_path.stat()
    return {
        "exists": True,
        "path": str(review_path),
        "updated_at_utc": _format_iso_timestamp(stat_info.st_mtime),
        "label": "Review ready",
        "preview": _review_preview(review_path),
    }


def summarize_edge_workspace(workspace: str | Path) -> dict[str, Any]:
    workspace_path = Path(workspace).expanduser().resolve()
    if not (workspace_path / "patient_state.db").is_file():
        return {}
    status = load_runtime_status(workspace_path)
    bundle_dir = workspace_path / "live_bundle"
    certification = _certification_summary(bundle_dir)
    review = _review_summary(bundle_dir)
    pid_alive = is_process_alive(status.get("pid"))
    api_host = status.get("api_host") or "127.0.0.1"
    api_port = status.get("api_port") or 8765
    dashboard_url = f"http://{api_host}:{api_port}/dashboard"
    kiosk_url = f"http://{api_host}:{api_port}/kiosk"

    return {
        **status,
        "workspace": str(workspace_path),
        "pid_alive": pid_alive,
        "dashboard_url": dashboard_url,
        "kiosk_url": kiosk_url,
        "api_url": f"http://{api_host}:{api_port}",
        "workspace_size_mb": round(_directory_size_bytes(workspace_path) / (1024 * 1024), 3),
        "bundle_size_mb": round(_directory_size_bytes(bundle_dir) / (1024 * 1024), 3),
        "last_heartbeat_utc": status.get("updated_at_utc"),
        "certification": certification,
        "review": review,
        "artifacts": {
            "database": str(workspace_path / "patient_state.db"),
            "snapshot": str(workspace_path / "simulator_snapshot.json"),
            "log": str(workspace_path / "patient.log"),
            "bundle_dir": str(bundle_dir),
        },
    }


def create_edge_bundle(
    workspace: str | Path,
    *,
    output_path: str | Path,
    include_log: bool = True,
    include_database: bool = True,
) -> dict[str, Any]:
    workspace_path = Path(workspace).expanduser().resolve()
    bundle_output = Path(output_path).expanduser().resolve()
    bundle_output.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_edge_workspace(workspace_path)
    bundle_root = workspace_path.name

    with zipfile.ZipFile(bundle_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{bundle_root}/EDGE_BUNDLE_SUMMARY.json",
            json.dumps(summary, indent=2, sort_keys=True),
        )
        archive.writestr(
            f"{bundle_root}/README.txt",
            "\n".join(
                [
                    "IINTS edge runtime bundle",
                    "",
                    "Use this bundle on a workstation to inspect:",
                    "- the persistent SQLite state",
                    "- the live bundle CSV and manifests",
                    "- certification artifacts if present",
                    "- realism review markdown if present",
                ]
            )
            + "\n",
        )

        for path in workspace_path.rglob("*"):
            if not path.is_file():
                continue
            if not include_log and path.name == "patient.log":
                continue
            if not include_database and path.name == "patient_state.db":
                continue
            relative = path.relative_to(workspace_path)
            archive.write(path, arcname=f"{bundle_root}/{relative.as_posix()}")

    return {
        "archive": str(bundle_output),
        "workspace": str(workspace_path),
        "summary": summary,
    }


def render_edge_update_script(*, profile: str = "edge", version_pin: str | None = None) -> str:
    target = version_pin or _sdk_version()
    spec = f'iints-sdk-python35[{profile},mdmp]=={target}' if version_pin else f'iints-sdk-python35[{profile},mdmp]'
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "python -m pip install -U pip",
            f'python -m pip install -U "{spec}"',
            'hash -r || true',
            "iints doctor --smoke-run || true",
            "",
        ]
    )


def write_edge_update_script(
    output_path: str | Path,
    *,
    profile: str = "edge",
    version_pin: str | None = None,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_edge_update_script(profile=profile, version_pin=version_pin), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _render_default_algorithm(algo_name: str, author_name: str) -> str:
    template = files("iints.templates").joinpath("default_algorithm.py").read_text(encoding="utf-8")
    return template.replace("{{ALGO_NAME}}", algo_name).replace("{{AUTHOR_NAME}}", author_name)


def export_edge_setup(
    output_dir: str | Path,
    *,
    board: str = "raspberry_pi",
    workspace_name: str = "patient_runtime",
    scenario_profile: str = "normal_day",
    patient_config: str = "default_patient",
    patient_model_type: str = "auto",
    mode: str = "demo-time",
    speed: float = 60.0,
    api_host: str = "127.0.0.1",
    api_port: int = 8765,
    seed: int | None = None,
    service_name: str = "iints-digital-patient",
    user_name: str | None = None,
    include_uno_bridge: bool = False,
) -> dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    algorithms_dir = root / "algorithms"
    workspace = root / workspace_name
    algorithms_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    algo_path = algorithms_dir / "example_algorithm.py"
    algo_path.write_text(_render_default_algorithm("EdgeDemoAlgorithm", "IINTS SBC Setup"), encoding="utf-8")

    profile = get_runtime_scenario_profile(scenario_profile)
    config = PatientRuntimeConfig(
        workspace=str(workspace),
        algo_path=str(algo_path),
        patient_config=patient_config,
        patient_model_type=patient_model_type,
        scenario_profile=profile.name,
        mode=mode,
        speed=speed,
        api_host=api_host,
        api_port=api_port,
        seed=seed if seed is not None else profile.default_seed,
    )
    config.config_path.write_text(json.dumps(config.to_json(), indent=2, sort_keys=True), encoding="utf-8")

    run_script = root / "run_edge_patient.sh"
    run_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'cd "$(dirname "$0")"',
                "iints patient start \\",
                "  --algo algorithms/example_algorithm.py \\",
                f"  --workspace {workspace_name} \\",
                f"  --scenario-profile {profile.name} \\",
                f"  --mode {mode} \\",
                f"  --speed {speed:g}x",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_script.chmod(run_script.stat().st_mode | stat.S_IXUSR)

    kiosk_script = root / "launch_kiosk.sh"
    kiosk_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'URL="http://{api_host}:{api_port}/kiosk"',
                'if command -v xdg-open >/dev/null 2>&1; then',
                '  xdg-open "$URL"',
                'elif command -v open >/dev/null 2>&1; then',
                '  open "$URL"',
                "else",
                '  echo "Open $URL in your browser."',
                "fi",
                "",
            ]
        ),
        encoding="utf-8",
    )
    kiosk_script.chmod(kiosk_script.stat().st_mode | stat.S_IXUSR)

    update_script = write_edge_update_script(root / "update_edge_runtime.sh", profile="edge")
    service_paths = write_service_artifacts(
        config,
        output_path=workspace / f"{service_name}.service",
        service_name=service_name,
        user_name=user_name,
    )

    if board == "uno_q":
        guide_lines = [
            "# IINTS Edge Setup",
            "",
            f"Board profile: `{board}`",
            f"Scenario profile: `{profile.name}`",
            f"Workspace: `{workspace}`",
            "",
            "This setup has two parts:",
            "",
            "1. the Linux side runs the IINTS digital patient",
            "2. the STM32 side runs the simple LED / buzzer bridge sketch",
            "",
            "## 1. Start the Linux side",
            "",
            "```bash",
            "iints edge up --project-dir .",
            "```",
            "",
            "Check that the runtime is alive:",
            "",
            "```bash",
            "iints edge status --project-dir .",
            "```",
            "",
            "## 2. Open the kiosk dashboard",
            "",
            "```bash",
            "iints edge kiosk --project-dir .",
            "```",
            "",
            "## 3. Flash the STM32 bridge sketch",
            "",
            "CLI path with Arduino CLI installed:",
            "",
            "```bash",
            "iints edge bridge-flash --project-dir . --port /dev/ttyACM0 --fqbn <your-board-fqbn>",
            "```",
            "",
            "Or open this file in Arduino IDE:",
            "",
            "```text",
            "uno_q_bridge/iints_supervisor_bridge.ino",
            "```",
            "",
            "Then:",
            "",
            "- upload it to the STM32 side of the UNO Q",
            "- open Serial Monitor at `115200` baud",
            "- set line ending to `Newline`",
            "- confirm the sketch prints `IINTS UNO Q supervisor bridge ready`",
            "",
            "## 4. Manual bridge test",
            "",
            "Send these serial messages one by one:",
            "",
            "```text",
            "OK",
            "OVERRIDE",
            "CRITICAL",
            "```",
            "",
            "CLI shortcut:",
            "",
            "```bash",
            "iints edge bridge-test --port /dev/ttyACM0",
            "```",
            "",
            "Expected behavior:",
            "",
            "- `OK`: built-in status LED on",
            "- `OVERRIDE`: red LED on",
            "- `CRITICAL`: red LED on and buzzer chirps",
            "- note: on some UNO Q boards the built-in status LED looks blue instead of green",
            "- note: the bridge test ends on the last state you send, so LEDs can stay lit until the next `OK`",
            "",
            "## 5. Live bridge forwarding",
            "",
            "```bash",
            "iints edge bridge-run --project-dir . --port /dev/ttyACM0",
            "```",
            "",
            "This command follows the Linux-side runtime state and sends `OK`, `OVERRIDE`, or `CRITICAL` to the STM32 side.",
            "",
            "## 6. Recommended first success target",
            "",
            "1. Linux runtime works",
            "2. bridge sketch works",
            "3. bridge test works",
            "4. bridge forwarder works",
            "",
            "## Service install",
            "",
            f"- Service file: `{service_paths['service_file']}`",
            f"- Install notes: `{service_paths['install_notes']}`",
            "",
            "## Update edge runtime",
            "",
            "```bash",
            "./update_edge_runtime.sh",
            "```",
            "",
            "## Useful live commands",
            "",
            "- `iints edge status --project-dir .`",
            "- `iints edge kiosk --project-dir .`",
            "- `iints edge reset --project-dir .`",
            "- `iints edge stop --project-dir .`",
            "- `iints edge bundle --project-dir . --output edge_bundle.zip`",
            "",
        ]
    else:
        guide_lines = [
            "# IINTS Edge Setup",
            "",
            f"Board profile: `{board}`",
            f"Scenario profile: `{profile.name}`",
            f"Workspace: `{workspace}`",
            "",
            "## First run",
            "",
            "```bash",
            "iints edge up --project-dir .",
            "```",
            "",
            "## Open the kiosk dashboard",
            "",
            "```bash",
            "iints edge kiosk --project-dir .",
            "```",
            "",
            "## Service install",
            "",
            f"- Service file: `{service_paths['service_file']}`",
            f"- Install notes: `{service_paths['install_notes']}`",
            "",
            "## Update edge runtime",
            "",
            "```bash",
            "./update_edge_runtime.sh",
            "```",
            "",
            "## Useful live commands",
            "",
            "- `iints edge status --project-dir .`",
            "- `iints edge kiosk --project-dir .`",
            "- `iints edge reset --project-dir .`",
            "- `iints edge stop --project-dir .`",
            "- `iints edge service --project-dir .`",
            "- `iints edge bundle --project-dir . --output edge_bundle.zip`",
            "",
        ]

    setup_guide = root / "EDGE_SETUP.md"
    setup_guide.write_text("\n".join(guide_lines) + "\n", encoding="utf-8")

    outputs = {
        "root": str(root),
        "algorithm": str(algo_path),
        "workspace": str(workspace),
        "config": str(config.config_path),
        "run_script": str(run_script),
        "kiosk_script": str(kiosk_script),
        "update_script": str(update_script),
        "service_file": service_paths["service_file"],
        "service_notes": service_paths["install_notes"],
        "setup_guide": str(setup_guide),
    }

    if board == "uno_q" or include_uno_bridge:
        bridge = export_uno_q_bridge(root / "uno_q_bridge")
        outputs["uno_q_bridge"] = bridge["output_dir"]

    return outputs
