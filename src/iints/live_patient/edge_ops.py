from __future__ import annotations

import json
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

from .runtime import PatientRuntimeConfig, get_runtime_scenario_profile, is_process_alive, load_runtime_status
from .service_export import (
    write_makerfaire_autostart_artifacts,
    write_service_artifacts,
    write_uno_q_bridge_service_artifact,
)
from .long_study import render_edge_long_study_config_template
from .uno_q import UNO_Q_BRIDGE_BAUDRATE, export_uno_q_bridge


def _sdk_version() -> str:
    try:
        return version("iints-sdk-python35")
    except PackageNotFoundError:  # pragma: no cover - source tree fallback
        import iints as iints_sdk

        return getattr(iints_sdk, "__version__", "source")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
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
    except (OSError, UnicodeDecodeError):
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


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _ssh_destination(host: str, user_name: str | None = None) -> str:
    return f"{user_name}@{host}" if user_name else host


def _ssh_command(destination: str, *, port: int, remote_command: str) -> list[str]:
    command = ["ssh"]
    if port != 22:
        command.extend(["-p", str(port)])
    command.extend([destination, "bash", "-lc", remote_command])
    return command


def _remote_path_expr(path: str) -> str:
    stripped = path.strip()
    if stripped == "~":
        return "$HOME"
    if stripped.startswith("~/"):
        return "$HOME/" + stripped[2:]
    return shlex.quote(stripped)


def _run_process(
    command: list[str],
    *,
    step: str,
    timeout_seconds: float = 300.0,
    retries: int = 0,
    cwd: Path | None = None,
) -> str:
    attempt = 0
    while True:
        attempt += 1
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(cwd) if cwd is not None else None,
            )
            return (result.stdout or "").strip()
        except FileNotFoundError as exc:
            raise RuntimeError(f"{step} failed because `{command[0]}` is not installed or not on PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            if attempt <= retries + 1:
                if attempt <= retries:
                    continue
            raise RuntimeError(
                f"{step} timed out after {timeout_seconds:.0f}s. "
                "Check the hostname, SSH reachability, and whether the Pi is still online."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or f"exit code {exc.returncode}"
            if attempt <= retries:
                continue
            raise RuntimeError(f"{step} failed: {detail}") from exc


def _raspberry_pi_connect_notes(project_root: Path) -> str:
    return "\n".join(
        [
            "# Remote Access",
            "",
            "The safest remote presentation path for Raspberry Pi is Raspberry Pi Connect.",
            "",
            "## Recommended remote workflow",
            "",
            "1. Keep the IINTS dashboard bound to `127.0.0.1` on the Pi.",
            "2. Use Raspberry Pi Connect screen sharing for the kiosk or desktop.",
            "3. Use Raspberry Pi Connect remote shell or SSH for maintenance commands.",
            "4. Only expose the dashboard API to the network if another machine truly must call it directly.",
            "",
            "## Raspberry Pi Connect checklist",
            "",
            "- `rpi-connect status`",
            "- `rpi-connect shell on`",
            "- `rpi-connect vnc on`",
            "- `loginctl enable-linger`",
            "",
            "If screen sharing is unavailable after reboot, enable Desktop Autologin in Raspberry Pi OS.",
            "",
            "## Remote project root",
            "",
            f"`{project_root}`",
            "",
            "## One-command booth start on the Pi",
            "",
            "```bash",
            "./start_makerfaire_patient.sh",
            "```",
            "",
            "## Optional remote maintenance",
            "",
            "- `iints edge status --project-dir .`",
            "- `iints edge reset --project-dir .`",
            "- `iints edge stop --project-dir .`",
            "- `iints makerfaire watchdog --project-dir .`",
            "",
        ]
    )


def _render_edge_offline_install_script(*, package_spec: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'ROOT="$(cd "$(dirname "$0")" && pwd)"',
            'cd "$ROOT"',
            'python3 -m venv .venv',
            'source .venv/bin/activate',
            'python -m pip install -U pip',
            f'python -m pip install --no-index --find-links "$ROOT/wheelhouse" "{package_spec}"',
            "",
            'echo ""',
            'echo "[IINTS] Offline edge environment is ready."',
            'echo "[IINTS] Project scaffold: $ROOT/edge_project"',
            'echo "[IINTS] Next step: cd $ROOT/edge_project"',
            'echo "[IINTS] Then run: iints makerfaire up --project-dir ."',
            "",
        ]
    )


def _render_edge_offline_install_guide(*, bundle_root_name: str, package_spec: str, board: str) -> str:
    return "\n".join(
        [
            "# Offline Edge Install",
            "",
            "Use this bundle when the venue Wi-Fi is unreliable and you need a USB-stick install path.",
            "",
            "## What is inside",
            "",
            "- `wheelhouse/` with the SDK wheel and dependency wheels",
            f"- `{bundle_root_name}/edge_project/` with a ready-to-run project scaffold",
            "- `install_offline_edge.sh` to create a virtual environment and install the SDK without internet",
            "",
            "## Install on the Raspberry Pi",
            "",
            "```bash",
            "tar -xzf iints_offline.tar.gz",
            f"cd {bundle_root_name}",
            "./install_offline_edge.sh",
            "source .venv/bin/activate",
            "cd edge_project",
            "iints makerfaire up --project-dir .",
            "```",
            "",
            f"Offline package spec: `{package_spec}`",
            f"Board profile baked into the scaffold: `{board}`",
            "",
            "If you are using Raspberry Pi Connect, keep the dashboard on `127.0.0.1` and present it through screen sharing instead of opening the API to the LAN.",
            "",
        ]
    )


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
    uno_bridge_port: str | None = None,
    uno_bridge_baudrate: int = UNO_Q_BRIDGE_BAUDRATE,
    uno_bridge_service_name: str = "iints-uno-q-bridge",
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

    makerfaire_script = root / "start_makerfaire_patient.sh"
    makerfaire_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'cd "$(dirname "$0")"',
                "iints makerfaire up \\",
                "  --project-dir . \\",
                "  --scenario-profile expo_hot_start \\",
                '  "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    makerfaire_script.chmod(makerfaire_script.stat().st_mode | stat.S_IXUSR)

    update_script = write_edge_update_script(root / "update_edge_runtime.sh", profile="edge")
    service_paths = write_service_artifacts(
        config,
        output_path=workspace / f"{service_name}.service",
        service_name=service_name,
        user_name=user_name,
    )
    makerfaire_autostart = write_makerfaire_autostart_artifacts(
        config,
        project_root=root,
        service_path=Path(service_paths["service_file"]),
        service_name=service_name,
        user_name=service_paths["user_name"],
        cli_path=Path(service_paths["python_path"]).with_name("iints"),
    )
    bridge_service_outputs: dict[str, str] | None = None

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
            *(
                [
                    f"- UNO Q bridge service: `./{uno_bridge_service_name}.service`",
                ]
                if uno_bridge_port
                else []
            ),
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
            "- `iints edge long-study --project-dir . --config edge_long_study.yaml`",
            "- `iints edge study-export --project-dir . --input-dir results/long_study --output results/long_study_export.zip`",
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
            "- `iints edge long-study --project-dir . --config edge_long_study.yaml`",
            "- `iints edge study-export --project-dir . --input-dir results/long_study --output results/long_study_export.zip`",
            "",
        ]
        if uno_bridge_port:
            guide_lines.extend(
                [
                    "## Optional UNO Q bridge",
                    "",
                    f"- generated service: `./{uno_bridge_service_name}.service`",
                    f"- fixed serial port: `{uno_bridge_port}`",
                    "- install it through `./install_makerfaire_autostart.sh` after you confirm the board is connected on the Pi",
                    "",
                ]
            )

    setup_guide = root / "EDGE_SETUP.md"
    setup_guide.write_text("\n".join(guide_lines) + "\n", encoding="utf-8")

    makerfaire_guide = root / "MAKERFAIRE_START.md"
    makerfaire_guide.write_text(
        "\n".join(
            [
                "# Maker Faire Startup",
                "",
                "Use this path when the Raspberry Pi is acting as your show-ready virtual patient.",
                "",
                "## One-command startup",
                "",
                "```bash",
                "./start_makerfaire_patient.sh",
                "```",
                "",
                "That script runs:",
                "",
                "```bash",
                "iints makerfaire up --project-dir . --scenario-profile expo_hot_start",
                "```",
                "",
                "## What it does",
                "",
                "- loads the generated edge runtime config",
                "- starts the persistent digital patient if it is not already running",
                "- resets the patient into a booth-safe profile by default",
                "- opens the kiosk in hardened full-screen mode when Chromium is available",
                "- prints the kiosk URL you should show on the Pi display",
                "- tells you which commands to use for status, reset, and stop",
                "",
                "## Daily booth routine",
                "",
                "1. Power on the Pi",
                "2. Open a terminal in this project folder",
                "3. Run `./start_makerfaire_patient.sh`",
                "4. Open the kiosk URL on the Pi screen",
                "5. Use `iints edge reset --project-dir .` between visitor sessions if needed",
                "6. If the booth patient looks dead, run `iints makerfaire watchdog --project-dir .`",
                "",
                "## Useful commands",
                "",
                "- `iints makerfaire up --project-dir .`",
                "- `iints makerfaire autostart --project-dir .`",
                "- `iints makerfaire watchdog --project-dir .`",
                "- `iints edge status --project-dir .`",
                "- `iints edge kiosk --project-dir .`",
                "- `iints edge reset --project-dir .`",
                "- `iints edge stop --project-dir .`",
                "",
                "## Full event checklist",
                "",
                "- `MAKERFAIRE_CHECKLIST.md`",
                "",
                "## If you are also using an Arduino UNO Q",
                "",
                "Run the Linux-side patient first, then start the bridge in a second terminal:",
                "",
                "```bash",
                "iints edge bridge-run --project-dir . --port /dev/ttyACM0",
                "```",
                "",
                "## Autostart option",
                "",
                f"Service file: `{service_paths['service_file']}`",
                f"Autostart guide: `{makerfaire_autostart['guide']}`",
                f"Autostart installer: `{makerfaire_autostart['install_script']}`",
                f"Kiosk desktop entry: `{makerfaire_autostart['desktop_entry']}`",
                f"Watchdog script: `{makerfaire_autostart['watchdog_script']}`",
                f"Watchdog timer: `{makerfaire_autostart['watchdog_timer']}`",
                "",
                "If you want the Pi to come up directly into the digital patient, use the generated autostart installer after you have tested the normal command-line flow.",
                "On Raspberry Pi OS, also enable Desktop Autologin so the kiosk browser can launch after login.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    makerfaire_checklist = root / "MAKERFAIRE_CHECKLIST.md"
    makerfaire_checklist.write_text(
        "\n".join(
            [
                "# Maker Faire Pi Checklist",
                "",
                "Use this checklist on the actual Raspberry Pi you will bring to the event.",
                "",
                "## Day Before The Event",
                "",
                "1. Run `./update_edge_runtime.sh` if you want the latest SDK on the Pi.",
                "2. Run `./start_makerfaire_patient.sh` and confirm the kiosk opens.",
                "3. Run `iints edge reset --project-dir .` and make sure the patient cleanly resets.",
                "4. If you want auto-boot, run `./install_makerfaire_autostart.sh` and reboot once to test it.",
                "5. If you are using the UNO Q, test `iints edge bridge-run --project-dir . --port /dev/ttyACM0` separately.",
                "6. Pack the Pi power supply, HDMI cable, keyboard, and one backup mouse.",
                "",
                "## When You Arrive At The Venue",
                "",
                "1. Power on the Pi and wait for Raspberry Pi OS desktop login.",
                "2. If autostart is enabled, wait for the kiosk to open by itself.",
                "3. If autostart is not enabled, open a terminal and run `./start_makerfaire_patient.sh`.",
                "4. Confirm the kiosk URL is visible and the patient looks alive.",
                "5. Keep one terminal open for booth control commands only.",
                "",
                "## Between Visitors",
                "",
                "1. Run `iints edge reset --project-dir .`",
                "2. Confirm the kiosk returns to the booth-safe state.",
                "3. If you changed the scenario on purpose, go back to `expo_hot_start` before the next visitor.",
                "",
                "## Fast Recovery",
                "",
                "1. Check status: `iints edge status --project-dir .`",
                "2. Run the booth watchdog: `iints makerfaire watchdog --project-dir .`",
                "3. If needed, rerun `./start_makerfaire_patient.sh`",
                "4. If the browser closed, rerun `iints edge kiosk --project-dir .`",
                "5. If the Pi is really stuck, reboot it and let autostart recover the setup.",
                "",
                "## End Of The Day",
                "",
                "1. Stop the runtime cleanly: `iints edge stop --project-dir .`",
                "2. Optional: export a runtime bundle with `iints edge bundle --project-dir . --output edge_bundle.zip`",
                "3. Power the Pi down normally before unplugging it.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    remote_access_guide = root / "EDGE_REMOTE_ACCESS.md"
    remote_access_guide.write_text(_raspberry_pi_connect_notes(root) + "\n", encoding="utf-8")

    long_study_template = root / "edge_long_study.yaml"
    long_study_template.write_text(
        render_edge_long_study_config_template(),
        encoding="utf-8",
    )

    outputs = {
        "root": str(root),
        "algorithm": str(algo_path),
        "workspace": str(workspace),
        "config": str(config.config_path),
        "run_script": str(run_script),
        "kiosk_script": str(kiosk_script),
        "makerfaire_script": str(makerfaire_script),
        "makerfaire_guide": str(makerfaire_guide),
        "makerfaire_kiosk_script": makerfaire_autostart["kiosk_script"],
        "makerfaire_desktop_entry": makerfaire_autostart["desktop_entry"],
        "makerfaire_autostart_script": makerfaire_autostart["install_script"],
        "makerfaire_autostart_guide": makerfaire_autostart["guide"],
        "makerfaire_watchdog_script": makerfaire_autostart["watchdog_script"],
        "makerfaire_watchdog_service": makerfaire_autostart["watchdog_service"],
        "makerfaire_watchdog_timer": makerfaire_autostart["watchdog_timer"],
        "makerfaire_checklist": str(makerfaire_checklist),
        "remote_access_guide": str(remote_access_guide),
        "long_study_template": str(long_study_template),
        "update_script": str(update_script),
        "service_file": service_paths["service_file"],
        "service_notes": service_paths["install_notes"],
        "setup_guide": str(setup_guide),
    }

    if board == "uno_q" or include_uno_bridge:
        bridge = export_uno_q_bridge(root / "uno_q_bridge")
        outputs["uno_q_bridge"] = bridge["output_dir"]
        if uno_bridge_port:
            bridge_service_outputs = write_uno_q_bridge_service_artifact(
                project_root=root,
                output_path=root / f"{uno_bridge_service_name}.service",
                user_name=service_paths["user_name"],
                cli_path=Path(service_paths["python_path"]).with_name("iints"),
                port=uno_bridge_port,
                baudrate=uno_bridge_baudrate,
                workspace_name=workspace_name,
                service_name=uno_bridge_service_name,
                patient_service_name=service_name,
            )
            outputs["uno_bridge_service"] = bridge_service_outputs["service_file"]
            outputs["uno_bridge_service_notes"] = bridge_service_outputs["install_notes"]

    return outputs


def deploy_edge_project(
    *,
    host: str,
    user_name: str | None = None,
    ssh_port: int = 22,
    remote_dir: str = "~/iints_pi_demo",
    local_output_dir: str | Path = "iints_pi_demo",
    board: str = "raspberry_pi",
    workspace_name: str = "patient_runtime",
    scenario_profile: str = "expo_hot_start",
    patient_config: str = "default_patient",
    patient_model_type: str = "auto",
    mode: str = "demo-time",
    speed: float = 60.0,
    api_host: str = "127.0.0.1",
    api_port: int = 8765,
    seed: int | None = None,
    service_name: str = "iints-digital-patient",
    include_uno_bridge: bool = False,
    uno_bridge_port: str | None = None,
    uno_bridge_baudrate: int = UNO_Q_BRIDGE_BAUDRATE,
    uno_bridge_service_name: str = "iints-uno-q-bridge",
    install_autostart: bool = True,
    start_runtime: bool = True,
    enable_connect_linger: bool = True,
    flash_uno_bridge: bool = False,
    uno_fqbn: str | None = None,
    arduino_cli: str = "arduino-cli",
    dry_run: bool = False,
    ssh_timeout_seconds: float = 300.0,
    ssh_retries: int = 1,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if progress_callback is not None:
        progress_callback("Writing local edge scaffold")
    local_root = Path(local_output_dir).expanduser().resolve()
    outputs = export_edge_setup(
        local_root,
        board=board,
        workspace_name=workspace_name,
        scenario_profile=scenario_profile,
        patient_config=patient_config,
        patient_model_type=patient_model_type,
        mode=mode,
        speed=speed,
        api_host=api_host,
        api_port=api_port,
        seed=seed,
        service_name=service_name,
        user_name=user_name,
        include_uno_bridge=include_uno_bridge or board == "uno_q" or uno_bridge_port is not None,
        uno_bridge_port=uno_bridge_port,
        uno_bridge_baudrate=uno_bridge_baudrate,
        uno_bridge_service_name=uno_bridge_service_name,
    )

    destination = _ssh_destination(host, user_name)
    remote_project_dir = remote_dir.rstrip("/")
    remote_dir_expr = _remote_path_expr(remote_project_dir)

    setup_parts = [
        "iints",
        "edge",
        "setup",
        "--output-dir",
        "__REMOTE_DIR__",
        "--board",
        board,
        "--workspace-name",
        workspace_name,
        "--scenario-profile",
        scenario_profile,
        "--patient-config",
        patient_config,
        "--patient-model",
        patient_model_type,
        "--mode",
        mode,
        "--speed",
        f"{speed:g}x",
        "--api-host",
        api_host,
        "--api-port",
        str(api_port),
        "--service-name",
        service_name,
    ]
    if seed is not None:
        setup_parts.extend(["--seed", str(seed)])
    if user_name:
        setup_parts.extend(["--user-name", user_name])
    if uno_bridge_port:
        setup_parts.extend(["--uno-bridge-port", uno_bridge_port])
        setup_parts.extend(["--uno-bridge-service-name", uno_bridge_service_name])
    setup_command = _shell_join(setup_parts).replace("__REMOTE_DIR__", '"$REMOTE_DIR"')

    remote_lines = [
        "set -euo pipefail",
        f"REMOTE_DIR={remote_dir_expr}",
        "python3 -m pip install -U pip",
        'python3 -m pip install -U "iints-sdk-python35[edge,mdmp]"',
        "hash -r || true",
        "mkdir -p \"$REMOTE_DIR\"",
        setup_command,
    ]
    remote_lines = [
        *remote_lines,
        "cd \"$REMOTE_DIR\"",
        "chmod +x ./update_edge_runtime.sh ./start_makerfaire_patient.sh ./install_makerfaire_autostart.sh ./run_makerfaire_watchdog.sh || true",
    ]
    if enable_connect_linger:
        remote_lines.append("loginctl enable-linger \"$USER\" >/dev/null 2>&1 || true")
    if install_autostart:
        remote_lines.append("./install_makerfaire_autostart.sh")
    if flash_uno_bridge:
        if not uno_bridge_port or not uno_fqbn:
            raise ValueError("Remote UNO Q flashing requires both `uno_bridge_port` and `uno_fqbn`.")
        remote_lines.append(
            _shell_join(
                [
                    "iints",
                    "edge",
                    "bridge-flash",
                    "--project-dir",
                    ".",
                    "--port",
                    uno_bridge_port,
                    "--fqbn",
                    uno_fqbn,
                    "--arduino-cli",
                    arduino_cli,
                ]
            )
        )
    if start_runtime:
        remote_lines.append(_shell_join(["iints", "makerfaire", "up", "--project-dir", ".", "--scenario-profile", scenario_profile]))
    remote_lines.append("iints edge status --project-dir .")

    ssh_command = _ssh_command(destination, port=ssh_port, remote_command="\n".join(remote_lines))
    if dry_run:
        remote_stdout = ""
    else:
        if progress_callback is not None:
            progress_callback(f"Provisioning {destination} over SSH")
        remote_stdout = _run_process(
            ssh_command,
            step=f"Remote deploy to {destination}",
            timeout_seconds=ssh_timeout_seconds,
            retries=ssh_retries,
        )
    if progress_callback is not None:
        progress_callback("Preparing remote maintenance commands")

    return {
        "host": host,
        "destination": destination,
        "ssh_port": ssh_port,
        "remote_dir": remote_project_dir,
        "local_output_dir": str(local_root),
        "board": board,
        "scenario_profile": scenario_profile,
        "uno_bridge_enabled": bool(uno_bridge_port),
        "deploy_stdout": remote_stdout,
        "artifacts": outputs,
        "dry_run": dry_run,
        "deploy_command": _shell_join(ssh_command),
        "remote_commands": {
            "status": _shell_join(["ssh", *(["-p", str(ssh_port)] if ssh_port != 22 else []), destination, "bash", "-lc", f"REMOTE_DIR={remote_dir_expr}\ncd \"$REMOTE_DIR\" && iints edge status --project-dir ."]),
            "reset": _shell_join(["ssh", *(["-p", str(ssh_port)] if ssh_port != 22 else []), destination, "bash", "-lc", f"REMOTE_DIR={remote_dir_expr}\ncd \"$REMOTE_DIR\" && iints edge reset --project-dir ."]),
            "stop": _shell_join(["ssh", *(["-p", str(ssh_port)] if ssh_port != 22 else []), destination, "bash", "-lc", f"REMOTE_DIR={remote_dir_expr}\ncd \"$REMOTE_DIR\" && iints edge stop --project-dir ."]),
        },
    }


def build_edge_offline_bundle(
    output_path: str | Path,
    *,
    board: str = "raspberry_pi",
    workspace_name: str = "patient_runtime",
    scenario_profile: str = "expo_hot_start",
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
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    bundle_output = Path(output_path).expanduser().resolve()
    bundle_output.parent.mkdir(parents=True, exist_ok=True)
    bundle_root_name = bundle_output.stem.replace(".tar", "") or "iints_offline"
    package_spec = f'iints-sdk-python35[edge,mdmp]=={_sdk_version()}'

    with tempfile.TemporaryDirectory(prefix="iints-edge-offline-") as temp_dir:
        staging_root = Path(temp_dir) / bundle_root_name
        wheelhouse = staging_root / "wheelhouse"
        project_root = staging_root / "edge_project"
        wheelhouse.mkdir(parents=True, exist_ok=True)
        if progress_callback is not None:
            progress_callback("Scaffolding offline edge project")
        export_edge_setup(
            project_root,
            board=board,
            workspace_name=workspace_name,
            scenario_profile=scenario_profile,
            patient_config=patient_config,
            patient_model_type=patient_model_type,
            mode=mode,
            speed=speed,
            api_host=api_host,
            api_port=api_port,
            seed=seed,
            service_name=service_name,
            user_name=user_name,
            include_uno_bridge=include_uno_bridge or board == "uno_q",
        )
        if progress_callback is not None:
            progress_callback("Building offline wheelhouse")
        _run_process(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--wheel-dir",
                str(wheelhouse),
                ".[edge,mdmp]",
            ],
            step="Build offline wheelhouse",
            timeout_seconds=900.0,
            cwd=_repo_root(),
        )
        wheel_files = sorted(wheelhouse.glob("*.whl"))
        if not wheel_files:
            raise RuntimeError(
                "Offline wheelhouse build did not produce any wheels. "
                "Run the command on a machine with internet access once so pip can resolve the dependencies."
            )

        install_script = staging_root / "install_offline_edge.sh"
        install_script.write_text(
            _render_edge_offline_install_script(package_spec=package_spec),
            encoding="utf-8",
        )
        install_script.chmod(install_script.stat().st_mode | stat.S_IXUSR)

        install_guide = staging_root / "OFFLINE_INSTALL.md"
        install_guide.write_text(
            _render_edge_offline_install_guide(
                bundle_root_name=bundle_root_name,
                package_spec=package_spec,
                board=board,
            )
            + "\n",
            encoding="utf-8",
        )

        if progress_callback is not None:
            progress_callback("Packing offline bundle")
        with tarfile.open(bundle_output, "w:gz") as archive:
            archive.add(staging_root, arcname=bundle_root_name)

    return {
        "archive": str(bundle_output),
        "bundle_root_name": bundle_root_name,
        "install_script": f"{bundle_root_name}/install_offline_edge.sh",
        "install_guide": f"{bundle_root_name}/OFFLINE_INSTALL.md",
        "package_spec": package_spec,
    }


def run_remote_edge_command(
    *,
    host: str,
    user_name: str | None = None,
    ssh_port: int = 22,
    remote_dir: str = "~/iints_pi_demo",
    action: str = "status",
    scenario_profile: str | None = None,
    seed: int | None = None,
    timeout_seconds: float = 60.0,
    retries: int = 0,
) -> dict[str, str]:
    destination = _ssh_destination(host, user_name)
    remote_dir_expr = _remote_path_expr(remote_dir)
    normalized = action.strip().lower()
    if normalized == "status":
        command = 'cd "$REMOTE_DIR" && iints edge status --project-dir .'
    elif normalized == "stop":
        command = 'cd "$REMOTE_DIR" && iints edge stop --project-dir .'
    elif normalized == "reset":
        reset_parts = ["iints", "edge", "reset", "--project-dir", "."]
        if scenario_profile:
            reset_parts.extend(["--scenario-profile", scenario_profile])
        if seed is not None:
            reset_parts.extend(["--seed", str(seed)])
        command = f'cd "$REMOTE_DIR" && {_shell_join(reset_parts)}'
    else:
        raise ValueError(f"Unsupported remote edge action: {action}")

    remote_script = "\n".join(
        [
            "set -euo pipefail",
            f"REMOTE_DIR={remote_dir_expr}",
            command,
        ]
    )
    stdout = _run_process(
        _ssh_command(destination, port=ssh_port, remote_command=remote_script),
        step=f"Remote edge {normalized} on {destination}",
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    return {
        "destination": destination,
        "action": normalized,
        "stdout": stdout,
    }
