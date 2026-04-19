from __future__ import annotations

import os
import sys
from pathlib import Path

from .runtime import PatientRuntimeConfig


def service_file_text(config: PatientRuntimeConfig, *, service_name: str, user_name: str, python_path: str) -> str:
    return f"""[Unit]
Description=IINTS Digital Patient ({service_name})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user_name}
WorkingDirectory={config.workspace_path.parent}
Environment=PYTHONUNBUFFERED=1
ExecStart={python_path} -m iints.live_patient.daemon --config {config.config_path}
Restart=always
RestartSec=5
KillSignal=SIGINT

[Install]
WantedBy=multi-user.target
"""


def service_instructions_text(service_path: Path, service_name: str) -> str:
    return "\n".join(
        [
            "Copy this service onto the device systemd path:",
            f"  sudo cp {service_path} /etc/systemd/system/{service_name}.service",
            "Reload and enable it:",
            "  sudo systemctl daemon-reload",
            f"  sudo systemctl enable {service_name}.service",
            f"  sudo systemctl start {service_name}.service",
            "Check status:",
            f"  systemctl status {service_name}.service",
        ]
    )


def makerfaire_kiosk_script_text(config: PatientRuntimeConfig) -> str:
    kiosk_url = f"http://{config.api_host}:{config.api_port}/kiosk"
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'cd "$(dirname "$0")"',
            f'URL="{kiosk_url}"',
            'PROFILE_DIR="$PWD/.makerfaire-kiosk-profile"',
            'LOG_FILE="${TMPDIR:-/tmp}/iints-makerfaire-kiosk.log"',
            "",
            "# Give the daemon a moment to come up after login.",
            "sleep 10",
            "",
            "mkdir -p \"$PROFILE_DIR\"",
            "",
            "# If curl is available, wait until the kiosk endpoint answers before opening the browser.",
            "if command -v curl >/dev/null 2>&1; then",
            "  for _ in $(seq 1 20); do",
            '    if curl -fsS "$URL" >/dev/null 2>&1; then',
            "      break",
            "    fi",
            "    sleep 2",
            "  done",
            "fi",
            "",
            "# Disable screen blanking when a desktop session is available.",
            'if [ -n "${DISPLAY:-}" ] && command -v xset >/dev/null 2>&1; then',
            "  xset s off || true",
            "  xset -dpms || true",
            "  xset s noblank || true",
            "fi",
            "",
            "# Hide the mouse cursor after a short idle period if unclutter is installed.",
            'if [ -n "${DISPLAY:-}" ] && command -v unclutter >/dev/null 2>&1; then',
            "  pkill -f \"unclutter -idle 1\" >/dev/null 2>&1 || true",
            "  nohup unclutter -idle 1 >/dev/null 2>&1 &",
            "fi",
            "",
            "# Prefer Chromium in real kiosk mode for the Maker Faire screen.",
            "for BROWSER in chromium-browser chromium google-chrome google-chrome-stable; do",
            "  if command -v \"$BROWSER\" >/dev/null 2>&1; then",
            "    pkill -f \"$PROFILE_DIR\" >/dev/null 2>&1 || true",
            "    sleep 1",
            "    nohup \"$BROWSER\" \\",
            "      --kiosk \\",
            "      --start-fullscreen \\",
            "      --start-maximized \\",
            "      --no-first-run \\",
            "      --no-default-browser-check \\",
            "      --disable-session-crashed-bubble \\",
            "      --disable-infobars \\",
            "      --overscroll-history-navigation=0 \\",
            "      --disable-features=Translate,MediaRouter,OptimizationHints,AutofillServerCommunication \\",
            "      --password-store=basic \\",
            "      --user-data-dir=\"$PROFILE_DIR\" \\",
            "      \"$URL\" >\"$LOG_FILE\" 2>&1 &",
            "    exit 0",
            "  fi",
            "done",
            "",
            "# Firefox can also do kiosk mode if Chromium is missing.",
            "if command -v firefox >/dev/null 2>&1; then",
            "  nohup firefox --kiosk \"$URL\" >\"$LOG_FILE\" 2>&1 &",
            "  exit 0",
            "fi",
            "",
            'if command -v xdg-open >/dev/null 2>&1; then',
            '  xdg-open "$URL"',
            'elif command -v open >/dev/null 2>&1; then',
            '  open "$URL"',
            "else",
            '  echo "Open $URL in your browser."',
            "fi",
            "",
        ]
    )


def makerfaire_desktop_entry_text(project_root: Path) -> str:
    return "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            "Version=1.0",
            "Name=IINTS Maker Faire Kiosk",
            "Comment=Open the IINTS kiosk after Raspberry Pi desktop login",
            f"Path={project_root}",
            "Exec=/bin/bash -lc ./open_makerfaire_kiosk.sh",
            "Terminal=false",
            "X-GNOME-Autostart-enabled=true",
            "X-GNOME-Autostart-Delay=15",
            "StartupNotify=false",
            "",
        ]
    )


def _cli_path_from_python(python_path: str | Path) -> str:
    candidate = Path(python_path).expanduser().resolve().with_name("iints")
    if candidate.is_file():
        return str(candidate)
    return "iints"


def makerfaire_watchdog_name(service_name: str) -> str:
    return f"{service_name}-watchdog"


def makerfaire_watchdog_script_text(
    *,
    cli_path: str,
    scenario_profile: str,
    seed: int | None,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        f'CLI="{cli_path}"',
        "",
        '"$CLI" makerfaire watchdog \\',
        "  --project-dir . \\",
        f"  --scenario-profile {scenario_profile} \\",
    ]
    if seed is not None:
        lines.append(f"  --seed {seed} \\")
    lines.extend(
        [
            "  --quiet",
            "",
        ]
    )
    return "\n".join(lines)


def makerfaire_watchdog_service_text(
    *,
    project_root: Path,
    watchdog_name: str,
    user_name: str,
) -> str:
    return "\n".join(
        [
            "[Unit]",
            f"Description=IINTS Maker Faire Watchdog ({watchdog_name})",
            "After=network-online.target",
            "Wants=network-online.target",
            "",
            "[Service]",
            "Type=oneshot",
            f"User={user_name}",
            f"WorkingDirectory={project_root}",
            "ExecStart=/bin/bash -lc ./run_makerfaire_watchdog.sh",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ]
    )


def makerfaire_watchdog_timer_text(*, watchdog_name: str, interval_seconds: int = 30) -> str:
    return "\n".join(
        [
            "[Unit]",
            f"Description=IINTS Maker Faire Watchdog Timer ({watchdog_name})",
            "",
            "[Timer]",
            "OnBootSec=45",
            f"OnUnitActiveSec={interval_seconds}",
            f"Unit={watchdog_name}.service",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        ]
    )


def makerfaire_install_script_text(*, service_name: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'ROOT="$(cd "$(dirname "$0")" && pwd)"',
            f'SERVICE_NAME="{service_name}"',
            'WATCHDOG_NAME="${SERVICE_NAME}-watchdog"',
            'SERVICE_SRC="$ROOT/patient_runtime/${SERVICE_NAME}.service"',
            'WATCHDOG_SERVICE_SRC="$ROOT/${WATCHDOG_NAME}.service"',
            'WATCHDOG_TIMER_SRC="$ROOT/${WATCHDOG_NAME}.timer"',
            'DESKTOP_SRC="$ROOT/iints-makerfaire-kiosk.desktop"',
            'AUTOSTART_DIR="$HOME/.config/autostart"',
            "",
            'if [ ! -f "$SERVICE_SRC" ]; then',
            '  echo "Missing service file: $SERVICE_SRC" >&2',
            "  exit 1",
            "fi",
            'if [ ! -f "$WATCHDOG_SERVICE_SRC" ]; then',
            '  echo "Missing watchdog service: $WATCHDOG_SERVICE_SRC" >&2',
            "  exit 1",
            "fi",
            'if [ ! -f "$WATCHDOG_TIMER_SRC" ]; then',
            '  echo "Missing watchdog timer: $WATCHDOG_TIMER_SRC" >&2',
            "  exit 1",
            "fi",
            'if [ ! -f "$DESKTOP_SRC" ]; then',
            '  echo "Missing desktop entry: $DESKTOP_SRC" >&2',
            "  exit 1",
            "fi",
            "",
            'echo "Installing systemd service..."',
            'sudo cp "$SERVICE_SRC" "/etc/systemd/system/${SERVICE_NAME}.service"',
            'sudo cp "$WATCHDOG_SERVICE_SRC" "/etc/systemd/system/${WATCHDOG_NAME}.service"',
            'sudo cp "$WATCHDOG_TIMER_SRC" "/etc/systemd/system/${WATCHDOG_NAME}.timer"',
            "sudo systemctl daemon-reload",
            'sudo systemctl enable "${SERVICE_NAME}.service"',
            'sudo systemctl restart "${SERVICE_NAME}.service"',
            'sudo systemctl enable "${WATCHDOG_NAME}.timer"',
            'sudo systemctl restart "${WATCHDOG_NAME}.timer"',
            "",
            'echo "Installing desktop autostart entry..."',
            'mkdir -p "$AUTOSTART_DIR"',
            'cp "$DESKTOP_SRC" "$AUTOSTART_DIR/iints-makerfaire-kiosk.desktop"',
            "",
            'echo "Done. Reboot the Pi and confirm the kiosk opens automatically."',
            'echo "The watchdog timer will now re-check the booth runtime in the background."',
            'echo "If the browser does not open after reboot, enable Desktop Autologin in Raspberry Pi Configuration and try again."',
            "",
        ]
    )


def makerfaire_autostart_instructions_text(
    *,
    project_root: Path,
    service_path: Path,
    desktop_entry_path: Path,
    kiosk_script_path: Path,
    install_script_path: Path,
    watchdog_script_path: Path,
    watchdog_service_path: Path,
    watchdog_timer_path: Path,
    service_name: str,
) -> str:
    watchdog_name = makerfaire_watchdog_name(service_name)
    return "\n".join(
        [
            "# Maker Faire Autostart",
            "",
            "Use this when you want the Raspberry Pi to boot straight into the digital patient booth setup.",
            "",
            "## What gets installed",
            "",
            f"- systemd service: `{service_path}`",
            f"- kiosk opener script: `{kiosk_script_path}`",
            f"- desktop autostart entry: `{desktop_entry_path}`",
            f"- watchdog script: `{watchdog_script_path}`",
            f"- watchdog service: `{watchdog_service_path}`",
            f"- watchdog timer: `{watchdog_timer_path}`",
            "",
            "## Recommended path",
            "",
            "1. Test the normal booth command first:",
            "",
            "```bash",
            "./start_makerfaire_patient.sh",
            "```",
            "",
            "2. Install the autostart files:",
            "",
            "```bash",
            "./install_makerfaire_autostart.sh",
            "```",
            "",
            "3. Make sure Raspberry Pi OS is set to desktop auto-login.",
            "",
            "4. Reboot and confirm:",
            "",
            f"- `{service_name}.service` is active",
            f"- `{watchdog_name}.timer` is active",
            "- the kiosk browser opens by itself after login",
            "- `iints edge reset --project-dir .` still works between visitors",
            "",
            "## Manual install commands",
            "",
            "```bash",
            f"sudo cp {service_path} /etc/systemd/system/{service_name}.service",
            f"sudo cp {watchdog_service_path} /etc/systemd/system/{watchdog_name}.service",
            f"sudo cp {watchdog_timer_path} /etc/systemd/system/{watchdog_name}.timer",
            "sudo systemctl daemon-reload",
            f"sudo systemctl enable {service_name}.service",
            f"sudo systemctl restart {service_name}.service",
            f"sudo systemctl enable {watchdog_name}.timer",
            f"sudo systemctl restart {watchdog_name}.timer",
            "mkdir -p ~/.config/autostart",
            f"cp {desktop_entry_path} ~/.config/autostart/iints-makerfaire-kiosk.desktop",
            "```",
            "",
            "## Project root",
            "",
            f"`{project_root}`",
            "",
            "## Helper script",
            "",
            f"`{install_script_path}`",
            "",
        ]
    )


def write_service_artifacts(
    config: PatientRuntimeConfig,
    *,
    output_path: Path,
    service_name: str = "iints-digital-patient",
    user_name: str | None = None,
    python_path: Path | None = None,
) -> dict[str, str]:
    target_path = output_path.expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    chosen_user = user_name or os.getenv("USER") or "pi"
    chosen_python = str((python_path or Path(sys.executable)).expanduser().resolve())

    target_path.write_text(
        service_file_text(config, service_name=service_name, user_name=chosen_user, python_path=chosen_python),
        encoding="utf-8",
    )
    instructions_path = target_path.with_suffix(".INSTALL.txt")
    instructions_path.write_text(service_instructions_text(target_path, service_name), encoding="utf-8")
    return {
        "service_file": str(target_path),
        "install_notes": str(instructions_path),
        "service_name": service_name,
        "user_name": chosen_user,
        "python_path": chosen_python,
    }


def write_makerfaire_autostart_artifacts(
    config: PatientRuntimeConfig,
    *,
    project_root: Path,
    service_path: Path,
    service_name: str = "iints-digital-patient",
    user_name: str | None = None,
    cli_path: str | Path | None = None,
) -> dict[str, str]:
    root = project_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved_user = user_name or os.getenv("USER") or "pi"
    resolved_cli = _cli_path_from_python(cli_path or Path(sys.executable))
    watchdog_name = makerfaire_watchdog_name(service_name)

    kiosk_script_path = root / "open_makerfaire_kiosk.sh"
    kiosk_script_path.write_text(makerfaire_kiosk_script_text(config), encoding="utf-8")
    kiosk_script_path.chmod(kiosk_script_path.stat().st_mode | 0o100)

    desktop_entry_path = root / "iints-makerfaire-kiosk.desktop"
    desktop_entry_path.write_text(makerfaire_desktop_entry_text(root), encoding="utf-8")

    install_script_path = root / "install_makerfaire_autostart.sh"
    install_script_path.write_text(
        makerfaire_install_script_text(service_name=service_name),
        encoding="utf-8",
    )
    install_script_path.chmod(install_script_path.stat().st_mode | 0o100)

    watchdog_script_path = root / "run_makerfaire_watchdog.sh"
    watchdog_script_path.write_text(
        makerfaire_watchdog_script_text(
            cli_path=resolved_cli,
            scenario_profile=config.scenario_profile,
            seed=config.seed,
        ),
        encoding="utf-8",
    )
    watchdog_script_path.chmod(watchdog_script_path.stat().st_mode | 0o100)

    watchdog_service_path = root / f"{watchdog_name}.service"
    watchdog_service_path.write_text(
        makerfaire_watchdog_service_text(
            project_root=root,
            watchdog_name=watchdog_name,
            user_name=resolved_user,
        ),
        encoding="utf-8",
    )

    watchdog_timer_path = root / f"{watchdog_name}.timer"
    watchdog_timer_path.write_text(
        makerfaire_watchdog_timer_text(watchdog_name=watchdog_name),
        encoding="utf-8",
    )

    guide_path = root / "MAKERFAIRE_AUTOSTART.md"
    guide_path.write_text(
        makerfaire_autostart_instructions_text(
            project_root=root,
            service_path=service_path.expanduser().resolve(),
            desktop_entry_path=desktop_entry_path,
            kiosk_script_path=kiosk_script_path,
            install_script_path=install_script_path,
            watchdog_script_path=watchdog_script_path,
            watchdog_service_path=watchdog_service_path,
            watchdog_timer_path=watchdog_timer_path,
            service_name=service_name,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "kiosk_script": str(kiosk_script_path),
        "desktop_entry": str(desktop_entry_path),
        "install_script": str(install_script_path),
        "watchdog_script": str(watchdog_script_path),
        "watchdog_service": str(watchdog_service_path),
        "watchdog_timer": str(watchdog_timer_path),
        "guide": str(guide_path),
    }
