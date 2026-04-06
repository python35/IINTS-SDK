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
