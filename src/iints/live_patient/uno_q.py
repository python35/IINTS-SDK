from __future__ import annotations

import shutil
import subprocess
import time
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping


UNO_Q_BRIDGE_BAUDRATE = 115200
UNO_Q_BRIDGE_STATES = ("OK", "OVERRIDE", "CRITICAL")
UNO_Q_BRIDGE_READY_BANNER = "IINTS UNO Q supervisor bridge ready"
UNO_Q_BRIDGE_BOOT_DELAY_SECONDS = 1.2
UNO_Q_BRIDGE_READ_POLL_SECONDS = 0.1


def _require_pyserial():
    try:
        import serial  # type: ignore
        from serial.tools import list_ports  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised through callers
        raise ImportError(
            "UNO Q bridge commands require the optional serial stack. "
            "Install `pyserial` or use the `iints-sdk-python35[edge,mdmp]` profile."
        ) from exc
    return serial, list_ports


def _normalize_bridge_state(state: str) -> str:
    normalized = str(state).strip().upper()
    if normalized not in UNO_Q_BRIDGE_STATES:
        raise ValueError(
            f"Unsupported UNO Q bridge state '{state}'. Use one of: "
            + ", ".join(UNO_Q_BRIDGE_STATES)
        )
    return normalized


def list_uno_q_serial_ports() -> list[str]:
    try:
        _, list_ports = _require_pyserial()
    except ImportError:
        return []
    return sorted(port.device for port in list_ports.comports())


def resolve_uno_q_port(port: str | None) -> str:
    if port and str(port).strip().lower() != "auto":
        return str(port).strip()

    ports = list_uno_q_serial_ports()
    if len(ports) == 1:
        return ports[0]
    if not ports:
        raise ValueError("No serial ports detected. Pass --port explicitly.")
    raise ValueError(
        "Multiple serial ports detected. Pass --port explicitly. "
        f"Detected: {', '.join(ports)}"
    )


def uno_q_bridge_environment_report(*, arduino_cli: str = "arduino-cli") -> dict[str, Any]:
    serial_available = True
    serial_error = None
    try:
        _require_pyserial()
    except ImportError as exc:
        serial_available = False
        serial_error = str(exc)

    ports = list_uno_q_serial_ports() if serial_available else []
    return {
        "pyserial_available": serial_available,
        "pyserial_error": serial_error,
        "serial_ports": ports,
        "arduino_cli_path": shutil.which(arduino_cli),
    }


def export_uno_q_bridge(output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    template_root = files("iints.templates").joinpath("uno_q")
    sketch_path = target / "iints_supervisor_bridge.ino"
    readme_path = target / "README.md"
    protocol_path = target / "bridge_protocol.txt"

    sketch_path.write_text(template_root.joinpath("iints_supervisor_bridge.ino").read_text(encoding="utf-8"), encoding="utf-8")
    readme_path.write_text(template_root.joinpath("README.md").read_text(encoding="utf-8"), encoding="utf-8")
    protocol_path.write_text(
        "\n".join(
            [
                "IINTS UNO Q serial bridge protocol",
                f"Baud rate: {UNO_Q_BRIDGE_BAUDRATE}",
                f"Startup banner: {UNO_Q_BRIDGE_READY_BANNER}",
                "Messages:",
                "  OK",
                "  OVERRIDE",
                "  CRITICAL",
                "Acknowledgements:",
                "  STATE=OK",
                "  STATE=OVERRIDE",
                "  STATE=CRITICAL",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(target),
        "sketch": str(sketch_path),
        "readme": str(readme_path),
        "protocol": str(protocol_path),
    }


def bridge_state_from_runtime_status(status: Mapping[str, Any] | None) -> str:
    if not status:
        return "OVERRIDE"

    daemon_status = str(status.get("daemon_status", "") or "").strip().lower()
    if daemon_status not in {"running", "paused", "starting"}:
        return "OVERRIDE"

    glucose = status.get("last_glucose_mgdl")
    try:
        glucose_value = float(glucose) if glucose is not None else None
    except (TypeError, ValueError):
        glucose_value = None

    safety_reason = str(status.get("last_safety_reason", "") or "").strip()
    safety_reason_upper = safety_reason.upper()

    if "CRITICAL" in safety_reason_upper:
        return "CRITICAL"
    if glucose_value is not None and glucose_value <= 70.0:
        return "CRITICAL"
    if bool(status.get("paused")):
        return "OVERRIDE"
    if safety_reason:
        return "OVERRIDE"
    return "OK"


def _decode_serial_line(raw: bytes) -> str | None:
    text = raw.decode("utf-8", errors="replace").strip()
    return text or None


def _response_matches_state(response: str, state: str) -> bool:
    normalized_response = str(response).strip().upper()
    normalized_state = _normalize_bridge_state(state)
    return normalized_response in {
        normalized_state,
        f"STATE={normalized_state}",
        f"ACK={normalized_state}",
        f"ACK: {normalized_state}",
    }


def _pick_bridge_response(lines: list[str], state: str) -> str | None:
    if not lines:
        return None
    for line in lines:
        if _response_matches_state(line, state):
            return line
    for line in lines:
        if line.upper().startswith("STATE=") or line.upper().startswith("ACK"):
            return line
    return lines[-1]


def _read_serial_lines(connection: Any, *, duration_seconds: float, stop_on_match: str | None = None) -> list[str]:
    deadline = time.monotonic() + max(duration_seconds, UNO_Q_BRIDGE_READ_POLL_SECONDS)
    lines: list[str] = []
    while time.monotonic() < deadline:
        raw = connection.readline()
        if raw:
            line = _decode_serial_line(raw)
            if line:
                lines.append(line)
                if stop_on_match is not None and _response_matches_state(line, stop_on_match):
                    break
            continue
        time.sleep(0.02)
    return lines


@contextmanager
def _uno_q_serial_connection(
    port: str,
    *,
    baudrate: int,
    timeout_seconds: float,
    boot_delay_seconds: float = UNO_Q_BRIDGE_BOOT_DELAY_SECONDS,
):
    serial, _ = _require_pyserial()
    with serial.Serial(  # type: ignore[attr-defined]
        port,
        baudrate=baudrate,
        timeout=min(timeout_seconds, UNO_Q_BRIDGE_READ_POLL_SECONDS),
        write_timeout=timeout_seconds,
    ) as connection:
        try:
            connection.reset_input_buffer()
            connection.reset_output_buffer()
        except (AttributeError, OSError):
            pass
        if boot_delay_seconds > 0:
            time.sleep(boot_delay_seconds)
        startup_lines = _read_serial_lines(connection, duration_seconds=0.35)
        yield connection, startup_lines


def _send_state_over_connection(
    connection: Any,
    state: str,
    *,
    timeout_seconds: float,
    expect_response: bool,
) -> dict[str, Any]:
    normalized_state = _normalize_bridge_state(state)
    try:
        connection.reset_input_buffer()
    except (AttributeError, OSError):
        pass
    payload = f"{normalized_state}\n".encode("utf-8")
    connection.write(payload)
    connection.flush()

    response_lines: list[str] = []
    response = None
    if expect_response:
        response_lines = _read_serial_lines(
            connection,
            duration_seconds=timeout_seconds,
            stop_on_match=normalized_state,
        )
        response = _pick_bridge_response(response_lines, normalized_state)

    return {
        "state": normalized_state,
        "response": response,
        "response_lines": response_lines,
    }


def send_uno_q_bridge_state(
    port: str | None,
    state: str,
    *,
    baudrate: int = UNO_Q_BRIDGE_BAUDRATE,
    timeout_seconds: float = 1.5,
    expect_response: bool = True,
) -> dict[str, Any]:
    resolved_port = resolve_uno_q_port(port)
    with _uno_q_serial_connection(
        resolved_port,
        baudrate=baudrate,
        timeout_seconds=timeout_seconds,
    ) as (connection, startup_lines):
        send_result = _send_state_over_connection(
            connection,
            state,
            timeout_seconds=timeout_seconds,
            expect_response=expect_response,
        )

    return {
        "port": resolved_port,
        "state": send_result["state"],
        "baudrate": baudrate,
        "response": send_result["response"],
        "startup_lines": startup_lines,
        "response_lines": send_result["response_lines"],
    }


def run_uno_q_bridge_test(
    port: str | None,
    *,
    baudrate: int = UNO_Q_BRIDGE_BAUDRATE,
    delay_seconds: float = 0.75,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    resolved_port = resolve_uno_q_port(port)
    timeout_seconds = max(delay_seconds, 0.5)
    with _uno_q_serial_connection(
        resolved_port,
        baudrate=baudrate,
        timeout_seconds=timeout_seconds,
    ) as (connection, startup_lines):
        for index, state in enumerate(UNO_Q_BRIDGE_STATES):
            send_result = _send_state_over_connection(
                connection,
                state,
                timeout_seconds=timeout_seconds,
                expect_response=True,
            )
            results.append(
                {
                    "port": resolved_port,
                    "state": send_result["state"],
                    "baudrate": baudrate,
                    "response": send_result["response"],
                    "response_lines": send_result["response_lines"],
                    "startup_lines": startup_lines if index == 0 else [],
                }
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)
    return results


def run_uno_q_bridge_forwarder(
    workspace: str | Path,
    port: str | None,
    *,
    baudrate: int = UNO_Q_BRIDGE_BAUDRATE,
    poll_interval: float = 1.0,
    once: bool = False,
    max_cycles: int | None = None,
) -> dict[str, Any]:
    from .runtime import load_runtime_status

    workspace_path = Path(workspace).expanduser().resolve()
    resolved_port = resolve_uno_q_port(port)
    messages_sent = 0
    last_state = None
    last_payload: dict[str, Any] | None = None
    cycles = 0

    with _uno_q_serial_connection(
        resolved_port,
        baudrate=baudrate,
        timeout_seconds=max(poll_interval, 0.5),
    ) as (connection, startup_lines):
        while True:
            status = load_runtime_status(workspace_path)
            state = bridge_state_from_runtime_status(status)
            if state != last_state:
                send_result = _send_state_over_connection(
                    connection,
                    state,
                    timeout_seconds=max(poll_interval, 0.5),
                    expect_response=False,
                )
                last_payload = {
                    "port": resolved_port,
                    "state": send_result["state"],
                    "baudrate": baudrate,
                    "response": send_result["response"],
                    "startup_lines": startup_lines if messages_sent == 0 else [],
                    "response_lines": send_result["response_lines"],
                }
                last_state = state
                messages_sent += 1

            cycles += 1
            if once or (max_cycles is not None and cycles >= max_cycles):
                return {
                    "workspace": str(workspace_path),
                    "port": resolved_port,
                    "state": last_state,
                    "messages_sent": messages_sent,
                    "last_payload": last_payload,
                }

            time.sleep(max(poll_interval, 0.1))


def flash_uno_q_bridge(
    sketch_dir: str | Path,
    *,
    port: str,
    fqbn: str,
    arduino_cli: str = "arduino-cli",
) -> dict[str, Any]:
    cli_path = shutil.which(arduino_cli)
    if cli_path is None:
        raise FileNotFoundError(
            f"`{arduino_cli}` was not found on PATH. Install Arduino CLI or pass --arduino-cli."
        )

    sketch_root = Path(sketch_dir).expanduser().resolve()
    if sketch_root.is_file():
        sketch_root = sketch_root.parent
    sketch_file = sketch_root / "iints_supervisor_bridge.ino"
    if not sketch_file.is_file():
        raise FileNotFoundError(f"UNO Q bridge sketch not found in {sketch_root}")

    resolved_port = resolve_uno_q_port(port)

    compile_cmd = [cli_path, "compile", "--fqbn", fqbn, str(sketch_root)]
    upload_cmd = [cli_path, "upload", "--fqbn", fqbn, "-p", resolved_port, str(sketch_root)]

    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
    upload_result = subprocess.run(upload_cmd, capture_output=True, text=True, check=True)

    return {
        "arduino_cli": cli_path,
        "sketch_dir": str(sketch_root),
        "port": resolved_port,
        "fqbn": fqbn,
        "compile_stdout": compile_result.stdout,
        "compile_stderr": compile_result.stderr,
        "upload_stdout": upload_result.stdout,
        "upload_stderr": upload_result.stderr,
    }
