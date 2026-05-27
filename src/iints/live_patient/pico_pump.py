from __future__ import annotations

import hashlib
import json
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any


PICO_PUMP_BAUDRATE = 115200
PICO_PUMP_READY_BANNER = "IINTS Pico Pump Bench firmware ready"
PICO_PUMP_CONFIRMATION = "I understand this is bench-only and not for human use"
PICO_PUMP_DEVICE_ID = "iints-pico-pump-bench"


DEFAULT_PICO_PUMP_SAFETY_CONTRACT: dict[str, Any] = {
    "schema_version": "1.0",
    "target": "raspberry_pi_pico_bench",
    "scope": "bench_only_not_for_human_or_animal_use",
    "hardware_actuation_enabled": False,
    "requires_external_supervisor": True,
    "requires_manual_bench_arm": True,
    "max_command_units": 0.0,
    "max_units_per_hour": 0.0,
    "allowed_serial_commands": ["PING", "STATUS", "LOCKOUT"],
    "blocked_serial_commands": ["DOSE", "BOLUS", "BASAL", "PRIME"],
    "notes": [
        "This contract is intentionally locked to non-actuating bench firmware.",
        "Use LEDs, a dummy load, or a disconnected mechanism only.",
        "Do not connect to a reservoir, infusion set, person, or animal.",
    ],
}


BENCH_ALGORITHM_TEMPLATE = '''from iints import AlgorithmInput, InsulinAlgorithm


class PicoBenchAlgorithm(InsulinAlgorithm):
    """Conservative starter algorithm for bench-only pump experiments.

    The SDK can simulate this algorithm, package it, and send it to the Pico
    bench firmware as source. The generated Pico firmware does not actuate a
    pump motor; it only reports status over USB serial.
    """

    def predict_insulin(self, data: AlgorithmInput):
        glucose = float(data.current_glucose)
        insulin_units = 0.0
        reason = "bench lockout: no insulin delivery"

        if glucose > 180.0:
            insulin_units = min((glucose - 140.0) / 80.0, 0.25)
            reason = "virtual correction proposal capped for simulation only"
        if glucose < 90.0:
            insulin_units = 0.0
            reason = "hypoglycemia guard: hold insulin"

        return {
            "total_insulin_delivered": insulin_units,
            "bolus_insulin": insulin_units,
            "basal_insulin": 0.0,
            "primary_reason": reason,
        }
'''


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree_contents(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def export_pico_pump_firmware(output_dir: str | Path) -> dict[str, str]:
    """Export non-actuating Pico bench firmware templates."""

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    template_root = files("iints.templates").joinpath("pico_pump")
    _copy_tree_contents(Path(str(template_root)), target)
    return {
        "output_dir": str(target),
        "code": str(target / "code.py"),
        "readme": str(target / "README.md"),
        "protocol": str(target / "serial_protocol.txt"),
    }


def create_pico_pump_lab(output_dir: str | Path, *, algorithm_path: str | Path | None = None) -> dict[str, str]:
    """Create a complete bench-only Pico pump research workspace."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    algorithms_dir = root / "algorithms"
    firmware_dir = root / "firmware" / "pico_pump_bench"
    bundles_dir = root / "bundles"
    evidence_dir = root / "evidence"
    scripts_dir = root / "scripts"
    for directory in (algorithms_dir, firmware_dir, bundles_dir, evidence_dir, scripts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if algorithm_path is None:
        algorithm_target = algorithms_dir / "pico_bench_algorithm.py"
        algorithm_target.write_text(BENCH_ALGORITHM_TEMPLATE, encoding="utf-8")
    else:
        source_algorithm = Path(algorithm_path).expanduser().resolve()
        if not source_algorithm.is_file():
            raise FileNotFoundError(f"Algorithm file not found: {source_algorithm}")
        algorithm_target = algorithms_dir / source_algorithm.name
        shutil.copy2(source_algorithm, algorithm_target)

    contract_path = root / "safety_contract.json"
    _write_json(contract_path, DEFAULT_PICO_PUMP_SAFETY_CONTRACT)
    firmware_outputs = export_pico_pump_firmware(firmware_dir)

    readme_path = root / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# IINTS Pico Pump Lab",
                "",
                "This workspace is for bench-only research with a Raspberry Pi Pico-style controller.",
                "It is not a medical device workflow and must not be connected to a person, animal,",
                "insulin reservoir, infusion set, or active pumping mechanism.",
                "",
                "## Flow",
                "",
                "1. edit `algorithms/pico_bench_algorithm.py` or copy in your own SDK algorithm",
                "2. simulate and validate the algorithm inside IINTS",
                "3. package a bench-only bundle with `iints edge pump package`",
                "4. copy the locked firmware bundle to a Pico/CircuitPython drive with `iints edge pump upload --write`",
                "5. run `iints edge pump serial-test` to confirm the board is alive and still locked",
                "",
                "## Package Command",
                "",
                "```bash",
                "iints edge pump package \\",
                f"  --algorithm {algorithm_target} \\",
                "  --output-dir bundles/pico_bench_bundle \\",
                "  --safety-contract safety_contract.json",
                "```",
                "",
                "## Upload Command",
                "",
                "```bash",
                "iints edge pump upload \\",
                "  --bundle-dir bundles/pico_bench_bundle \\",
                "  --mount-dir /Volumes/CIRCUITPY \\",
                f"  --bench-only-confirm \"{PICO_PUMP_CONFIRMATION}\" \\",
                "  --write",
                "```",
                "",
                "Use a dummy load or LEDs only. The generated firmware does not contain motor actuation code.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    package_script = scripts_dir / "package_bench_bundle.sh"
    package_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd \"$(dirname \"$0\")/..\"",
                "iints edge pump package --algorithm algorithms/pico_bench_algorithm.py --output-dir bundles/pico_bench_bundle --safety-contract safety_contract.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    package_script.chmod(0o755)

    return {
        "output_dir": str(root),
        "algorithm": str(algorithm_target),
        "safety_contract": str(contract_path),
        "firmware_dir": firmware_outputs["output_dir"],
        "readme": str(readme_path),
        "package_script": str(package_script),
    }


def _load_safety_contract(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_PICO_PUMP_SAFETY_CONTRACT)
    contract_path = Path(path).expanduser().resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(f"Safety contract not found: {contract_path}")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Safety contract must be a JSON object.")
    return payload


def _validate_bench_contract(contract: dict[str, Any]) -> None:
    if contract.get("hardware_actuation_enabled") is not False:
        raise ValueError("Pico pump bundles must keep hardware_actuation_enabled=false.")
    if float(contract.get("max_command_units", 0.0) or 0.0) != 0.0:
        raise ValueError("Pico pump bench bundles must use max_command_units=0.0.")
    if float(contract.get("max_units_per_hour", 0.0) or 0.0) != 0.0:
        raise ValueError("Pico pump bench bundles must use max_units_per_hour=0.0.")


def build_pico_pump_bundle(
    algorithm_path: str | Path,
    output_dir: str | Path,
    *,
    safety_contract_path: str | Path | None = None,
    label: str = "pico_pump_bench",
) -> dict[str, Any]:
    """Package an SDK algorithm with locked Pico bench firmware and a manifest."""

    source_algorithm = Path(algorithm_path).expanduser().resolve()
    if not source_algorithm.is_file():
        raise FileNotFoundError(f"Algorithm file not found: {source_algorithm}")

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    firmware_dir = target / "firmware"
    firmware_outputs = export_pico_pump_firmware(firmware_dir)

    algorithm_target = target / "algorithm.py"
    shutil.copy2(source_algorithm, algorithm_target)
    contract = _load_safety_contract(safety_contract_path)
    _validate_bench_contract(contract)
    contract_target = target / "safety_contract.json"
    _write_json(contract_target, contract)

    manifest = {
        "schema_version": "1.0",
        "label": label,
        "target": "raspberry_pi_pico_bench",
        "scope": "bench_only_not_for_human_or_animal_use",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm_file": "algorithm.py",
        "algorithm_sha256": _sha256_file(algorithm_target),
        "firmware_entrypoint": "firmware/code.py",
        "safety_contract": "safety_contract.json",
        "hardware_actuation_enabled": False,
        "upload_requires_confirmation": PICO_PUMP_CONFIRMATION,
        "ready_banner": PICO_PUMP_READY_BANNER,
    }
    manifest_path = target / "manifest.json"
    _write_json(manifest_path, manifest)

    (target / "README.md").write_text(
        "\n".join(
            [
                "# IINTS Pico Pump Bench Bundle",
                "",
                "This bundle can be copied to a Pico-style board for bench-only serial testing.",
                "The firmware is deliberately locked and does not actuate a motor or deliver insulin.",
                "",
                "Files:",
                "",
                "- `algorithm.py`: source snapshot from the SDK algorithm under test",
                "- `firmware/code.py`: non-actuating USB serial bench firmware",
                "- `safety_contract.json`: zero-delivery contract",
                "- `manifest.json`: hashes and reproducibility metadata",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(target),
        "algorithm": str(algorithm_target),
        "algorithm_sha256": manifest["algorithm_sha256"],
        "firmware": firmware_outputs,
        "safety_contract": str(contract_target),
        "manifest": str(manifest_path),
    }


def _load_bundle_manifest(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pico pump manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pico pump manifest must be a JSON object.")
    if payload.get("target") != "raspberry_pi_pico_bench":
        raise ValueError("Only raspberry_pi_pico_bench bundles are supported.")
    if payload.get("hardware_actuation_enabled") is not False:
        raise ValueError("Refusing to upload a bundle with hardware actuation enabled.")
    return payload


def bench_test_pico_pump_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Validate a Pico pump bundle before any board upload is attempted."""

    source = Path(bundle_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {source}")
    manifest = _load_bundle_manifest(source)
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, message: str, **details: Any) -> None:
        checks.append({"name": name, "passed": passed, "message": message, "details": details})

    algorithm_path = source / str(manifest.get("algorithm_file", "algorithm.py"))
    firmware_path = source / str(manifest.get("firmware_entrypoint", "firmware/code.py"))
    contract_path = source / str(manifest.get("safety_contract", "safety_contract.json"))
    manifest_path = source / "manifest.json"

    add_check("manifest", manifest_path.is_file(), "Manifest file is present.", path=str(manifest_path))
    add_check("algorithm", algorithm_path.is_file(), "Algorithm source snapshot is present.", path=str(algorithm_path))
    add_check("firmware", firmware_path.is_file(), "Locked Pico firmware entrypoint is present.", path=str(firmware_path))
    add_check("safety_contract", contract_path.is_file(), "Zero-delivery safety contract is present.", path=str(contract_path))

    if algorithm_path.is_file():
        observed_hash = _sha256_file(algorithm_path)
        expected_hash = str(manifest.get("algorithm_sha256", ""))
        add_check(
            "algorithm_sha256",
            bool(expected_hash) and observed_hash == expected_hash,
            "Algorithm source hash matches the manifest.",
            expected=expected_hash,
            observed=observed_hash,
        )

    if contract_path.is_file():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            _validate_bench_contract(contract)
            add_check("contract_lockout", True, "Safety contract keeps hardware actuation locked out.")
        except Exception as exc:
            add_check("contract_lockout", False, "Safety contract failed lockout validation.", error=str(exc))

    add_check(
        "scope",
        manifest.get("scope") == "bench_only_not_for_human_or_animal_use",
        "Manifest declares bench-only scope.",
        scope=manifest.get("scope"),
    )

    passed = all(check["passed"] for check in checks)
    report = {
        "passed": passed,
        "bundle_dir": str(source),
        "scope": "bench_only_not_for_human_or_animal_use",
        "manifest": manifest,
        "checks": checks,
        "required_next_step": (
            "Run serial-test after upload and keep the hardware disconnected from insulin or people."
            if passed
            else "Fix the failing bundle checks before upload."
        ),
    }
    _write_json(source / "bench_test_report.json", report)
    return report


def upload_pico_pump_bundle(
    bundle_dir: str | Path,
    mount_dir: str | Path,
    *,
    bench_only_confirmation: str,
    write: bool = False,
) -> dict[str, Any]:
    """Copy a locked bench bundle to a mounted Pico/CircuitPython-style drive."""

    source = Path(bundle_dir).expanduser().resolve()
    target = Path(mount_dir).expanduser().resolve()
    if bench_only_confirmation != PICO_PUMP_CONFIRMATION:
        raise ValueError(f"Refusing upload without exact confirmation: {PICO_PUMP_CONFIRMATION!r}")
    if not source.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {source}")
    manifest = _load_bundle_manifest(source)

    target_name = target.name.upper()
    if target_name == "RPI-RP2":
        raise ValueError(
            "The Pico is in BOOTSEL/UF2 mode (`RPI-RP2`). Install MicroPython/CircuitPython first, "
            "then mount a writable firmware filesystem such as `CIRCUITPY`, or use --mount-dir for a test folder."
        )
    if write and not target.is_dir():
        raise FileNotFoundError(f"Mount directory not found: {target}")

    copy_plan = [
        (source / "firmware" / "code.py", target / "code.py"),
        (source / "algorithm.py", target / "iints_algorithm.py"),
        (source / "safety_contract.json", target / "iints_safety_contract.json"),
        (source / "manifest.json", target / "iints_pump_manifest.json"),
    ]
    for source_file, _ in copy_plan:
        if not source_file.is_file():
            raise FileNotFoundError(f"Bundle file missing: {source_file}")

    copied: list[str] = []
    if write:
        for source_file, destination in copy_plan:
            shutil.copy2(source_file, destination)
            copied.append(str(destination))

    return {
        "bundle_dir": str(source),
        "mount_dir": str(target),
        "write": write,
        "copied": copied,
        "planned": [str(destination) for _, destination in copy_plan],
        "manifest": manifest,
    }


def _require_pyserial():
    try:
        import serial  # type: ignore
        from serial.tools import list_ports  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised through callers
        raise ImportError(
            "Pico pump serial commands require pyserial. Install the edge profile with "
            '`python -m pip install -U "iints-sdk-python35[edge,mdmp]"`.'
        ) from exc
    return serial, list_ports


@contextmanager
def _pico_serial_connection(port: str, *, baudrate: int, timeout_seconds: float):
    serial, _ = _require_pyserial()
    with serial.Serial(  # type: ignore[attr-defined]
        port,
        baudrate=baudrate,
        timeout=timeout_seconds,
        write_timeout=timeout_seconds,
    ) as connection:
        try:
            connection.reset_input_buffer()
            connection.reset_output_buffer()
        except (AttributeError, OSError):
            pass
        time.sleep(0.4)
        yield connection


def _serial_exchange(connection: Any, message: str) -> str | None:
    connection.write((message.strip() + "\n").encode("utf-8"))
    connection.flush()
    raw = connection.readline()
    if not raw:
        return None
    return raw.decode("utf-8", errors="replace").strip() or None


def run_pico_pump_serial_self_test(
    port: str,
    *,
    baudrate: int = PICO_PUMP_BAUDRATE,
    timeout_seconds: float = 1.5,
) -> list[dict[str, str | None]]:
    """Run a non-actuating serial smoke test against the Pico bench firmware."""

    results: list[dict[str, str | None]] = []
    with _pico_serial_connection(port, baudrate=baudrate, timeout_seconds=timeout_seconds) as connection:
        for command in ("PING", "STATUS", "LOCKOUT"):
            response = _serial_exchange(connection, command)
            results.append({"command": command, "response": response})
    return results
