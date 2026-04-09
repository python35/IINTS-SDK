from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_cli_prefix(args: argparse.Namespace) -> list[str]:
    if args.python_module:
        return [sys.executable, "-m", args.python_module]
    if args.iints_bin:
        return [args.iints_bin]
    return ["iints"]


def _run(
    prefix: Sequence[str],
    command: Sequence[str],
    *,
    env: dict[str, str],
    cwd: Path | None = None,
    expect_exit_code: int = 0,
) -> subprocess.CompletedProcess[str]:
    full_cmd = [*prefix, *command]
    pretty = " ".join(shlex.quote(part) for part in full_cmd)
    print(f"$ {pretty}")
    completed = subprocess.run(
        full_cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)
    if completed.returncode != expect_exit_code:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {pretty}")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-check the installed edge CLI from a built wheel.")
    parser.add_argument(
        "--python-module",
        help="Run the CLI as `python -m <module>` using the current interpreter. Useful inside an isolated wheel venv.",
    )
    parser.add_argument("--iints-bin", help="Path to the installed `iints` executable.")
    args = parser.parse_args()

    prefix = _build_cli_prefix(args)

    with tempfile.TemporaryDirectory(prefix="iints_edge_cli_smoke_") as tmp_dir:
        root = Path(tmp_dir)
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))

        _run(prefix, ["--help"], env=env)
        _run(prefix, ["edge", "--help"], env=env)

        doctor_pi = _run(prefix, ["edge", "doctor", "--board", "raspberry_pi"], env=env)
        _assert("Raspberry Pi Edge Check" in doctor_pi.stdout, "Raspberry Pi doctor output was not rendered.")

        doctor_uno = _run(prefix, ["edge", "doctor", "--board", "uno_q"], env=env)
        _assert("Arduino UNO Q Edge Check" in doctor_uno.stdout, "UNO Q doctor output was not rendered.")
        _assert("USB serial support" in doctor_uno.stdout, "UNO Q doctor did not report serial bridge readiness.")

        pi_dir = root / "iints_pi_demo"
        _run(prefix, ["edge", "setup", "--output-dir", str(pi_dir), "--board", "raspberry_pi"], env=env)
        _assert((pi_dir / "run_edge_patient.sh").is_file(), "Pi setup did not create run_edge_patient.sh.")
        _assert(
            (pi_dir / "patient_runtime" / "patient_runtime_config.json").is_file(),
            "Pi setup did not create patient_runtime_config.json.",
        )
        _run(
            prefix,
            ["edge", "up", "--project-dir", str(pi_dir), "--foreground", "--max-steps", "1"],
            env=env,
        )
        status_pi = _run(prefix, ["edge", "status", "--project-dir", str(pi_dir)], env=env)
        _assert("IINTS Edge Runtime Status" in status_pi.stdout, "Pi status output did not render.")
        _assert((pi_dir / "patient_runtime" / "patient_state.db").is_file(), "Pi runtime did not create patient_state.db.")

        uno_dir = root / "iints_uno_q_demo"
        _run(prefix, ["edge", "setup", "--output-dir", str(uno_dir), "--board", "uno_q"], env=env)
        _assert(
            (uno_dir / "uno_q_bridge" / "iints_supervisor_bridge.ino").is_file(),
            "UNO Q setup did not create the bridge sketch.",
        )
        _assert((uno_dir / "EDGE_SETUP.md").is_file(), "UNO Q setup did not create EDGE_SETUP.md.")
        _run(
            prefix,
            ["edge", "up", "--project-dir", str(uno_dir), "--foreground", "--max-steps", "1"],
            env=env,
        )
        status_uno = _run(prefix, ["edge", "status", "--project-dir", str(uno_dir)], env=env)
        _assert("IINTS Edge Runtime Status" in status_uno.stdout, "UNO Q status output did not render.")
        _run(prefix, ["edge", "bridge-test", "--help"], env=env)
        _run(prefix, ["edge", "bridge-run", "--help"], env=env)
        _run(prefix, ["edge", "bridge-flash", "--help"], env=env)

        print("Installed edge CLI smoke passed for Raspberry Pi and Arduino UNO Q.")


if __name__ == "__main__":
    main()
