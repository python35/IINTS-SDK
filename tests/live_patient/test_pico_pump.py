from __future__ import annotations

import json
from pathlib import Path

import pytest

from iints.live_patient.pico_pump import (
    PICO_PUMP_CONFIRMATION,
    build_pico_pump_bundle,
    create_pico_pump_lab,
    upload_pico_pump_bundle,
)


def test_create_pico_pump_lab_exports_locked_workspace(tmp_path: Path) -> None:
    outputs = create_pico_pump_lab(tmp_path / "pico_lab")

    root = Path(outputs["output_dir"])
    assert (root / "algorithms" / "pico_bench_algorithm.py").is_file()
    assert (root / "firmware" / "pico_pump_bench" / "code.py").is_file()
    contract = json.loads((root / "safety_contract.json").read_text(encoding="utf-8"))
    assert contract["hardware_actuation_enabled"] is False
    assert contract["max_command_units"] == 0.0
    assert "bench-only" in (root / "README.md").read_text(encoding="utf-8")


def test_build_and_upload_pico_pump_bundle_requires_bench_contract(tmp_path: Path) -> None:
    algorithm = tmp_path / "algorithm.py"
    algorithm.write_text("class DemoAlgorithm:\n    pass\n", encoding="utf-8")
    bundle = build_pico_pump_bundle(algorithm, tmp_path / "bundle")

    manifest = json.loads(Path(bundle["manifest"]).read_text(encoding="utf-8"))
    assert manifest["target"] == "raspberry_pi_pico_bench"
    assert manifest["hardware_actuation_enabled"] is False
    assert manifest["algorithm_sha256"] == bundle["algorithm_sha256"]

    mount = tmp_path / "CIRCUITPY"
    mount.mkdir()
    dry_run = upload_pico_pump_bundle(
        tmp_path / "bundle",
        mount,
        bench_only_confirmation=PICO_PUMP_CONFIRMATION,
        write=False,
    )
    assert dry_run["copied"] == []
    assert str(mount / "code.py") in dry_run["planned"]

    written = upload_pico_pump_bundle(
        tmp_path / "bundle",
        mount,
        bench_only_confirmation=PICO_PUMP_CONFIRMATION,
        write=True,
    )
    assert (mount / "code.py").is_file()
    assert (mount / "iints_pump_manifest.json").is_file()
    assert str(mount / "code.py") in written["copied"]


def test_pico_pump_upload_refuses_missing_confirmation(tmp_path: Path) -> None:
    algorithm = tmp_path / "algorithm.py"
    algorithm.write_text("class DemoAlgorithm:\n    pass\n", encoding="utf-8")
    build_pico_pump_bundle(algorithm, tmp_path / "bundle")
    mount = tmp_path / "CIRCUITPY"
    mount.mkdir()

    with pytest.raises(ValueError, match="Refusing upload"):
        upload_pico_pump_bundle(
            tmp_path / "bundle",
            mount,
            bench_only_confirmation="yes",
            write=True,
        )


def test_pico_pump_bundle_rejects_actuating_contract(tmp_path: Path) -> None:
    algorithm = tmp_path / "algorithm.py"
    algorithm.write_text("class DemoAlgorithm:\n    pass\n", encoding="utf-8")
    contract = tmp_path / "unsafe.json"
    contract.write_text(
        json.dumps({"hardware_actuation_enabled": True, "max_command_units": 1.0}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hardware_actuation_enabled=false"):
        build_pico_pump_bundle(algorithm, tmp_path / "bundle", safety_contract_path=contract)
