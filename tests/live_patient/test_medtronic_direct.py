from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.live_patient.medtronic_direct import (
    DirectPumpConfig,
    PumpSnapshot,
    SimulatedMedtronicPumpTransport,
    stream_direct_pump_snapshots,
    write_direct_pump_snapshot,
)


def test_direct_pump_config_rejects_impersonation_modes() -> None:
    with pytest.raises(ValueError):
        DirectPumpConfig(identity_mode="spoof_mobile_app")

    with pytest.raises(ValueError):
        DirectPumpConfig(identity_mode="emulate_sensor")


def test_pump_snapshot_rejects_command_like_payload() -> None:
    with pytest.raises(ValueError):
        PumpSnapshot.from_mapping(
            {
                "timestamp": "2026-05-25T10:00:00Z",
                "glucoseMgDl": 120,
                "bolusCommand": {"units": 1.0},
            }
        )


def test_simulated_direct_pump_stream_returns_valid_snapshots() -> None:
    snapshots = list(
        stream_direct_pump_snapshots(
            DirectPumpConfig(transport="simulated", simulated_seed=7),
            samples=3,
            poll_seconds=0,
        )
    )

    assert len(snapshots) == 3
    assert snapshots[0].source == "medtronic_direct_pump"
    assert snapshots[1].timestamp > snapshots[0].timestamp
    assert all(snapshot.glucose_mgdl > 0 for snapshot in snapshots)


def test_write_direct_pump_snapshot_writes_iints_outputs(tmp_path: Path) -> None:
    transport = SimulatedMedtronicPumpTransport(seed=3)
    transport.connect()
    snapshots = [transport.read_snapshot(), transport.read_snapshot()]
    transport.disconnect()

    outputs = write_direct_pump_snapshot(snapshots, tmp_path)

    assert Path(outputs["timeline_csv"]).is_file()
    assert Path(outputs["standard_csv"]).is_file()
    latest = json.loads(Path(outputs["latest_json"]).read_text(encoding="utf-8"))
    assert latest["rows"] == 2
    assert latest["latest"]["source"] == "medtronic_direct_pump"


def test_cli_medtronic_pump_direct_simulated_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "direct"

    result = CliRunner().invoke(
        app,
        [
            "medtronic-pump-direct",
            "--transport",
            "simulated",
            "--samples",
            "2",
            "--poll-seconds",
            "0",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "pump_timeline.csv").is_file()
    assert (output_dir / "cgm_standard.csv").is_file()
    assert (output_dir / "pump_latest.json").is_file()
