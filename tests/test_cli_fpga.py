from __future__ import annotations

from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def test_fpga_cli_setup_and_mock_simulate(tmp_path) -> None:
    lab_dir = tmp_path / "fpga_lab"
    setup_result = runner.invoke(app, ["fpga", "setup", "--output-dir", str(lab_dir)])

    assert setup_result.exit_code == 0
    assert "IINTS FPGA Lab" in setup_result.stdout
    assert (lab_dir / "rtl" / "iints_fpga_safety_core.v").is_file()

    run_dir = tmp_path / "fpga_run"
    simulate_result = runner.invoke(
        app,
        [
            "fpga",
            "simulate",
            "--events",
            str(lab_dir / "scenarios" / "fpga_demo_events.json"),
            "--output-dir",
            str(run_dir),
        ],
    )

    assert simulate_result.exit_code == 0
    assert "IINTS FPGA Mode Run" in simulate_result.stdout
    assert (run_dir / "fpga_comparison.json").is_file()
    assert (run_dir / "fpga_report.md").is_file()

    compare_result = runner.invoke(app, ["fpga", "compare", "--run-dir", str(run_dir)])
    assert compare_result.exit_code == 0
    assert "IINTS FPGA Comparison" in compare_result.stdout

    report_result = runner.invoke(app, ["fpga", "report", "--run-dir", str(run_dir)])
    assert report_result.exit_code == 0
    assert "IINTS FPGA Report" in report_result.stdout


def test_fpga_cli_doctor_and_demo(tmp_path) -> None:
    doctor_result = runner.invoke(app, ["fpga", "doctor"])

    assert doctor_result.exit_code == 0
    assert "IINTS FPGA Doctor" in doctor_result.stdout
    assert "NOT A MEDICAL DEVICE" in doctor_result.stdout

    demo_dir = tmp_path / "fpga_demo"
    demo_result = runner.invoke(app, ["fpga", "demo", "--output-dir", str(demo_dir)])

    assert demo_result.exit_code == 0
    assert "IINTS FPGA Demo" in demo_result.stdout
    assert (demo_dir / "lab" / "rtl" / "iints_fpga_safety_core.v").is_file()
    assert (demo_dir / "lab" / "FPGA_STORY.md").is_file()
    assert (demo_dir / "events.csv").is_file()
    assert (demo_dir / "results.json").is_file()
    assert (demo_dir / "manifest.json").is_file()
    assert (demo_dir / "report.md").is_file()
    assert (demo_dir / "fpga_report.md").is_file()


def test_fpga_cli_start_and_replay(tmp_path) -> None:
    start_dir = tmp_path / "fpga_start"
    start_result = runner.invoke(app, ["fpga", "start", "--output-dir", str(start_dir)])

    assert start_result.exit_code == 0
    assert "IINTS FPGA Start" in start_result.stdout
    assert (start_dir / "lab" / "bridge" / "fpga_jsonline_bridge.py").is_file()
    assert (start_dir / "lab" / "testbench" / "iints_fpga_safety_core_tb.v").is_file()
    assert (start_dir / "report.md").is_file()

    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "\n".join(
            [
                "time_minutes,glucose_actual_mgdl,carb_intake_grams,delivered_insulin_units",
                "0,118,0,0",
                "5,108,0,0.4",
                "10,92,0,0",
                "15,68,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "fpga_replay"
    replay_result = runner.invoke(
        app,
        [
            "fpga",
            "replay",
            "--results-csv",
            str(results_csv),
            "--output-dir",
            str(replay_dir),
        ],
    )

    assert replay_result.exit_code == 0
    assert "IINTS FPGA Mode Run" in replay_result.stdout
    assert (replay_dir / "fpga_events_from_results.json").is_file()
    assert (replay_dir / "fpga_comparison.json").is_file()


def test_fpga_cli_export_events(tmp_path) -> None:
    results_csv = tmp_path / "results.csv"
    events_json = tmp_path / "events.json"
    results_csv.write_text(
        "\n".join(
            [
                "time_minutes,glucose_actual_mgdl,carb_intake_grams,delivered_insulin_units",
                "0,130,0,0",
                "5,150,20,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "fpga",
            "export-events",
            "--results-csv",
            str(results_csv),
            "--output-events",
            str(events_json),
        ],
    )

    assert result.exit_code == 0
    assert events_json.is_file()
    assert "FPGA events written" in result.stdout
