from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.safety.openfda_safety import (
    FDA_RECALL_REGISTRY,
    run_fda_safety_benchmark,
    simulate_fda_failure_scenario,
)

runner = CliRunner()


def test_fda_recall_registry_completeness():
    assert len(FDA_RECALL_REGISTRY) >= 5
    case_ids = {c.case_id for c in FDA_RECALL_REGISTRY}
    assert "FDA-2024-TANDEM-AUTOBOLUS" in case_ids
    assert "FDA-2024-MINIMED-BATTERY-DEPLETION" in case_ids
    assert "FDA-2023-INFUSION-OCCLUSION-BURST" in case_ids
    assert "FDA-2023-OMNIPOD-IOB-DESYNC" in case_ids
    assert "FDA-2024-DEXCOM-SILENT-CRASH" in case_ids


def test_simulate_fda_failure_scenario_tandem():
    case = next(c for c in FDA_RECALL_REGISTRY if c.case_id == "FDA-2024-TANDEM-AUTOBOLUS")
    # Unmitigated run
    df_unmit, m_unmit = simulate_fda_failure_scenario(case, enable_supervisor=False)
    assert not m_unmit.adverse_event_prevented
    assert m_unmit.min_glucose_mgdl < 54.0

    # Supervised run
    df_sup, m_sup = simulate_fda_failure_scenario(case, enable_supervisor=True)
    assert m_sup.adverse_event_prevented
    assert m_sup.min_glucose_mgdl >= 54.0
    assert m_sup.supervisor_intervened


def test_simulate_fda_minimed_battery_depletion():
    case = next(c for c in FDA_RECALL_REGISTRY if c.case_id == "FDA-2024-MINIMED-BATTERY-DEPLETION")
    df_sup, m_sup = simulate_fda_failure_scenario(case, enable_supervisor=True)
    assert m_sup.hazard_detected
    assert m_sup.detection_latency_minutes <= 10.0


def test_run_fda_safety_benchmark(tmp_path: Path):
    out_dir = tmp_path / "fda_safety_test"
    report = run_fda_safety_benchmark(output_dir=out_dir)

    assert report.total_cases_evaluated >= 5
    assert report.unmitigated_adverse_event_rate_pct > report.supervised_adverse_event_rate_pct
    assert report.hazard_detection_rate_pct == 100.0
    assert (out_dir / "fda_safety_benchmark_summary.csv").is_file()
    assert (out_dir / "fda_safety_benchmark_summary.json").is_file()
    assert (out_dir / "FDA_ADVERSE_EVENTS_SAFETY_REPORT.md").is_file()


def test_cli_safety_fda_commands(tmp_path: Path):
    # Test fda-list
    res_list = runner.invoke(app, ["safety", "fda-list"])
    assert res_list.exit_code == 0
    assert "Tandem" in res_list.output
    assert "MiniMed" in res_list.output

    # Test fda-benchmark
    out_dir = tmp_path / "cli_fda_out"
    res_bench = runner.invoke(app, ["safety", "fda-benchmark", "--output-dir", str(out_dir)])
    assert res_bench.exit_code == 0
    assert "Safety Benchmark Completed Successfully" in res_bench.output
    assert (out_dir / "FDA_ADVERSE_EVENTS_SAFETY_REPORT.md").is_file()
