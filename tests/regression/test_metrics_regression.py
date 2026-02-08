import json
from pathlib import Path

from iints.analysis.clinical_metrics import ClinicalMetricsCalculator
from iints.data.importer import import_cgm_dataframe, load_demo_dataframe


def _compute_metrics() -> dict:
    demo_df = load_demo_dataframe()
    standard_df = import_cgm_dataframe(demo_df, data_format="generic", source="demo")
    duration_hours = standard_df["timestamp"].max() / 60.0
    calculator = ClinicalMetricsCalculator()
    metrics = calculator.calculate(
        glucose=standard_df["glucose"],
        duration_hours=duration_hours,
    ).to_dict()
    # Round for stability
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}


def test_demo_metrics_regression():
    snapshot_path = Path("tests/regression/golden_metrics_demo.json")
    metrics = _compute_metrics()

    if not snapshot_path.exists():
        snapshot_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        # First run creates snapshot for CI usage
        return

    expected = json.loads(snapshot_path.read_text())
    assert metrics == expected
