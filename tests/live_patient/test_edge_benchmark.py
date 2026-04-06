from __future__ import annotations

from iints.live_patient.edge_benchmark import run_edge_benchmark


ALGO_TEMPLATE = '''from iints import InsulinAlgorithm, AlgorithmInput, AlgorithmMetadata


class DemoPID(InsulinAlgorithm):
    def __init__(self):
        super().__init__()
        self.set_algorithm_metadata(AlgorithmMetadata(name="DemoPID", version="1.0.0"))

    def predict_insulin(self, data: AlgorithmInput):
        self.why_log = []
        dose = 0.6 if data.current_glucose > 110 else 0.0
        return {
            "total_insulin_delivered": dose,
            "basal_insulin": dose,
            "bolus_insulin": 0.0,
            "correction_bolus": 0.0,
            "meal_bolus": 0.0,
        }
'''


def test_run_edge_benchmark_returns_core_metrics(tmp_path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text(ALGO_TEMPLATE, encoding="utf-8")

    payload = run_edge_benchmark(
        algo_path=algo,
        scenario_profile="normal_day",
        steps=4,
        api_port=8799,
        platform_name="test-platform",
    )

    assert payload["platform"] == "test-platform"
    assert payload["runtime"]["steps_per_second"] > 0
    assert payload["dashboard"]["status_response_ms"]["mean_ms"] >= 0
    assert payload["latest_status"]["scenario_profile"] == "normal_day"
