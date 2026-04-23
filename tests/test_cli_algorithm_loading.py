from __future__ import annotations

import sys

from iints.cli.cli import _load_algorithm_instance_silent


def test_load_algorithm_instance_silent_does_not_leave_temp_modules(tmp_path) -> None:
    algo_path = tmp_path / "injected_algo.py"
    algo_path.write_text(
        """
class InjectedAlgo(iints.InsulinAlgorithm):
    def predict_insulin(self, data):
        return {"total_insulin_delivered": 0.0}
""".strip(),
        encoding="utf-8",
    )

    before = {name for name in sys.modules if name.startswith("_iints_user_algo_")}
    instance = _load_algorithm_instance_silent(algo_path)
    after = {name for name in sys.modules if name.startswith("_iints_user_algo_")}

    assert instance.__class__.__name__ == "InjectedAlgo"
    assert after == before
