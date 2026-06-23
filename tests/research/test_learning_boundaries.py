from __future__ import annotations

import numpy as np
import pytest

from iints.learning.learning_system import LearningSystem


def test_legacy_random_mock_learning_is_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    system = LearningSystem()

    with pytest.raises(RuntimeError, match="Random mock learning is disabled"):
        system.simulate_learning_process("demo", [100.0, 110.0, 120.0])


def test_synthetic_teacher_is_reproducible_and_bounded() -> None:
    pytest.importorskip("torch")
    from iints.learning.autonomous_optimizer import ClinicalTeacher

    first_x, first_y = ClinicalTeacher(seed=123).generate_clinical_training_data(64)
    second_x, second_y = ClinicalTeacher(seed=123).generate_clinical_training_data(64)

    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)
    assert np.all(first_y >= 0.0)
    assert np.all(first_y <= 1.0)
