from __future__ import annotations

import pytest

from iints.core.units import bounded_value, finite_value, nonnegative_value


def test_unit_contracts_accept_valid_values() -> None:
    assert finite_value(120, name="glucose", unit="mg/dL") == 120.0
    assert nonnegative_value(0.2, name="insulin", unit="U/step") == 0.2
    assert bounded_value(5, name="time_step", unit="min", minimum=1, maximum=15) == 5.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "bad"])
def test_unit_contracts_reject_invalid_values_with_units(value: object) -> None:
    with pytest.raises(ValueError, match="mg/dL"):
        finite_value(value, name="glucose", unit="mg/dL")


def test_nonnegative_and_bounded_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="U/step"):
        nonnegative_value(-0.1, name="insulin", unit="U/step")
    with pytest.raises(ValueError, match="min"):
        bounded_value(30, name="time_step", unit="min", minimum=1, maximum=15)
