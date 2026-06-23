from __future__ import annotations

import ast
import inspect
import textwrap

from iints.ai.assistant import IINTSAssistant
from iints.ai.prompts import SYSTEM_PROMPT, TASK_TEMPLATES


def test_predict_insulin_contains_no_language_model_call() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(IINTSAssistant.predict_insulin)))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "complete" not in called_attributes
    assert "_run_task" not in called_attributes
    assert "calculate_deterministic_dose" in called_attributes or any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "calculate_deterministic_dose"
        for node in ast.walk(tree)
    )


def test_prompts_forbid_numeric_and_diagnostic_authority() -> None:
    combined = "\n".join([SYSTEM_PROMPT, *TASK_TEMPLATES.values()]).lower()

    assert "never calculate" in combined
    assert "all metrics and controller outputs must come from deterministic sdk code" in combined
    assert "final_dose" not in combined
    assert "final_glucagon_dose_mg" not in combined
    assert "diagnose the" not in combined
