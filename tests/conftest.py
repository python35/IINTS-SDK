from pathlib import Path
import sys

import pytest

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"

if src_path.exists():
    sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give rich a fixed, wide console so CLI assertions test content, not layout.

    Without this, rich wraps tables to the ambient terminal width and truncates
    cell values with an ellipsis, so tests asserting that a path or filename
    appears in the output pass or fail depending on the window they run in.
    """
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("TERM", "dumb")
