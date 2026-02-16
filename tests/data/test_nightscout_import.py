import importlib.util
import pytest

from iints.data.nightscout import NightscoutConfig, import_nightscout


def test_import_nightscout_requires_dependency():
    has_py_nightscout = (
        importlib.util.find_spec("py_nightscout") is not None
        or importlib.util.find_spec("nightscout") is not None
    )
    if has_py_nightscout:
        pytest.skip("py-nightscout installed; skip dependency error check.")

    with pytest.raises(ImportError):
        import_nightscout(NightscoutConfig(url="https://example.com"))
