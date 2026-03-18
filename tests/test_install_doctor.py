from importlib import metadata

from iints_sdk_python35_doctor import diagnose_install


class _Module:
    __file__ = "/tmp/site-packages/iints/__init__.py"
    __version__ = "0.1.1"


def test_diagnose_install_detects_package_conflict(monkeypatch):
    monkeypatch.setattr(
        metadata,
        "packages_distributions",
        lambda: {"iints": ["iints", "iints-sdk-python35"]},
    )

    def fake_version(name: str) -> str:
        versions = {"iints": "0.1.1", "iints-sdk-python35": "1.1.2"}
        if name in versions:
            return versions[name]
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", fake_version)
    monkeypatch.setattr("iints_sdk_python35_doctor.importlib.import_module", lambda name: _Module())

    diagnosis = diagnose_install()

    assert diagnosis.has_conflict is True
    assert diagnosis.installed_sdk_version == "1.1.2"
    assert diagnosis.installed_legacy_version == "0.1.1"
    assert diagnosis.imported_module_version == "0.1.1"
