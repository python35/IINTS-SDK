from __future__ import annotations

import importlib
from dataclasses import dataclass

try:
    from importlib import metadata
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    import importlib_metadata as metadata  # type: ignore


RECOMMENDED_VERSION = "1.3.1"


@dataclass(frozen=True)
class InstallDiagnosis:
    owners: list[str]
    installed_sdk_version: str | None
    installed_legacy_version: str | None
    imported_module_path: str | None
    imported_module_version: str | None

    @property
    def has_conflict(self) -> bool:
        return "iints" in self.owners and "iints-sdk-python35" in self.owners


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def diagnose_install() -> InstallDiagnosis:
    owners = sorted(set(metadata.packages_distributions().get("iints", [])))
    imported_module_path: str | None = None
    imported_module_version: str | None = None

    try:
        module = importlib.import_module("iints")
        imported_module_path = getattr(module, "__file__", None)
        imported_module_version = getattr(module, "__version__", None)
    except Exception:
        pass

    return InstallDiagnosis(
        owners=owners,
        installed_sdk_version=_distribution_version("iints-sdk-python35"),
        installed_legacy_version=_distribution_version("iints"),
        imported_module_path=imported_module_path,
        imported_module_version=imported_module_version,
    )


def main() -> int:
    diagnosis = diagnose_install()
    owners_text = ", ".join(diagnosis.owners) if diagnosis.owners else "none"

    print("IINTS SDK Install Doctor")
    print("========================")
    print(f"`iints` package owners: {owners_text}")
    print(f"`iints-sdk-python35` version: {diagnosis.installed_sdk_version or 'not installed'}")
    print(f"`iints` legacy version: {diagnosis.installed_legacy_version or 'not installed'}")
    print(f"Imported module path: {diagnosis.imported_module_path or 'not importable'}")
    print(f"Imported module version: {diagnosis.imported_module_version or 'unknown'}")
    print()

    if diagnosis.has_conflict:
        print("Conflict detected: both `iints` and `iints-sdk-python35` claim the `iints` package.")
        print("This is the usual reason `iints ai ...` is missing even after an SDK upgrade.")
        print()
        print("Fix it with:")
        print("  python -m pip uninstall -y iints iints-sdk-python35")
        print(f"  python -m pip install -U \"iints-sdk-python35[mdmp]=={RECOMMENDED_VERSION}\"")
        print("  hash -r")
        return 1

    if diagnosis.installed_sdk_version is None:
        print("`iints-sdk-python35` is not installed in this environment.")
        print("Install it with:")
        print(f"  python -m pip install -U \"iints-sdk-python35[mdmp]=={RECOMMENDED_VERSION}\"")
        return 1

    print("No package ownership conflict detected.")
    print("If `iints ai` is still missing, reinstall the SDK in the active virtual environment:")
    print(f"  python -m pip install -U \"iints-sdk-python35[mdmp]=={RECOMMENDED_VERSION}\"")
    print("  hash -r")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
