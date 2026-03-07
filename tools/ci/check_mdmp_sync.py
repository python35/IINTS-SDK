from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import yaml


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    os.environ["IINTS_MDMP_BACKEND"] = "mdmp_core"

    try:
        from mdmp_core.contracts import load_contract as load_external_contract
        from mdmp_core.runner import ContractRunner as ExternalRunner
        from mdmp_core.runner import MDMP_GRADE_ORDER as EXTERNAL_GRADE_ORDER
    except Exception as exc:  # pragma: no cover - depends on CI environment
        raise RuntimeError(
            "Standalone mdmp_core is not importable. Install mdmp-protocol before running this check."
        ) from exc

    from iints.mdmp.backend import (
        MDMP_GRADE_ORDER as SDK_GRADE_ORDER,
        active_mdmp_backend,
        build_mdmp_dashboard_html,
        load_mdmp_contract,
        run_mdmp_validation,
    )

    _assert(active_mdmp_backend() == "mdmp_core", "SDK did not activate mdmp_core backend")
    _assert(tuple(SDK_GRADE_ORDER) == tuple(EXTERNAL_GRADE_ORDER), "Grade order mismatch between SDK and mdmp_core")

    contract_payload = {
        "schema": {
            "name": "sync_contract",
            "version": "1.0",
            "industry": "health",
            "columns": [
                {"name": "timestamp", "type": "datetime", "required": True},
                {"name": "glucose", "type": "float", "bounds": [40, 400], "required": True},
            ],
        },
        "consent": {
            "ai_training_allowed": True,
            "jurisdiction": "GDPR",
            "anonymized": True,
        },
    }

    with tempfile.TemporaryDirectory(prefix="mdmp_sync_") as tmp_dir:
        root = Path(tmp_dir)
        contract_path = root / "contract.yaml"
        contract_path.write_text(yaml.safe_dump(contract_payload, sort_keys=False), encoding="utf-8")

        df = pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-01T00:00:00Z",
                    "2026-03-01T00:05:00Z",
                    "2026-03-01T00:10:00Z",
                ],
                "glucose": [110.0, 126.0, 118.0],
            }
        )

        external_contract = load_external_contract(contract_path)
        external_result = ExternalRunner(external_contract).run(df)

        sdk_contract = load_mdmp_contract(contract_path)
        sdk_result = run_mdmp_validation(sdk_contract, df, apply_builtin_transforms=False)

    _assert(
        sdk_result.dataset_fingerprint_sha256 == external_result.dataset_fingerprint_sha256,
        "Dataset fingerprint mismatch between SDK and mdmp_core",
    )
    _assert(
        sdk_result.contract_fingerprint_sha256 == external_result.contract_fingerprint_sha256,
        "Contract fingerprint mismatch between SDK and mdmp_core",
    )
    _assert(sdk_result.mdmp_grade == external_result.grade, "Grade mismatch between SDK and mdmp_core")
    _assert(
        sdk_result.mdmp_protocol_version == external_result.protocol_version,
        "Protocol version mismatch between SDK and mdmp_core",
    )
    _assert(bool(sdk_result.checks), "SDK validation returned zero checks")

    html = build_mdmp_dashboard_html(sdk_result.to_dict(), title="MDMP Sync Check")
    _assert("<html" in html.lower(), "Dashboard renderer did not return HTML")
    _assert("MDMP Sync Check" in html, "Dashboard title missing in rendered HTML")

    print("MDMP sync check passed: SDK backend is aligned with standalone mdmp_core.")


if __name__ == "__main__":
    main()
