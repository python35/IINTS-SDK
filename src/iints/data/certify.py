from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from iints.mdmp.backend import (
    MDMPValidationResult,
    build_mdmp_dashboard_html,
    load_mdmp_contract,
    run_mdmp_validation,
)


def certify_dataset(
    contract: Any | str | Path,
    dataframe: pd.DataFrame,
    *,
    apply_builtin_transforms: bool = True,
) -> MDMPValidationResult:
    """Run the bundled data-certification flow on a DataFrame."""
    resolved_contract = load_mdmp_contract(Path(contract)) if isinstance(contract, (str, Path)) else contract
    return run_mdmp_validation(
        resolved_contract,
        dataframe,
        apply_builtin_transforms=apply_builtin_transforms,
    )


def certify_csv(
    contract_path: str | Path,
    input_csv: str | Path,
    *,
    apply_builtin_transforms: bool = True,
) -> MDMPValidationResult:
    """Load a CSV and certify it against an IINTS data contract."""
    contract = load_mdmp_contract(Path(contract_path))
    dataframe = pd.read_csv(input_csv)
    return certify_dataset(
        contract,
        dataframe,
        apply_builtin_transforms=apply_builtin_transforms,
    )


def render_certification_dashboard(
    report: Mapping[str, Any] | MDMPValidationResult,
    *,
    title: str = "IINTS Data Certification Dashboard",
) -> str:
    """Render a single-file HTML dashboard from a certification report."""
    payload = report.to_dict() if isinstance(report, MDMPValidationResult) else dict(report)
    return build_mdmp_dashboard_html(payload, title=title)


def write_certification_report(
    report: Mapping[str, Any] | MDMPValidationResult,
    output_path: str | Path,
) -> Path:
    """Write a certification report JSON file."""
    resolved = Path(output_path)
    payload = report.to_dict() if isinstance(report, MDMPValidationResult) else dict(report)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2))
    return resolved


def write_certification_dashboard(
    report: Mapping[str, Any] | MDMPValidationResult,
    output_path: str | Path,
    *,
    title: str = "IINTS Data Certification Dashboard",
) -> Path:
    """Write a certification dashboard HTML file."""
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(render_certification_dashboard(report, title=title))
    return resolved
