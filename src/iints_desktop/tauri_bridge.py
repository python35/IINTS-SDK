from __future__ import annotations

import argparse
import base64
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from iints_desktop.engine import (
    get_desktop_environment,
    list_desktop_presets,
    read_run_history,
    run_demo_preset,
)
from iints_desktop.evidence_connectors import list_evidence_connectors
from iints_desktop.local_ai import (
    ask_local_ai,
    check_local_ai,
    list_local_ai_models,
    start_local_ai_stack,
)
from iints_desktop.results import load_results_preview
from iints_desktop.update import get_desktop_update_info


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _emit(payload: dict[str, Any]) -> int:
    sys.stdout.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")
    return 0


def _ok(data: dict[str, Any]) -> int:
    return _emit({"ok": True, "data": data})


def _fail(message: str, *, details: str | None = None) -> int:
    return _emit({"ok": False, "error": message, "details": details})


def _status(_args: argparse.Namespace) -> int:
    env = get_desktop_environment(qt_available=False)
    return _ok(
        {
            "sdk_version": env.sdk_version,
            "python_executable": sys.executable,
            "bridge": "iints_desktop.tauri_bridge",
            "research_only": True,
            "medical_device": False,
        }
    )


def _workflows(_args: argparse.Namespace) -> int:
    workflows = [
        {
            "key": preset.key,
            "title": preset.title,
            "preset_name": preset.preset_name,
            "audience": preset.audience,
            "description": preset.description,
            "expected_output": preset.expected_output,
            "talk_track": list(preset.talk_track),
        }
        for preset in list_desktop_presets()
    ]
    return _ok({"workflows": workflows})


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _diagnostics(_args: argparse.Namespace) -> int:
    env = get_desktop_environment(qt_available=_module_available("PySide6"))
    optional_modules = {
        "pandas": _module_available("pandas"),
        "matplotlib": _module_available("matplotlib"),
        "plotly": _module_available("plotly"),
        "roadrunner": _module_available("roadrunner"),
        "fmpy": _module_available("fmpy"),
        "mdmp_core.crypto": _module_available("mdmp_core.crypto"),
        "PySide6": _module_available("PySide6"),
    }
    recommended_checks = []
    if not optional_modules["pandas"]:
        recommended_checks.append("Install the SDK with the desktop-all extra so CSV previews can load.")
    if not optional_modules["matplotlib"]:
        recommended_checks.append("Install matplotlib for generated preview graphs.")
    if not optional_modules["mdmp_core.crypto"]:
        recommended_checks.append("Install the mdmp extra before creating signed MDMP certificates.")
    if not optional_modules["roadrunner"]:
        recommended_checks.append(
            "Install the desktop-all or mechanistic extra to execute SBML models; safe structural inspection remains available."
        )
    if not optional_modules["fmpy"]:
        recommended_checks.append(
            "Install the desktop-all or fmi extra only when trusted FMU execution is needed; static inspection remains available."
        )
    ollama_path = shutil.which("ollama")
    if ollama_path is None:
        recommended_checks.append("Install Ollama if you want local AI analysis.")

    return _ok(
        {
            "sdk_version": env.sdk_version,
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "cwd": Path.cwd(),
            "iints_python_env": os.getenv("IINTS_PYTHON"),
            "research_only": True,
            "medical_device": False,
            "optional_modules": optional_modules,
            "ollama_on_path": ollama_path is not None,
            "ollama_path": ollama_path,
            "recommended_checks": recommended_checks,
        }
    )


def _update_info(_args: argparse.Namespace) -> int:
    return _ok(asdict(get_desktop_update_info()))


def _image_data_url(path: Path) -> str | None:
    try:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:image/png;base64,{payload}"


def _molecules(_args: argparse.Namespace) -> int:
    from iints_desktop.molecules import list_molecule_assets, pae_html_path

    molecules = []
    for molecule in list_molecule_assets():
        pae_path = pae_html_path(molecule.pae_target) if molecule.pae_target else None
        molecules.append(
            {
                "key": molecule.key,
                "title": molecule.title,
                "uniprot_id": molecule.uniprot_id,
                "image_path": molecule.image_path,
                "image_data_url": _image_data_url(molecule.image_path),
                "structure_path": molecule.structure_path,
                "explanation": molecule.explanation,
                "sdk_link": molecule.sdk_link,
                "pae_target": molecule.pae_target,
                "pae_note": molecule.pae_note,
                "pae_path": pae_path,
                "pae_exists": bool(pae_path and pae_path.exists()),
            }
        )
    return _ok({"molecules": molecules})


def _evidence_connectors(_args: argparse.Namespace) -> int:
    connectors = [asdict(connector) for connector in list_evidence_connectors()]
    return _ok(
        {
            "connectors": connectors,
            "research_only": True,
            "medical_device": False,
        }
    )


def _genomics_sim(args: argparse.Namespace) -> int:
    from iints.research.genomics_engine import GenomicsEngine

    html_path, metadata = GenomicsEngine.run_multi_scale_simulation(
        gene=args.gene,
        variant=args.variant,
        out_dir=Path(args.output_dir).expanduser().resolve() / "structural",
        duration_minutes=max(60, min(int(args.duration_minutes), 24 * 60)),
    )
    return _ok(
        {
            "html_path": html_path,
            "output_dir": html_path.parent,
            "metadata": metadata,
            "research_only": True,
            "medical_device": False,
        }
    )


def _tissue_stress(args: argparse.Namespace) -> int:
    from iints.research.tissue_stressor import TissueStressor

    muscle_scalar = max(0.0, min(float(args.muscle_percent), 100.0)) / 100.0
    liver_scalar = max(0.0, min(float(args.liver_percent), 100.0)) / 100.0
    # TissueStressor uses Rich for CLI feedback. The Tauri bridge must emit
    # exactly one JSON object on stdout, so capture incidental console text.
    with contextlib.redirect_stdout(io.StringIO()):
        html_path, metadata = TissueStressor.run_stress_test(
            muscle_scalar=muscle_scalar,
            liver_scalar=liver_scalar,
            output_dir=Path(args.output_dir).expanduser().resolve() / "structural",
        )
    return _ok(
        {
            "html_path": html_path,
            "output_dir": html_path.parent,
            "metadata": metadata,
            "research_only": True,
            "medical_device": False,
        }
    )


def _mechanistic_status(_args: argparse.Namespace) -> int:
    from iints.research.mechanistic_models import roadrunner_status

    return _ok({**roadrunner_status(), "inspection_available": True, "research_only": True})


def _mechanistic_inspect(args: argparse.Namespace) -> int:
    from iints.research.mechanistic_models import inspect_sbml_model, sbml_summary_payload

    summary = inspect_sbml_model(Path(args.model))
    return _ok(
        {
            "summary": sbml_summary_payload(summary),
            "research_only": True,
            "medical_device": False,
            "schema_validation_performed": False,
        }
    )


def _mechanistic_run(args: argparse.Namespace) -> int:
    from iints.research.mechanistic_models import run_sbml_model

    result = run_sbml_model(
        Path(args.model),
        Path(args.output_dir),
        start=float(args.start),
        end=float(args.end),
        points=int(args.points),
        variables=args.variable or (),
        source_url=args.source_url,
        model_license=args.model_license,
    )
    return _ok(
        {
            **asdict(result),
            "research_only": True,
            "medical_device": False,
        }
    )


def _cross_scale_status(_args: argparse.Namespace) -> int:
    from iints.research.cellml_models import opencor_status
    from iints.research.copasi_models import copasi_status
    from iints.research.fmi_models import fmpy_status

    return _ok(
        {
            "copasi": copasi_status(),
            "opencor": opencor_status(),
            "fmpy": fmpy_status(),
            "static_inspection": {"copasi": True, "cellml": True, "fmu": True},
            "bindingdb": {"available": True, "network_required": True, "verified_tls": True},
            "research_only": True,
            "medical_device": False,
        }
    )


def _copasi_inspect(args: argparse.Namespace) -> int:
    from iints.research.copasi_models import copasi_summary_payload, inspect_copasi_model

    summary = inspect_copasi_model(Path(args.model))
    return _ok({"summary": copasi_summary_payload(summary), "research_only": True, "medical_device": False})


def _copasi_run(args: argparse.Namespace) -> int:
    from iints.research.copasi_models import run_copasi_model

    result = run_copasi_model(
        Path(args.model),
        Path(args.output_dir),
        scheduled_task=args.task,
        timeout_seconds=int(args.timeout_seconds),
        allow_external_execution=bool(args.allow_external_execution),
    )
    return _ok({**asdict(result), "research_only": True, "medical_device": False})


def _cellml_inspect(args: argparse.Namespace) -> int:
    from iints.research.cellml_models import cellml_summary_payload, inspect_cellml_model

    summary = inspect_cellml_model(Path(args.model))
    return _ok({"summary": cellml_summary_payload(summary), "research_only": True, "medical_device": False})


def _cellml_validate(args: argparse.Namespace) -> int:
    from iints.research.cellml_models import validate_cellml_model

    result = validate_cellml_model(
        Path(args.model),
        Path(args.output_dir),
        timeout_seconds=int(args.timeout_seconds),
    )
    return _ok({**asdict(result), "research_only": True, "medical_device": False})


def _fmi_inspect(args: argparse.Namespace) -> int:
    from iints.research.fmi_models import fmu_summary_payload, inspect_fmu_model

    summary = inspect_fmu_model(Path(args.model))
    return _ok({"summary": fmu_summary_payload(summary), "research_only": True, "medical_device": False})


def _fmi_run(args: argparse.Namespace) -> int:
    from iints.research.fmi_models import run_fmu_model

    result = run_fmu_model(
        Path(args.model),
        Path(args.output_dir),
        start=float(args.start),
        end=float(args.end),
        output_interval=float(args.output_interval),
        variables=args.variable or (),
        timeout_seconds=int(args.timeout_seconds),
        allow_native_execution=bool(args.trust_native_code),
    )
    return _ok({**asdict(result), "research_only": True, "medical_device": False})


def _binding_query(args: argparse.Namespace) -> int:
    from iints.research.binding_evidence import query_bindingdb_uniprot

    result = query_bindingdb_uniprot(
        args.uniprot,
        Path(args.output_dir),
        cutoff_nm=int(args.cutoff_nm),
        max_records=int(args.max_records),
        timeout_seconds=int(args.timeout_seconds),
    )
    return _ok({**asdict(result), "research_only": True, "medical_device": False})


def _run(args: argparse.Namespace) -> int:
    result = run_demo_preset(
        output_dir=Path(args.output_dir),
        desktop_preset_key=args.workflow_key,
        seed=int(args.seed),
    )
    return _ok(
        {
            "run_id": result.run_id,
            "workflow_title": result.workflow_title,
            "preset_name": result.preset_name,
            "seed": result.seed,
            "output_dir": result.output_dir,
            "results_csv": result.results_csv,
            "report_pdf": result.report_pdf,
            "config_path": result.config_path,
            "summary": result.summary,
        }
    )


def _preview(args: argparse.Namespace) -> int:
    preview = load_results_preview(Path(args.csv), max_rows=int(args.max_rows))
    return _ok(
        {
            "csv_path": preview.csv_path,
            "row_count": preview.row_count,
            "columns": preview.columns,
            "rows": preview.rows,
            "metrics": preview.metrics,
            "graph_path": preview.graph_path,
        }
    )


def _history(args: argparse.Namespace) -> int:
    limit = max(1, min(int(args.limit), 200))
    entries = read_run_history(Path(args.output_dir), limit=limit)
    return _ok(
        {
            "output_dir": Path(args.output_dir).expanduser(),
            "limit": limit,
            "history": entries,
        }
    )


def _mdmp_certify(args: argparse.Namespace) -> int:
    from iints_desktop.mdmp import create_desktop_mdmp_certificate

    result = create_desktop_mdmp_certificate(
        Path(args.csv),
        quick=not bool(args.full),
        quick_rows=max(1, int(args.quick_rows)),
    )
    return _ok(
        {
            "certificate_path": result.certificate_path,
            "report_path": result.report_path,
            "public_key_path": result.public_key_path,
            "grade": result.grade,
            "compliance_score": result.compliance_score,
            "row_count": result.row_count,
            "quick": not bool(args.full),
        }
    )


def _academic_bundle(args: argparse.Namespace) -> int:
    from iints.research.academic_bundle import build_academic_bundle

    result = build_academic_bundle(
        Path(args.run_dir),
        title=args.title,
        description=args.description,
        creator_name=args.creator,
        creator_orcid=args.orcid,
        license_id=args.license_id,
        source_ids=args.source_id or (),
    )
    return _ok(
        {
            **asdict(result),
            "research_only": True,
            "medical_device": False,
        }
    )


def _ai_check(args: argparse.Namespace) -> int:
    status = check_local_ai(model=args.model, host=args.host)
    return _ok(
        {
            "available": status.available,
            "message": status.message,
            "resolved_model": status.resolved_model,
        }
    )


def _ai_models(args: argparse.Namespace) -> int:
    return _ok({"models": list_local_ai_models(host=args.host)})


def _ai_start(args: argparse.Namespace) -> int:
    result = start_local_ai_stack(
        model=args.model,
        host=args.host,
        pull_missing_model=not bool(args.no_pull),
    )
    return _ok(
        {
            "available": result.available,
            "message": result.message,
            "resolved_model": result.resolved_model,
            "started_process": result.started_process,
            "pulled_model": result.pulled_model,
        }
    )


def _ai_ask(args: argparse.Namespace) -> int:
    answer = ask_local_ai(
        question=args.question,
        model=args.model,
        host=args.host,
        result_csv=args.csv,
    )
    return _ok(
        {
            "answer": answer.answer,
            "model": answer.model,
            "context_used": answer.context_used,
            "policy_violations": list(answer.policy_violations),
            "policy_warnings": list(answer.policy_warnings),
            "policy_action": answer.policy_action,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m iints_desktop.tauri_bridge",
        description="JSON bridge used by the experimental Tauri desktop shell.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    status = subcommands.add_parser("status")
    status.set_defaults(func=_status)

    workflows = subcommands.add_parser("workflows")
    workflows.set_defaults(func=_workflows)

    diagnostics = subcommands.add_parser("diagnostics")
    diagnostics.set_defaults(func=_diagnostics)

    update_info = subcommands.add_parser("update-info")
    update_info.set_defaults(func=_update_info)

    molecules = subcommands.add_parser("molecules")
    molecules.set_defaults(func=_molecules)

    evidence = subcommands.add_parser("evidence-connectors")
    evidence.set_defaults(func=_evidence_connectors)

    genomics = subcommands.add_parser("genomics-sim")
    genomics.add_argument("--gene", default="INSR")
    genomics.add_argument("--variant", required=True)
    genomics.add_argument("--output-dir", required=True)
    genomics.add_argument("--duration-minutes", type=int, default=360)
    genomics.set_defaults(func=_genomics_sim)

    tissue = subcommands.add_parser("tissue-stress")
    tissue.add_argument("--muscle-percent", type=float, default=30.0)
    tissue.add_argument("--liver-percent", type=float, default=100.0)
    tissue.add_argument("--output-dir", required=True)
    tissue.set_defaults(func=_tissue_stress)

    mechanistic_status = subcommands.add_parser("mechanistic-status")
    mechanistic_status.set_defaults(func=_mechanistic_status)

    mechanistic_inspect = subcommands.add_parser("mechanistic-inspect")
    mechanistic_inspect.add_argument("--model", required=True)
    mechanistic_inspect.set_defaults(func=_mechanistic_inspect)

    mechanistic_run = subcommands.add_parser("mechanistic-run")
    mechanistic_run.add_argument("--model", required=True)
    mechanistic_run.add_argument("--output-dir", required=True)
    mechanistic_run.add_argument("--start", type=float, default=0.0)
    mechanistic_run.add_argument("--end", type=float, default=1440.0)
    mechanistic_run.add_argument("--points", type=int, default=289)
    mechanistic_run.add_argument("--variable", action="append", default=[])
    mechanistic_run.add_argument("--source-url", default=None)
    mechanistic_run.add_argument("--model-license", default="NOASSERTION")
    mechanistic_run.set_defaults(func=_mechanistic_run)

    cross_scale_status = subcommands.add_parser("cross-scale-status")
    cross_scale_status.set_defaults(func=_cross_scale_status)

    copasi_inspect = subcommands.add_parser("copasi-inspect")
    copasi_inspect.add_argument("--model", required=True)
    copasi_inspect.set_defaults(func=_copasi_inspect)

    copasi_run = subcommands.add_parser("copasi-run")
    copasi_run.add_argument("--model", required=True)
    copasi_run.add_argument("--output-dir", required=True)
    copasi_run.add_argument("--task", default=None)
    copasi_run.add_argument("--timeout-seconds", type=int, default=900)
    copasi_run.add_argument("--allow-external-execution", action="store_true")
    copasi_run.set_defaults(func=_copasi_run)

    cellml_inspect = subcommands.add_parser("cellml-inspect")
    cellml_inspect.add_argument("--model", required=True)
    cellml_inspect.set_defaults(func=_cellml_inspect)

    cellml_validate = subcommands.add_parser("cellml-validate")
    cellml_validate.add_argument("--model", required=True)
    cellml_validate.add_argument("--output-dir", required=True)
    cellml_validate.add_argument("--timeout-seconds", type=int, default=120)
    cellml_validate.set_defaults(func=_cellml_validate)

    fmi_inspect = subcommands.add_parser("fmi-inspect")
    fmi_inspect.add_argument("--model", required=True)
    fmi_inspect.set_defaults(func=_fmi_inspect)

    fmi_run = subcommands.add_parser("fmi-run")
    fmi_run.add_argument("--model", required=True)
    fmi_run.add_argument("--output-dir", required=True)
    fmi_run.add_argument("--start", type=float, default=0.0)
    fmi_run.add_argument("--end", type=float, default=60.0)
    fmi_run.add_argument("--output-interval", type=float, default=0.1)
    fmi_run.add_argument("--variable", action="append", default=[])
    fmi_run.add_argument("--timeout-seconds", type=int, default=300)
    fmi_run.add_argument("--trust-native-code", action="store_true")
    fmi_run.set_defaults(func=_fmi_run)

    binding_query = subcommands.add_parser("binding-query")
    binding_query.add_argument("--uniprot", default="P06213")
    binding_query.add_argument("--output-dir", required=True)
    binding_query.add_argument("--cutoff-nm", type=int, default=10_000)
    binding_query.add_argument("--max-records", type=int, default=5_000)
    binding_query.add_argument("--timeout-seconds", type=int, default=30)
    binding_query.set_defaults(func=_binding_query)

    run = subcommands.add_parser("run")
    run.add_argument("--workflow-key", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--seed", type=int, default=42)
    run.set_defaults(func=_run)

    preview = subcommands.add_parser("preview")
    preview.add_argument("--csv", required=True)
    preview.add_argument("--max-rows", type=int, default=80)
    preview.set_defaults(func=_preview)

    history = subcommands.add_parser("history")
    history.add_argument("--output-dir", required=True)
    history.add_argument("--limit", type=int, default=25)
    history.set_defaults(func=_history)

    mdmp = subcommands.add_parser("mdmp-certify")
    mdmp.add_argument("--csv", required=True)
    mdmp.add_argument("--quick-rows", type=int, default=5000)
    mdmp.add_argument("--full", action="store_true")
    mdmp.set_defaults(func=_mdmp_certify)

    academic = subcommands.add_parser("academic-bundle")
    academic.add_argument("--run-dir", required=True)
    academic.add_argument("--title", default=None)
    academic.add_argument("--description", default=None)
    academic.add_argument("--creator", default=None)
    academic.add_argument("--orcid", default=None)
    academic.add_argument("--license", dest="license_id", default="NOASSERTION")
    academic.add_argument("--source-id", action="append", default=[])
    academic.set_defaults(func=_academic_bundle)

    ai_check = subcommands.add_parser("ai-check")
    ai_check.add_argument("--model", default="ministral-3:8b")
    ai_check.add_argument("--host", default=None)
    ai_check.set_defaults(func=_ai_check)

    ai_models = subcommands.add_parser("ai-models")
    ai_models.add_argument("--host", default=None)
    ai_models.set_defaults(func=_ai_models)

    ai_start = subcommands.add_parser("ai-start")
    ai_start.add_argument("--model", default="ministral-3:8b")
    ai_start.add_argument("--host", default=None)
    ai_start.add_argument("--no-pull", action="store_true")
    ai_start.set_defaults(func=_ai_start)

    ai_ask = subcommands.add_parser("ai-ask")
    ai_ask.add_argument("--question", required=True)
    ai_ask.add_argument("--model", default="ministral-3:8b")
    ai_ask.add_argument("--host", default=None)
    ai_ask.add_argument("--csv", default=None)
    ai_ask.set_defaults(func=_ai_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        return _fail(str(exc), details=traceback.format_exc())


if __name__ == "__main__":
    raise SystemExit(main())
