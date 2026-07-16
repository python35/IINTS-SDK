from __future__ import annotations

import argparse
import json
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
from iints_desktop.local_ai import (
    ask_local_ai,
    check_local_ai,
    list_local_ai_models,
    start_local_ai_stack,
)
from iints_desktop.results import load_results_preview


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
