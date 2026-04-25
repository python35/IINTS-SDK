from __future__ import annotations

from typing import Any

from .api import create_patient_app
from .edge_ops import create_edge_bundle, export_edge_setup, summarize_edge_workspace, write_edge_update_script
from .long_study import (
    create_edge_study_snapshot,
    export_edge_study_archive,
    load_edge_long_study_config,
    render_edge_long_study_config_template,
    run_edge_long_study,
)
from .runtime import (
    LivePatientDaemon,
    PatientRuntimeConfig,
    PatientRuntimeStore,
    get_runtime_scenario_profile,
    is_process_alive,
    list_runtime_scenario_profiles,
    load_runtime_status,
)
from .uno_q import export_uno_q_bridge


def run_edge_benchmark(*args: Any, **kwargs: Any) -> Any:
    from .edge_benchmark import run_edge_benchmark as _run_edge_benchmark

    return _run_edge_benchmark(*args, **kwargs)


__all__ = [
    "create_edge_bundle",
    "export_edge_setup",
    "summarize_edge_workspace",
    "write_edge_update_script",
    "render_edge_long_study_config_template",
    "load_edge_long_study_config",
    "run_edge_long_study",
    "create_edge_study_snapshot",
    "export_edge_study_archive",
    "create_patient_app",
    "run_edge_benchmark",
    "export_uno_q_bridge",
    "LivePatientDaemon",
    "PatientRuntimeConfig",
    "PatientRuntimeStore",
    "get_runtime_scenario_profile",
    "list_runtime_scenario_profiles",
    "load_runtime_status",
    "is_process_alive",
]
