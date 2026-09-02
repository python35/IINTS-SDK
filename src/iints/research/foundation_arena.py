"""Evidence-backed comparison of CGM representation models.

The arena deliberately contains no built-in model scores. Every displayed
number must come from a supplied evaluation artifact, and models are ranked
only when they were evaluated with the same benchmark contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


FOUNDATION_ARENA_SCHEMA = "iints.foundation-arena.evaluation.v1"
_DIRECTIONS = {"higher", "lower"}


@dataclass(frozen=True)
class ArenaMetric:
    """One measured metric with enough metadata to interpret its direction."""

    value: float
    unit: str
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"metric direction must be one of {sorted(_DIRECTIONS)}, "
                f"got {self.direction!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelArenaMetrics:
    """A model evaluation loaded from a traceable benchmark artifact."""

    model_name: str
    architecture: str
    latent_dimension: int
    implementation_kind: str
    checkpoint_sha256: str
    benchmark_id: str
    task: str
    cohort_id: str
    split_id: str
    split_strategy: str
    group_disjoint: bool
    n_groups: int
    n_samples: int
    seed: int
    metrics: Mapping[str, ArenaMetric]
    source_path: Path
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metrics"] = {
            name: metric.to_dict() for name, metric in self.metrics.items()
        }
        data["source_path"] = str(self.source_path)
        return data


@dataclass(frozen=True)
class FoundationArenaReport:
    """Aggregate comparison built entirely from supplied evidence artifacts."""

    total_models_evaluated: int
    models: Sequence[ModelArenaMetrics]
    benchmark_id: str
    comparable: bool
    metric_leaders: Mapping[str, str]
    report_md_path: Path
    summary_json_path: Path
    comparison_csv_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_models_evaluated": self.total_models_evaluated,
            "models": [model.to_dict() for model in self.models],
            "benchmark_id": self.benchmark_id,
            "comparable": self.comparable,
            "metric_leaders": dict(self.metric_leaders),
            "report_md_path": str(self.report_md_path),
            "summary_json_path": str(self.summary_json_path),
            "comparison_csv_path": str(self.comparison_csv_path),
        }


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"foundation arena artifact requires object field {key!r}")
    return value


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"foundation arena artifact requires non-empty field {key!r}")
    return value.strip()


def _validate_sha256(value: str, field: str) -> str:
    normalized = value.lower().strip()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256 digest")
    return normalized


def load_foundation_evaluation(path: Path | str) -> ModelArenaMetrics:
    """Load and validate one local evaluation artifact.

    The artifact's own hash is calculated here; it is never trusted from input.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"foundation evaluation artifact not found: {source}")
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid foundation evaluation JSON: {source}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("foundation evaluation root must be a JSON object")
    if payload.get("schema_version") != FOUNDATION_ARENA_SCHEMA:
        raise ValueError(
            f"unsupported foundation evaluation schema: {payload.get('schema_version')!r}; "
            f"expected {FOUNDATION_ARENA_SCHEMA!r}"
        )

    model = _required_mapping(payload, "model")
    evaluation = _required_mapping(payload, "evaluation")
    metric_payload = _required_mapping(payload, "metrics")
    if not metric_payload:
        raise ValueError("foundation evaluation must contain at least one measured metric")

    metrics: dict[str, ArenaMetric] = {}
    for name, raw_metric in metric_payload.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("metric names must be non-empty strings")
        if not isinstance(raw_metric, Mapping):
            raise ValueError(f"metric {name!r} must be an object")
        try:
            value = float(raw_metric["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"metric {name!r} requires a numeric value") from exc
        metrics[name] = ArenaMetric(
            value=value,
            unit=str(raw_metric.get("unit", "")),
            direction=_required_text(raw_metric, "direction"),
        )

    try:
        latent_dimension = int(model["latent_dimension"])
        n_groups = int(evaluation["n_groups"])
        n_samples = int(evaluation["n_samples"])
        seed = int(evaluation["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "model.latent_dimension and evaluation n_groups/n_samples/seed "
            "must be integers"
        ) from exc
    if latent_dimension <= 0 or n_groups <= 0 or n_samples <= 0:
        raise ValueError("latent_dimension, n_groups, and n_samples must be positive")

    group_disjoint = evaluation.get("group_disjoint")
    if not isinstance(group_disjoint, bool):
        raise ValueError("evaluation.group_disjoint must be a boolean")

    return ModelArenaMetrics(
        model_name=_required_text(model, "name"),
        architecture=_required_text(model, "architecture"),
        latent_dimension=latent_dimension,
        implementation_kind=_required_text(model, "implementation_kind"),
        checkpoint_sha256=_validate_sha256(
            _required_text(model, "checkpoint_sha256"), "model.checkpoint_sha256"
        ),
        benchmark_id=_required_text(evaluation, "benchmark_id"),
        task=_required_text(evaluation, "task"),
        cohort_id=_required_text(evaluation, "cohort_id"),
        split_id=_required_text(evaluation, "split_id"),
        split_strategy=_required_text(evaluation, "split_strategy"),
        group_disjoint=group_disjoint,
        n_groups=n_groups,
        n_samples=n_samples,
        seed=seed,
        metrics=metrics,
        source_path=source,
        source_sha256=sha256(raw).hexdigest(),
    )


def _ensure_comparable(models: Sequence[ModelArenaMetrics]) -> str:
    benchmark_ids = {model.benchmark_id for model in models}
    if len(benchmark_ids) != 1:
        detail = ", ".join(sorted(benchmark_ids))
        raise ValueError(
            "foundation arena artifacts are not comparable: benchmark_id differs "
            f"({detail}). Evaluate every model on the same cohort, task, and split."
        )
    if not all(model.group_disjoint for model in models):
        offenders = ", ".join(
            model.model_name for model in models if not model.group_disjoint
        )
        raise ValueError(
            "foundation arena ranking requires group-disjoint evaluation; "
            f"non-compliant artifacts: {offenders}"
        )
    return next(iter(benchmark_ids))


def _common_metrics(models: Sequence[ModelArenaMetrics]) -> list[str]:
    common = set(models[0].metrics)
    for model in models[1:]:
        common.intersection_update(model.metrics)
    valid: list[str] = []
    for name in sorted(common):
        signatures = {
            (model.metrics[name].unit, model.metrics[name].direction) for model in models
        }
        if len(signatures) != 1:
            raise ValueError(
                f"metric {name!r} has inconsistent unit or direction across artifacts"
            )
        valid.append(name)
    if not valid:
        raise ValueError("foundation arena artifacts have no comparable metric in common")
    return valid


def _leader(models: Sequence[ModelArenaMetrics], metric_name: str) -> str:
    direction = models[0].metrics[metric_name].direction
    key = lambda model: model.metrics[metric_name].value
    winner = max(models, key=key) if direction == "higher" else min(models, key=key)
    return winner.model_name


def _format_metric(metric: ArenaMetric) -> str:
    unit = f" {metric.unit}" if metric.unit else ""
    return f"{metric.value:.4g}{unit}"


def run_foundation_model_arena(
    output_dir: Path | str = "results/foundation_arena",
    evaluation_artifacts: Sequence[Path | str] | None = None,
    *,
    n_benchmark_trials: int | None = None,
) -> FoundationArenaReport:
    """Compare real model evaluations under one benchmark contract.

    ``n_benchmark_trials`` is accepted only as a migration aid. It cannot
    generate evidence and has no effect; callers must supply artifacts.
    """

    artifacts = list(evaluation_artifacts or [])
    if not artifacts:
        suffix = (
            f" --n-trials={n_benchmark_trials} cannot create measured results."
            if n_benchmark_trials is not None
            else ""
        )
        raise ValueError(
            "foundation-arena requires one or more --result evaluation artifacts;"
            + suffix
        )

    models = [load_foundation_evaluation(path) for path in artifacts]
    model_names = [model.model_name for model in models]
    if len(set(model_names)) != len(model_names):
        raise ValueError("each foundation arena artifact must identify a unique model")

    benchmark_id = _ensure_comparable(models)
    metric_names = _common_metrics(models)
    leaders = {name: _leader(models, name) for name in metric_names}

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "foundation_arena_comparison.csv"
    report_path = out_dir / "FOUNDATION_MODEL_ARENA_REPORT.md"
    summary_path = out_dir / "foundation_arena_summary.json"

    rows: list[dict[str, Any]] = []
    for model in models:
        row: dict[str, Any] = {
            "model_name": model.model_name,
            "architecture": model.architecture,
            "latent_dimension": model.latent_dimension,
            "implementation_kind": model.implementation_kind,
            "checkpoint_sha256": model.checkpoint_sha256,
            "benchmark_id": model.benchmark_id,
            "cohort_id": model.cohort_id,
            "split_id": model.split_id,
            "group_disjoint": model.group_disjoint,
            "n_groups": model.n_groups,
            "n_samples": model.n_samples,
            "seed": model.seed,
            "source_path": str(model.source_path),
            "source_sha256": model.source_sha256,
        }
        for name in metric_names:
            row[name] = model.metrics[name].value
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    header = "| Model | Architecture | " + " | ".join(metric_names) + " |"
    separator = "| :--- | :--- | " + " | ".join(":---:" for _ in metric_names) + " |"
    table_rows = []
    for model in models:
        values = " | ".join(_format_metric(model.metrics[name]) for name in metric_names)
        table_rows.append(f"| {model.model_name} | {model.architecture} | {values} |")
    leader_rows = "\n".join(
        f"- `{name}` ({models[0].metrics[name].direction} is better): **{leader}**"
        for name, leader in leaders.items()
    )
    provenance_rows = "\n".join(
        f"- **{model.model_name}:** `{model.source_path}`; "
        f"SHA-256 `{model.source_sha256}`; checkpoint `{model.checkpoint_sha256}`"
        for model in models
    )
    report_path.write_text(
        "\n".join(
            [
                "# CGM Foundation Model Arena",
                "",
                "This report contains only values loaded from supplied evaluation artifacts. "
                "It does not contain built-in literature values or synthetic benchmark scores.",
                "",
                f"- **Benchmark contract:** `{benchmark_id}`",
                f"- **Task:** `{models[0].task}`",
                f"- **Cohort:** `{models[0].cohort_id}`",
                f"- **Split:** `{models[0].split_id}` ({models[0].split_strategy})",
                "- **Leakage guard:** group-disjoint for every model",
                "",
                "## Measured Results",
                "",
                header,
                separator,
                *table_rows,
                "",
                "## Metric Leaders",
                "",
                leader_rows,
                "",
                "Leaders are descriptive for this exact benchmark only. They are not claims of "
                "clinical superiority or general performance.",
                "",
                "## Provenance",
                "",
                provenance_rows,
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = FoundationArenaReport(
        total_models_evaluated=len(models),
        models=models,
        benchmark_id=benchmark_id,
        comparable=True,
        metric_leaders=leaders,
        report_md_path=report_path,
        summary_json_path=summary_path,
        comparison_csv_path=csv_path,
    )
    summary_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


__all__ = [
    "FOUNDATION_ARENA_SCHEMA",
    "ArenaMetric",
    "ModelArenaMetrics",
    "FoundationArenaReport",
    "load_foundation_evaluation",
    "run_foundation_model_arena",
]
