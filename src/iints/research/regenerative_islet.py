from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


_RESOURCE_NAME = "regenerative_protein_panels.json"
_UNIPROT_ACCESSION = re.compile(r"^[A-Z0-9]{6,10}$")


@dataclass(frozen=True)
class RegenerativeProteinTarget:
    """One protein target in a research-only evidence panel."""

    gene_symbol: str
    uniprot_accession: str
    role: str
    evidence_requirements: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegenerativeProteinPanel:
    """A multi-protein panel tied to one explicit biological question."""

    key: str
    question: str
    targets: tuple[RegenerativeProteinTarget, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegenerativeEvidencePlan:
    """Evidence requests for a panel, without automated biological scoring."""

    panel: RegenerativeProteinPanel
    source_matrix: Mapping[str, tuple[str, ...]]
    scientific_boundaries: Mapping[str, bool]
    forbidden_inferences: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel": self.panel.to_dict(),
            "source_matrix": {
                key: list(value) for key, value in self.source_matrix.items()
            },
            "scientific_boundaries": dict(self.scientific_boundaries),
            "forbidden_inferences": list(self.forbidden_inferences),
        }


@dataclass(frozen=True)
class RegenerativeComparisonResult:
    """Paths and high-level status for one descriptive protein comparison."""

    output_dir: Path
    comparison_csv: Path
    report_json: Path
    report_md: Path
    figure_html: Path | None
    target_count: int
    observed_target_count: int
    status: str


_COMPARISON_REQUIRED_COLUMNS = {
    "gene_symbol",
    "group",
    "sample_id",
    "value",
    "unit",
    "scale",
    "source_id",
}
_SUPPORTED_SCALES = {"linear", "log2"}


def _load_payload(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        resource = resources.files("iints").joinpath("data").joinpath(_RESOURCE_NAME)
        text = resource.read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("regenerative protein panel registry must be a JSON object")
    return payload


def _validate_boundaries(payload: Mapping[str, Any]) -> dict[str, bool]:
    raw = payload.get("scientific_boundaries")
    if not isinstance(raw, Mapping):
        raise ValueError("scientific_boundaries must be an object")
    if not all(isinstance(value, bool) for value in raw.values()):
        raise ValueError("scientific_boundaries values must be JSON booleans")
    boundaries = {str(key): value for key, value in raw.items()}
    mandatory = {
        "research_only": True,
        "automatic_physiology_mapping_allowed": False,
        "alphafold_is_structure_confidence_only": True,
        "network_scores_are_not_causal_effects": True,
        "protein_markers_are_not_patient_outcomes": True,
        "unknown_or_conflicting_evidence_fails_closed": True,
    }
    for key, expected in mandatory.items():
        if boundaries.get(key) is not expected:
            raise ValueError(
                f"scientific boundary {key!r} must remain {expected!r}"
            )
    return boundaries


def _validate_sources(payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    raw = payload.get("evidence_sources")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("evidence_sources must be a non-empty object")
    sources: dict[str, tuple[str, ...]] = {}
    for requirement, providers in raw.items():
        if not isinstance(providers, list) or not providers:
            raise ValueError(
                f"evidence source {requirement!r} must list at least one provider"
            )
        normalized = tuple(str(provider).strip() for provider in providers)
        if not all(normalized):
            raise ValueError(f"evidence source {requirement!r} contains an empty provider")
        sources[str(requirement)] = normalized
    return sources


def _parse_target(
    raw: Mapping[str, Any],
    *,
    known_requirements: set[str],
) -> RegenerativeProteinTarget:
    symbol = str(raw.get("gene_symbol", "")).strip()
    accession = str(raw.get("uniprot_accession", "")).strip()
    role = str(raw.get("role", "")).strip()
    requirements_raw = raw.get("evidence_requirements")

    if not symbol or not role:
        raise ValueError("each regenerative target needs a gene_symbol and role")
    if not _UNIPROT_ACCESSION.fullmatch(accession):
        raise ValueError(f"invalid UniProt accession for {symbol}: {accession!r}")
    if raw.get("direct_physiology_mapping", False):
        raise ValueError(
            f"direct physiology mapping is forbidden for regenerative target {symbol}"
        )
    if not isinstance(requirements_raw, list) or not requirements_raw:
        raise ValueError(f"target {symbol} needs evidence_requirements")

    requirements = tuple(str(value) for value in requirements_raw)
    unknown = sorted(set(requirements) - known_requirements)
    if unknown:
        raise ValueError(
            f"target {symbol} references unknown evidence requirements: {unknown}"
        )
    if len(requirements) != len(set(requirements)):
        raise ValueError(f"target {symbol} has duplicate evidence requirements")

    return RegenerativeProteinTarget(
        gene_symbol=symbol,
        uniprot_accession=accession,
        role=role,
        evidence_requirements=requirements,
    )


def load_regenerative_protein_panels(
    path: Path | None = None,
) -> dict[str, RegenerativeProteinPanel]:
    """Load and validate the bundled regenerative-islet protein panels."""

    payload = _load_payload(path)
    _validate_boundaries(payload)
    sources = _validate_sources(payload)
    raw_panels = payload.get("panels")
    if not isinstance(raw_panels, Mapping) or not raw_panels:
        raise ValueError("panels must be a non-empty object")

    panels: dict[str, RegenerativeProteinPanel] = {}
    for key, raw_panel in raw_panels.items():
        if not isinstance(raw_panel, Mapping):
            raise ValueError(f"panel {key!r} must be an object")
        question = str(raw_panel.get("question", "")).strip()
        raw_targets = raw_panel.get("targets")
        if not question or not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError(f"panel {key!r} needs a question and targets")
        targets = tuple(
            _parse_target(raw, known_requirements=set(sources))
            for raw in raw_targets
            if isinstance(raw, Mapping)
        )
        if len(targets) != len(raw_targets):
            raise ValueError(f"panel {key!r} contains a non-object target")
        symbols = [target.gene_symbol for target in targets]
        accessions = [target.uniprot_accession for target in targets]
        if len(symbols) != len(set(symbols)):
            raise ValueError(f"panel {key!r} contains duplicate gene symbols")
        if len(accessions) != len(set(accessions)):
            raise ValueError(f"panel {key!r} contains duplicate UniProt accessions")
        panel_key = str(key)
        panels[panel_key] = RegenerativeProteinPanel(
            key=panel_key,
            question=question,
            targets=targets,
        )
    return panels


def get_regenerative_protein_panel(
    key: str,
    path: Path | None = None,
) -> RegenerativeProteinPanel:
    panels = load_regenerative_protein_panels(path)
    try:
        return panels[key]
    except KeyError as exc:
        choices = ", ".join(sorted(panels))
        raise KeyError(f"unknown regenerative protein panel {key!r}; choose: {choices}") from exc


def build_regenerative_evidence_plan(
    panel_key: str,
    path: Path | None = None,
) -> RegenerativeEvidencePlan:
    """
    Build a source plan while deliberately refusing automated efficacy scoring.

    The returned object tells callers what evidence must be collected. It does
    not translate protein observations into graft or patient-model parameters.
    Such mappings require an explicit, reviewed assay-to-parameter model.
    """

    payload = _load_payload(path)
    boundaries = _validate_boundaries(payload)
    sources = _validate_sources(payload)
    panel = get_regenerative_protein_panel(panel_key, path)
    requirements = {
        requirement
        for target in panel.targets
        for requirement in target.evidence_requirements
    }
    forbidden = payload.get("forbidden_inferences")
    if not isinstance(forbidden, list) or not forbidden:
        raise ValueError("forbidden_inferences must be a non-empty list")
    return RegenerativeEvidencePlan(
        panel=panel,
        source_matrix={key: sources[key] for key in sorted(requirements)},
        scientific_boundaries=boundaries,
        forbidden_inferences=tuple(str(item) for item in forbidden),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_comparison_frame(path: Path) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"regenerative comparison dataset not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        raise ValueError("comparison dataset must be CSV or Parquet")
    missing = sorted(_COMPARISON_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"comparison dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("comparison dataset is empty")

    data = frame.copy()
    for column in ("gene_symbol", "group", "sample_id", "unit", "scale", "source_id"):
        data[column] = data[column].astype(str).str.strip()
        if (data[column] == "").any():
            raise ValueError(f"comparison column {column!r} contains empty values")
    data["gene_symbol"] = data["gene_symbol"].str.upper()
    data["scale"] = data["scale"].str.lower()
    unsupported = sorted(set(data["scale"]) - _SUPPORTED_SCALES)
    if unsupported:
        raise ValueError(
            f"unsupported measurement scales {unsupported}; use 'linear' or 'log2'"
        )
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    values = data["value"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("comparison values must all be finite numbers")
    if ((data["scale"] == "linear") & (data["value"] < 0.0)).any():
        raise ValueError("linear-scale comparison values must be non-negative")

    duplicate = data.duplicated(
        subset=["gene_symbol", "group", "sample_id"], keep=False
    )
    if duplicate.any():
        rows = data.loc[duplicate, ["gene_symbol", "group", "sample_id"]]
        example = rows.head(3).to_dict(orient="records")
        raise ValueError(
            "each sample must contain one protein-level value per gene and group; "
            f"aggregate peptide rows first. Examples: {example}"
        )
    return data


def _log2_difference(
    test_values: np.ndarray,
    reference_values: np.ndarray,
    *,
    scale: str,
) -> float | None:
    test_median = float(np.median(test_values))
    reference_median = float(np.median(reference_values))
    if scale == "log2":
        return test_median - reference_median
    if test_median <= 0.0 or reference_median <= 0.0:
        return None
    return float(np.log2(test_median / reference_median))


def _bootstrap_log2_interval(
    test_values: np.ndarray,
    reference_values: np.ndarray,
    *,
    scale: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    if samples < 1 or len(test_values) < 3 or len(reference_values) < 3:
        return None, None
    estimates: list[float] = []
    for _ in range(samples):
        test_draw = rng.choice(test_values, size=len(test_values), replace=True)
        reference_draw = rng.choice(
            reference_values, size=len(reference_values), replace=True
        )
        estimate = _log2_difference(test_draw, reference_draw, scale=scale)
        if estimate is not None and np.isfinite(estimate):
            estimates.append(float(estimate))
    if len(estimates) < max(20, samples // 2):
        return None, None
    low, high = np.percentile(np.asarray(estimates), [2.5, 97.5])
    return float(low), float(high)


def _comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Regenerative Islet Protein Comparison",
        "",
        "> Research-only descriptive comparison. This is not evidence of treatment efficacy, "
        "cell-product release, transplant suitability, or clinical safety.",
        "",
        "## Design",
        "",
        f"- Test group: `{report['test_group']}`",
        f"- Reference group: `{report['reference_group']}`",
        f"- Comparison design: `{report['comparison_design']}`",
        f"- Review status: `{report['status']}`",
        f"- Normalization note: {report['normalization_note']}",
        f"- Input SHA-256: `{report['input']['sha256']}`",
        "",
        "## Panel Coverage",
        "",
        "| Panel | Observed | Targets | Coverage | Median absolute log2 difference |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["panel_summaries"]:
        distance = row["median_absolute_log2_difference"]
        distance_text = "n/a" if distance is None else f"{distance:.3f}"
        lines.append(
            f"| `{row['panel']}` | {row['observed_targets']} | {row['target_count']} | "
            f"{row['coverage_pct']:.1f}% | {distance_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Log2 differences are descriptive effect estimates, not proof of biological equivalence.",
            "- A bootstrap interval is emitted only with at least three independent samples per group.",
            "- Missing proteins remain missing and are never imputed by AI.",
            "- Cross-source comparisons require batch-effect and normalization review.",
            "- Function, immune response, tumorigenicity, and graft survival require independent assays.",
            "- No protein result in this bundle changes an IINTS physiological parameter automatically.",
            "",
            "## Warnings",
            "",
        ]
    )
    warnings = report.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- No input-schema warnings; biological review is still required.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `protein_comparison.csv`: target-level descriptive estimates",
            "- `comparison_report.json`: machine-readable provenance and summaries",
            "- `comparison_forest.html`: optional aggregate-only interactive forest plot",
            "",
        ]
    )
    return "\n".join(lines)


def _write_comparison_figure(
    comparison: pd.DataFrame,
    output_path: Path,
    *,
    descriptive_margin_log2: float,
) -> Path | None:
    plotted = comparison.dropna(subset=["log2_difference"]).copy()
    if plotted.empty:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    plotted["label"] = plotted["gene_symbol"] + " | " + plotted["panel"]
    low_error = (
        plotted["log2_difference"] - plotted["ci95_low"]
    ).where(plotted["ci95_low"].notna(), 0.0)
    high_error = (
        plotted["ci95_high"] - plotted["log2_difference"]
    ).where(plotted["ci95_high"].notna(), 0.0)
    figure = go.Figure(
        go.Scatter(
            x=plotted["log2_difference"],
            y=plotted["label"],
            mode="markers",
            marker={"size": 8, "color": "#155e75"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": high_error,
                "arrayminus": low_error,
                "color": "#475569",
                "thickness": 1,
            },
            customdata=np.column_stack(
                [
                    plotted["test_n"],
                    plotted["reference_n"],
                    plotted["unit"],
                    plotted["scale"],
                    plotted["status"],
                ]
            ),
            hovertemplate=(
                "%{y}<br>log2 difference=%{x:.3f}<br>test n=%{customdata[0]}"
                "<br>reference n=%{customdata[1]}<br>unit=%{customdata[2]}"
                "<br>scale=%{customdata[3]}<br>status=%{customdata[4]}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0.0, line_color="#0f172a", line_width=1)
    figure.add_vline(
        x=-descriptive_margin_log2, line_color="#94a3b8", line_dash="dot"
    )
    figure.add_vline(
        x=descriptive_margin_log2, line_color="#94a3b8", line_dash="dot"
    )
    figure.update_layout(
        title="SC-islet versus reference protein abundance",
        xaxis_title="Descriptive log2 difference (test - reference)",
        yaxis_title="Protein and evidence panel",
        template="plotly_white",
        height=max(520, 30 * len(plotted) + 180),
        showlegend=False,
    )
    figure.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    return output_path


def compare_regenerative_islet_proteomics(
    data_path: Path,
    output_dir: Path,
    *,
    test_group: str = "sc_islet",
    reference_group: str = "primary_islet",
    panel_keys: Sequence[str] = (),
    normalization_note: str = "not supplied",
    descriptive_margin_log2: float = 0.5,
    bootstrap_samples: int = 2_000,
    seed: int = 42,
) -> RegenerativeComparisonResult:
    """
    Compare protein-level observations without inferring treatment efficacy.

    Input values must already be normalized within a defensible experimental
    design. The function never converts abundance or structure scores into
    transplant-model parameters.
    """

    if test_group == reference_group:
        raise ValueError("test_group and reference_group must be different")
    if not np.isfinite(descriptive_margin_log2) or descriptive_margin_log2 <= 0.0:
        raise ValueError("descriptive_margin_log2 must be positive and finite")
    if bootstrap_samples < 0 or bootstrap_samples > 100_000:
        raise ValueError("bootstrap_samples must be between 0 and 100000")

    source_path = Path(data_path).expanduser().resolve()
    frame = _load_comparison_frame(source_path)
    available_groups = set(frame["group"])
    missing_groups = {test_group, reference_group} - available_groups
    if missing_groups:
        raise ValueError(
            f"comparison groups not found: {sorted(missing_groups)}; "
            f"available groups: {sorted(available_groups)}"
        )

    all_panels = load_regenerative_protein_panels()
    selected_keys = list(panel_keys) if panel_keys else list(all_panels)
    unknown_panels = sorted(set(selected_keys) - set(all_panels))
    if unknown_panels:
        raise ValueError(f"unknown regenerative panels: {unknown_panels}")
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("panel_keys must not contain duplicates")

    target_rows: list[tuple[str, RegenerativeProteinTarget]] = []
    seen_symbols: set[str] = set()
    for panel_key in selected_keys:
        for target in all_panels[panel_key].targets:
            if target.gene_symbol in seen_symbols:
                raise ValueError(
                    f"target {target.gene_symbol} occurs in more than one selected panel"
                )
            seen_symbols.add(target.gene_symbol)
            target_rows.append((panel_key, target))

    relevant = frame[
        frame["group"].isin({test_group, reference_group})
        & frame["gene_symbol"].isin(seen_symbols)
    ].copy()
    test_sources = sorted(set(relevant.loc[relevant["group"] == test_group, "source_id"]))
    reference_sources = sorted(
        set(relevant.loc[relevant["group"] == reference_group, "source_id"])
    )
    comparison_design = (
        "within-source"
        if test_sources and test_sources == reference_sources
        else "cross-source"
    )

    warnings: list[str] = []
    note = normalization_note.strip() or "not supplied"
    if note == "not supplied":
        warnings.append(
            "No normalization note was supplied; quantitative comparability requires review."
        )
    if comparison_design == "cross-source":
        warnings.append(
            "Test and reference source IDs differ; batch and platform effects may dominate the comparison."
        )
    if "batch_id" not in frame.columns:
        warnings.append(
            "No batch_id column was supplied; batch-aware validation cannot be performed."
        )

    rng = np.random.default_rng(seed)
    comparison_rows: list[dict[str, Any]] = []
    for panel_key, target in target_rows:
        target_frame = relevant[relevant["gene_symbol"] == target.gene_symbol]
        test_frame = target_frame[target_frame["group"] == test_group]
        reference_frame = target_frame[target_frame["group"] == reference_group]
        row: dict[str, Any] = {
            "panel": panel_key,
            "gene_symbol": target.gene_symbol,
            "uniprot_accession": target.uniprot_accession,
            "role": target.role,
            "test_group": test_group,
            "reference_group": reference_group,
            "test_n": int(len(test_frame)),
            "reference_n": int(len(reference_frame)),
            "unit": None,
            "scale": None,
            "test_median": None,
            "reference_median": None,
            "log2_difference": None,
            "ci95_low": None,
            "ci95_high": None,
            "within_descriptive_margin": None,
            "status": "missing",
        }
        if test_frame.empty or reference_frame.empty:
            comparison_rows.append(row)
            continue

        units = sorted(set(target_frame["unit"]))
        scales = sorted(set(target_frame["scale"]))
        if len(units) != 1 or len(scales) != 1:
            raise ValueError(
                f"target {target.gene_symbol} has incompatible units or scales "
                f"across groups: units={units}, scales={scales}"
            )
        unit = units[0]
        scale = scales[0]
        test_values = test_frame["value"].to_numpy(dtype=float)
        reference_values = reference_frame["value"].to_numpy(dtype=float)
        difference = _log2_difference(test_values, reference_values, scale=scale)
        ci_low, ci_high = _bootstrap_log2_interval(
            test_values,
            reference_values,
            scale=scale,
            samples=bootstrap_samples,
            rng=rng,
        )
        if difference is None:
            status = "not_estimable_zero_median"
        elif len(test_values) < 3 or len(reference_values) < 3:
            status = "insufficient_replication"
        else:
            status = "descriptive_estimate"
        row.update(
            {
                "unit": unit,
                "scale": scale,
                "test_median": float(np.median(test_values)),
                "reference_median": float(np.median(reference_values)),
                "log2_difference": difference,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "within_descriptive_margin": (
                    None
                    if difference is None
                    else bool(abs(difference) <= descriptive_margin_log2)
                ),
                "status": status,
            }
        )
        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    panel_summaries: list[dict[str, Any]] = []
    for panel_key in selected_keys:
        panel_frame = comparison[comparison["panel"] == panel_key]
        observed = panel_frame[panel_frame["status"] != "missing"]
        distances = pd.to_numeric(observed["log2_difference"], errors="coerce").abs().dropna()
        panel_summaries.append(
            {
                "panel": panel_key,
                "target_count": int(len(panel_frame)),
                "observed_targets": int(len(observed)),
                "coverage_pct": float(100.0 * len(observed) / max(1, len(panel_frame))),
                "median_absolute_log2_difference": (
                    None if distances.empty else float(distances.median())
                ),
                "insufficient_replication_count": int(
                    (observed["status"] == "insufficient_replication").sum()
                ),
            }
        )

    observed_count = int((comparison["status"] != "missing").sum())
    missing_target_count = int(len(comparison) - observed_count)
    if missing_target_count:
        warnings.append(
            f"{missing_target_count} selected panel targets are missing from one or both groups."
        )
    if observed_count == 0:
        status = "insufficient_data"
        warnings.append("No selected target had observations in both comparison groups.")
    elif warnings or (comparison["status"] == "insufficient_replication").any():
        status = "review_required"
    else:
        status = "ready_for_descriptive_review"

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    comparison_csv = destination / "protein_comparison.csv"
    report_json = destination / "comparison_report.json"
    report_md = destination / "comparison_report.md"
    figure_candidate = destination / "comparison_forest.html"
    comparison.to_csv(comparison_csv, index=False)
    figure_html = _write_comparison_figure(
        comparison,
        figure_candidate,
        descriptive_margin_log2=descriptive_margin_log2,
    )

    report: dict[str, Any] = {
        "schema_version": "iints_regenerative_protein_comparison_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "not_for_treatment_or_cell_product_release": True,
        "automatic_physiology_mapping_performed": False,
        "status": status,
        "test_group": test_group,
        "reference_group": reference_group,
        "comparison_design": comparison_design,
        "normalization_note": note,
        "descriptive_margin_log2": float(descriptive_margin_log2),
        "equivalence_claim_performed": False,
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
        "input": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "row_count": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "test_source_ids": test_sources,
            "reference_source_ids": reference_sources,
        },
        "selected_panels": selected_keys,
        "target_count": int(len(comparison)),
        "observed_target_count": observed_count,
        "panel_summaries": panel_summaries,
        "warnings": warnings,
        "artifacts": {
            "comparison_csv": str(comparison_csv),
            "report_md": str(report_md),
            "figure_html": None if figure_html is None else str(figure_html),
        },
        "interpretation_boundary": (
            "Protein abundance proximity is not proof of beta-cell maturity, function, "
            "immune safety, graft survival, or clinical efficacy."
        ),
    }
    report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_md.write_text(_comparison_markdown(report), encoding="utf-8")
    return RegenerativeComparisonResult(
        output_dir=destination,
        comparison_csv=comparison_csv,
        report_json=report_json,
        report_md=report_md,
        figure_html=figure_html,
        target_count=int(len(comparison)),
        observed_target_count=observed_count,
        status=status,
    )
