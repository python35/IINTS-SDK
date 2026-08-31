from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from iints.research.regenerative_islet import (
    _COMPARISON_REQUIRED_COLUMNS,
    _SUPPORTED_SCALES,
    load_regenerative_protein_panels,
)


@dataclass(frozen=True)
class ProteomicsImportResult:
    """Outcome and metadata from a proteomics dataset standardization."""

    output_path: Path
    row_count: int
    gene_count: int
    sample_count: int
    group_counts: Mapping[str, int]
    source_sha256: str
    output_sha256: str
    metadata_sha256: str | None
    input_format: str
    quality_filters: tuple[str, ...]
    manifest_path: Path
    target_panel_coverage: Mapping[str, tuple[int, int]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sample_metadata(path: Path | str) -> dict[str, dict[str, str]]:
    """
    Load sample annotations mapping sample identifiers to experimental metadata.

    Expected columns in CSV/TSV:
      - sample_id (or sample, name, run)
      - group (e.g. 'sc_islet', 'primary_islet', 'control')
      - batch_id (optional, e.g. 'batch_1')
      - source_id (optional, e.g. 'PXD001539')
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"sample metadata file not found: {source}")

    suffix = source.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        df = pd.read_csv(source, sep="\t")
    elif suffix == ".csv":
        df = pd.read_csv(source)
    elif suffix == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("sample metadata JSON must be a dict mapping sample_id to metadata")
        return {str(k): {str(ik): str(iv) for ik, iv in v.items()} for k, v in raw.items()}
    else:
        raise ValueError("sample metadata file must be CSV, TSV, or JSON")

    # Detect sample_id column
    id_col = None
    for candidate in ("sample_id", "sample", "sample_name", "run", "raw_file", "file_name"):
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(
            f"sample metadata missing sample identifier column; available: {list(df.columns)}"
        )

    # Detect group column
    group_col = None
    for candidate in ("group", "condition", "treatment", "cell_type", "type"):
        if candidate in df.columns:
            group_col = candidate
            break
    if group_col is None:
        raise ValueError(
            f"sample metadata missing group column; available: {list(df.columns)}"
        )

    metadata: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        sample_id = str(row[id_col]).strip()
        if not sample_id or sample_id == "nan":
            continue
        entry: dict[str, str] = {
            "group": str(row[group_col]).strip(),
        }
        for opt_col in ("batch_id", "batch", "replicate", "source_id", "source"):
            if opt_col in df.columns and pd.notna(row[opt_col]):
                key = "batch_id" if "batch" in opt_col or opt_col == "replicate" else "source_id"
                entry[key] = str(row[opt_col]).strip()
        metadata[sample_id] = entry

    if not metadata:
        raise ValueError("no valid sample metadata records found")
    return metadata


def _extract_primary_gene(raw_gene: Any, raw_protein: Any = None) -> str | None:
    """Extract a declared gene symbol without relabelling accessions as genes."""
    if pd.notna(raw_gene) and str(raw_gene).strip() and str(raw_gene).strip() != "nan":
        parts = re.split(r"[;,\s|/]+", str(raw_gene).strip())
        for p in parts:
            clean = p.strip().upper()
            if clean and clean != "NAN":
                return clean
    # Protein accessions require an explicit, versioned mapping resource. Treating
    # e.g. P01308 as a gene symbol would silently corrupt panel coverage.
    _ = raw_protein
    return None


def _resolve_sample_id(sample_raw: str, metadata: Mapping[str, Mapping[str, Any]]) -> str | None:
    if sample_raw in metadata:
        return sample_raw
    clean_name = Path(sample_raw).stem
    if clean_name in metadata:
        return clean_name
    matches = [
        sample_id
        for sample_id in metadata
        if sample_id in sample_raw or sample_raw in sample_id
    ]
    if len(matches) > 1:
        raise ValueError(
            f"sample identifier '{sample_raw}' ambiguously matches metadata samples {sorted(matches)}"
        )
    return matches[0] if matches else None


def _validate_scale(scale: str) -> str:
    normalized = str(scale).strip().lower()
    if normalized not in _SUPPORTED_SCALES:
        raise ValueError(
            f"unsupported proteomics scale '{scale}'; expected one of {sorted(_SUPPORTED_SCALES)}"
        )
    return normalized


def import_maxquant_protein_groups(
    protein_groups_path: Path | str,
    sample_metadata: Path | str | Mapping[str, Mapping[str, Any]],
    *,
    intensity_prefix: str = "LFQ intensity ",
    gene_column: str = "Gene names",
    protein_id_column: str = "Majority protein IDs",
    default_source_id: str = "MaxQuant",
    default_unit: str = "LFQ intensity",
    default_scale: str = "linear",
    filter_contaminants: bool = True,
    filter_reverse: bool = True,
    filter_only_identified_by_site: bool = True,
) -> pd.DataFrame:
    """
    Ingest MaxQuant proteinGroups.txt and map to regenerative comparator contract.
    """
    path = Path(protein_groups_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MaxQuant file not found: {path}")

    default_scale = _validate_scale(default_scale)
    meta = (
        load_sample_metadata(sample_metadata)
        if isinstance(sample_metadata, (str, Path))
        else {str(k): dict(v) for k, v in sample_metadata.items()}
    )

    df = pd.read_csv(path, sep="\t", low_memory=False)

    if filter_contaminants:
        for col in ("Potential contaminant", "Contaminant"):
            if col in df.columns:
                df = df[df[col] != "+"]
    if filter_reverse:
        if "Reverse" in df.columns:
            df = df[df["Reverse"] != "+"]
    if filter_only_identified_by_site:
        for col in ("Only identified by site", "Only identified by site+"):
            if col in df.columns:
                df = df[df[col] != "+"]

    if gene_column not in df.columns and protein_id_column not in df.columns:
        raise ValueError(
            f"neither gene column '{gene_column}' nor protein ID column '{protein_id_column}' found in MaxQuant table"
        )

    # Find sample intensity columns
    intensity_cols: dict[str, str] = {}
    for sample_id in meta:
        exact_col = f"{intensity_prefix}{sample_id}"
        if exact_col in df.columns:
            intensity_cols[sample_id] = exact_col
        elif sample_id in df.columns:
            intensity_cols[sample_id] = sample_id

    if not intensity_cols:
        # Fallback: scan all columns starting with prefix
        for col in df.columns:
            if col.startswith(intensity_prefix):
                sid = col[len(intensity_prefix) :].strip()
                if sid in meta:
                    intensity_cols[sid] = col

    if not intensity_cols:
        raise ValueError(
            f"no matching sample intensity columns found with prefix '{intensity_prefix}' for declared metadata samples"
        )

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_gene = row.get(gene_column, None)
        raw_prot = row.get(protein_id_column, None)
        gene_symbol = _extract_primary_gene(raw_gene, raw_prot)
        if not gene_symbol:
            continue

        for sample_id, col_name in intensity_cols.items():
            val = row.get(col_name, np.nan)
            if pd.isna(val):
                continue
            numeric_val = float(val)
            if not np.isfinite(numeric_val) or (default_scale == "linear" and numeric_val <= 0.0):
                continue

            s_meta = meta[sample_id]
            rows.append({
                "gene_symbol": gene_symbol,
                "group": str(s_meta.get("group", "unknown")),
                "sample_id": str(sample_id),
                "value": numeric_val,
                "unit": str(s_meta.get("unit", default_unit)),
                "scale": str(s_meta.get("scale", default_scale)).lower(),
                "source_id": str(s_meta.get("source_id", default_source_id)),
                "batch_id": str(s_meta.get("batch_id", "batch_1")),
            })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("no quantified protein measurements extracted from MaxQuant file")

    aggregated = (
        result_df.groupby(["gene_symbol", "group", "sample_id", "unit", "scale", "source_id", "batch_id"], as_index=False)
        .agg({"value": "median"})
    )
    return aggregated


def import_diann_report(
    report_path: Path | str,
    sample_metadata: Path | str | Mapping[str, Mapping[str, Any]],
    *,
    gene_column: str = "Genes",
    protein_id_column: str = "Protein.Group",
    sample_column: str = "Run",
    intensity_column: str = "PG.MaxLFQ",
    default_source_id: str = "DIA-NN",
    default_unit: str = "MaxLFQ intensity",
    default_scale: str = "linear",
    qvalue_column: str = "PG.Q.Value",
    max_qvalue: float | None = 0.01,
    require_qvalue: bool = True,
    format_name: str = "DIA-NN",
) -> pd.DataFrame:
    """
    Ingest a DIA-NN long-form report into regenerative comparator format.
    """
    path = Path(report_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{format_name} report not found: {path}")

    default_scale = _validate_scale(default_scale)
    meta = (
        load_sample_metadata(sample_metadata)
        if isinstance(sample_metadata, (str, Path))
        else {str(k): dict(v) for k, v in sample_metadata.items()}
    )

    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)

    for req_col in (sample_column, intensity_column):
        if req_col not in df.columns:
            raise ValueError(f"required {format_name} column '{req_col}' not found in report")

    if gene_column not in df.columns and protein_id_column not in df.columns:
        raise ValueError(
            f"neither gene column '{gene_column}' nor protein ID column '{protein_id_column}' found in {format_name} table"
        )

    if max_qvalue is not None and not 0.0 <= float(max_qvalue) <= 1.0:
        raise ValueError("max_qvalue must be between 0 and 1")
    if require_qvalue and qvalue_column not in df.columns:
        raise ValueError(
            f"required {format_name} quality column '{qvalue_column}' not found; "
            "pass an explicit verified column or disable the filter deliberately"
        )
    if qvalue_column in df.columns and max_qvalue is not None:
        qvalues = pd.to_numeric(df[qvalue_column], errors="coerce")
        df = df[qvalues.notna() & (qvalues <= float(max_qvalue))]

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        sample_raw = str(row[sample_column]).strip()
        sample_id = _resolve_sample_id(sample_raw, meta)

        if not sample_id:
            continue

        raw_gene = row.get(gene_column, None)
        raw_prot = row.get(protein_id_column, None)
        gene_symbol = _extract_primary_gene(raw_gene, raw_prot)
        if not gene_symbol:
            continue

        val = row.get(intensity_column, np.nan)
        if pd.isna(val):
            continue
        numeric_val = float(val)
        if not np.isfinite(numeric_val) or (default_scale == "linear" and numeric_val <= 0.0):
            continue

        s_meta = meta[sample_id]
        rows.append({
            "gene_symbol": gene_symbol,
            "group": str(s_meta.get("group", "unknown")),
            "sample_id": str(sample_id),
            "value": numeric_val,
            "unit": str(s_meta.get("unit", default_unit)),
            "scale": str(s_meta.get("scale", default_scale)).lower(),
            "source_id": str(s_meta.get("source_id", default_source_id)),
            "batch_id": str(s_meta.get("batch_id", "batch_1")),
        })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError(f"no quantified protein measurements extracted from {format_name} report")

    aggregated = (
        result_df.groupby(["gene_symbol", "group", "sample_id", "unit", "scale", "source_id", "batch_id"], as_index=False)
        .agg({"value": "median"})
    )
    return aggregated


def import_spectronaut_report(
    report_path: Path | str,
    sample_metadata: Path | str | Mapping[str, Mapping[str, Any]],
    *,
    gene_column: str = "PG.Genes",
    protein_id_column: str = "PG.ProteinGroups",
    sample_column: str = "R.FileName",
    intensity_column: str = "PG.Quantity",
    default_source_id: str = "Spectronaut",
    default_unit: str = "protein-group quantity",
    default_scale: str = "linear",
    qvalue_column: str = "PG.Qvalue",
    max_qvalue: float | None = 0.01,
    require_qvalue: bool = True,
) -> pd.DataFrame:
    """Ingest a Spectronaut protein-group report using explicit column semantics."""

    return import_diann_report(
        report_path,
        sample_metadata,
        gene_column=gene_column,
        protein_id_column=protein_id_column,
        sample_column=sample_column,
        intensity_column=intensity_column,
        default_source_id=default_source_id,
        default_unit=default_unit,
        default_scale=default_scale,
        qvalue_column=qvalue_column,
        max_qvalue=max_qvalue,
        require_qvalue=require_qvalue,
        format_name="Spectronaut",
    )


def import_wide_proteomics_matrix(
    matrix_path: Path | str,
    sample_metadata: Path | str | Mapping[str, Mapping[str, Any]],
    *,
    gene_column: str = "gene_symbol",
    protein_id_column: str = "uniprot_id",
    default_source_id: str = "PRIDE_matrix",
    default_unit: str = "normalized_intensity",
    default_scale: str = "linear",
) -> pd.DataFrame:
    """
    Ingest a wide table (proteins as rows, samples as columns).
    """
    path = Path(matrix_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"matrix file not found: {path}")

    default_scale = _validate_scale(default_scale)
    meta = (
        load_sample_metadata(sample_metadata)
        if isinstance(sample_metadata, (str, Path))
        else {str(k): dict(v) for k, v in sample_metadata.items()}
    )

    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)

    id_col = None
    for cand in (gene_column, "gene", "genes", "gene_name", "symbol", protein_id_column, "protein_id", "accession", "uniprot"):
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        raise ValueError(f"gene or protein column not found in matrix; columns: {list(df.columns)}")
    if id_col in {protein_id_column, "protein_id", "accession", "uniprot"}:
        raise ValueError(
            "wide matrix contains protein accessions but no declared gene-symbol column; "
            "provide a reviewed accession-to-gene mapping before import"
        )

    sample_cols = [col for col in df.columns if col in meta]
    if not sample_cols:
        raise ValueError(
            f"none of the metadata sample IDs {sorted(meta)} match matrix columns {list(df.columns)}"
        )

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        gene_symbol = _extract_primary_gene(row[id_col])
        if not gene_symbol:
            continue

        for sample_id in sample_cols:
            val = row[sample_id]
            if pd.isna(val):
                continue
            try:
                numeric_val = float(val)
            except (ValueError, TypeError):
                continue

            if not np.isfinite(numeric_val) or (default_scale == "linear" and numeric_val < 0.0):
                continue

            s_meta = meta[sample_id]
            rows.append({
                "gene_symbol": gene_symbol,
                "group": str(s_meta.get("group", "unknown")),
                "sample_id": str(sample_id),
                "value": numeric_val,
                "unit": str(s_meta.get("unit", default_unit)),
                "scale": str(s_meta.get("scale", default_scale)).lower(),
                "source_id": str(s_meta.get("source_id", default_source_id)),
                "batch_id": str(s_meta.get("batch_id", "batch_1")),
            })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("no valid measurements extracted from wide proteomics matrix")

    aggregated = (
        result_df.groupby(["gene_symbol", "group", "sample_id", "unit", "scale", "source_id", "batch_id"], as_index=False)
        .agg({"value": "median"})
    )
    return aggregated


def import_standard_long_proteomics(path: Path | str) -> pd.DataFrame:
    """Load an already standardized long table and enforce the comparator contract."""

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(source)
    else:
        separator = "\t" if source.suffix.lower() in {".tsv", ".txt"} else ","
        df = pd.read_csv(source, sep=separator, low_memory=False)
    missing = sorted(_COMPARISON_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"standard long proteomics table missing required columns: {missing}")
    return df.copy()


def _table_columns(path: Path) -> set[str]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return set(pd.read_parquet(path).columns)
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return set(pd.read_csv(path, sep=separator, nrows=0).columns)


def detect_proteomics_format(
    path: Path | str,
    sample_metadata: Path | str | Mapping[str, Mapping[str, Any]],
    *,
    intensity_prefix: str = "LFQ intensity ",
) -> str:
    """Detect only unambiguous, schema-backed formats; never guess from filenames."""

    source = Path(path).expanduser().resolve()
    columns = _table_columns(source)
    metadata = (
        load_sample_metadata(sample_metadata)
        if isinstance(sample_metadata, (str, Path))
        else {str(key): dict(value) for key, value in sample_metadata.items()}
    )
    matches: list[str] = []
    if _COMPARISON_REQUIRED_COLUMNS.issubset(columns):
        matches.append("standard_long")
    if (
        {"Gene names", "Majority protein IDs"}.intersection(columns)
        and any(column.startswith(intensity_prefix) for column in columns)
    ):
        matches.append("maxquant")
    if {"Run", "PG.MaxLFQ"}.issubset(columns) and (
        "PG.Q.Value" in columns or "PG.Qvalue" in columns
    ):
        matches.append("diann")
    if {"R.FileName", "PG.Quantity"}.issubset(columns) and (
        "PG.Qvalue" in columns or "PG.Q.Value" in columns
    ):
        matches.append("spectronaut")
    gene_columns = {"gene_symbol", "gene", "genes", "gene_name", "symbol"}
    if columns.intersection(gene_columns) and columns.intersection(metadata):
        matches.append("wide_matrix")

    if len(matches) != 1:
        detail = ", ".join(matches) if matches else "none"
        raise ValueError(
            "proteomics format could not be detected unambiguously "
            f"(matches: {detail}); specify --format explicitly after reviewing the source schema"
        )
    return matches[0]


def import_and_validate_proteomics(
    data_path: Path | str,
    sample_metadata: Path | str | Mapping[str, Mapping[str, Any]],
    output_path: Path | str,
    *,
    input_format: str = "auto",
    default_source_id: str = "PRIDE",
    default_unit: str = "normalized_intensity",
    default_scale: str = "linear",
    intensity_prefix: str = "LFQ intensity ",
) -> ProteomicsImportResult:
    """
    High-level entry point to ingest and standardize any proteomics dataset
    into the strict IINTS regenerative islet comparator contract.
    """
    in_path = Path(data_path).expanduser().resolve()
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = input_format.lower().strip()
    if fmt == "auto":
        fmt = detect_proteomics_format(
            in_path,
            sample_metadata,
            intensity_prefix=intensity_prefix,
        )

    quality_filters: tuple[str, ...]

    if fmt == "maxquant":
        df = import_maxquant_protein_groups(
            in_path,
            sample_metadata,
            intensity_prefix=intensity_prefix,
            default_source_id=default_source_id,
            default_unit=default_unit,
            default_scale=default_scale,
        )
        quality_filters = (
            "Potential contaminant != +",
            "Reverse != +",
            "Only identified by site != +",
            "positive finite intensity",
        )
    elif fmt == "diann":
        df = import_diann_report(
            in_path,
            sample_metadata,
            default_source_id=default_source_id,
            default_unit=default_unit,
            default_scale=default_scale,
        )
        quality_filters = ("PG.Q.Value <= 0.01", "positive finite intensity")
    elif fmt == "spectronaut":
        df = import_spectronaut_report(
            in_path,
            sample_metadata,
            default_source_id=default_source_id,
            default_unit=default_unit,
            default_scale=default_scale,
        )
        quality_filters = ("PG.Qvalue <= 0.01", "positive finite intensity")
    elif fmt in {"wide_matrix", "wide", "matrix"}:
        df = import_wide_proteomics_matrix(
            in_path,
            sample_metadata,
            default_source_id=default_source_id,
            default_unit=default_unit,
            default_scale=default_scale,
        )
        fmt = "wide_matrix"
        quality_filters = ("declared gene-symbol column", "finite non-negative intensity")
    elif fmt in {"standard_long", "long", "generic_long"}:
        df = import_standard_long_proteomics(in_path)
        fmt = "standard_long"
        quality_filters = ("existing comparator contract validated",)
    else:
        raise ValueError(
            f"unsupported proteomics format '{input_format}'; use 'auto', 'maxquant', "
            "'diann', 'spectronaut', 'wide_matrix', or 'standard_long'"
        )

    # Validate output frame schema
    missing = sorted(_COMPARISON_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"standardized dataset missing required columns: {missing}")

    if df.empty:
        raise ValueError("standardized dataset contains zero rows")
    invalid_scales = sorted(set(df["scale"].astype(str).str.lower()) - set(_SUPPORTED_SCALES))
    if invalid_scales:
        raise ValueError(f"standardized dataset contains unsupported scales: {invalid_scales}")

    # Export
    if out_path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    # Compute coverage of regenerative panels
    panels = load_regenerative_protein_panels()
    coverage: dict[str, tuple[int, int]] = {}
    found_symbols = set(df["gene_symbol"])
    for p_key, panel in panels.items():
        total_targets = len(panel.targets)
        observed = sum(1 for t in panel.targets if t.gene_symbol in found_symbols)
        coverage[p_key] = (observed, total_targets)

    group_counts = {
        str(key): int(value)
        for key, value in df["group"].value_counts().to_dict().items()
    }
    metadata_path = (
        Path(sample_metadata).expanduser().resolve()
        if isinstance(sample_metadata, (str, Path))
        else None
    )
    manifest_path = out_path.with_suffix(out_path.suffix + ".import_manifest.json")
    output_sha256 = _sha256(out_path)
    metadata_sha256 = _sha256(metadata_path) if metadata_path is not None else None
    manifest = {
        "schema_version": 1,
        "input_format": fmt,
        "source_file": in_path.name,
        "source_sha256": _sha256(in_path),
        "sample_metadata_file": metadata_path.name if metadata_path is not None else None,
        "sample_metadata_sha256": metadata_sha256,
        "output_file": out_path.name,
        "output_sha256": output_sha256,
        "quality_filters": list(quality_filters),
        "row_count": int(len(df)),
        "gene_count": int(df["gene_symbol"].nunique()),
        "sample_count": int(df["sample_id"].nunique()),
        "group_counts": group_counts,
        "target_panel_coverage": {
            key: {"observed": value[0], "total": value[1]}
            for key, value in coverage.items()
        },
        "research_only": True,
        "medical_device": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return ProteomicsImportResult(
        output_path=out_path,
        row_count=len(df),
        gene_count=int(df["gene_symbol"].nunique()),
        sample_count=int(df["sample_id"].nunique()),
        group_counts=group_counts,
        source_sha256=manifest["source_sha256"],
        output_sha256=output_sha256,
        metadata_sha256=metadata_sha256,
        input_format=fmt,
        quality_filters=quality_filters,
        manifest_path=manifest_path,
        target_panel_coverage=coverage,
    )


__all__ = [
    "ProteomicsImportResult",
    "load_sample_metadata",
    "import_maxquant_protein_groups",
    "import_diann_report",
    "import_spectronaut_report",
    "import_wide_proteomics_matrix",
    "import_standard_long_proteomics",
    "detect_proteomics_format",
    "import_and_validate_proteomics",
]
