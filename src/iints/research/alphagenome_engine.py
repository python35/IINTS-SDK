"""Google AlphaGenome regulatory, splicing, and variant-impact evidence connector.

This module integrates Google DeepMind's AlphaGenome and AlphaGenome Atlas
into the IINTS-AF research workflow. It supports:

1. Validated AlphaGenome Atlas deep-link construction for single variants
   (`chr:pos:ref>alt`), genomic intervals (`chr:start-end`), and curated
   diabetes loci (`INS`, `INSR`, `SLC2A4`, `GCGR`, `GCK`, `HNF1A`), enforcing
   the Atlas multi-modality Assay-group filter rule and splicing + RNA-seq
   co-plotting rule.
2. Live AlphaGenome Variant Impact (AVI) scoring and 18-modality feature
   attribution via `alphagenome.atlas.atlas` when `ALPHAGENOME_API_KEY` is
   configured, with deterministic offline fallback to Atlas deep-link evidence
   bundles when run without network/API credentials.
3. Saturation mutagenesis / regional window scanning (`chr:start-end`, 1-based
   closed coordinates converted to 0-based half-open `[start - 1, end)` for
   the AlphaGenome SDK).

Research and clinical safety boundary
-------------------------------------
AlphaGenome and AlphaGenome AVI scores are strictly research tools. High AVI
Phred or raw scores reflect predicted molecular/functional effects (such as
splicing disruption, chromatin accessibility shifts, or transcription factor
binding changes) and must NEVER be interpreted as clinical diagnoses,
treatment recommendations, insulin dosing adjustments, or automatic
physiological simulator scalars (`supports_quantitative_functional_scalar` is
always `False`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, field
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from typing import Any
import urllib.parse

from .external_models_common import safe_stem, timestamp_token, utc_now, write_json
from iints.utils.csv_safety import sanitize_csv_mapping


BASE_ATLAS_URL = "https://deepmind.google.com/science/alphagenome/atlas"
ATLAS_TRACK_PREDICTIONS_URL = "https://deepmind.google.com/science/alphagenome/atlas/track-predictions"
ALPHAGENOME_DOCS_URL = "https://deepmind.google.com/science/alphagenome/"
ALPHAGENOME_GRPC_ADDRESS = "dns:///gdmscience.googleapis.com:443"
ALPHAGENOME_SCHEMA_VERSION = "1.0"
MAX_REGION_SCAN_BP = 1_000

_VARIANT_PATTERN = re.compile(
    r"^(?:chr)?([0-9]{1,2}|X|Y|M|MT):([0-9,]+):([ACGTN]+)>([ACGTN]+)$",
    re.IGNORECASE,
)
_INTERVAL_PATTERN = re.compile(
    r"^(?:chr)?([0-9]{1,2}|X|Y|M|MT):([0-9,]+)-([0-9,]+)$",
    re.IGNORECASE,
)

AVI_FEATURE_METADATA: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "MERGED_SPLICING": (
        "Splicing",
        "Splicing",
        ("SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS"),
    ),
    "MAX_ABS_RNA_SEQ": ("RNA-seq", "Transcription / RNA-seq", ("RNA_SEQ",)),
    "MAX_ABS_ATAC": ("ATAC-seq", "Chromatin Accessibility", ("ATAC",)),
    "MAX_ABS_DNASE": ("DNASE-seq", "Chromatin Accessibility", ("DNASE",)),
    "MAX_ABS_CHIP_TF": ("ChIP-TF", "Transcription Factor Binding", ("CHIP_TF",)),
    "MAX_ABS_CHIP_HISTONE": ("ChIP-Histone", "Histone Modification", ("CHIP_HISTONE",)),
    "MAX_ABS_CAGE": ("CAGE", "Transcription / RNA-seq", ("CAGE",)),
    "MAX_ABS_PROCAP": ("PRO-cap", "Transcription / RNA-seq", ("PROCAP",)),
    "MAX_ABS_POLYADENYLATION": ("Polyadenylation", "Transcription / RNA-seq", ("POLYADENYLATION",)),
    "MAX_ABS_CONTACT_MAPS": ("3D Genome Contacts", "3D Genome Organization", ("CONTACT_MAPS",)),
    "ALPHAMISSENSE": ("AlphaMissense", "Protein Impact", ()),
    "CACTUS_241_WAY": ("Cactus", "Evolutionary Conservation", ()),
    "PROTEIN_TERMINATION": ("Protein Termination", "Coding Consequence", ()),
    "START_LOST": ("Start Lost", "Coding Consequence", ()),
    "STOP_LOST": ("Stop Lost", "Coding Consequence", ()),
    "PHASTCONS_470_WAY": ("PhastCons 470", "Evolutionary Conservation", ()),
    "IS_INSERTION": ("Insertion", "Structural Variant", ()),
    "IS_DELETION": ("Deletion", "Structural Variant", ()),
}

_MODALITY_FILTER_MAP: dict[str, str] = {
    "RNA_SEQ": "RNA-seq",
    "DNASE": "DNase",
    "ATAC": "ATAC-seq",
    "CHIP_TF": "ChIP-TF",
    "CHIP_HISTONE": "ChIP-Histone",
    "CAGE": "CAGE",
    "PROCAP": "PRO-cap",
    "POLYADENYLATION": "Polyadenylation",
    "SPLICE_JUNCTIONS": "Splice junctions",
    "SPLICE_SITE_USAGE": "Splice site usage",
    "SPLICE_SITES": "Splice sites",
}

_SPLICING_MODALITIES = {"SPLICE_JUNCTIONS", "SPLICE_SITE_USAGE", "SPLICE_SITES"}

ALPHAGENOME_LIMITATIONS: tuple[str, ...] = (
    "AlphaGenome and AlphaGenome Variant Impact (AVI) scores are strictly non-clinical research tools.",
    "High AVI Phred or raw scores indicate predicted molecular/functional impact (e.g. splicing, chromatin, TF binding), never clinical pathogenicity or disease penetrance.",
    "Predicted regulatory or splicing effects must not be converted automatically into IINTS patient-model parameters (insulin sensitivity, EGP, PK/PD, or dosing).",
    "When inspecting splicing variants, continuous RNA-seq expression must be reviewed alongside splice junction and splice site usage tracks.",
    "No output in this bundle may be used for medical diagnosis, clinical decision-making, or insulin therapy adjustment.",
)


@dataclass(frozen=True)
class DiabetesGeneLocus:
    """Curated GRCh38 (GENCODE v46) locus metadata for a diabetes-relevant gene."""

    gene_symbol: str
    ensembl_gene_id: str
    uniprot_id: str
    chromosome: str
    start_1_based: int
    end_1_based: int
    strand: str
    primary_biosample: str
    key_transcription_factors: tuple[str, ...]
    default_modalities: tuple[str, ...]
    representative_variant: str
    sdk_physiological_context: str

    @property
    def interval_1_based(self) -> str:
        return f"{self.chromosome}:{self.start_1_based}-{self.end_1_based}"

    @property
    def interval_0_based_half_open(self) -> tuple[int, int]:
        return (self.start_1_based - 1, self.end_1_based)


DIABETES_ALPHAGENOME_LOCI: dict[str, DiabetesGeneLocus] = {
    "INS": DiabetesGeneLocus(
        gene_symbol="INS",
        ensembl_gene_id="ENSG00000254647",
        uniprot_id="P01308",
        chromosome="chr11",
        start_1_based=2_159_779,
        end_1_based=2_161_209,
        strand="-",
        primary_biosample="Pancreas",
        key_transcription_factors=("PDX1", "MAFA", "NEUROD1"),
        default_modalities=("RNA_SEQ", "DNASE", "CHIP_TF", "SPLICE_JUNCTIONS"),
        representative_variant="chr11:2160494:C>T",
        sdk_physiological_context=(
            "Links to endogenous beta-cell insulin biosynthesis, C-peptide co-secretion, "
            "and absolute insulin-deficiency stress scenarios."
        ),
    ),
    "INSR": DiabetesGeneLocus(
        gene_symbol="INSR",
        ensembl_gene_id="ENSG00000171105",
        uniprot_id="P06213",
        chromosome="chr19",
        start_1_based=7_112_255,
        end_1_based=7_294_034,
        strand="-",
        primary_biosample="Muscle_Skeletal",
        key_transcription_factors=("SP1", "FOXO1", "CEBPB"),
        default_modalities=("RNA_SEQ", "DNASE", "CHIP_TF", "SPLICE_JUNCTIONS"),
        representative_variant="chr19:7125512:C>T",
        sdk_physiological_context=(
            "Links to insulin receptor signalling and the illustrative molecular_affinity_scalar "
            "in Hovorka/Bergman receptor-resistance stress tests."
        ),
    ),
    "SLC2A4": DiabetesGeneLocus(
        gene_symbol="SLC2A4",
        ensembl_gene_id="ENSG00000181856",
        uniprot_id="P14672",
        chromosome="chr17",
        start_1_based=7_281_629,
        end_1_based=7_288_284,
        strand="+",
        primary_biosample="Muscle_Skeletal",
        key_transcription_factors=("MEF2A", "MYOD1", "KLF15"),
        default_modalities=("RNA_SEQ", "DNASE", "CHIP_TF"),
        representative_variant="chr17:7284500:G>A",
        sdk_physiological_context=(
            "Encodes GLUT4; links to insulin-stimulated peripheral glucose disposal in the "
            "skeletal-muscle and adipose compartments."
        ),
    ),
    "GCGR": DiabetesGeneLocus(
        gene_symbol="GCGR",
        ensembl_gene_id="ENSG00000215644",
        uniprot_id="P47871",
        chromosome="chr17",
        start_1_based=81_804_155,
        end_1_based=81_814_921,
        strand="+",
        primary_biosample="Liver",
        key_transcription_factors=("HNF4A", "FOXA2", "CREB1"),
        default_modalities=("RNA_SEQ", "DNASE", "CHIP_TF"),
        representative_variant="chr17:81808500:C>A",
        sdk_physiological_context=(
            "Encodes the glucagon receptor; links to hepatic glucose production (EGP), "
            "glycogenolysis, and dual-hormone hypoglycemia rescue dynamics."
        ),
    ),
    "GCK": DiabetesGeneLocus(
        gene_symbol="GCK",
        ensembl_gene_id="ENSG00000106633",
        uniprot_id="P35557",
        chromosome="chr7",
        start_1_based=44_143_213,
        end_1_based=44_189_439,
        strand="-",
        primary_biosample="Pancreas",
        key_transcription_factors=("PDX1", "HNF4A", "NEUROD1"),
        default_modalities=("RNA_SEQ", "DNASE", "CHIP_TF", "SPLICE_JUNCTIONS"),
        representative_variant="chr7:44149800:G>A",
        sdk_physiological_context=(
            "Encodes glucokinase (hexokinase IV); provides molecular context for beta-cell "
            "glucose-sensing threshold shifts and hepatic glucose phosphorylation."
        ),
    ),
    "HNF1A": DiabetesGeneLocus(
        gene_symbol="HNF1A",
        ensembl_gene_id="ENSG00000135100",
        uniprot_id="P20823",
        chromosome="chr12",
        start_1_based=120_978_543,
        end_1_based=121_002_512,
        strand="+",
        primary_biosample="Pancreas",
        key_transcription_factors=("HNF4A", "FOXA2", "PDX1"),
        default_modalities=("RNA_SEQ", "DNASE", "CHIP_TF", "SPLICE_JUNCTIONS"),
        representative_variant="chr12:120994314:C>T",
        sdk_physiological_context=(
            "Encodes hepatocyte nuclear factor 1-alpha; provides transcriptional network "
            "context for beta-cell maturation and insulin secretory capacity."
        ),
    ),
}


@dataclass(frozen=True)
class AlphaGenomeVariantScore:
    """One variant record annotated with AlphaGenome AVI context and Atlas deep-link."""

    variant: str
    chromosome: str
    position: int
    ref: str
    alt: str
    avi_phred: float | None
    avi_raw: float | None
    avi_quantile: float | None
    top_percentile: float | None
    impact_band: str
    top_modality: str | None
    top_feature_importance: float | None
    atlas_url: str
    query_mode: str
    overlapping_diabetes_gene: str | None = None
    feature_importances: dict[str, float] = field(default_factory=dict)
    track_attributions: dict[str, dict[str, Any]] = field(default_factory=dict)
    supports_quantitative_functional_scalar: bool = False

    def to_csv_row(self, *, rank: int | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {}
        if rank is not None:
            row["rank"] = rank
        row.update(
            {
                "variant": self.variant,
                "chromosome": self.chromosome,
                "position": self.position,
                "ref": self.ref,
                "alt": self.alt,
                "avi_phred": self.avi_phred,
                "avi_raw": self.avi_raw,
                "avi_quantile": self.avi_quantile,
                "top_percentile": self.top_percentile,
                "impact_band": self.impact_band,
                "top_modality": self.top_modality,
                "top_feature_importance": self.top_feature_importance,
                "overlapping_diabetes_gene": self.overlapping_diabetes_gene or "",
                "query_mode": self.query_mode,
                "atlas_url": self.atlas_url,
                "supports_quantitative_functional_scalar": False,
            }
        )
        return row


@dataclass(frozen=True)
class AlphaGenomeEvidenceResult:
    """Filesystem bundle returned by AlphaGenome variant or region queries."""

    output_dir: Path
    records_csv: Path
    evidence_json: Path
    report_md: Path
    query_type: str
    query_target: str
    query_mode: str
    record_count: int
    atlas_url: str


def _clean_chrom(chrom: str) -> str:
    stripped = chrom.strip()
    if stripped.lower().startswith("chr"):
        suffix = stripped[3:].upper()
    else:
        suffix = stripped.upper()
    if suffix == "MT":
        suffix = "M"
    return f"chr{suffix}"


def normalize_genomic_variant(variant_str: str) -> dict[str, Any]:
    """Validate and normalize a 1-based genomic variant in `chr:pos:ref>alt` format."""

    cleaned = variant_str.strip()
    match = _VARIANT_PATTERN.fullmatch(cleaned)
    if match is None:
        raise ValueError(
            f"Invalid genomic variant '{variant_str}'. Expected 1-based 'chr:pos:ref>alt' "
            "(for example 'chr11:2160494:C>T'). Colon-separated alleles and rsIDs are not accepted."
        )
    chrom = _clean_chrom(match.group(1))
    position = int(match.group(2).replace(",", ""))
    if position < 1:
        raise ValueError("Genomic variant position must be a 1-based positive integer.")
    ref = match.group(3).upper()
    alt = match.group(4).upper()
    if ref == alt:
        raise ValueError(f"Reference and alternate alleles must differ in '{variant_str}'.")
    normalized = f"{chrom}:{position}:{ref}>{alt}"
    return {
        "variant": normalized,
        "chromosome": chrom,
        "position": position,
        "ref": ref,
        "alt": alt,
    }


def normalize_genomic_interval(interval_str: str) -> dict[str, Any]:
    """Validate a 1-based closed genomic interval (`chr:start-end`) and compute 0-based half-open bounds."""

    cleaned = interval_str.strip()
    match = _INTERVAL_PATTERN.fullmatch(cleaned)
    if match is None:
        raise ValueError(
            f"Invalid genomic interval '{interval_str}'. Expected 1-based closed 'chr:start-end' "
            "(for example 'chr11:2159779-2161209')."
        )
    chrom = _clean_chrom(match.group(1))
    start_1 = int(match.group(2).replace(",", ""))
    end_1 = int(match.group(3).replace(",", ""))
    if start_1 < 1 or end_1 < start_1:
        raise ValueError(
            f"Invalid interval bounds in '{interval_str}': start must be >= 1 and end >= start."
        )
    width_bp = end_1 - start_1 + 1
    return {
        "interval": f"{chrom}:{start_1}-{end_1}",
        "chromosome": chrom,
        "start_1_based": start_1,
        "end_1_based": end_1,
        "start_0_based": start_1 - 1,
        "end_0_based_half_open": end_1,
        "width_bp": width_bp,
    }


def find_overlapping_diabetes_gene(chromosome: str, position_1_based: int) -> str | None:
    """Return the curated diabetes gene symbol if `chromosome:position` overlaps its locus."""

    norm_chrom = _clean_chrom(chromosome)
    for symbol, locus in DIABETES_ALPHAGENOME_LOCI.items():
        if locus.chromosome == norm_chrom and locus.start_1_based <= position_1_based <= locus.end_1_based:
            return symbol
    return None


def _normalize_modalities(
    modalities: Sequence[str] | str,
    *,
    enforce_splicing_rna_coplot: bool = True,
) -> list[str]:
    if isinstance(modalities, str):
        raw_items = [item.strip().upper() for item in modalities.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip().upper() for item in modalities if str(item).strip()]
    deduped: list[str] = []
    for item in raw_items:
        if item not in deduped:
            deduped.append(item)
    if enforce_splicing_rna_coplot and any(mod in _SPLICING_MODALITIES for mod in deduped):
        if "RNA_SEQ" not in deduped:
            deduped.insert(0, "RNA_SEQ")
    return deduped


def build_atlas_filter_string(
    *,
    biosample: str | None = None,
    modalities: Sequence[str] = (),
    tfs: Sequence[str] = (),
) -> str | None:
    """Construct the AlphaGenome Atlas `f=` filter parameter.

    Enforces the mandatory Assay-group OR-logic rule: when transcription factor
    filters (`ASSAY_TRANSCRIPTOR_FACTOR`) are supplied, `RNA_SEQ` and `DNASE`
    scorer modalities are automatically included so RNA-seq and DNase tracks do
    not fail the Assay group evaluation and disappear from the heatmap.
    """

    filter_parts: list[str] = []
    if biosample and biosample.strip():
        filter_parts.append(f"BIOSAMPLE_NAME:{biosample.strip()}")

    clean_tfs = [tf.strip() for tf in tfs if str(tf).strip()]
    active_modalities = _normalize_modalities(modalities, enforce_splicing_rna_coplot=False)
    if clean_tfs:
        for required_mod in ("RNA_SEQ", "DNASE"):
            if required_mod not in active_modalities:
                active_modalities.append(required_mod)

    for mod in active_modalities:
        display_name = _MODALITY_FILTER_MAP.get(mod, mod)
        filter_parts.append(f"SCORER_MODALITY:{display_name}")

    for tf in clean_tfs:
        filter_parts.append(f"ASSAY_TRANSCRIPTOR_FACTOR:{tf}")

    return ",".join(filter_parts) if filter_parts else None


def build_atlas_layout_items(
    *,
    include_avi: bool = True,
    modalities: Sequence[str] = ("RNA_SEQ", "DNASE", "CHIP_TF"),
    pinned_tracks: Sequence[str] = (),
) -> str:
    """Construct the AlphaGenome Atlas `lItems=` layout string."""

    items: list[str] = []
    if include_avi:
        items.append("avi")
    for mod in _normalize_modalities(modalities, enforce_splicing_rna_coplot=True):
        items.append(f"section:{mod}")
    for pinned in pinned_tracks:
        if str(pinned).strip():
            items.append(str(pinned).strip())
    return ",".join(items) if items else "avi"


def build_atlas_variant_url(
    variant: str,
    *,
    biosample: str | None = None,
    modalities: Sequence[str] = ("RNA_SEQ", "DNASE", "CHIP_TF"),
    filter_modalities: Sequence[str] = (),
    tfs: Sequence[str] = (),
    include_avi: bool = True,
    pinned_tracks: Sequence[str] = (),
) -> str:
    """Construct a validated AlphaGenome Atlas single-variant exploration URL."""

    parsed = normalize_genomic_variant(variant)
    l_items = build_atlas_layout_items(
        include_avi=include_avi,
        modalities=modalities,
        pinned_tracks=pinned_tracks,
    )
    query: dict[str, str] = {
        "q": parsed["variant"],
        "m": "variant",
        "lItems": l_items,
    }
    f_str = build_atlas_filter_string(
        biosample=biosample,
        modalities=filter_modalities,
        tfs=tfs,
    )
    if f_str:
        query["f"] = f_str
    return f"{BASE_ATLAS_URL}?" + urllib.parse.urlencode(
        query, quote_via=urllib.parse.quote, safe=":,-"
    )


def build_atlas_locus_url(
    interval: str,
    *,
    biosample: str | None = None,
    modalities: Sequence[str] = ("RNA_SEQ", "DNASE", "CHIP_TF"),
    filter_modalities: Sequence[str] = (),
    tfs: Sequence[str] = (),
    include_avi: bool = True,
    viewport_interval: str | None = None,
) -> str:
    """Construct a validated AlphaGenome Atlas genomic locus URL."""

    parsed = normalize_genomic_interval(interval)
    l_items = build_atlas_layout_items(
        include_avi=include_avi,
        modalities=modalities,
    )
    query: dict[str, str] = {
        "q": parsed["interval"],
        "m": "locus",
        "lItems": l_items,
    }
    if viewport_interval:
        vp = normalize_genomic_interval(viewport_interval)
        query["i"] = vp["interval"]
    f_str = build_atlas_filter_string(
        biosample=biosample,
        modalities=filter_modalities,
        tfs=tfs,
    )
    if f_str:
        query["f"] = f_str
    return f"{BASE_ATLAS_URL}?" + urllib.parse.urlencode(
        query, quote_via=urllib.parse.quote, safe=":,-"
    )


def list_diabetes_alphagenome_loci() -> list[dict[str, Any]]:
    """Return curated diabetes gene loci with prebuilt AlphaGenome Atlas links."""

    cards: list[dict[str, Any]] = []
    for symbol, locus in DIABETES_ALPHAGENOME_LOCI.items():
        locus_url = build_atlas_locus_url(
            locus.interval_1_based,
            biosample=locus.primary_biosample,
            modalities=locus.default_modalities,
            tfs=locus.key_transcription_factors,
        )
        variant_url = build_atlas_variant_url(
            locus.representative_variant,
            biosample=locus.primary_biosample,
            modalities=locus.default_modalities,
            tfs=locus.key_transcription_factors,
        )
        cards.append(
            {
                **asdict(locus),
                "interval_1_based": locus.interval_1_based,
                "interval_0_based_half_open": list(locus.interval_0_based_half_open),
                "atlas_locus_url": locus_url,
                "atlas_representative_variant_url": variant_url,
                "supports_quantitative_functional_scalar": False,
            }
        )
    return cards


def get_diabetes_gene_atlas_context(gene: str) -> dict[str, Any] | None:
    """Return AlphaGenome Atlas locus and regulatory metadata for a diabetes gene symbol."""

    normalized = gene.strip().upper()
    locus = DIABETES_ALPHAGENOME_LOCI.get(normalized)
    if locus is None:
        return None
    return {
        "gene_symbol": locus.gene_symbol,
        "ensembl_gene_id": locus.ensembl_gene_id,
        "uniprot_id": locus.uniprot_id,
        "interval_1_based": locus.interval_1_based,
        "interval_0_based_half_open": list(locus.interval_0_based_half_open),
        "strand": locus.strand,
        "primary_biosample": locus.primary_biosample,
        "key_transcription_factors": list(locus.key_transcription_factors),
        "default_modalities": list(locus.default_modalities),
        "representative_variant": locus.representative_variant,
        "atlas_locus_url": build_atlas_locus_url(
            locus.interval_1_based,
            biosample=locus.primary_biosample,
            modalities=locus.default_modalities,
            tfs=locus.key_transcription_factors,
        ),
        "atlas_representative_variant_url": build_atlas_variant_url(
            locus.representative_variant,
            biosample=locus.primary_biosample,
            modalities=locus.default_modalities,
            tfs=locus.key_transcription_factors,
        ),
        "sdk_physiological_context": locus.sdk_physiological_context,
        "supports_quantitative_functional_scalar": False,
    }


def _has_env_key_in_dotenv() -> bool:
    env_path = Path.home() / ".env"
    if not env_path.is_file():
        return False
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            if key.strip() == "ALPHAGENOME_API_KEY" and val.strip().strip("'\""):
                return True
    except OSError:
        return False
    return False


def _resolve_api_key() -> str | None:
    direct = os.environ.get("ALPHAGENOME_API_KEY", "").strip()
    if direct:
        return direct
    env_path = Path.home() / ".env"
    if not env_path.is_file():
        return None
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            if key.strip() == "ALPHAGENOME_API_KEY":
                cleaned = val.strip().strip("'\"")
                if cleaned:
                    return cleaned
    except OSError:
        return None
    return None


def alphagenome_status() -> dict[str, Any]:
    """Report AlphaGenome SDK, API key presence (never value), and offline capabilities."""

    sdk_installed = importlib.util.find_spec("alphagenome") is not None
    api_key_configured = bool(os.environ.get("ALPHAGENOME_API_KEY", "").strip()) or _has_env_key_in_dotenv()
    gtf_path = os.environ.get("ALPHAGENOME_GTF_PATH", "").strip()
    gtf_configured = bool(gtf_path and Path(gtf_path).expanduser().is_file())
    return {
        "engine": "Google AlphaGenome",
        "sdk_installed": sdk_installed,
        "api_key_configured": api_key_configured,
        "live_api_ready": bool(sdk_installed and api_key_configured),
        "offline_atlas_links_available": True,
        "gtf_feather_configured": gtf_configured,
        "curated_diabetes_genes": sorted(DIABETES_ALPHAGENOME_LOCI.keys()),
        "atlas_base_url": BASE_ATLAS_URL,
        "docs_url": ALPHAGENOME_DOCS_URL,
        "research_only": True,
        "medical_device": False,
        "supports_quantitative_functional_scalar": False,
    }


def classify_avi_phred(phred: float | None) -> tuple[str, float | None]:
    """Return `(impact_band, top_percentile)` for a calibrated AVI Phred score."""

    if phred is None or not math.isfinite(phred):
        return "unscored_offline", None
    top_percentile = round((10.0 ** (-float(phred) / 10.0)) * 100.0, 6)
    if phred >= 40.0:
        band = "extreme_top_0_01pct"
    elif phred >= 30.0:
        band = "very_high_top_0_1pct"
    elif phred >= 20.0:
        band = "high_top_1pct"
    elif phred >= 15.0:
        band = "moderate_top_3_16pct"
    elif phred >= 10.0:
        band = "nominal_top_10pct"
    else:
        band = "low_bottom_90pct"
    return band, top_percentile


def _cdf_to_phred(cdf_quantile: float) -> tuple[float, float]:
    tail = max(1e-7, 1.0 - float(cdf_quantile))
    return tail, -10.0 * math.log10(tail)


def format_avi_feature_name(feature_key: str | None) -> str:
    if not feature_key:
        return ""
    meta = AVI_FEATURE_METADATA.get(feature_key)
    return meta[0] if meta is not None else feature_key


def _create_live_atlas_client(api_key: str) -> Any:
    from alphagenome.atlas import atlas  # type: ignore[import-not-found]

    return atlas.create(api_key)


def _score_variant_with_client(
    parsed_variant: dict[str, Any],
    client: Any,
    *,
    biosample: str | None = None,
    modalities: Sequence[str] = ("RNA_SEQ", "DNASE", "CHIP_TF"),
    tfs: Sequence[str] = (),
) -> AlphaGenomeVariantScore:
    """Query a single variant against an AlphaGenome `AtlasClient` (or compatible test stub)."""

    try:
        from alphagenome.data import genome  # type: ignore[import-not-found]

        variant_obj: Any = genome.Variant(
            chromosome=parsed_variant["chromosome"],
            position=int(parsed_variant["position"]),
            reference_bases=parsed_variant["ref"],
            alternate_bases=parsed_variant["alt"],
        )
    except ImportError:
        variant_obj = parsed_variant["variant"]

    scores = client.query_variant(
        variant_obj,
        requested_scorers=["AVI_SCORE", "AVI_SCORE_FEATURE_IMPORTANCE"],
    )

    raw_score: float | None = None
    quantile_val: float | None = None
    phred_val: float | None = None
    feature_importances: dict[str, float] = {}
    top_modality: str | None = None
    top_feature_val: float | None = None

    if isinstance(scores, Mapping) and "AVI_SCORE" in scores:
        avi_adata = scores["AVI_SCORE"]
        if getattr(avi_adata, "X", None) is not None and avi_adata.X.size > 0:
            raw_score = round(float(avi_adata.X.reshape(-1)[0]), 6)
        layers = getattr(avi_adata, "layers", {}) or {}
        if "quantiles" in layers:
            cdf_q = float(layers["quantiles"].reshape(-1)[0])
            tail_q, phred_raw = _cdf_to_phred(cdf_q)
            quantile_val = round(tail_q, 6)
            phred_val = round(phred_raw, 4)

    if isinstance(scores, Mapping) and "AVI_SCORE_FEATURE_IMPORTANCE" in scores:
        fi_adata = scores["AVI_SCORE_FEATURE_IMPORTANCE"]
        if getattr(fi_adata, "X", None) is not None and fi_adata.X.size > 0:
            values = [float(v) for v in fi_adata.X.reshape(-1)]
            var_df = getattr(fi_adata, "var", None)
            if var_df is not None and hasattr(var_df, "columns") and "name" in var_df.columns:
                names = [str(n) for n in var_df["name"]]
            elif hasattr(fi_adata, "var_names"):
                names = [str(n) for n in fi_adata.var_names]
            else:
                names = [f"FEATURE_{i}" for i in range(len(values))]
            if len(names) == len(values) and values:
                feature_importances = {
                    name: round(val, 6) for name, val in zip(names, values)
                }
                best_idx = max(range(len(values)), key=lambda idx: abs(values[idx]))
                top_modality = format_avi_feature_name(names[best_idx])
                top_feature_val = round(values[best_idx], 6)

    impact_band, top_percentile = classify_avi_phred(phred_val)
    overlapping_gene = find_overlapping_diabetes_gene(
        parsed_variant["chromosome"], int(parsed_variant["position"])
    )
    effective_mods = list(modalities)
    if top_modality == "Splicing" and "SPLICE_JUNCTIONS" not in effective_mods:
        effective_mods.append("SPLICE_JUNCTIONS")
    atlas_url = build_atlas_variant_url(
        parsed_variant["variant"],
        biosample=biosample,
        modalities=effective_mods,
        tfs=tfs,
    )
    return AlphaGenomeVariantScore(
        variant=parsed_variant["variant"],
        chromosome=parsed_variant["chromosome"],
        position=int(parsed_variant["position"]),
        ref=parsed_variant["ref"],
        alt=parsed_variant["alt"],
        avi_phred=phred_val,
        avi_raw=raw_score,
        avi_quantile=quantile_val,
        top_percentile=top_percentile,
        impact_band=impact_band,
        top_modality=top_modality,
        top_feature_importance=top_feature_val,
        atlas_url=atlas_url,
        query_mode="live_api",
        overlapping_diabetes_gene=overlapping_gene,
        feature_importances=feature_importances,
    )


def _offline_variant_score(
    parsed_variant: dict[str, Any],
    *,
    biosample: str | None = None,
    modalities: Sequence[str] = ("RNA_SEQ", "DNASE", "CHIP_TF"),
    tfs: Sequence[str] = (),
) -> AlphaGenomeVariantScore:
    overlapping_gene = find_overlapping_diabetes_gene(
        parsed_variant["chromosome"], int(parsed_variant["position"])
    )
    locus = DIABETES_ALPHAGENOME_LOCI.get(overlapping_gene) if overlapping_gene else None
    effective_biosample = biosample or (locus.primary_biosample if locus else None)
    effective_tfs = list(tfs) if tfs else (list(locus.key_transcription_factors) if locus else [])
    effective_mods = list(modalities) if modalities else (list(locus.default_modalities) if locus else ["RNA_SEQ", "DNASE", "CHIP_TF"])
    atlas_url = build_atlas_variant_url(
        parsed_variant["variant"],
        biosample=effective_biosample,
        modalities=effective_mods,
        tfs=effective_tfs,
    )
    return AlphaGenomeVariantScore(
        variant=parsed_variant["variant"],
        chromosome=parsed_variant["chromosome"],
        position=int(parsed_variant["position"]),
        ref=parsed_variant["ref"],
        alt=parsed_variant["alt"],
        avi_phred=None,
        avi_raw=None,
        avi_quantile=None,
        top_percentile=None,
        impact_band="unscored_offline",
        top_modality=None,
        top_feature_importance=None,
        atlas_url=atlas_url,
        query_mode="atlas_link_only",
        overlapping_diabetes_gene=overlapping_gene,
    )


def _write_alphagenome_bundle(
    *,
    output_dir: Path,
    stem_prefix: str,
    query_type: str,
    query_target: str,
    query_mode: str,
    primary_atlas_url: str,
    records: Sequence[AlphaGenomeVariantScore],
    extra_metadata: Mapping[str, Any] | None = None,
) -> AlphaGenomeEvidenceResult:
    output_root = output_dir.expanduser().resolve()
    run_dir = output_root / f"alphagenome_{safe_stem(stem_prefix)}_{timestamp_token()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    csv_path = run_dir / "alphagenome_variants.csv"
    fieldnames = [
        "rank",
        "variant",
        "chromosome",
        "position",
        "ref",
        "alt",
        "avi_phred",
        "avi_raw",
        "avi_quantile",
        "top_percentile",
        "impact_band",
        "top_modality",
        "top_feature_importance",
        "overlapping_diabetes_gene",
        "query_mode",
        "atlas_url",
        "supports_quantitative_functional_scalar",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, record in enumerate(records, start=1):
            writer.writerow(sanitize_csv_mapping(record.to_csv_row(rank=rank)))

    evidence_path = run_dir / "alphagenome_evidence.json"
    payload: dict[str, Any] = {
        "schema_version": ALPHAGENOME_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "research_only": True,
        "medical_device": False,
        "supports_quantitative_functional_scalar": False,
        "csv_formula_protection": True,
        "source": {
            "name": "Google AlphaGenome & AlphaGenome Atlas",
            "atlas_base_url": BASE_ATLAS_URL,
            "docs_url": ALPHAGENOME_DOCS_URL,
            "query_type": query_type,
            "query_target": query_target,
            "query_mode": query_mode,
            "primary_atlas_url": primary_atlas_url,
        },
        "record_count": len(records),
        "records": [asdict(record) for record in records],
        "limitations": list(ALPHAGENOME_LIMITATIONS),
    }
    if extra_metadata:
        payload["context"] = dict(extra_metadata)
    write_json(evidence_path, payload)

    report_lines = [
        "# IINTS Google AlphaGenome Evidence Bundle",
        "",
        f"- Query type: `{query_type}`",
        f"- Query target: `{query_target}`",
        f"- Execution mode: `{query_mode}`",
        f"- Variants exported: `{len(records)}`",
        f"- Primary AlphaGenome Atlas Deep-Link: [View in AlphaGenome Atlas]({primary_atlas_url})",
        "",
        "## Variant Summary & Mandatory Atlas Deep-Links",
        "",
        "| Rank | Variant | AVI Phred | Top Percentile | Top Modality | Diabetes Locus | Atlas Deep-Link |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for rank, rec in enumerate(records[:25], start=1):
        phred_text = f"**{rec.avi_phred:.2f}**" if rec.avi_phred is not None else "N/A (offline)"
        pct_text = f"Top {rec.top_percentile:.4f}%" if rec.top_percentile is not None else "N/A"
        mod_text = f"`{rec.top_modality}`" if rec.top_modality else "N/A"
        gene_text = f"`{rec.overlapping_diabetes_gene}`" if rec.overlapping_diabetes_gene else "—"
        report_lines.append(
            f"| {rank} | `{rec.variant}` | {phred_text} | {pct_text} | {mod_text} | {gene_text} | [View in Atlas]({rec.atlas_url}) |"
        )
    report_lines.extend(
        [
            "",
            "## Research & Clinical Safety Boundary",
            "",
            "AlphaGenome and AlphaGenome Variant Impact (AVI) models are strictly non-clinical research tools.",
            "A high AVI Phred score reflects predicted molecular or regulatory impact (such as splicing change,",
            "chromatin accessibility shift, or transcription factor binding alteration) and must NEVER be",
            "extrapolated to clinical disease causation, patient prognosis, or insulin dosing.",
            "",
            "No value in this bundle modifies IINTS virtual-patient parameters (`insulin_sensitivity`,",
            "`molecular_affinity_scalar`, EGP, or PK/PD) automatically.",
            "",
        ]
    )
    report_path = run_dir / "ALPHAGENOME_REVIEW.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return AlphaGenomeEvidenceResult(
        output_dir=run_dir,
        records_csv=csv_path,
        evidence_json=evidence_path,
        report_md=report_path,
        query_type=query_type,
        query_target=query_target,
        query_mode=query_mode,
        record_count=len(records),
        atlas_url=primary_atlas_url,
    )


def query_alphagenome_variants(
    variants: Sequence[str] | str,
    output_dir: Path,
    *,
    biosample: str | None = None,
    modalities: Sequence[str] = ("RNA_SEQ", "DNASE", "CHIP_TF"),
    tfs: Sequence[str] = (),
    offline: bool = False,
    require_live_api: bool = False,
    _client: Any | None = None,
) -> AlphaGenomeEvidenceResult:
    """Query one or more `chr:pos:ref>alt` variants via AlphaGenome and export an evidence bundle."""

    if isinstance(variants, str):
        raw_list = [v.strip() for v in variants.split(",") if v.strip()]
    else:
        raw_list = [str(v).strip() for v in variants if str(v).strip()]
    if not raw_list:
        raise ValueError("At least one genomic variant in 'chr:pos:ref>alt' format is required.")
    if len(raw_list) > 500:
        raise ValueError("Batch size exceeds the 500-variant interactive safety limit.")

    parsed_variants = [normalize_genomic_variant(item) for item in raw_list]

    client = _client
    if client is None and not offline:
        api_key = _resolve_api_key()
        sdk_available = importlib.util.find_spec("alphagenome") is not None
        if api_key and sdk_available:
            client = _create_live_atlas_client(api_key)
        elif require_live_api:
            raise RuntimeError(
                "Live AlphaGenome API query requested, but either the 'alphagenome' package "
                "is not installed or ALPHAGENOME_API_KEY is not set in the environment / ~/.env."
            )
    elif require_live_api and offline:
        raise ValueError("Cannot specify both offline=True and require_live_api=True.")

    scored_records: list[AlphaGenomeVariantScore] = []
    if client is not None and not offline:
        for parsed in parsed_variants:
            scored_records.append(
                _score_variant_with_client(
                    parsed,
                    client,
                    biosample=biosample,
                    modalities=modalities,
                    tfs=tfs,
                )
            )
        scored_records.sort(
            key=lambda r: (
                r.avi_phred if r.avi_phred is not None else -999.0,
                r.avi_raw if r.avi_raw is not None else -999.0,
            ),
            reverse=True,
        )
        query_mode = "live_api"
    else:
        for parsed in parsed_variants:
            scored_records.append(
                _offline_variant_score(
                    parsed,
                    biosample=biosample,
                    modalities=modalities,
                    tfs=tfs,
                )
            )
        query_mode = "atlas_link_only"

    primary_url = scored_records[0].atlas_url
    target_label = parsed_variants[0]["variant"] if len(parsed_variants) == 1 else f"{len(parsed_variants)}_variants"
    return _write_alphagenome_bundle(
        output_dir=output_dir,
        stem_prefix=target_label,
        query_type="variant",
        query_target=",".join(p["variant"] for p in parsed_variants),
        query_mode=query_mode,
        primary_atlas_url=primary_url,
        records=scored_records,
        extra_metadata={
            "biosample": biosample,
            "modalities": _normalize_modalities(modalities),
            "tfs": list(tfs),
        },
    )


def scan_alphagenome_region(
    region: str,
    output_dir: Path,
    *,
    biosample: str | None = None,
    modalities: Sequence[str] = ("RNA_SEQ", "DNASE", "CHIP_TF"),
    tfs: Sequence[str] = (),
    min_phred: float = 15.0,
    top_k: int = 20,
    offline: bool = False,
    require_live_api: bool = False,
    _client: Any | None = None,
) -> AlphaGenomeEvidenceResult:
    """Scan a 1-based closed genomic interval (`chr:start-end`) with AlphaGenome AVI."""

    parsed_interval = normalize_genomic_interval(region)
    if parsed_interval["width_bp"] > MAX_REGION_SCAN_BP and not offline:
        raise ValueError(
            f"Region width ({parsed_interval['width_bp']} bp) exceeds the {MAX_REGION_SCAN_BP} bp "
            "live saturation-mutagenesis limit."
        )
    if top_k < 1 or top_k > 500:
        raise ValueError("top_k must be between 1 and 500.")

    locus_url = build_atlas_locus_url(
        parsed_interval["interval"],
        biosample=biosample,
        modalities=modalities,
        tfs=tfs,
    )

    client = _client
    if client is None and not offline:
        api_key = _resolve_api_key()
        sdk_available = importlib.util.find_spec("alphagenome") is not None
        if api_key and sdk_available:
            client = _create_live_atlas_client(api_key)
        elif require_live_api:
            raise RuntimeError(
                "Live AlphaGenome region scan requested, but either the 'alphagenome' package "
                "is not installed or ALPHAGENOME_API_KEY is not set."
            )

    records: list[AlphaGenomeVariantScore] = []
    if client is not None and not offline:
        try:
            from alphagenome.data import genome  # type: ignore[import-not-found]

            interval_obj: Any = genome.Interval(
                chromosome=parsed_interval["chromosome"],
                start=int(parsed_interval["start_0_based"]),
                end=int(parsed_interval["end_0_based_half_open"]),
            )
        except ImportError:
            interval_obj = parsed_interval

        results = client.query_interval(
            interval_obj,
            requested_scorers=["AVI_SCORE", "AVI_SCORE_FEATURE_IMPORTANCE"],
        )
        avi_adata = results.get("AVI_SCORE") if isinstance(results, Mapping) else None
        fi_adata = results.get("AVI_SCORE_FEATURE_IMPORTANCE") if isinstance(results, Mapping) else None

        if avi_adata is not None and getattr(avi_adata, "obs", None) is not None and "variant" in avi_adata.obs:
            variant_objs = list(avi_adata.obs["variant"])
            raw_scores = avi_adata.X.reshape(-1) if getattr(avi_adata, "X", None) is not None else None
            layers = getattr(avi_adata, "layers", {}) or {}
            quantiles = layers["quantiles"].reshape(-1) if "quantiles" in layers else None
            fi_matrix = fi_adata.X if (fi_adata is not None and getattr(fi_adata, "X", None) is not None) else None
            var_df = getattr(fi_adata, "var", None) if fi_adata is not None else None
            if var_df is not None and hasattr(var_df, "columns") and "name" in var_df.columns:
                fi_names = [str(n) for n in var_df["name"]]
            elif fi_adata is not None and hasattr(fi_adata, "var_names"):
                fi_names = [str(n) for n in fi_adata.var_names]
            else:
                fi_names = []

            for idx, var_item in enumerate(variant_objs):
                parsed_var = normalize_genomic_variant(str(var_item))
                raw_val = round(float(raw_scores[idx]), 6) if raw_scores is not None else None
                phred_val: float | None = None
                quantile_val: float | None = None
                if quantiles is not None:
                    q_tail, p_score = _cdf_to_phred(float(quantiles[idx]))
                    quantile_val = round(q_tail, 6)
                    phred_val = round(p_score, 4)
                if phred_val is not None and phred_val < min_phred:
                    continue

                fi_dict: dict[str, float] = {}
                top_mod: str | None = None
                top_val: float | None = None
                if fi_matrix is not None and len(fi_names) == fi_matrix.shape[1]:
                    row_vals = [float(v) for v in fi_matrix[idx].reshape(-1)]
                    fi_dict = {name: round(val, 6) for name, val in zip(fi_names, row_vals)}
                    best_i = max(range(len(row_vals)), key=lambda i: abs(row_vals[i]))
                    top_mod = format_avi_feature_name(fi_names[best_i])
                    top_val = round(row_vals[best_i], 6)

                band, top_pct = classify_avi_phred(phred_val)
                overlapping_gene = find_overlapping_diabetes_gene(
                    parsed_var["chromosome"], int(parsed_var["position"])
                )
                records.append(
                    AlphaGenomeVariantScore(
                        variant=parsed_var["variant"],
                        chromosome=parsed_var["chromosome"],
                        position=int(parsed_var["position"]),
                        ref=parsed_var["ref"],
                        alt=parsed_var["alt"],
                        avi_phred=phred_val,
                        avi_raw=raw_val,
                        avi_quantile=quantile_val,
                        top_percentile=top_pct,
                        impact_band=band,
                        top_modality=top_mod,
                        top_feature_importance=top_val,
                        atlas_url=build_atlas_variant_url(
                            parsed_var["variant"],
                            biosample=biosample,
                            modalities=modalities,
                            tfs=tfs,
                        ),
                        query_mode="live_api",
                        overlapping_diabetes_gene=overlapping_gene,
                        feature_importances=fi_dict,
                    )
                )

        records.sort(
            key=lambda r: (
                r.avi_phred if r.avi_phred is not None else -999.0,
                r.avi_raw if r.avi_raw is not None else -999.0,
            ),
            reverse=True,
        )
        records = records[:top_k]
        query_mode = "live_api"
    else:
        query_mode = "atlas_link_only"

    return _write_alphagenome_bundle(
        output_dir=output_dir,
        stem_prefix=parsed_interval["interval"],
        query_type="region",
        query_target=parsed_interval["interval"],
        query_mode=query_mode,
        primary_atlas_url=locus_url,
        records=records,
        extra_metadata={
            "interval_1_based": parsed_interval["interval"],
            "interval_0_based_half_open": [
                parsed_interval["start_0_based"],
                parsed_interval["end_0_based_half_open"],
            ],
            "width_bp": parsed_interval["width_bp"],
            "min_phred": min_phred,
            "top_k": top_k,
            "biosample": biosample,
            "modalities": _normalize_modalities(modalities),
            "tfs": list(tfs),
        },
    )
