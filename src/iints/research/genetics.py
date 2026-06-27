"""ClinVar-backed genotype stressors for educational digital-twin experiments.

The functions in this module fetch public ClinVar summaries and map a small
curated set of diabetes-relevant genes onto simulator parameters.  The mapping
is intentionally conservative and explanatory: it is not diagnostic genetics,
not a medical-device function, and not used for treatment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import ssl
from typing import Any, Final
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from rich.console import Console
from rich.table import Table

console = Console()

NCBI_EUTILS: Final = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT: Final = "IINTS-AF-SDK/genetics-research"
GENE_EFFECTS: Final[dict[str, tuple[str, ...]]] = {
    "INSR": (
        "Insulin receptor disruption: model insulin sensitivity is reduced for a severe-resistance stress test.",
        "EGP pressure is increased to emulate difficult-to-control hyperglycemia in the virtual patient.",
        "Use only as an educational edge-case; ClinVar records do not calibrate an individual patient.",
    ),
    "INS": (
        "Insulin gene disruption: endogenous insulin secretion is disabled in the virtual patient.",
        "The scenario behaves like an absolute insulin-deficiency stress test.",
        "Use only as an educational phenotype mapping, not as variant interpretation.",
    ),
}


class ClinVarError(RuntimeError):
    """Raised when a ClinVar request or response cannot be interpreted."""


@dataclass(frozen=True)
class ClinVarVariant:
    """Small public ClinVar summary suitable for CLI display."""

    uid: str
    title: str
    clinical_significance: str
    trait: str


def fetch_clinvar_pathogenic(gene: str, *, retmax: int = 5) -> list[ClinVarVariant]:
    """Fetch pathogenic/likely pathogenic ClinVar records for a gene symbol."""

    normalized_gene = gene.strip().upper()
    if not normalized_gene:
        raise ClinVarError("Gene symbol cannot be empty.")

    search_term = f"{normalized_gene}[gene] AND (pathogenic[clinsig] OR likely pathogenic[clinsig])"
    esearch_url = (
        f"{NCBI_EUTILS}/esearch.fcgi?db=clinvar&term={quote(search_term)}"
        f"&retmode=json&retmax={max(1, min(retmax, 25))}"
    )
    try:
        search_data = _download_json(esearch_url)
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not isinstance(id_list, list) or not id_list:
            return []

        ids = ",".join(str(uid) for uid in id_list)
        esummary_url = f"{NCBI_EUTILS}/esummary.fcgi?db=clinvar&id={quote(ids)}&retmode=json"
        summary_data = _download_json(esummary_url)
    except (ClinVarError, HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ClinVarError(f"ClinVar request failed for {normalized_gene}: {exc}") from exc

    result_root = summary_data.get("result", {})
    if not isinstance(result_root, dict):
        raise ClinVarError(f"ClinVar summary payload is malformed for {normalized_gene}.")

    variants: list[ClinVarVariant] = []
    for uid in id_list:
        uid_text = str(uid)
        doc = result_root.get(uid_text, {})
        if not isinstance(doc, dict):
            continue
        variants.append(
            ClinVarVariant(
                uid=uid_text,
                title=str(doc.get("title") or "Unknown ClinVar variant"),
                clinical_significance=_clinical_significance(doc),
                trait=_trait_name(doc),
            )
        )
    return variants


def simulate_mutation(gene: str) -> list[ClinVarVariant]:
    """Fetch ClinVar records and print the SDK's deterministic stress-test mapping."""

    normalized_gene = gene.strip().upper()
    console.print(f"[yellow]Fetching public ClinVar variant summaries for {normalized_gene}...[/yellow]")

    try:
        variants = fetch_clinvar_pathogenic(normalized_gene)
    except ClinVarError as exc:
        console.print(f"[red]{exc}[/red]")
        return []

    if not variants:
        console.print(f"[red]No pathogenic or likely pathogenic ClinVar records found for {normalized_gene}.[/red]")
        return []

    table = Table(title=f"ClinVar pathogenic/likely pathogenic records: {normalized_gene}")
    table.add_column("UID", style="cyan", no_wrap=True)
    table.add_column("Variant", style="white")
    table.add_column("Clinical significance", style="yellow")
    table.add_column("Condition/trait", style="red")
    for variant in variants:
        table.add_row(variant.uid, variant.title, variant.clinical_significance, variant.trait)
    console.print(table)

    console.print("\n[bold magenta]Deterministic SDK stress-test mapping[/bold magenta]")
    effects = GENE_EFFECTS.get(normalized_gene)
    if effects is None:
        console.print(
            f"[yellow]{normalized_gene} has no curated simulator-parameter mapping yet. "
            "The ClinVar evidence is displayed, but the virtual patient is not retuned.[/yellow]"
        )
    else:
        for effect in effects:
            console.print(f"  - {effect}")
    console.print("[dim]Research/education only: this is not variant interpretation or clinical genetics.[/dim]")
    return variants


def _download_json(url: str) -> dict[str, Any]:
    if not url.startswith("https://"):
        raise ClinVarError(f"Refusing to download non-HTTPS ClinVar resource: {url}")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30, context=_verified_https_context()) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ClinVarError("ClinVar returned a non-object JSON payload.")
    return payload


def _verified_https_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _clinical_significance(doc: dict[str, Any]) -> str:
    value = doc.get("clinical_significance")
    if isinstance(value, dict):
        description = value.get("description")
        if description:
            return str(description)
    if isinstance(value, str) and value:
        return value
    return "pathogenic/likely pathogenic"


def _trait_name(doc: dict[str, Any]) -> str:
    trait_set = doc.get("trait_set")
    if isinstance(trait_set, list) and trait_set:
        first_trait = trait_set[0]
        if isinstance(first_trait, dict):
            for key in ("trait_name", "name"):
                value = first_trait.get(key)
                if value:
                    return str(value)
    trait = doc.get("trait")
    return str(trait) if trait else "not specified"
