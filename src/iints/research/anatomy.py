"""GTEx tissue-expression renders for anatomy-aware model explanations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import ssl
from typing import Any, Final
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from rich.console import Console

console = Console()

GTEX_API: Final = "https://gtexportal.org/api/v2"
USER_AGENT: Final = "IINTS-AF-SDK/anatomy-research"
DEFAULT_OUTPUT_DIR: Final = Path("results") / "structural"
GENE_ALIASES: Final[dict[str, str]] = {
    "GLUT4": "SLC2A4",
    "GLUCAGON-RECEPTOR": "GCGR",
    "GLUCAGON_RECEPTOR": "GCGR",
    "INSULIN-RECEPTOR": "INSR",
    "INSULIN_RECEPTOR": "INSR",
}


class GTExError(RuntimeError):
    """Raised when GTEx lookup or rendering fails."""


@dataclass(frozen=True)
class TissueExpression:
    """Median GTEx expression for one tissue."""

    tissue: str
    median_tpm: float


@dataclass(frozen=True)
class ExpressionRenderResult:
    """Interactive expression artifact produced for one gene."""

    requested_gene: str
    official_gene: str
    gencode_id: str
    html_path: Path
    tissues: tuple[TissueExpression, ...]


def official_gene_symbol(gene: str) -> str:
    """Normalize common aliases to symbols used by GTEx."""

    stripped = gene.strip()
    if not stripped:
        raise GTExError("Gene symbol cannot be empty.")
    return GENE_ALIASES.get(stripped.upper(), stripped.upper())


def resolve_gtex_gencode_id(gene_symbol: str) -> str:
    """Resolve a gene symbol to the versioned GTEx/GENCODE identifier."""

    official = official_gene_symbol(gene_symbol)
    url = f"{GTEX_API}/reference/gene?format=json&geneId={quote(official)}&datasetId=gtex_v8"
    payload = _download_json(url)
    genes = payload.get("data", [])
    if not isinstance(genes, list) or not genes:
        raise GTExError(f"No GTEx GENCODE mapping found for {official}.")
    first = genes[0]
    if not isinstance(first, dict):
        raise GTExError(f"GTEx mapping payload is malformed for {official}.")
    gencode_id = first.get("gencodeId")
    if not isinstance(gencode_id, str) or not gencode_id:
        raise GTExError(f"GTEx mapping for {official} has no gencodeId.")
    return gencode_id


def fetch_gtex_expression(gene_symbol: str) -> list[TissueExpression]:
    """Fetch median GTEx v8 expression across tissues for a gene symbol."""

    official = official_gene_symbol(gene_symbol)
    gencode_id = resolve_gtex_gencode_id(official)
    return fetch_gtex_expression_by_gencode(official, gencode_id)


def fetch_gtex_expression_by_gencode(official_gene: str, gencode_id: str) -> list[TissueExpression]:
    """Fetch median GTEx v8 expression using an already-resolved GENCODE ID."""

    url = (
        f"{GTEX_API}/expression/medianGeneExpression?datasetId=gtex_v8"
        f"&format=json&gencodeId={quote(gencode_id)}"
    )
    payload = _download_json(url)
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise GTExError(f"GTEx expression payload is malformed for {official_gene}.")

    tissues: list[TissueExpression] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tissue = str(row.get("tissueSiteDetailId") or row.get("tissueSiteDetail") or "Unknown tissue")
        try:
            median = float(row.get("median") or 0.0)
        except (TypeError, ValueError):
            median = 0.0
        tissues.append(TissueExpression(tissue=tissue, median_tpm=median))
    return sorted(tissues, key=lambda item: item.median_tpm, reverse=True)


def render_expression(gene: str, *, output_dir: Path = DEFAULT_OUTPUT_DIR) -> ExpressionRenderResult | None:
    """Render an interactive GTEx tissue-expression bar chart."""

    official = official_gene_symbol(gene)
    console.print(f"[yellow]Fetching GTEx tissue expression for {gene} (official symbol: {official})...[/yellow]")
    try:
        gencode_id = resolve_gtex_gencode_id(official)
        tissues = fetch_gtex_expression_by_gencode(official, gencode_id)
    except GTExError as exc:
        console.print(f"[red]{exc}[/red]")
        return None
    if not tissues:
        console.print(f"[red]No GTEx expression data found for {official}.[/red]")
        return None

    try:
        import plotly.graph_objects as go
    except ImportError:
        console.print('[red]Plotly not installed. Install with: python -m pip install -U -e ".[research]"[/red]')
        return None

    fig = go.Figure(
        data=[
            go.Bar(
                x=[item.tissue for item in tissues],
                y=[item.median_tpm for item in tissues],
                marker_color="#2f6f9f",
                hovertemplate="%{x}<br>Median TPM: %{y:.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"GTEx v8 median tissue expression - {official}",
        xaxis_title="Tissue",
        yaxis_title="Median TPM",
        xaxis_tickangle=-45,
        template="plotly_white",
        height=820,
        margin={"l": 70, "r": 30, "t": 80, "b": 220},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    html_out = output_dir / f"{official}_expression.html"
    fig.write_html(str(html_out), include_plotlyjs=True, full_html=True)
    console.print(f"[green]Saved interactive GTEx expression plot: {html_out}[/green]")
    console.print("[dim]Research/education only: expression evidence explains model compartments; it does not calibrate a patient.[/dim]")
    return ExpressionRenderResult(
        requested_gene=gene,
        official_gene=official,
        gencode_id=gencode_id,
        html_path=html_out,
        tissues=tuple(tissues),
    )


def _download_json(url: str) -> dict[str, Any]:
    if not url.startswith("https://"):
        raise GTExError(f"Refusing to download non-HTTPS GTEx resource: {url}")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=45, context=_verified_https_context()) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise GTExError(f"GTEx request failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise GTExError("GTEx returned a non-object JSON payload.")
    return payload


def _verified_https_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())
