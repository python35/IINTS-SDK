"""Optional structural-biology renders for research and educational explanation.

This module keeps AlphaFold retrieval and PyMOL isolated from the physiological
simulation engine.  The rendered structures are explanatory assets only; they
do not provide inputs to dosing, safety, or treatment logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rich.console import Console


console = Console()

ALPHAFOLD_API_TEMPLATE: Final = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
DEFAULT_CACHE_DIR: Final = Path(".cache") / "alphafold"
DEFAULT_OUTPUT_DIR: Final = Path("results") / "structural"
TARGETS: Final[dict[str, str]] = {
    "insulin-mutation": "P01308",
    "glucagon": "P01275",
    "glut4": "P14672",
    "insulin-receptor": "P06213",
}


class StructuralRenderError(RuntimeError):
    """Raised when a structure cannot be safely downloaded or rendered."""


@dataclass(frozen=True)
class StructuralRenderResult:
    """Paths emitted by a successful isolated PyMOL render."""

    target: str
    uniprot_id: str
    cif_path: Path
    png_path: Path
    session_path: Path


@dataclass(frozen=True)
class PAEHeatmapResult:
    """Interactive Predicted Aligned Error artifact emitted by Plotly."""

    target: str
    uniprot_id: str
    pae_url: str
    html_path: Path
    residue_count: int
    max_predicted_aligned_error: float


def is_valid_mmcif(path: Path) -> bool:
    """Check that a cached file is a mmCIF payload, not an HTTP error page."""

    try:
        prefix = path.read_text(encoding="utf-8", errors="replace")[:256].lstrip()
    except OSError:
        return False
    return prefix.startswith("data_")


def download_alphafold_cif(uniprot_id: str, out_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Download and atomically cache one AlphaFold mmCIF with TLS verification."""

    out_dir.mkdir(parents=True, exist_ok=True)
    cif_path = out_dir / f"{uniprot_id}.cif"
    if is_valid_mmcif(cif_path):
        console.print(f"[green]Using cached AlphaFold mmCIF for {uniprot_id}[/green]")
        return cif_path
    if cif_path.exists():
        console.print(f"[yellow]Replacing invalid cached mmCIF for {uniprot_id}[/yellow]")

    metadata_url = ALPHAFOLD_API_TEMPLATE.format(uniprot_id=uniprot_id)
    request = Request(metadata_url, headers={"User-Agent": "IINTS-AF-SDK/structure-renderer"})
    console.print(f"[blue]Requesting AlphaFold metadata for {uniprot_id}...[/blue]")
    try:
        with urlopen(request, timeout=30, context=_verified_https_context()) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        cif_url = _extract_cif_url(payload, uniprot_id)
        cif_request = Request(cif_url, headers={"User-Agent": "IINTS-AF-SDK/structure-renderer"})
        with urlopen(cif_request, timeout=90, context=_verified_https_context()) as response:  # noqa: S310
            mmcif_text = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise StructuralRenderError(
            f"Could not download AlphaFold structure {uniprot_id}: {exc}"
        ) from exc

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".cif", prefix=f"{uniprot_id}-", dir=out_dir, delete=False
    ) as temporary:
        temporary.write(mmcif_text)
        temporary_path = Path(temporary.name)
    try:
        if not is_valid_mmcif(temporary_path):
            raise StructuralRenderError(
                f"AlphaFold returned a non-mmCIF response for {uniprot_id}; cache was not changed."
            )
        temporary_path.replace(cif_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    console.print(f"[green]Cached AlphaFold mmCIF: {cif_path}[/green]")
    return cif_path


def generate_pymol_script(*, target: str, cif_path: Path, png_path: Path, session_path: Path) -> str:
    """Create the isolated PEP 723 script used by ``uv run`` and PyMOL."""

    if target not in TARGETS:
        raise StructuralRenderError(f"Unknown structural target: {target}")

    quoted_cif = json.dumps(str(cif_path))
    quoted_target = json.dumps(target)
    quoted_png = json.dumps(str(png_path))
    quoted_session = json.dumps(str(session_path))
    script = f'''# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["pymol-open-source-whl"]
# ///

import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pymol

pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()
from pymol import cmd

cmd.reinitialize()
cmd.load({quoted_cif}, {quoted_target})
if cmd.count_atoms("all") == 0:
    raise RuntimeError("PyMOL did not load any atoms from the mmCIF input")

cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
'''
    if target == "insulin-mutation":
        script += f'''
cmd.show_as("cartoon")
cmd.color("gray80", {quoted_target})
cmd.select("mut_target", "resi 52+53")
cmd.show("sticks", "mut_target")
cmd.color("hotpink", "mut_target")
cmd.color("lightblue", "resi 25-54")
cmd.color("palegreen", "resi 90-110")
cmd.zoom("mut_target", buffer=5)
'''
    elif target == "glut4":
        script += '''
cmd.show_as("cartoon")
cmd.color("orange", "ss h")
cmd.color("yellow", "ss s")
cmd.color("gray", "ss l+''")
cmd.show("surface")
cmd.set("transparency", 0.6)
cmd.set("surface_color", "white")
cmd.orient()
'''
    else:
        script += '''
cmd.show_as("cartoon")
cmd.spectrum("b", "red_yellow_green_cyan_blue", minimum=50, maximum=90)
cmd.orient()
'''
    return script + f'''
cmd.png({quoted_png}, width=1600, height=1200, dpi=300)
cmd.save({quoted_session})
cmd.quit()
'''


def render_target(
    target: str,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    uv_binary: str = "uv",
) -> list[StructuralRenderResult]:
    """Render one target or all available targets through an isolated PyMOL runtime."""

    if target != "all" and target not in TARGETS:
        choices = ", ".join([*TARGETS, "all"])
        raise StructuralRenderError(f"Unknown target '{target}'. Choose one of: {choices}.")
    uv_path = shutil.which(uv_binary)
    if uv_path is None:
        raise StructuralRenderError(
            "The 'uv' executable is required for isolated PyMOL rendering. Install uv and retry."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    requested_targets = list(TARGETS) if target == "all" else [target]
    rendered: list[StructuralRenderResult] = []
    for target_name in requested_targets:
        uniprot_id = TARGETS[target_name]
        cif_path = download_alphafold_cif(uniprot_id, cache_dir)
        png_path = output_dir / f"{target_name}.png"
        session_path = output_dir / f"{target_name}.pse"
        script_text = generate_pymol_script(
            target=target_name,
            cif_path=cif_path,
            png_path=png_path,
            session_path=session_path,
        )

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".py", delete=False) as script_file:
            script_file.write(script_text)
            script_path = Path(script_file.name)
        console.print(f"[blue]Rendering {target_name} in isolated PyMOL...[/blue]")
        try:
            subprocess.run([uv_path, "run", "--python", "3.12", str(script_path)], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise StructuralRenderError(f"PyMOL rendering failed for {target_name}: {exc}") from exc
        finally:
            script_path.unlink(missing_ok=True)
        if not png_path.exists() or not session_path.exists():
            raise StructuralRenderError(f"PyMOL did not create expected artifacts for {target_name}.")
        result = StructuralRenderResult(target_name, uniprot_id, cif_path, png_path, session_path)
        rendered.append(result)
        console.print(f"[green]Rendered {target_name}: {png_path}[/green]")
    return rendered


def _extract_cif_url(payload: object, uniprot_id: str) -> str:
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise StructuralRenderError(f"AlphaFold returned no prediction metadata for {uniprot_id}.")
    cif_url = payload[0].get("cifUrl")
    if not isinstance(cif_url, str) or not cif_url.startswith("https://"):
        raise StructuralRenderError(f"AlphaFold metadata has no valid HTTPS mmCIF URL for {uniprot_id}.")
    return cif_url


def render_pae(
    target: str,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[PAEHeatmapResult]:
    """Render interactive AlphaFold Predicted Aligned Error heatmaps.

    The PAE matrix is an explanatory structural-biology artifact.  It is not
    used by the simulator, safety supervisor, dosing logic, or clinical report
    scoring.  The generated HTML is standalone and can be opened from the CLI or
    the desktop app.
    """

    if target != "all" and target not in TARGETS:
        choices = ", ".join([*TARGETS, "all"])
        raise StructuralRenderError(f"Unknown target '{target}'. Choose one of: {choices}.")

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise StructuralRenderError(
            "Plotly is required for interactive PAE heatmaps. "
            'Install it with: python -m pip install -U -e ".[research]"'
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    targets_to_run = list(TARGETS.keys()) if target == "all" else [target]
    rendered: list[PAEHeatmapResult] = []

    for t in targets_to_run:
        uniprot = TARGETS[t]
        console.print(f"[yellow]Fetching Predicted Aligned Error (PAE) matrix for {t} ({uniprot})...[/yellow]")
        metadata = _download_json(ALPHAFOLD_API_TEMPLATE.format(uniprot_id=uniprot))
        pae_doc_url = _extract_pae_url(metadata, uniprot)
        pae_payload = _download_json(pae_doc_url)
        pae_matrix, max_pae = _extract_pae_matrix(pae_payload, uniprot)
        residue_count = len(pae_matrix)
        residue_numbers = list(range(1, residue_count + 1))

        colorscale = [
            [0.0, "#005a32"],
            [0.45, "#74c476"],
            [0.75, "#d9f0d3"],
            [1.0, "#ffffff"],
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=pae_matrix,
                x=residue_numbers,
                y=residue_numbers,
                colorscale=colorscale,
                zmin=0,
                zmax=max(1.0, min(30.0, max_pae)),
                colorbar={"title": "PAE (A)"},
                hovertemplate=(
                    "Scored residue: %{x}<br>"
                    "Aligned residue: %{y}<br>"
                    "Expected position error: %{z:.1f} A<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=f"Predicted Aligned Error (PAE) - {t} / UniProt {uniprot}",
            xaxis_title="Scored residue",
            yaxis_title="Aligned residue",
            yaxis_autorange="reversed",
            template="plotly_white",
            width=900,
            height=840,
            margin={"l": 72, "r": 40, "t": 80, "b": 72},
        )

        html_out = output_dir / f"{t}_pae.html"
        fig.write_html(str(html_out), include_plotlyjs=True, full_html=True)
        result = PAEHeatmapResult(
            target=t,
            uniprot_id=uniprot,
            pae_url=pae_doc_url,
            html_path=html_out,
            residue_count=residue_count,
            max_predicted_aligned_error=max_pae,
        )
        rendered.append(result)
        console.print(f"[green]Saved interactive PAE heatmap: {html_out}[/green]")
    return rendered


def _download_json(url: str) -> object:
    if not url.startswith("https://"):
        raise StructuralRenderError(f"Refusing to download non-HTTPS structural resource: {url}")
    request = Request(url, headers={"User-Agent": "IINTS-AF-SDK/structure-renderer"})
    try:
        with urlopen(request, timeout=60, context=_verified_https_context()) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise StructuralRenderError(f"Could not download AlphaFold JSON resource: {exc}") from exc


def _verified_https_context() -> ssl.SSLContext:
    """Build a verified HTTPS context that also works with python.org macOS builds."""

    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _extract_pae_url(payload: object, uniprot_id: str) -> str:
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise StructuralRenderError(f"AlphaFold returned no prediction metadata for {uniprot_id}.")
    pae_url = payload[0].get("paeDocUrl")
    if not isinstance(pae_url, str) or not pae_url.startswith("https://"):
        raise StructuralRenderError(f"AlphaFold metadata has no valid HTTPS PAE URL for {uniprot_id}.")
    return pae_url


def _extract_pae_matrix(payload: object, uniprot_id: str) -> tuple[list[list[float]], float]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        document = payload[0]
    elif isinstance(payload, dict):
        document = payload
    else:
        raise StructuralRenderError(f"AlphaFold PAE document is not readable for {uniprot_id}.")

    raw_matrix = document.get("predicted_aligned_error")
    raw_max = document.get("max_predicted_aligned_error", 30.0)
    if not isinstance(raw_matrix, list) or not raw_matrix:
        raise StructuralRenderError(f"AlphaFold PAE document has no matrix for {uniprot_id}.")

    matrix: list[list[float]] = []
    for row in raw_matrix:
        if not isinstance(row, list) or len(row) != len(raw_matrix):
            raise StructuralRenderError(f"AlphaFold PAE matrix is not square for {uniprot_id}.")
        try:
            matrix.append([float(value) for value in row])
        except (TypeError, ValueError) as exc:
            raise StructuralRenderError(f"AlphaFold PAE matrix contains non-numeric values for {uniprot_id}.") from exc
    try:
        max_pae = float(raw_max)
    except (TypeError, ValueError):
        max_pae = 30.0
    return matrix, max_pae
