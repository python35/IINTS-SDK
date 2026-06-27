"""Physiology pathway renders for explanatory research assets.

STRING network images help users connect SDK model terms such as insulin
signalling or glucagon rescue to known protein interaction pathways.  These
figures are documentation and education artifacts only; they do not feed into
the simulator, controllers, safety checks, or treatment logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ssl
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from rich.console import Console


console = Console()
DEFAULT_OUTPUT_DIR: Final = Path("results") / "structural"
STRING_NETWORK_URL: Final = "https://string-db.org/api/highres_image/network"


class PhysiologyPathwayError(RuntimeError):
    """Raised when a pathway network cannot be downloaded safely."""


@dataclass(frozen=True)
class PathwayNetwork:
    """One preconfigured physiology network from STRING DB."""

    key: str
    genes: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class PathwayRenderResult:
    """Image artifact emitted by a successful STRING render."""

    network: str
    png_path: Path
    source_url: str


NETWORKS: Final[dict[str, PathwayNetwork]] = {
    "insulin-cascade": PathwayNetwork(
        key="insulin-cascade",
        genes=("INSR", "IRS1", "PIK3CA", "AKT1", "SLC2A4"),
        description="Insulin receptor signalling and AKT-mediated GLUT4 translocation.",
    ),
    "glucagon-rescue": PathwayNetwork(
        key="glucagon-rescue",
        genes=("GCGR", "GNAS", "PRKACA", "PYGL"),
        description="Glucagon/cAMP signalling involved in hepatic glycogenolysis.",
    ),
}


def fetch_string_network(network_name: str, out_dir: Path = DEFAULT_OUTPUT_DIR) -> PathwayRenderResult:
    """Download one high-resolution STRING confidence-network PNG."""

    network = NETWORKS.get(network_name)
    if network is None:
        choices = ", ".join([*NETWORKS, "all"])
        raise PhysiologyPathwayError(f"Unknown network '{network_name}'. Choose one of: {choices}.")

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{network.key}_string_network.png"
    query = urlencode(
        {
            "identifiers": "\r".join(network.genes),
            "species": "9606",
            "network_flavor": "confidence",
        }
    )
    api_url = f"{STRING_NETWORK_URL}?{query}"

    console.print(f"[yellow]Downloading physiological network '{network.key}' from STRING DB...[/yellow]")
    request = Request(api_url, headers={"User-Agent": "IINTS-AF-SDK/physiology-renderer"})
    try:
        with urlopen(request, timeout=90, context=_verified_https_context()) as response:  # noqa: S310
            image_data = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise PhysiologyPathwayError(f"Could not download STRING network '{network.key}': {exc}") from exc

    if not image_data.startswith(b"\x89PNG"):
        raise PhysiologyPathwayError(f"STRING did not return a PNG image for '{network.key}'.")
    png_path.write_bytes(image_data)
    console.print(f"[green]Saved physiological network: {png_path}[/green]")
    return PathwayRenderResult(network=network.key, png_path=png_path, source_url=api_url)


def render_pathways(network: str, *, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[PathwayRenderResult]:
    """Render one pathway network or all configured pathway networks."""

    if network != "all" and network not in NETWORKS:
        choices = ", ".join([*NETWORKS, "all"])
        raise PhysiologyPathwayError(f"Unknown network '{network}'. Choose one of: {choices}.")
    requested = list(NETWORKS) if network == "all" else [network]
    return [fetch_string_network(name, output_dir) for name in requested]


def _verified_https_context() -> ssl.SSLContext:
    """Build a verified HTTPS context that also works with python.org macOS builds."""

    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())
