"""ChEMBL-backed insulin analogue lookup with deterministic SDK PK mapping.

ChEMBL is used here for public molecule identity/context.  The absorption
parameters are curated SDK model defaults, not inferred by an LLM and not
computed from ChEMBL structure fields at runtime.
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
from rich.panel import Panel

console = Console()

CHEMBL_SEARCH_URL: Final = "https://www.ebi.ac.uk/chembl/api/data/molecule/search"
USER_AGENT: Final = "IINTS-AF-SDK/pharmacology-research"


class ChEMBLError(RuntimeError):
    """Raised when ChEMBL cannot be queried or parsed safely."""


@dataclass(frozen=True)
class ChEMBLMolecule:
    """Small molecule summary from ChEMBL search."""

    chembl_id: str
    preferred_name: str
    molecular_weight: str


@dataclass(frozen=True)
class InsulinPKProfile:
    """Deterministic SDK pharmacokinetic profile for an insulin class."""

    key: str
    label: str
    tmax_minutes: int
    duration_hours: tuple[float, float]
    simulator_note: str


INSULIN_PK_PROFILES: Final[dict[str, InsulinPKProfile]] = {
    "lispro": InsulinPKProfile("lispro", "rapid-acting analogue", 55, (3.0, 5.0), "fast subcutaneous absorption"),
    "aspart": InsulinPKProfile("aspart", "rapid-acting analogue", 55, (3.0, 5.0), "fast subcutaneous absorption"),
    "fiasp": InsulinPKProfile("fiasp", "faster aspart formulation", 45, (3.0, 5.0), "accelerated early absorption"),
    "glulisine": InsulinPKProfile("glulisine", "rapid-acting analogue", 55, (3.0, 5.0), "fast subcutaneous absorption"),
    "regular": InsulinPKProfile("regular", "short-acting human insulin", 120, (5.0, 8.0), "slower hexamer dissociation"),
    "human": InsulinPKProfile("human", "short-acting human insulin", 120, (5.0, 8.0), "slower hexamer dissociation"),
    "nph": InsulinPKProfile("nph", "intermediate-acting insulin", 360, (12.0, 18.0), "delayed basal-like absorption"),
    "glargine": InsulinPKProfile("glargine", "long-acting basal analogue", 1440, (20.0, 26.0), "flat basal depot profile"),
    "detemir": InsulinPKProfile("detemir", "long-acting basal analogue", 720, (12.0, 24.0), "albumin-binding basal profile"),
    "degludec": InsulinPKProfile("degludec", "ultra-long-acting basal analogue", 2500, (36.0, 42.0), "ultra-slow multi-hexamer depot"),
}


def fetch_chembl_drug(drug_name: str) -> ChEMBLMolecule | None:
    """Fetch a public ChEMBL molecule summary by drug name."""

    normalized = drug_name.strip()
    if not normalized:
        raise ChEMBLError("Drug name cannot be empty.")
    url = f"{CHEMBL_SEARCH_URL}?q={quote(normalized)}&format=json"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=30, context=_verified_https_context()) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ChEMBLError(f"ChEMBL request failed for {normalized}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ChEMBLError("ChEMBL returned a non-object JSON payload.")
    molecules = payload.get("molecules", [])
    if not isinstance(molecules, list) or not molecules:
        return None

    best = _select_best_match(molecules, normalized)
    if best is None:
        return None
    properties = best.get("molecule_properties") if isinstance(best, dict) else None
    if not isinstance(properties, dict):
        properties = {}
    return ChEMBLMolecule(
        chembl_id=str(best.get("molecule_chembl_id") or "unknown"),
        preferred_name=str(best.get("pref_name") or normalized.upper()),
        molecular_weight=str(properties.get("full_mwt") or "not available"),
    )


def sdk_pk_profile(drug_name: str) -> InsulinPKProfile:
    """Return a deterministic SDK PK profile for common insulin analogue names."""

    normalized = drug_name.strip().lower()
    for key, profile in INSULIN_PK_PROFILES.items():
        if key in normalized:
            return profile
    return InsulinPKProfile(
        key="rapid-default",
        label="rapid-acting insulin default",
        tmax_minutes=55,
        duration_hours=(3.0, 5.0),
        simulator_note="fallback rapid analogue profile; verify analogue-specific settings before research use",
    )


def analyze_insulin(drug_name: str) -> tuple[ChEMBLMolecule | None, InsulinPKProfile]:
    """Print ChEMBL context plus deterministic SDK absorption mapping."""

    normalized = drug_name.strip() or "lispro"
    console.print(f"[yellow]Fetching public ChEMBL record for {normalized}...[/yellow]")
    try:
        molecule = fetch_chembl_drug(normalized)
    except ChEMBLError as exc:
        console.print(f"[red]{exc}[/red]")
        molecule = None

    if molecule is None:
        console.print("[yellow]No ChEMBL molecule match found; continuing with curated SDK PK mapping.[/yellow]")
    else:
        console.print(f"\n[bold cyan]ChEMBL record:[/bold cyan] {molecule.preferred_name} ({molecule.chembl_id})")
        console.print(f"Molecular weight: {molecule.molecular_weight} Da")

    profile = sdk_pk_profile(normalized)
    report = f"""
[green]SDK insulin class:[/green] {profile.label}
[green]Deterministic simulator parameter:[/green] t_max,I = [bold]{profile.tmax_minutes} minutes[/bold]
[green]Typical action window used for context:[/green] {profile.duration_hours[0]:.0f}-{profile.duration_hours[1]:.0f} hours
[green]Model note:[/green] {profile.simulator_note}

[dim]Research/education only. ChEMBL identity is public pharmacology context; the SDK PK values are fixed model defaults and are not patient-specific dosing guidance.[/dim]
"""
    console.print(Panel(report, title="SDK Pharmacokinetic Mapping", expand=False))
    return molecule, profile


def _select_best_match(molecules: list[Any], drug_name: str) -> dict[str, Any] | None:
    normalized = drug_name.lower()
    fallback: dict[str, Any] | None = None
    for item in molecules:
        if not isinstance(item, dict):
            continue
        if fallback is None:
            fallback = item
        preferred_name = str(item.get("pref_name") or "").lower()
        if normalized in preferred_name:
            return item
    return fallback


def _verified_https_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())
