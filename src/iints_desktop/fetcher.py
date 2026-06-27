"""AlphaFold structure fetching logic for the desktop app."""

from __future__ import annotations

import json
from pathlib import Path
import ssl
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class FetchError(RuntimeError):
    """Raised when fetching a structure fails."""


def fetch_alphafold_structure(uniprot_id: str, output_dir: Path) -> Path:
    """Fetch the latest AlphaFold mmCIF structure for a given UniProt ID."""
    
    uniprot_id = uniprot_id.strip().upper()
    if not uniprot_id:
        raise FetchError("UniProt ID cannot be empty.")
        
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    
    try:
        req = Request(api_url, headers={"User-Agent": "IINTS-AF-Desktop/1.0"})
        with urlopen(req, timeout=15, context=_verified_https_context()) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            raise FetchError(f"No AlphaFold prediction found for UniProt ID: {uniprot_id}") from exc
        raise FetchError(f"AlphaFold API error ({exc.code}): {exc.reason}") from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise FetchError(f"Failed to communicate with AlphaFold API: {exc}") from exc

    if not isinstance(payload, list) or not payload:
        raise FetchError(f"Invalid payload format from AlphaFold for {uniprot_id}.")
        
    first_prediction = payload[0]
    cif_url = first_prediction.get("cifUrl")
    if not cif_url:
        raise FetchError(f"No mmCIF URL found in prediction for {uniprot_id}.")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{uniprot_id}_dynamic.cif"
    
    try:
        req = Request(cif_url, headers={"User-Agent": "IINTS-AF-Desktop/1.0"})
        with urlopen(req, timeout=30, context=_verified_https_context()) as response:  # noqa: S310
            cif_data = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise FetchError(f"Failed to download mmCIF structure: {exc}") from exc
        
    out_file.write_bytes(cif_data)
    return out_file


def _verified_https_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()
