from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import urllib.request


@dataclass
class TidepoolClient:
    """
    Lightweight client skeleton based on the Tidepool OpenAPI spec.
    Authentication and full endpoint coverage are intentionally left open
    so teams can wire in their preferred auth flow.
    """
    base_url: str = "https://api.tidepool.org"
    token: Optional[str] = None

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def get_json(self, path: str) -> Dict[str, Any]:
        if not self.token:
            raise RuntimeError("TidepoolClient requires an auth token. Provide one before calling.")
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req) as response:  # nosec - thin skeleton
            payload = response.read().decode("utf-8")
        return json.loads(payload)


def load_openapi_spec(path: str) -> Dict[str, Any]:
    """Load a local Tidepool OpenAPI JSON spec for reference tooling."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
