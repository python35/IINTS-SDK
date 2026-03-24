from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class FingerprintStore:
    path: Path

    def _load(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return {"datasets": {}, "fingerprints": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"datasets": {}, "fingerprints": {}}
        payload.setdefault("datasets", {})
        payload.setdefault("fingerprints", {})
        return payload

    def _save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def upsert(self, *, dataset_path: str, record: Dict[str, Any]) -> None:
        payload = self._load()
        key = str(Path(dataset_path))
        fingerprint = str(record.get("fingerprint", "")).strip()
        payload["datasets"][key] = record
        if fingerprint:
            payload["fingerprints"][fingerprint] = record
        self._save(payload)

    def get_by_dataset(self, dataset_path: str) -> Optional[Dict[str, Any]]:
        payload = self._load()
        return payload.get("datasets", {}).get(str(Path(dataset_path)))

    def get_by_fingerprint(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        payload = self._load()
        return payload.get("fingerprints", {}).get(fingerprint)
