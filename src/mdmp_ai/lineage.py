from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import pandas as pd
import yaml

from mdmp_core.contracts import load_contract
from mdmp_core.fingerprint import check_fingerprint, compute_fingerprint
from mdmp_core.runner import ContractRunner


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class DatasetRef:
    path: str
    fingerprint: str
    grade: str
    rows_used: int
    date_validated: str
    consent: str
    contract_fingerprint: str
    expires: str
    status: str = "valid"
    stale_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "fingerprint": self.fingerprint,
            "grade": self.grade,
            "rows_used": self.rows_used,
            "date_validated": self.date_validated,
            "consent": self.consent,
            "contract_fingerprint": self.contract_fingerprint,
            "expires": self.expires,
            "status": self.status,
            "stale_reason": self.stale_reason,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DatasetRef":
        return cls(
            path=str(payload.get("path", "")),
            fingerprint=str(payload.get("fingerprint", "")),
            grade=str(payload.get("grade", "draft")),
            rows_used=int(payload.get("rows_used", 0)),
            date_validated=str(payload.get("date_validated", "")),
            consent=str(payload.get("consent", "not_verified")),
            contract_fingerprint=str(payload.get("contract_fingerprint", "")),
            expires=str(payload.get("expires", "")),
            status=str(payload.get("status", "valid")),
            stale_reason=payload.get("stale_reason"),
        )


@dataclass
class LineageCard:
    model_name: str
    training_date: str
    datasets: List[DatasetRef] = field(default_factory=list)

    @property
    def stale_datasets(self) -> List[DatasetRef]:
        return [row for row in self.datasets if row.status == "stale"]

    @property
    def lineage_status(self) -> str:
        return "stale" if self.stale_datasets else "valid"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": {
                "name": self.model_name,
                "training_date": self.training_date,
                "lineage_status": self.lineage_status,
                "stale_lineage": bool(self.stale_datasets),
                "stale_datasets": [row.fingerprint for row in self.stale_datasets],
                "trained_on": [row.to_dict() for row in self.datasets],
            }
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LineageCard":
        model = payload.get("model", {}) if isinstance(payload, dict) else {}
        trained_on = model.get("trained_on", []) if isinstance(model, dict) else []
        rows: List[DatasetRef] = []
        if isinstance(trained_on, list):
            for entry in trained_on:
                if isinstance(entry, dict):
                    rows.append(DatasetRef.from_dict(entry))
        return cls(
            model_name=str(model.get("name", "unknown_model")),
            training_date=str(model.get("training_date", _now_iso())),
            datasets=rows,
        )

    def export(self, output_path: str | Path) -> Dict[str, Any]:
        payload = self.to_dict()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() in {".yaml", ".yml"}:
            output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        else:
            output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def refresh(self, *, data_dir: Optional[Path] = None) -> "LineageCard":
        for dataset in self.datasets:
            resolved_path = Path(dataset.path)
            if not resolved_path.is_file() and data_dir is not None:
                candidate = data_dir / dataset.path
                if candidate.is_file():
                    resolved_path = candidate
            if not resolved_path.is_file():
                dataset.status = "stale"
                dataset.stale_reason = "dataset_not_found"
                continue

            checked = check_fingerprint(
                {
                    "fingerprint": dataset.fingerprint,
                    "expires": dataset.expires,
                    "status": dataset.status,
                    "stale_reason": dataset.stale_reason,
                },
                data_path=resolved_path,
            )
            dataset.status = str(checked.get("status", "stale"))
            dataset.stale_reason = checked.get("stale_reason")
        return self


def load_lineage_card(path: str | Path) -> LineageCard:
    card_path = Path(path)
    raw = card_path.read_text(encoding="utf-8")
    if card_path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw) or {}
    else:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Lineage card root must be a mapping")
    return LineageCard.from_dict(payload)


@dataclass
class LineageTracker:
    datasets: list[DatasetRef] = field(default_factory=list)
    model_name: str | None = None

    def register_dataset(
        self,
        dataset_path: str | Path,
        *,
        contract: str | Path,
        expires_days: int = 365,
    ) -> Dict[str, Any]:
        dataset_path = Path(dataset_path)
        contract_path = Path(contract)
        df = pd.read_csv(dataset_path)
        contract_obj = load_contract(contract_path)
        validation = ContractRunner(contract_obj).run(df)
        fp_record = compute_fingerprint(dataset_path, expires_days=expires_days)

        row = DatasetRef(
            path=str(dataset_path),
            fingerprint=str(fp_record["fingerprint"]),
            grade=validation.grade,
            rows_used=validation.row_count,
            date_validated=validation.created_utc,
            consent="verified" if validation.consent_verified else "not_verified",
            contract_fingerprint=f"sha256:{validation.contract_fingerprint_sha256}",
            expires=str(fp_record["expires"]),
            status=str(fp_record["status"]),
            stale_reason=fp_record.get("stale_reason"),
        )
        self.datasets.append(row)
        return row.to_dict()

    def attach_to_model(self, model_name: str) -> None:
        self.model_name = model_name

    def export_card(self, output_path: str | Path) -> Dict[str, Any]:
        if not self.model_name:
            raise ValueError("Call attach_to_model() before export_card()")
        card = LineageCard(
            model_name=self.model_name,
            training_date=_now_iso(),
            datasets=self.datasets,
        )
        return card.export(output_path)
