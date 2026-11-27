"""Step 5 — Deterministic Model Registry for `iints.forecasting`."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from iints.forecasting.models.base import WindowForecaster
from iints.forecasting.models.gru import GRUGlucosePredictor
from iints.forecasting.models.persistence import PersistencePredictor
from iints.forecasting.models.ridge import RidgeCGMPredictor
from iints.forecasting.models.tcn import TCNGlucosePredictor
from iints.forecasting.schemas import BenchmarkTier, ModelRegistryEntry
from iints.sdk2.reproducibility import sha256_file

__all__ = ["ForecastingModelRegistry"]


class ForecastingModelRegistry:
    """File-backed versioned model registry storing checkpoints, checksums & provenance."""

    def __init__(self, registry_dir: Union[str, Path]) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_dir / "registry_index.json"

    def register(
        self,
        model: WindowForecaster,
        *,
        dataset_id: str,
        is_synthetic: bool,
        benchmark_tier: BenchmarkTier,
        dataset_sha256: str,
        input_length: int,
        elapsed_input_span_minutes: float,
        horizon_minutes: int,
        training_config: Dict[str, Any],
        split_counts: Dict[str, int],
        validation_metrics: Dict[str, float],
        test_metrics_available: bool = False,
        version: str = "1.0.0",
    ) -> ModelRegistryEntry:
        ckpt_name = f"{model.model_id}_v{version}.checkpoint.json"
        ckpt_path = model.save_checkpoint(self.registry_dir / ckpt_name)
        ckpt_sha = sha256_file(ckpt_path)

        entry = ModelRegistryEntry(
            model_id=model.model_id,
            version=version,
            architecture=model.architecture,
            dataset_id=dataset_id,
            is_synthetic=bool(is_synthetic),
            benchmark_tier=benchmark_tier,
            dataset_sha256=dataset_sha256,
            input_length=int(input_length),
            elapsed_input_span_minutes=float(elapsed_input_span_minutes),
            prediction_horizon_minutes=int(horizon_minutes),
            created_utc=datetime.now(timezone.utc).isoformat(),
            training_config=training_config,
            split_counts=split_counts,
            checkpoint_file=ckpt_name,
            checkpoint_sha256=ckpt_sha,
            validation_metrics={k: round(float(v), 4) for k, v in validation_metrics.items()},
            test_metrics_available=bool(test_metrics_available),
            clinical_validation="none",
        )

        entries = [e for e in self.list_entries() if not (e.model_id == entry.model_id and e.version == version)]
        entries.append(entry)
        self.index_path.write_text(
            json.dumps([e.model_dump(mode="json") for e in entries], indent=2),
            encoding="utf-8",
        )
        return entry

    def list_entries(self) -> List[ModelRegistryEntry]:
        if not self.index_path.exists():
            return []
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        return [ModelRegistryEntry.model_validate(item) for item in raw]

    def load_model(self, model_id: str, version: str = "1.0.0") -> WindowForecaster:
        matches = [e for e in self.list_entries() if e.model_id == model_id and e.version == version]
        if not matches:
            raise KeyError(f"Model '{model_id}' (v{version}) not found in registry {self.registry_dir}.")
        entry = matches[0]
        ckpt_path = self.registry_dir / entry.checkpoint_file
        actual_sha = sha256_file(ckpt_path)
        if actual_sha != entry.checkpoint_sha256:
            raise RuntimeError(
                f"Checkpoint SHA-256 mismatch for {ckpt_path.name}: expected {entry.checkpoint_sha256}, got {actual_sha}"
            )
        payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if entry.architecture == "persistence":
            return PersistencePredictor.from_checkpoint_dict(payload)
        if entry.architecture == "ridge":
            return RidgeCGMPredictor.from_checkpoint_dict(payload)
        if entry.architecture == "gru":
            return GRUGlucosePredictor.from_checkpoint_dict(payload)
        if entry.architecture == "tcn":
            return TCNGlucosePredictor.from_checkpoint_dict(payload)
        raise NotImplementedError(f"Unsupported architecture '{entry.architecture}'.")
