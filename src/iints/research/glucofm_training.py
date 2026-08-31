from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from iints.research.glucofm import (
    GLUCOFM_IMPLEMENTATION_KIND,
    GLUCOFM_MODEL_FAMILY,
    GLUCOFM_PAPER_REVISION,
    GLUCOFM_PAPER_URL,
    AlignedCGMWindow,
    GlucoFMCheckpointMetadata,
    GlucoFMConfig,
    GlucoFMPretrainer,
    align_cgm_window,
    augment_glucofm_batch,
    glucofm_ema_momentum,
    save_glucofm_checkpoint,
    sha256_file,
)


PRETRAINING_STATE_FORMAT = "iints.glucofm.pretraining.v1"


@dataclass(frozen=True)
class GlucoFMWindowCollection:
    values: np.ndarray
    observation_masks: np.ndarray
    absolute_grid_indices: np.ndarray
    subject_ids: tuple[str, ...]
    start_timestamps: tuple[str | None, ...]
    labels: tuple[str | None, ...]
    source_path: Path
    source_sha256: str
    glucose_column: str
    timestamp_column: str | None
    subject_column: str | None
    label_column: str | None

    @property
    def window_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def subject_count(self) -> int:
        return len(set(self.subject_ids))

    @property
    def mean_coverage(self) -> float:
        return float(np.mean(self.observation_masks))

    def subset(self, indices: Sequence[int]) -> GlucoFMWindowCollection:
        selected = np.asarray(indices, dtype=int)
        return GlucoFMWindowCollection(
            values=self.values[selected],
            observation_masks=self.observation_masks[selected],
            absolute_grid_indices=self.absolute_grid_indices[selected],
            subject_ids=tuple(self.subject_ids[index] for index in selected),
            start_timestamps=tuple(self.start_timestamps[index] for index in selected),
            labels=tuple(self.labels[index] for index in selected),
            source_path=self.source_path,
            source_sha256=self.source_sha256,
            glucose_column=self.glucose_column,
            timestamp_column=self.timestamp_column,
            subject_column=self.subject_column,
            label_column=self.label_column,
        )


@dataclass(frozen=True)
class GlucoFMTrainingResult:
    checkpoint_path: Path
    pretraining_state_path: Path
    report_path: Path
    history_path: Path
    window_manifest_path: Path
    train_windows: int
    validation_windows: int
    train_subjects: int
    validation_subjects: int
    best_validation_loss: float
    completed_epochs: int
    device: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "checkpoint_path",
            "pretraining_state_path",
            "report_path",
            "history_path",
            "window_manifest_path",
        ):
            payload[key] = str(payload[key])
        return payload


class _WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, collection: GlucoFMWindowCollection) -> None:
        self.values = torch.from_numpy(collection.values.astype(np.float32, copy=False))
        self.masks = torch.from_numpy(
            collection.observation_masks.astype(np.float32, copy=False)
        )
        self.absolute = torch.from_numpy(
            collection.absolute_grid_indices.astype(np.int64, copy=False)
        )

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.values[index], self.masks[index], self.absolute[index]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _infer_column(frame: pd.DataFrame, requested: str | None, candidates: Sequence[str]) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Column {requested!r} was not found")
        return requested
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _window_label(frame: pd.DataFrame, label_column: str | None) -> str | None:
    if label_column is None or label_column not in frame.columns:
        return None
    labels = frame[label_column].dropna().astype(str)
    if labels.empty:
        return None
    return str(labels.mode(dropna=True).iloc[0])


def _append_window(
    window: AlignedCGMWindow,
    *,
    subject_id: str,
    label: str | None,
    min_observations: int,
    values: list[np.ndarray],
    masks: list[np.ndarray],
    absolute: list[np.ndarray],
    subjects: list[str],
    starts: list[str | None],
    labels: list[str | None],
) -> None:
    if window.observed_count < min_observations:
        return
    values.append(window.values)
    masks.append(window.observation_mask)
    absolute.append(window.absolute_grid_indices)
    subjects.append(subject_id)
    starts.append(window.start_timestamp)
    labels.append(label)


def load_glucofm_windows(
    source: Path | str,
    *,
    glucose_column: str | None = None,
    timestamp_column: str | None = None,
    subject_column: str | None = "subject_id",
    label_column: str | None = None,
    max_gap_minutes: float = 60.0,
    min_observations: int = 48,
    binning: str = "floor",
    max_windows: int | None = None,
) -> GlucoFMWindowCollection:
    """Build non-overlapping, mask-preserving daily windows from a CGM table."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"CGM source not found: {source_path}")
    frame = _read_table(source_path)
    if frame.empty:
        raise ValueError("CGM source table is empty")
    glucose = _infer_column(
        frame,
        glucose_column,
        (
            "glucose",
            "cgm",
            "glucose_mgdl",
            "glucose_actual_mgdl",
            "glucose_dexcom",
            "glucose_libre",
        ),
    )
    if glucose is None:
        raise ValueError("Could not infer a glucose column; pass --glucose-column")
    timestamp = _infer_column(
        frame,
        timestamp_column,
        ("timestamp", "datetime", "date_time", "device_timestamp", "time"),
    )
    subject = subject_column if subject_column and subject_column in frame.columns else None
    if label_column and label_column not in frame.columns:
        raise ValueError(f"Label column {label_column!r} was not found")
    frame = frame.copy()
    frame[glucose] = pd.to_numeric(frame[glucose], errors="coerce")
    if subject is None:
        frame["__subject_id"] = "subject-unknown"
        subject = "__subject_id"

    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    absolute: list[np.ndarray] = []
    subjects: list[str] = []
    starts: list[str | None] = []
    labels: list[str | None] = []

    if timestamp is not None:
        frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce")
        frame = frame.dropna(subset=[timestamp]).sort_values(
            [subject, timestamp], kind="stable"
        )
        for subject_id, subject_frame in frame.groupby(subject, sort=True):
            ordered = subject_frame.sort_values(timestamp, kind="stable").copy()
            gaps = ordered[timestamp].diff().dt.total_seconds().div(60.0)
            ordered["__segment"] = (gaps > max_gap_minutes).cumsum()
            for _, segment in ordered.groupby("__segment", sort=True):
                if segment.empty:
                    continue
                first = pd.Timestamp(segment[timestamp].iloc[0])
                last = pd.Timestamp(segment[timestamp].iloc[-1])
                window_start = first
                while window_start <= last:
                    window_end = window_start + pd.Timedelta(hours=24)
                    daily = segment[
                        (segment[timestamp] >= window_start)
                        & (segment[timestamp] < window_end)
                    ]
                    if not daily.empty:
                        window = align_cgm_window(
                            daily[glucose].to_numpy(),
                            daily[timestamp].to_numpy(),
                            binning=binning,
                        )
                        _append_window(
                            window,
                            subject_id=str(subject_id),
                            label=_window_label(daily, label_column),
                            min_observations=min_observations,
                            values=values,
                            masks=masks,
                            absolute=absolute,
                            subjects=subjects,
                            starts=starts,
                            labels=labels,
                        )
                    if max_windows is not None and len(values) >= max_windows:
                        break
                    window_start = window_end
                if max_windows is not None and len(values) >= max_windows:
                    break
            if max_windows is not None and len(values) >= max_windows:
                break
    else:
        group_columns = [subject]
        if "window_id" in frame.columns:
            group_columns.append("window_id")
        for group_key, group in frame.groupby(group_columns, sort=True):
            subject_id = str(group_key[0] if isinstance(group_key, tuple) else group_key)
            raw = group[glucose].to_numpy(dtype=np.float32)
            chunks = [raw] if "window_id" in group_columns else [
                raw[index : index + 288] for index in range(0, len(raw), 288)
            ]
            for chunk in chunks:
                if len(chunk) != 288:
                    continue
                window = align_cgm_window(chunk)
                _append_window(
                    window,
                    subject_id=subject_id,
                    label=_window_label(group, label_column),
                    min_observations=min_observations,
                    values=values,
                    masks=masks,
                    absolute=absolute,
                    subjects=subjects,
                    starts=starts,
                    labels=labels,
                )
                if max_windows is not None and len(values) >= max_windows:
                    break
            if max_windows is not None and len(values) >= max_windows:
                break

    if not values:
        raise ValueError(
            "No eligible 24-hour windows were built. Check timestamps, gaps, and min_observations."
        )
    return GlucoFMWindowCollection(
        values=np.stack(values).astype(np.float32),
        observation_masks=np.stack(masks).astype(np.float32),
        absolute_grid_indices=np.stack(absolute).astype(np.int64),
        subject_ids=tuple(subjects),
        start_timestamps=tuple(starts),
        labels=tuple(labels),
        source_path=source_path,
        source_sha256=sha256_file(source_path),
        glucose_column=glucose,
        timestamp_column=timestamp,
        subject_column=None if subject == "__subject_id" else subject,
        label_column=label_column,
    )


def _split_windows(
    collection: GlucoFMWindowCollection,
    *,
    validation_fraction: float,
    seed: int,
    allow_single_subject: bool,
) -> tuple[GlucoFMWindowCollection, GlucoFMWindowCollection, str]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    subjects = np.array(sorted(set(collection.subject_ids)), dtype=object)
    rng = np.random.default_rng(seed)
    if len(subjects) >= 2:
        rng.shuffle(subjects)
        validation_count = min(
            len(subjects) - 1,
            max(1, int(math.ceil(len(subjects) * validation_fraction))),
        )
        validation_subjects = set(subjects[:validation_count].tolist())
        validation_indices = [
            index
            for index, subject in enumerate(collection.subject_ids)
            if subject in validation_subjects
        ]
        train_indices = [
            index
            for index, subject in enumerate(collection.subject_ids)
            if subject not in validation_subjects
        ]
        return (
            collection.subset(train_indices),
            collection.subset(validation_indices),
            "subject-disjoint",
        )
    if not allow_single_subject:
        raise ValueError(
            "At least two subjects are required for a subject-disjoint validation split. "
            "Use --allow-single-subject only for software smoke tests."
        )
    if collection.window_count < 2:
        raise ValueError("At least two windows are required for a smoke-test split")
    indices = np.arange(collection.window_count)
    rng.shuffle(indices)
    validation_count = min(
        collection.window_count - 1,
        max(1, int(math.ceil(collection.window_count * validation_fraction))),
    )
    return (
        collection.subset(indices[validation_count:]),
        collection.subset(indices[:validation_count]),
        "window-level-smoke-only",
    )


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _save_training_state(
    path: Path,
    *,
    model: GlucoFMPretrainer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    source_sha256: str,
) -> None:
    torch.save(
        {
            "format": PRETRAINING_STATE_FORMAT,
            "model_family": GLUCOFM_MODEL_FAMILY,
            "config": model.config.to_dict(),
            "pretrainer_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_validation_loss": float(best_validation_loss),
            "source_sha256": source_sha256,
        },
        path,
    )


def _load_training_state(
    path: Path,
    model: GlucoFMPretrainer,
    optimizer: torch.optim.Optimizer,
    *,
    expected_source_sha256: str,
) -> tuple[int, int, float]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Resume requires torch.load(weights_only=True)") from exc
    if not isinstance(payload, dict) or payload.get("format") != PRETRAINING_STATE_FORMAT:
        raise ValueError("Unsupported GlucoFM pretraining state")
    if payload.get("source_sha256") != expected_source_sha256:
        raise ValueError("Resume state was trained on a different dataset hash")
    model.load_state_dict(payload["pretrainer_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return (
        int(payload.get("epoch", 0)),
        int(payload.get("global_step", 0)),
        float(payload.get("best_validation_loss", float("inf"))),
    )


def pretrain_glucofm(
    source: Path | str,
    output_dir: Path | str,
    *,
    glucose_column: str | None = None,
    timestamp_column: str | None = None,
    subject_column: str | None = "subject_id",
    epochs: int = 120,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    sigma_learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    validation_fraction: float = 0.2,
    min_observations: int = 48,
    max_gap_minutes: float = 60.0,
    max_windows: int | None = None,
    seed: int = 42,
    device: str = "auto",
    allow_single_subject: bool = False,
    resume_state: Path | str | None = None,
    config: GlucoFMConfig | None = None,
) -> GlucoFMTrainingResult:
    """Train the independent GlucoFM v2 reproduction on unlabeled CGM windows."""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    collection = load_glucofm_windows(
        source,
        glucose_column=glucose_column,
        timestamp_column=timestamp_column,
        subject_column=subject_column,
        max_gap_minutes=max_gap_minutes,
        min_observations=min_observations,
        max_windows=max_windows,
    )
    train, validation, split_kind = _split_windows(
        collection,
        validation_fraction=validation_fraction,
        seed=seed,
        allow_single_subject=allow_single_subject,
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = _resolve_device(device)
    cfg = config or GlucoFMConfig()
    model = GlucoFMPretrainer(config=cfg).to(resolved_device)
    sigma_parameter = model.online_encoder.state_event_filter.rho
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter is not sigma_parameter
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": regular_parameters, "lr": learning_rate},
            {"params": [sigma_parameter], "lr": sigma_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    train_loader = DataLoader(
        _WindowDataset(train),
        batch_size=min(batch_size, train.window_count),
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        _WindowDataset(validation),
        batch_size=min(batch_size, validation.window_count),
        shuffle=False,
        num_workers=0,
    )
    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )
    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    if resume_state is not None:
        start_epoch, global_step, best_validation_loss = _load_training_state(
            Path(resume_state).expanduser().resolve(),
            model,
            optimizer,
            expected_source_sha256=collection.source_sha256,
        )

    history: list[dict[str, float | int]] = []
    best_online_state: dict[str, torch.Tensor] | None = None
    state_path = output / "glucofm_pretraining_state.pt"
    for epoch in range(start_epoch, epochs):
        model.train()
        model.target_encoder.eval()
        train_losses: list[float] = []
        train_mcr: list[float] = []
        train_td: list[float] = []
        for batch_values, batch_mask, batch_absolute in train_loader:
            batch_values = batch_values.to(resolved_device)
            batch_mask = batch_mask.to(resolved_device)
            batch_absolute = batch_absolute.to(resolved_device)
            batch_values, batch_mask = augment_glucofm_batch(
                batch_values, batch_mask
            )
            optimizer.zero_grad(set_to_none=True)
            result = model(batch_values, batch_mask, batch_absolute)
            if not torch.isfinite(result.loss):
                raise RuntimeError("Non-finite GlucoFM pretraining loss encountered")
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            momentum = glucofm_ema_momentum(
                global_step, total_steps, cfg.ema_momentum_initial
            )
            model.update_target(momentum)
            scheduler.step()
            train_losses.append(float(result.loss.detach().cpu()))
            train_mcr.append(float(result.masked_context_loss.detach().cpu()))
            train_td.append(float(result.temporal_dynamics_loss.detach().cpu()))

        model.eval()
        validation_losses: list[float] = []
        with torch.inference_mode():
            for batch_values, batch_mask, batch_absolute in validation_loader:
                result = model(
                    batch_values.to(resolved_device),
                    batch_mask.to(resolved_device),
                    batch_absolute.to(resolved_device),
                )
                validation_losses.append(float(result.loss.cpu()))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "masked_context_loss": float(np.mean(train_mcr)),
                "temporal_dynamics_loss": float(np.mean(train_td)),
                "validation_loss": validation_loss,
                "gaussian_sigma_grid_steps": float(
                    model.online_encoder.state_event_filter.sigma.detach().cpu()
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_online_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.online_encoder.state_dict().items()
            }
        _save_training_state(
            state_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            source_sha256=collection.source_sha256,
        )

    if best_online_state is None:
        best_online_state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in model.online_encoder.state_dict().items()
        }
    model.online_encoder.load_state_dict(best_online_state, strict=True)
    metadata = GlucoFMCheckpointMetadata(
        trained=True,
        training_epochs=epochs,
        dataset_sha256=collection.source_sha256,
        dataset_description=(
            f"{collection.window_count} mask-preserving 24-hour windows from "
            f"{collection.subject_count} dataset-defined subjects"
        ),
        code_revision=_git_revision(),
        notes=(
            "Independent method reproduction; not official Google weights. "
            f"Validation split: {split_kind}."
        ),
    )
    checkpoint_path = save_glucofm_checkpoint(
        output / "glucofm_encoder.pt", model.online_encoder.cpu(), metadata
    )
    history_path = output / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    window_manifest_path = output / "window_manifest.csv"
    pd.DataFrame(
        {
            "window_index": np.arange(collection.window_count),
            "subject_id": collection.subject_ids,
            "start_timestamp": collection.start_timestamps,
            "observed_count": collection.observation_masks.sum(axis=1).astype(int),
            "coverage": collection.observation_masks.mean(axis=1),
        }
    ).to_csv(window_manifest_path, index=False)
    report_path = output / "training_report.json"
    report = {
        "model_family": GLUCOFM_MODEL_FAMILY,
        "implementation_kind": GLUCOFM_IMPLEMENTATION_KIND,
        "paper_revision": GLUCOFM_PAPER_REVISION,
        "paper_url": GLUCOFM_PAPER_URL,
        "official_checkpoint": False,
        "research_only": True,
        "medical_device": False,
        "source_path": str(collection.source_path),
        "source_sha256": collection.source_sha256,
        "window_count": collection.window_count,
        "subject_count": collection.subject_count,
        "mean_coverage": collection.mean_coverage,
        "split_kind": split_kind,
        "train_windows": train.window_count,
        "validation_windows": validation.window_count,
        "train_subjects": train.subject_count,
        "validation_subjects": validation.subject_count,
        "best_validation_loss": best_validation_loss,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(resolved_device),
        "config": cfg.to_dict(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "pretraining_state": str(state_path),
        "warning": (
            "Loss values establish software/training behavior only. Downstream clinical utility "
            "must be evaluated separately with subject-disjoint labeled cohorts."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return GlucoFMTrainingResult(
        checkpoint_path=checkpoint_path,
        pretraining_state_path=state_path,
        report_path=report_path,
        history_path=history_path,
        window_manifest_path=window_manifest_path,
        train_windows=train.window_count,
        validation_windows=validation.window_count,
        train_subjects=train.subject_count,
        validation_subjects=validation.subject_count,
        best_validation_loss=best_validation_loss,
        completed_epochs=epochs,
        device=str(resolved_device),
    )


__all__ = [
    "PRETRAINING_STATE_FORMAT",
    "GlucoFMWindowCollection",
    "GlucoFMTrainingResult",
    "load_glucofm_windows",
    "pretrain_glucofm",
]
