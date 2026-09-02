from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResultPreview:
    """UI-ready summary of a results CSV."""

    csv_path: Path
    row_count: int
    columns: list[str]
    rows: list[list[str]]
    metrics: dict[str, str]
    graph_path: Path | None


@dataclass(frozen=True)
class RunThresholdScreen:
    """Deterministic threshold screen for one simulated result set.

    The screen describes the generated trace; it is not a clinical assessment
    and does not establish that the underlying physiology is valid.
    """

    status: str
    label: str
    flags: tuple[str, ...]
    metrics: dict[str, str]


TIME_COLUMNS = ("time_minutes", "timestamp", "time", "minute", "minutes")
GLUCOSE_COLUMNS = (
    "glucose_actual_mgdl",
    "glucose",
    "cgm_mgdl",
    "sensor_glucose_mgdl",
    "glucose_mgdl",
)
CARB_COLUMNS = ("carb_intake_grams", "carbs", "carbohydrates", "meal_carbs")
INSULIN_COLUMNS = ("delivered_insulin_units", "insulin", "insulin_units", "bolus_units")
SAFETY_COLUMNS = ("safety_triggered", "supervisor_triggered", "safety_intervention")


@dataclass(frozen=True)
class CompartmentTimeline:
    """Compartment contents and fluxes over a run, ready for a topology view.

    ``available`` is False for runs that carry no compartment columns -- every
    run produced before the simulator exported them, and any run whose patient
    backend publishes no schema. ``reason`` then explains which case it is, so
    the UI can say the layout is unavailable rather than draw an empty diagram.
    """

    available: bool
    reason: str
    schema: dict[str, Any]
    times: list[float]
    compartments: dict[str, list[float]]
    fluxes: dict[str, list[float]]
    flux_extremes: dict[str, tuple[float, float]]
    step_count: int
    stride: int
    # Canonical plasma glucose (mg/dL), aligned to `times` the same way every
    # other series is. Named distinctly from the GLUCOSE_COLUMNS candidate
    # list below (rather than "glucose_mgdl") to avoid confusion between the
    # two. Empty when the run has no recognizable glucose column -- the 3D
    # viewer's hypo-coloring degrades gracefully rather than crashing.
    plasma_glucose_mgdl: list[float]


def load_compartment_timeline(
    csv_path: str | Path, *, max_points: int = 400
) -> CompartmentTimeline:
    """Load the compartment/flux series exported alongside a results CSV.

    The schema is read from the sidecar the run wrote, never inferred from a
    model name, so the view describes the backend that actually produced the
    numbers.

    Downsampling takes every n-th step rather than averaging. Averaging fluxes
    over steps would report rates the ODE never computed, and these values are
    instantaneous rates at the end of a step, not amounts transferred during
    it. ``flux_extremes`` is computed over the full trace, so a scale derived
    from it does not depend on the stride.
    """

    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")

    empty: dict[str, list[float]] = {}
    schema_path = path.parent / "compartment_schema.json"
    if not schema_path.is_file():
        return CompartmentTimeline(
            available=False,
            reason=(
                "This run has no compartment_schema.json, so the compartment "
                "layout it used is unknown."
            ),
            schema={},
            times=[],
            compartments=empty,
            fluxes=empty,
            flux_extremes={},
            step_count=0,
            stride=1,
            plasma_glucose_mgdl=[],
        )

    import json

    import pandas as pd

    schema = json.loads(schema_path.read_text())
    df = pd.read_csv(path)

    state_columns = {
        column[len("patient_state_") :]: column
        for column in df.columns
        if str(column).startswith("patient_state_")
    }
    if not state_columns:
        return CompartmentTimeline(
            available=False,
            reason=(
                "This run exported no patient_state_* columns; it predates "
                "compartment export."
            ),
            schema=schema,
            times=[],
            compartments=empty,
            fluxes=empty,
            flux_extremes={},
            step_count=0,
            stride=1,
            plasma_glucose_mgdl=[],
        )

    flux_columns = {
        column[len("patient_flux_") :]: column
        for column in df.columns
        if str(column).startswith("patient_flux_")
    }

    time_column = next((name for name in TIME_COLUMNS if name in df.columns), None)
    times = (
        [float(value) for value in df[time_column].to_numpy()]
        if time_column is not None
        else [float(index) for index in range(len(df))]
    )

    step_count = len(df)
    stride = max(1, -(-step_count // max(1, max_points)))
    keep = slice(None, None, stride)

    extremes = {
        key: (float(df[column].min()), float(df[column].max()))
        for key, column in flux_columns.items()
    }

    glucose_column = _first_present(df.columns, GLUCOSE_COLUMNS)
    plasma_glucose_mgdl = (
        [float(value) for value in df[glucose_column].to_numpy()[keep]]
        if glucose_column is not None
        else []
    )

    return CompartmentTimeline(
        available=True,
        reason="",
        schema=schema,
        times=times[keep],
        compartments={
            key: [float(value) for value in df[column].to_numpy()[keep]]
            for key, column in state_columns.items()
        },
        fluxes={
            key: [float(value) for value in df[column].to_numpy()[keep]]
            for key, column in flux_columns.items()
        },
        flux_extremes=extremes,
        step_count=step_count,
        stride=stride,
        plasma_glucose_mgdl=plasma_glucose_mgdl,
    )


def load_results_preview(csv_path: str | Path, *, max_rows: int = 200) -> ResultPreview:
    """Load a CSV into bounded table rows plus a simple glucose graph."""

    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")

    import pandas as pd

    df = pd.read_csv(path)
    columns = [str(column) for column in df.columns]
    preview = df.head(max_rows).fillna("")
    rows = [[_cell_to_text(value) for value in row] for row in preview.to_numpy(dtype=object)]
    metrics = summarize_results_dataframe(df)
    graph_path = _write_glucose_graph(df, path)

    return ResultPreview(
        csv_path=path,
        row_count=int(len(df)),
        columns=columns,
        rows=rows,
        metrics=metrics,
        graph_path=graph_path,
    )


def summarize_results_dataframe(df: Any) -> dict[str, str]:
    """Return a small, deterministic summary used by both UI and AI context."""

    time_col = _first_present(df.columns, TIME_COLUMNS)
    glucose_col = _first_present(df.columns, GLUCOSE_COLUMNS)
    carb_col = _first_present(df.columns, CARB_COLUMNS)
    insulin_col = _first_present(df.columns, INSULIN_COLUMNS)
    safety_col = _first_present(df.columns, SAFETY_COLUMNS)

    metrics: dict[str, str] = {"Rows": str(len(df))}
    if time_col is not None:
        _add_time_metrics(metrics, df, time_col)
    if glucose_col is not None:
        import pandas as pd

        glucose = pd.to_numeric(df[glucose_col], errors="coerce").dropna()
        if glucose.empty:
            metrics["Glucose column"] = f"{glucose_col} (no numeric values)"
        else:
            metrics["Glucose column"] = glucose_col
            metrics["Mean glucose"] = f"{glucose.mean():.1f} mg/dL"
            metrics["Min glucose"] = f"{glucose.min():.1f} mg/dL"
            metrics["Max glucose"] = f"{glucose.max():.1f} mg/dL"
            if len(glucose) > 1 and float(glucose.mean()) > 0:
                metrics["Glucose CV"] = f"{glucose.std(ddof=1) / glucose.mean() * 100:.1f}%"
            metrics["Time in 70-180"] = f"{((glucose >= 70) & (glucose <= 180)).mean() * 100:.1f}%"
            metrics["Time below 70"] = f"{(glucose < 70).mean() * 100:.1f}%"
            metrics["Time above 180"] = f"{(glucose > 180).mean() * 100:.1f}%"
    if carb_col is not None:
        import pandas as pd

        carbs = pd.to_numeric(df[carb_col], errors="coerce").fillna(0.0)
        metrics["Total carbs"] = f"{carbs.sum():.1f} g"
    if insulin_col is not None:
        import pandas as pd

        insulin = pd.to_numeric(df[insulin_col], errors="coerce").fillna(0.0)
        metrics["Total insulin"] = f"{insulin.sum():.2f} U"
    if safety_col is not None:
        safety = df[safety_col].map(_to_bool)
        metrics["Safety-triggered samples"] = str(int(safety.sum()))
    return metrics


def build_ai_result_context(csv_path: str | Path | None) -> str:
    """Create a compact, non-raw-data context block for local AI questions."""

    if csv_path is None:
        return "No result CSV is currently loaded."
    preview = load_results_preview(csv_path, max_rows=20)
    metrics = "\n".join(f"- {key}: {value}" for key, value in preview.metrics.items())
    columns = ", ".join(preview.columns[:30])
    return (
        f"Loaded results CSV: {preview.csv_path}\n"
        f"Columns: {columns}\n"
        "AUTHORITATIVE DETERMINISTIC FACTS (computed by the SDK; quote values exactly):\n"
        f"{metrics}\n"
        "Do not calculate new statistics or introduce any quantitative claim that is not listed above. "
        "Do not infer treatment decisions. If a requested quantity is absent, state that it was not computed."
    )


def screen_results_csv(csv_path: str | Path) -> RunThresholdScreen:
    """Screen a result CSV for transparent glucose-range excursions."""

    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")

    import pandas as pd

    return screen_results_dataframe(pd.read_csv(path))


def screen_results_dataframe(df: Any) -> RunThresholdScreen:
    """Return descriptive range flags without making a clinical judgement."""

    import pandas as pd

    metrics = summarize_results_dataframe(df)
    glucose_col = _first_present(df.columns, GLUCOSE_COLUMNS)
    if glucose_col is None:
        return RunThresholdScreen(
            status="needs_review",
            label="Glucose column unavailable",
            flags=("No supported glucose column was available for threshold screening.",),
            metrics=metrics,
        )

    glucose = pd.to_numeric(df[glucose_col], errors="coerce").dropna()
    if glucose.empty:
        return RunThresholdScreen(
            status="needs_review",
            label="No numeric glucose samples",
            flags=("The detected glucose column contained no numeric samples.",),
            metrics=metrics,
        )

    flags: list[str] = []
    if bool((glucose < 54).any()):
        flags.append("The simulated trace contains samples below 54 mg/dL.")
    elif bool((glucose < 70).any()):
        flags.append("The simulated trace contains samples below 70 mg/dL.")
    if bool((glucose > 250).any()):
        flags.append("The simulated trace contains samples above 250 mg/dL.")
    elif bool((glucose > 180).any()):
        flags.append("The simulated trace contains samples above 180 mg/dL.")

    safety_col = _first_present(df.columns, SAFETY_COLUMNS)
    if safety_col is not None:
        safety_count = int(df[safety_col].map(_to_bool).sum())
        if safety_count:
            flags.append(f"Safety logic was marked active in {safety_count} sampled rows.")

    if flags:
        return RunThresholdScreen(
            status="excursions_present",
            label="Threshold excursions require review",
            flags=tuple(flags),
            metrics=metrics,
        )
    return RunThresholdScreen(
        status="no_excursions_detected",
        label="No configured threshold excursions detected",
        flags=(),
        metrics=metrics,
    )


def _add_time_metrics(metrics: dict[str, str], df: Any, time_col: str) -> None:
    """Add duration and cadence only when their units can be determined safely."""

    import pandas as pd

    numeric = pd.to_numeric(df[time_col], errors="coerce").dropna()
    minute_columns = {"time_minutes", "minute", "minutes"}
    if time_col.lower() in minute_columns and not numeric.empty:
        duration = float(numeric.max() - numeric.min())
        metrics["Duration"] = f"{duration:.1f} min"
        intervals = numeric.sort_values().diff().dropna()
        intervals = intervals[intervals > 0]
        if not intervals.empty:
            metrics["Median sample interval"] = f"{float(intervals.median()):.1f} min"
        return

    if time_col.lower() == "timestamp":
        # Bare numeric timestamps have ambiguous units (minutes, seconds, Unix
        # time, or sample index). Do not silently reinterpret them as datetimes.
        raw = df[time_col].dropna()
        if not raw.empty and pd.to_numeric(raw, errors="coerce").notna().all():
            return
        timestamps = pd.to_datetime(df[time_col], errors="coerce", utc=True).dropna()
        if not timestamps.empty:
            duration_minutes = (timestamps.max() - timestamps.min()).total_seconds() / 60.0
            metrics["Duration"] = f"{duration_minutes:.1f} min"
            intervals = timestamps.sort_values().diff().dropna().dt.total_seconds() / 60.0
            intervals = intervals[intervals > 0]
            if not intervals.empty:
                metrics["Median sample interval"] = f"{float(intervals.median()):.1f} min"


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _write_glucose_graph(df: Any, csv_path: Path) -> Path | None:
    time_col = _first_present(df.columns, TIME_COLUMNS)
    glucose_col = _first_present(df.columns, GLUCOSE_COLUMNS)
    if glucose_col is None:
        return None

    os.environ.setdefault("MPLCONFIGDIR", str(csv_path.parent / ".cache" / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pandas as pd

    y = pd.to_numeric(df[glucose_col], errors="coerce")
    valid = y.notna()
    if not bool(valid.any()):
        return None
    y = y[valid]
    x = df.loc[valid, time_col] if time_col is not None else range(len(y))
    preview_dir = csv_path.parent / ".desktop_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    graph_path = preview_dir / f"{csv_path.stem}_glucose.png"

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    ax.plot(x, y, color="#0f766e", linewidth=2.0)
    ax.axhspan(70, 180, color="#dcfce7", alpha=0.65, label="70-180 mg/dL")
    ax.axhline(70, color="#b91c1c", linewidth=1.0, linestyle="--")
    ax.axhline(180, color="#ca8a04", linewidth=1.0, linestyle="--")
    ax.set_title("Glucose Trace")
    ax.set_xlabel(time_col or "sample")
    ax.set_ylabel("mg/dL")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(graph_path)
    plt.close(fig)
    return graph_path


def _first_present(columns: Any, candidates: tuple[str, ...]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _cell_to_text(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
