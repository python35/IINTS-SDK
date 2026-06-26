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

    glucose_col = _first_present(df.columns, GLUCOSE_COLUMNS)
    carb_col = _first_present(df.columns, CARB_COLUMNS)
    insulin_col = _first_present(df.columns, INSULIN_COLUMNS)

    metrics: dict[str, str] = {"Rows": str(len(df))}
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
        f"Summary metrics:\n{metrics}\n"
        "Only use this summary for research explanation. Do not infer treatment decisions."
    )


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
