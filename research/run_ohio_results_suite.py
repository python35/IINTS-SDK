#!/usr/bin/env python3
"""Generate privacy-safe aggregate result artifacts from prepared OhioT1DM files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_LOW = 70.0
TARGET_HIGH = 180.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aggregate OhioT1DM result artifacts for IINTS research reports.")
    parser.add_argument("--train", type=Path, required=True, help="Prepared Ohio train CSV")
    parser.add_argument("--test", type=Path, required=True, help="Prepared Ohio test CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for aggregate artifacts")
    parser.add_argument("--max-public-subjects", type=int, default=12, help="Max anonymized subject rows in public tables")
    return parser.parse_args()


def _safe_subject_map(values: Iterable[object]) -> dict[str, str]:
    subjects = sorted({str(v) for v in values})
    return {subject: f"S{idx + 1:02d}" for idx, subject in enumerate(subjects)}


def _load(path: Path, split: str, subject_map: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["source_split"] = split
    if subject_map is None:
        subject_map = _safe_subject_map(df["subject_id"].unique())
    df["subject_label"] = df["subject_id"].astype(str).map(subject_map).fillna("S??")
    df["glucose_actual_mgdl"] = pd.to_numeric(df["glucose_actual_mgdl"], errors="coerce")
    df["carb_intake_grams"] = pd.to_numeric(df.get("carb_intake_grams", 0.0), errors="coerce").fillna(0.0)
    df["insulin_units"] = pd.to_numeric(df.get("insulin_units", 0.0), errors="coerce").fillna(0.0)
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["date_key"] = df.groupby(["source_split", "subject_label"])["timestamp"].transform(
        lambda s: (s.dt.normalize() - s.dt.normalize().min()).dt.days
    )
    return df


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    g = frame["glucose_actual_mgdl"].dropna().astype(float)
    if g.empty:
        return {}
    return {
        "rows": int(len(frame)),
        "subjects": int(frame["subject_label"].nunique()),
        "mean_glucose_mgdl": round(float(g.mean()), 3),
        "sd_glucose_mgdl": round(float(g.std(ddof=0)), 3),
        "min_glucose_mgdl": round(float(g.min()), 3),
        "max_glucose_mgdl": round(float(g.max()), 3),
        "tir_70_180_pct": round(float(((g >= TARGET_LOW) & (g <= TARGET_HIGH)).mean() * 100.0), 3),
        "time_below_70_pct": round(float((g < TARGET_LOW).mean() * 100.0), 3),
        "time_below_54_pct": round(float((g < 54.0).mean() * 100.0), 3),
        "time_above_180_pct": round(float((g > TARGET_HIGH).mean() * 100.0), 3),
        "time_above_250_pct": round(float((g > 250.0).mean() * 100.0), 3),
        "meal_events": int((frame["carb_intake_grams"] > 0).sum()),
        "insulin_event_steps": int((frame["insulin_units"] > 0).sum()),
        "total_carbs_grams": round(float(frame["carb_intake_grams"].sum()), 3),
        "total_insulin_units": round(float(frame["insulin_units"].sum()), 3),
    }


def _subject_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, subject), group in df.groupby(["source_split", "subject_label"], observed=False):
        row = {"source_split": split, "subject_label": subject}
        row.update(_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source_split", "subject_label"])


def _daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, subject, day_index), group in df.groupby(["source_split", "subject_label", "date_key"], observed=False):
        if len(group) < 48:
            continue
        row = {"source_split": split, "subject_label": subject, "day_index": int(day_index)}
        row.update(_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source_split", "subject_label", "day_index"])


def _agp(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["source_split", "minute_of_day"], observed=False)["glucose_actual_mgdl"]
    rows = []
    for (split, minute), series in grouped:
        clean = series.dropna().astype(float)
        if len(clean) < 5:
            continue
        rows.append(
            {
                "source_split": split,
                "minute_of_day": int(minute),
                "p05": float(clean.quantile(0.05)),
                "p25": float(clean.quantile(0.25)),
                "p50": float(clean.quantile(0.50)),
                "p75": float(clean.quantile(0.75)),
                "p95": float(clean.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_split", "minute_of_day"])


def _meal_response(df: pd.DataFrame, *, window_before: int = 30, window_after: int = 240) -> pd.DataFrame:
    rows = []
    for (split, subject), group in df.sort_values("timestamp").groupby(["source_split", "subject_label"], observed=False):
        meals = group[group["carb_intake_grams"] > 0]
        for _, meal in meals.iterrows():
            t0 = meal["timestamp"]
            rel = (group["timestamp"] - t0).dt.total_seconds() / 60.0
            around = group[(rel >= -window_before) & (rel <= window_after)].copy()
            if len(around) < 12:
                continue
            baseline = around[(rel.loc[around.index] >= -window_before) & (rel.loc[around.index] <= 0)][
                "glucose_actual_mgdl"
            ].median()
            if not np.isfinite(baseline):
                continue
            around["relative_minutes"] = rel.loc[around.index].round().astype(int)
            around["delta_glucose_mgdl"] = around["glucose_actual_mgdl"] - float(baseline)
            around["meal_carbs_grams"] = float(meal["carb_intake_grams"])
            around["source_split"] = split
            around["subject_label"] = subject
            rows.append(around[["source_split", "subject_label", "relative_minutes", "delta_glucose_mgdl", "meal_carbs_grams"]])
    if not rows:
        return pd.DataFrame(columns=["source_split", "relative_minutes", "median_delta_mgdl", "p25", "p75", "meal_count"])
    events = pd.concat(rows, ignore_index=True)
    agg = (
        events.groupby(["source_split", "relative_minutes"], observed=False)["delta_glucose_mgdl"]
        .agg(median_delta_mgdl="median", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75), meal_count="count")
        .reset_index()
    )
    return agg.sort_values(["source_split", "relative_minutes"])


def _style(ax: plt.Axes, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22)


def _save_split_tir(split_metrics: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    x = np.arange(len(split_metrics))
    ax.bar(x, split_metrics["tir_70_180_pct"], color="#2E7D32", label="70-180")
    ax.bar(x, split_metrics["time_below_70_pct"], bottom=split_metrics["tir_70_180_pct"], color="#C62828", label="<70")
    ax.bar(
        x,
        split_metrics["time_above_180_pct"],
        bottom=split_metrics["tir_70_180_pct"] + split_metrics["time_below_70_pct"],
        color="#F9A825",
        label=">180",
    )
    ax.set_xticks(x, split_metrics["source_split"])
    ax.set_ylim(0, 100)
    _style(ax, "OhioT1DM time-in-range by split", "% of CGM readings")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _save_subject_distribution(subject_metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=180, sharey=True)
    for ax, split in zip(axes, ["train", "test"]):
        sub = subject_metrics[subject_metrics["source_split"] == split].copy()
        sub = sub.sort_values("tir_70_180_pct")
        ax.barh(sub["subject_label"], sub["tir_70_180_pct"], color="#1565C0")
        ax.axvline(70, color="#555", linestyle="--", linewidth=1)
        ax.set_xlim(0, 100)
        _style(ax, f"{split} subject TIR", "")
        ax.set_xlabel("TIR 70-180 (%)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _save_agp(agp: pd.DataFrame, split: str, output: Path) -> None:
    sub = agp[agp["source_split"] == split].copy()
    x = sub["minute_of_day"].to_numpy(dtype=float) / 60.0
    fig, ax = plt.subplots(figsize=(9.8, 4.8), dpi=180)
    ax.axhspan(TARGET_LOW, TARGET_HIGH, color="#DDEFE1", alpha=0.9, label="Target range")
    ax.fill_between(x, sub["p05"].to_numpy(float), sub["p95"].to_numpy(float), color="#B8C7DD", alpha=0.55, label="5-95%")
    ax.fill_between(x, sub["p25"].to_numpy(float), sub["p75"].to_numpy(float), color="#5F7EA8", alpha=0.55, label="25-75%")
    ax.plot(x, sub["p50"].to_numpy(float), color="#111827", linewidth=1.8, label="Median")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_xlabel("Hour of day")
    _style(ax, f"OhioT1DM aggregate AGP ({split})", "Glucose (mg/dL)")
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _save_meal_response(meal_response: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    colors = {"train": "#0F766E", "test": "#B45309"}
    for split, sub in meal_response.groupby("source_split", observed=False):
        x = sub["relative_minutes"].to_numpy(float)
        ax.fill_between(x, sub["p25"].to_numpy(float), sub["p75"].to_numpy(float), color=colors.get(split, "#666"), alpha=0.18)
        ax.plot(x, sub["median_delta_mgdl"].to_numpy(float), color=colors.get(split, "#666"), linewidth=1.8, label=split)
    ax.axvline(0, color="#333", linestyle="--", linewidth=1)
    ax.axhline(0, color="#777", linewidth=0.8)
    ax.set_xlabel("Minutes from meal")
    _style(ax, "Median post-meal glucose response", "Delta glucose (mg/dL)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _save_daily_distribution(daily_metrics: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    values = [
        daily_metrics.loc[daily_metrics["source_split"] == split, "tir_70_180_pct"].dropna().to_numpy(float)
        for split in ["train", "test"]
    ]
    ax.boxplot(values, tick_labels=["train", "test"], patch_artist=True, boxprops={"facecolor": "#C7D2FE"})
    ax.set_ylim(0, 100)
    _style(ax, "Daily TIR distribution", "TIR 70-180 (%)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{float(value):.2f}")
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    # Build a shared anonymized subject map across both splits.
    raw_subjects = []
    for path in [args.train, args.test]:
        raw_subjects.extend(pd.read_csv(path, usecols=["subject_id"])["subject_id"].astype(str).unique().tolist())
    subject_map = _safe_subject_map(raw_subjects)

    train = _load(args.train, "train", subject_map)
    test = _load(args.test, "test", subject_map)
    df = pd.concat([train, test], ignore_index=True)

    split_metrics = pd.DataFrame([{ "source_split": split, **_metrics(group)} for split, group in df.groupby("source_split", observed=False)])
    subject_metrics = _subject_metrics(df)
    daily_metrics = _daily_metrics(df)
    agp = _agp(df)
    meal_response = _meal_response(df)

    split_metrics.to_csv(tables / "split_metrics.csv", index=False)
    subject_metrics.to_csv(tables / "subject_metrics_anonymized.csv", index=False)
    daily_metrics.to_csv(tables / "daily_metrics_anonymized.csv", index=False)
    agp.to_csv(tables / "aggregate_agp.csv", index=False)
    meal_response.to_csv(tables / "meal_response_profile.csv", index=False)

    _save_split_tir(split_metrics, figures / "split_time_in_range.png")
    _save_subject_distribution(subject_metrics, figures / "subject_tir_distribution.png")
    _save_agp(agp, "train", figures / "agp_train.png")
    _save_agp(agp, "test", figures / "agp_test.png")
    _save_meal_response(meal_response, figures / "meal_response_profile.png")
    _save_daily_distribution(daily_metrics, figures / "daily_tir_distribution.png")

    manifest = {
        "source": "OhioT1DM full local XML release prepared into IINTS standard schema",
        "privacy": "Raw subject ids and timestamps are not exported in public tables; subject labels are anonymized S01..S12.",
        "train_path": str(args.train),
        "test_path": str(args.test),
        "outputs": {
            "tables": sorted(p.name for p in tables.glob("*.csv")),
            "figures": sorted(p.name for p in figures.glob("*.png")),
        },
        "split_metrics": split_metrics.to_dict(orient="records"),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    report = [
        "# OhioT1DM Full Aggregate Result Suite",
        "",
        "This folder contains privacy-safe aggregate outputs generated from the full local OhioT1DM XML release.",
        "Raw XML, prepared CSVs, model checkpoints, and these result folders are ignored by git.",
        "",
        "## Key Metrics",
        "",
        _markdown_table(split_metrics),
        "",
        "## Figures",
        "",
    ]
    for fig in sorted(figures.glob("*.png")):
        report.append(f"- `figures/{fig.name}`")
    report.extend([
        "",
        "## Tables",
        "",
    ])
    for table in sorted(tables.glob("*.csv")):
        report.append(f"- `tables/{table.name}`")
    report.extend([
        "",
        "## Public-Use Note",
        "",
        "Use these outputs for EUCYS/docs/research summaries, not the raw Ohio rows. If a public artifact needs subject-level content, use `subject_label` only and avoid exact dates/timestamps.",
    ])
    (args.output_dir / "OHIO_RESULTS_REPORT.md").write_text("\n".join(report) + "\n")
    print(f"Saved Ohio result suite: {args.output_dir}")


if __name__ == "__main__":
    main()
