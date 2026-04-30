from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from iints.analysis.clinical_metrics import ClinicalMetricsCalculator

from .importer import import_cgm_csv
from .quality_checker import DataQualityChecker


RealismStatus = Literal["passed", "warning", "failed", "skipped"]
RealismVerdict = Literal["likely_realistic", "needs_review", "likely_unrealistic"]
REALISM_VERDICT_ORDER: tuple[RealismVerdict, ...] = (
    "likely_unrealistic",
    "needs_review",
    "likely_realistic",
)


@dataclass(frozen=True)
class RealismCheck:
    code: str
    title: str
    status: RealismStatus
    severity: str
    detail: str
    score_impact: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "status": self.status,
            "severity": self.severity,
            "detail": self.detail,
            "score_impact": self.score_impact,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class MealResponse:
    meal_time_minutes: float
    carbs_grams: float
    baseline_glucose_mgdl: float
    peak_glucose_mgdl: float
    rise_mgdl: float
    peak_lag_minutes: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "meal_time_minutes": round(self.meal_time_minutes, 3),
            "carbs_grams": round(self.carbs_grams, 3),
            "baseline_glucose_mgdl": round(self.baseline_glucose_mgdl, 3),
            "peak_glucose_mgdl": round(self.peak_glucose_mgdl, 3),
            "rise_mgdl": round(self.rise_mgdl, 3),
            "peak_lag_minutes": round(self.peak_lag_minutes, 3),
        }


@dataclass(frozen=True)
class RealismReport:
    verdict: RealismVerdict
    realism_score: float
    summary: str
    metrics: Dict[str, Any]
    checks: List[RealismCheck]
    meal_responses: List[MealResponse]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "realism_score": round(self.realism_score, 4),
            "summary": self.summary,
            "metrics": self.metrics,
            "checks": [check.to_dict() for check in self.checks],
            "meal_responses": [response.to_dict() for response in self.meal_responses],
            "warnings": self.warnings,
        }


def _round_or_none(value: float | int | None, digits: int = 3) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def _median_interval_minutes(timestamps: pd.Series) -> float:
    diffs = timestamps.diff().dropna()
    positive = diffs[diffs > 0]
    if positive.empty:
        return 0.0
    return float(positive.median())


def _meal_rows(df: pd.DataFrame, *, min_meal_grams: float) -> pd.DataFrame:
    carbs = pd.to_numeric(df.get("carbs", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return df.loc[carbs >= min_meal_grams].copy()


def _insulin_event_count(df: pd.DataFrame, *, min_units: float = 0.3) -> int:
    insulin = pd.to_numeric(df.get("insulin", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return int((insulin >= min_units).sum())


def _evaluate_meal_responses(
    df: pd.DataFrame,
    *,
    min_meal_grams: float,
    pre_window_minutes: float = 30.0,
    post_peak_start_minutes: float = 20.0,
    post_peak_end_minutes: float = 180.0,
) -> List[MealResponse]:
    meals = _meal_rows(df, min_meal_grams=min_meal_grams)
    responses: List[MealResponse] = []
    if meals.empty:
        return responses

    for row in meals.itertuples(index=False):
        meal_time = float(getattr(row, "timestamp"))
        carbs = float(getattr(row, "carbs"))
        pre = df[(df["timestamp"] < meal_time) & (df["timestamp"] >= meal_time - pre_window_minutes)]
        post = df[
            (df["timestamp"] >= meal_time + post_peak_start_minutes)
            & (df["timestamp"] <= meal_time + post_peak_end_minutes)
        ]
        if len(pre) < 2 or post.empty:
            continue
        baseline = float(pd.to_numeric(pre["glucose"], errors="coerce").median())
        peak_idx = pd.to_numeric(post["glucose"], errors="coerce").idxmax()
        peak_glucose = float(pd.to_numeric(pd.Series([post.loc[peak_idx, "glucose"]]), errors="coerce").iloc[0])
        peak_time = float(pd.to_numeric(pd.Series([post.loc[peak_idx, "timestamp"]]), errors="coerce").iloc[0])
        responses.append(
            MealResponse(
                meal_time_minutes=meal_time,
                carbs_grams=carbs,
                baseline_glucose_mgdl=baseline,
                peak_glucose_mgdl=peak_glucose,
                rise_mgdl=peak_glucose - baseline,
                peak_lag_minutes=peak_time - meal_time,
            )
        )
    return responses


def _check_quality_basics(df: pd.DataFrame, *, expected_interval_minutes: int) -> tuple[RealismCheck, Dict[str, Any]]:
    report = DataQualityChecker(expected_interval=expected_interval_minutes).check(df[["timestamp", "glucose"]].copy())
    critical_anomalies = [anomaly for anomaly in report.anomalies if anomaly.severity == "high"]
    impossible_values = sum(1 for anomaly in critical_anomalies if anomaly.anomaly_type == "impossible_value")
    rapid_changes = sum(1 for anomaly in critical_anomalies if anomaly.anomaly_type == "rapid_change")
    long_gaps = sum(1 for gap in report.gaps if gap.duration_minutes >= expected_interval_minutes * 6)
    metrics = {
        "quality_overall_score": _round_or_none(report.overall_score, 4),
        "gap_count": len(report.gaps),
        "anomaly_count": len(report.anomalies),
        "critical_anomaly_count": len(critical_anomalies),
        "impossible_value_count": impossible_values,
        "rapid_change_count": rapid_changes,
        "long_gap_count": long_gaps,
    }
    if impossible_values > 0 or rapid_changes > 3:
        return (
            RealismCheck(
                code="quality_basics",
                title="Cadence and sensor sanity",
                status="failed",
                severity="critical",
                detail=(
                    f"Detected {impossible_values} impossible value(s) and {rapid_changes} implausible rapid-change event(s); "
                    "the trace likely contains data-quality artifacts."
                ),
                score_impact=0.35,
                metrics=metrics,
            ),
            metrics,
        )
    if rapid_changes > 0 or long_gaps > 0 or report.overall_score < 0.85:
        return (
            RealismCheck(
                code="quality_basics",
                title="Cadence and sensor sanity",
                status="warning",
                severity="warning",
                detail=(
                    f"Found {rapid_changes} rapid-change warning(s), {long_gaps} long gap(s), "
                    f"and an overall quality score of {report.overall_score:.2f}; this trace is usable but needs review."
                ),
                score_impact=0.12,
                metrics=metrics,
            ),
            metrics,
        )
    return (
        RealismCheck(
            code="quality_basics",
            title="Cadence and sensor sanity",
            status="passed",
            severity="info",
            detail=f"Sampling cadence and basic physiological bounds look coherent (quality score {report.overall_score:.2f}).",
            metrics=metrics,
        ),
        metrics,
    )


def _check_variability(
    metrics: Dict[str, Any],
    *,
    meal_count: int,
) -> RealismCheck:
    sd = float(metrics["sd_mgdl"])
    cv = float(metrics["cv_pct"])
    glucose_range = float(metrics["glucose_range_mgdl"])
    max_glucose = float(metrics["max_glucose_mgdl"])
    min_glucose = float(metrics["min_glucose_mgdl"])
    detail = (
        f"Mean {metrics['mean_glucose_mgdl']:.1f} mg/dL, SD {sd:.1f}, CV {cv:.1f}%, range {glucose_range:.1f} mg/dL "
        f"(min {min_glucose:.1f}, max {max_glucose:.1f})."
    )
    if sd < 10.0 or glucose_range < 40.0 or (meal_count >= 3 and max_glucose < 145.0):
        return RealismCheck(
            code="glucose_variability",
            title="Daily variability envelope",
            status="failed",
            severity="critical",
            detail=f"Trace looks too flat to resemble a full T1D day. {detail}",
            score_impact=0.28,
            metrics=metrics,
        )
    if sd < 14.0 or glucose_range < 55.0 or cv < 12.0 or cv > 45.0:
        return RealismCheck(
            code="glucose_variability",
            title="Daily variability envelope",
            status="warning",
            severity="warning",
            detail=f"Variability is on the edge of plausibility and should be reviewed. {detail}",
            score_impact=0.12,
            metrics=metrics,
        )
    return RealismCheck(
        code="glucose_variability",
        title="Daily variability envelope",
        status="passed",
        severity="info",
        detail=f"Variability sits in a believable day-scale envelope. {detail}",
        metrics=metrics,
    )


def _check_event_balance(*, meal_count: int, insulin_event_count: int) -> RealismCheck:
    metrics = {
        "meal_count": meal_count,
        "insulin_event_count": insulin_event_count,
    }
    if meal_count == 0:
        return RealismCheck(
            code="event_balance",
            title="Meal and insulin annotation balance",
            status="skipped",
            severity="info",
            detail="No meal annotations were present, so meal/insulin balance could not be judged.",
            metrics=metrics,
        )
    if meal_count >= 2 and insulin_event_count == 0:
        return RealismCheck(
            code="event_balance",
            title="Meal and insulin annotation balance",
            status="failed",
            severity="critical",
            detail="Meals are annotated but no insulin events are present; that is suspicious for a meal-aware T1D trace.",
            score_impact=0.24,
            metrics=metrics,
        )
    if meal_count >= 3 and insulin_event_count < max(1, meal_count // 2):
        return RealismCheck(
            code="event_balance",
            title="Meal and insulin annotation balance",
            status="warning",
            severity="warning",
            detail="Meal annotations greatly outnumber insulin events; verify that boluses or basal estimates were preserved.",
            score_impact=0.10,
            metrics=metrics,
        )
    return RealismCheck(
        code="event_balance",
        title="Meal and insulin annotation balance",
        status="passed",
        severity="info",
        detail="Meal and insulin annotations move together in a believable way.",
        metrics=metrics,
    )


def _check_meal_response(
    responses: Sequence[MealResponse],
    *,
    meal_count: int,
) -> RealismCheck:
    if meal_count == 0:
        return RealismCheck(
            code="meal_response",
            title="Post-prandial response shape",
            status="skipped",
            severity="info",
            detail="No meal annotations were present, so post-prandial realism was not evaluated.",
        )
    if not responses:
        return RealismCheck(
            code="meal_response",
            title="Post-prandial response shape",
            status="warning",
            severity="warning",
            detail="Meal annotations exist, but there was not enough surrounding data to judge the glucose response.",
            score_impact=0.08,
            metrics={"assessed_meals": 0, "meal_count": meal_count},
        )

    rises = np.array([response.rise_mgdl for response in responses], dtype=float)
    lags = np.array([response.peak_lag_minutes for response in responses], dtype=float)
    responding = (rises >= 15.0) & (lags >= 20.0) & (lags <= 180.0)
    response_ratio = float(responding.mean()) if len(responding) else 0.0
    metrics = {
        "assessed_meals": len(responses),
        "responding_meals": int(responding.sum()),
        "response_ratio": _round_or_none(response_ratio, 4),
        "median_rise_mgdl": _round_or_none(float(np.median(rises))),
        "median_peak_lag_minutes": _round_or_none(float(np.median(lags))),
    }
    if response_ratio < 0.34 or float(np.median(rises)) < 10.0:
        return RealismCheck(
            code="meal_response",
            title="Post-prandial response shape",
            status="failed",
            severity="critical",
            detail=(
                f"Meals rarely produce a believable glucose excursion. "
                f"Only {int(responding.sum())}/{len(responses)} assessed meals showed a >=15 mg/dL rise with a plausible lag."
            ),
            score_impact=0.26,
            metrics=metrics,
        )
    if response_ratio < 0.6 or float(np.median(rises)) < 15.0:
        return RealismCheck(
            code="meal_response",
            title="Post-prandial response shape",
            status="warning",
            severity="warning",
            detail=(
                f"Meal responses are present but weaker or less consistent than expected: "
                f"{int(responding.sum())}/{len(responses)} assessed meals responded clearly."
            ),
            score_impact=0.10,
            metrics=metrics,
        )
    return RealismCheck(
        code="meal_response",
        title="Post-prandial response shape",
        status="passed",
        severity="info",
        detail=(
            f"Meal excursions look believable: {int(responding.sum())}/{len(responses)} assessed meals "
            f"showed a clear post-prandial rise."
        ),
        metrics=metrics,
    )


def _check_overnight_shape(df: pd.DataFrame, *, overall_sd: float) -> RealismCheck:
    minute_of_day = pd.to_numeric(df["timestamp"], errors="coerce").mod(1440.0)
    overnight = df[(minute_of_day >= 0.0) & (minute_of_day < 360.0)]
    if len(overnight) < 12:
        return RealismCheck(
            code="overnight_shape",
            title="Overnight shape",
            status="skipped",
            severity="info",
            detail="Not enough overnight coverage to judge the night-time pattern.",
        )
    overnight_glucose = pd.to_numeric(overnight["glucose"], errors="coerce").dropna()
    overnight_range = float(overnight_glucose.max() - overnight_glucose.min())
    metrics = {
        "overnight_range_mgdl": _round_or_none(overnight_range),
        "overnight_samples": int(len(overnight_glucose)),
    }
    if overnight_range < 8.0 and overall_sd < 12.0:
        return RealismCheck(
            code="overnight_shape",
            title="Overnight shape",
            status="failed",
            severity="critical",
            detail="The overnight section is almost perfectly flat, which usually reads as synthetic rather than physiological.",
            score_impact=0.16,
            metrics=metrics,
        )
    if overnight_range > 85.0:
        return RealismCheck(
            code="overnight_shape",
            title="Overnight shape",
            status="warning",
            severity="warning",
            detail="The overnight section is extremely volatile; double-check whether this is physiology or data corruption.",
            score_impact=0.08,
            metrics=metrics,
        )
    return RealismCheck(
        code="overnight_shape",
        title="Overnight shape",
        status="passed",
        severity="info",
        detail="The overnight section has a believable amount of drift and stability.",
        metrics=metrics,
    )


def realism_verdict_meets_minimum(verdict: str, minimum: str) -> bool:
    return REALISM_VERDICT_ORDER.index(verdict) >= REALISM_VERDICT_ORDER.index(minimum)


def validate_realism_dataset(
    dataframe: pd.DataFrame,
    *,
    expected_interval_minutes: int = 5,
    min_meal_grams: float = 10.0,
) -> RealismReport:
    if "timestamp" not in dataframe.columns or "glucose" not in dataframe.columns:
        raise ValueError("Realism validation requires at least 'timestamp' and 'glucose' columns.")

    df = dataframe.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
    if "carbs" not in df.columns:
        df["carbs"] = 0.0
    if "insulin" not in df.columns:
        df["insulin"] = 0.0
    df["carbs"] = pd.to_numeric(df["carbs"], errors="coerce").fillna(0.0)
    df["insulin"] = pd.to_numeric(df["insulin"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["timestamp", "glucose"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("Realism validation requires at least one valid glucose row.")

    duration_hours = float((df["timestamp"].max() - df["timestamp"].min()) / 60.0) if len(df) > 1 else 0.0
    calculator = ClinicalMetricsCalculator()
    clinical = calculator.calculate(glucose=df["glucose"], timestamp=df["timestamp"], duration_hours=max(duration_hours, expected_interval_minutes / 60.0))
    meal_count = int((df["carbs"] >= min_meal_grams).sum())
    insulin_events = _insulin_event_count(df)
    median_interval = _median_interval_minutes(df["timestamp"])
    responses = _evaluate_meal_responses(df, min_meal_grams=min_meal_grams)

    summary_metrics: Dict[str, Any] = {
        "row_count": int(len(df)),
        "duration_hours": _round_or_none(duration_hours),
        "median_interval_minutes": _round_or_none(median_interval),
        "mean_glucose_mgdl": _round_or_none(clinical.mean_glucose),
        "sd_mgdl": _round_or_none(clinical.sd),
        "cv_pct": _round_or_none(clinical.cv),
        "tir_70_180_pct": _round_or_none(clinical.tir_70_180),
        "tir_above_180_pct": _round_or_none(clinical.tir_above_180),
        "tir_below_70_pct": _round_or_none(clinical.tir_below_70),
        "min_glucose_mgdl": _round_or_none(float(df["glucose"].min())),
        "max_glucose_mgdl": _round_or_none(float(df["glucose"].max())),
        "glucose_range_mgdl": _round_or_none(float(df["glucose"].max() - df["glucose"].min())),
        "meal_count": meal_count,
        "insulin_event_count": insulin_events,
        "assessed_meal_responses": len(responses),
    }

    quality_check, quality_metrics = _check_quality_basics(df, expected_interval_minutes=expected_interval_minutes)
    summary_metrics.update(quality_metrics)
    checks = [
        quality_check,
        _check_variability(summary_metrics, meal_count=meal_count),
        _check_event_balance(meal_count=meal_count, insulin_event_count=insulin_events),
        _check_meal_response(responses, meal_count=meal_count),
        _check_overnight_shape(df, overall_sd=float(clinical.sd)),
    ]

    realism_score = max(0.0, 1.0 - sum(check.score_impact for check in checks))
    failed_checks = [check for check in checks if check.status == "failed"]
    warning_checks = [check for check in checks if check.status == "warning"]
    if realism_score < 0.45 or len(failed_checks) >= 2:
        verdict: RealismVerdict = "likely_unrealistic"
    elif failed_checks or len(warning_checks) >= 2 or realism_score < 0.75:
        verdict = "needs_review"
    else:
        verdict = "likely_realistic"

    warnings = [check.detail for check in checks if check.status in {"warning", "failed"}]
    summary = (
        f"{verdict.replace('_', ' ')}: mean {clinical.mean_glucose:.1f} mg/dL, CV {clinical.cv:.1f}%, "
        f"{meal_count} meal event(s), {insulin_events} insulin event(s), realism score {realism_score:.2f}."
    )
    return RealismReport(
        verdict=verdict,
        realism_score=realism_score,
        summary=summary,
        metrics=summary_metrics,
        checks=checks,
        meal_responses=responses,
        warnings=warnings,
    )


def validate_realism_csv(
    input_csv: str | Path,
    *,
    data_format: str = "generic",
    column_map: Optional[Dict[str, str]] = None,
    time_unit: str = "minutes",
    source: Optional[str] = None,
    expected_interval_minutes: int = 5,
    min_meal_grams: float = 10.0,
) -> RealismReport:
    df = import_cgm_csv(
        input_csv,
        data_format=data_format,
        column_map=column_map,
        time_unit=time_unit,
        source=source,
    )
    return validate_realism_dataset(
        df,
        expected_interval_minutes=expected_interval_minutes,
        min_meal_grams=min_meal_grams,
    )


def write_realism_report(
    report: RealismReport,
    output_path: str | Path,
) -> Path:
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return resolved
