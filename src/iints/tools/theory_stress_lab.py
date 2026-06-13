from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

import numpy as np

from iints.core.devices.models import PumpModel, SensorModel
from iints.core.patient.advanced_metabolic_model import AdvancedMetabolicModel
from iints.core.supervisor import IndependentSupervisor

CheckStatus = Literal["passed", "warning", "failed"]


@dataclass(frozen=True)
class TheoryCheckResult:
    code: str
    title: str
    status: CheckStatus
    score: float
    severity: str
    detail: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    weak_point: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "status": self.status,
            "score": round(float(self.score), 4),
            "severity": self.severity,
            "detail": self.detail,
            "metrics": _json_safe(self.metrics),
            "weak_point": self.weak_point,
        }


@dataclass(frozen=True)
class TheoryStressReport:
    profile: str
    seed: int
    duration_minutes: float
    generated_at_unix: float
    overall_score: float
    verdict: str
    checks: List[TheoryCheckResult]
    output_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "seed": self.seed,
            "duration_minutes": self.duration_minutes,
            "generated_at_unix": self.generated_at_unix,
            "overall_score": round(float(self.overall_score), 4),
            "verdict": self.verdict,
            "checks": [check.to_dict() for check in self.checks],
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True)
class TraceRow:
    time_min: float
    glucose: float
    ffa: float
    ketones: float
    beta_mass: float
    iob: float
    cob: float
    delivered_insulin: float
    carbs: float
    fat: float
    protein: float
    sensor: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_min": self.time_min,
            "glucose": self.glucose,
            "ffa": self.ffa,
            "ketones": self.ketones,
            "beta_mass": self.beta_mass,
            "iob": self.iob,
            "cob": self.cob,
            "delivered_insulin": self.delivered_insulin,
            "carbs": self.carbs,
            "fat": self.fat,
            "protein": self.protein,
            "sensor": self.sensor,
        }


@dataclass(frozen=True)
class ScenarioSummary:
    name: str
    rows: List[TraceRow]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def values(self, key: str) -> List[float]:
        return [float(row.to_dict()[key]) for row in self.rows]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if hasattr(value, "name") and hasattr(value, "value"):
        return getattr(value, "value")
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _status_from_score(score: float, *, fail_below: float = 0.5, warn_below: float = 0.85) -> CheckStatus:
    if score < fail_below:
        return "failed"
    if score < warn_below:
        return "warning"
    return "passed"


def _severity(status: CheckStatus) -> str:
    return {"passed": "info", "warning": "warning", "failed": "critical"}[status]


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _advanced_state(patient: AdvancedMetabolicModel) -> tuple[float, float, float]:
    state = patient.get_patient_state()
    return (
        float(state.get("plasma_ffa_mmol_L", 0.0)),
        float(state.get("plasma_ketones_mmol_L", 0.0)),
        float(state.get("residual_beta_cell_mass", 0.0)),
    )


def _simulate_patient(
    *,
    name: str,
    minutes: int,
    time_step: float,
    seed: int,
    initial_glucose: float = 120.0,
    initial_ffa: float = 0.4,
    initial_ketones: float = 0.1,
    initial_beta_mass: float = 0.0,
    basal_insulin_rate: float = 0.8,
    insulin_schedule: Optional[Dict[int, float]] = None,
    carb_schedule: Optional[Dict[int, float]] = None,
    fat_schedule: Optional[Dict[int, float]] = None,
    protein_schedule: Optional[Dict[int, float]] = None,
    exercise_windows: Optional[Sequence[tuple[int, int, float]]] = None,
    illness_windows: Optional[Sequence[tuple[int, int, float]]] = None,
    pump_dropout_windows: Optional[Sequence[tuple[int, int]]] = None,
    sensor: Optional[SensorModel] = None,
) -> ScenarioSummary:
    _ = np.random.default_rng(seed)
    patient = AdvancedMetabolicModel(
        initial_glucose=initial_glucose,
        initial_ffa=initial_ffa,
        initial_ketones=initial_ketones,
        initial_beta_mass=initial_beta_mass,
        basal_insulin_rate=basal_insulin_rate,
    )
    insulin_schedule = insulin_schedule or {}
    carb_schedule = carb_schedule or {}
    fat_schedule = fat_schedule or {}
    protein_schedule = protein_schedule or {}
    exercise_windows = exercise_windows or []
    illness_windows = illness_windows or []
    pump_dropout_windows = pump_dropout_windows or []

    rows: List[TraceRow] = []
    for minute in range(0, minutes + 1, int(time_step)):
        exercising = next((intensity for start, end, intensity in exercise_windows if start <= minute < end), 0.0)
        if exercising > 0:
            patient.start_exercise(float(exercising))
        else:
            patient.stop_exercise()

        illness = next((severity for start, end, severity in illness_windows if start <= minute < end), 0.0)
        if illness > 0:
            patient.start_illness(float(illness))
        else:
            patient.stop_illness()

        scheduled_bolus = float(insulin_schedule.get(minute, 0.0))
        scheduled_carbs = float(carb_schedule.get(minute, 0.0))
        scheduled_fat = float(fat_schedule.get(minute, 0.0))
        scheduled_protein = float(protein_schedule.get(minute, 0.0))
        dropout = any(start <= minute < end for start, end in pump_dropout_windows)

        # basal_insulin_rate is U/hour; update() expects units delivered this step.
        # Without this conversion, normal scenarios accidentally become pump-failure scenarios.
        basal_for_step = max(0.0, float(basal_insulin_rate)) * float(time_step) / 60.0
        delivered_insulin = 0.0 if dropout else scheduled_bolus + basal_for_step

        glucose = float(
            patient.update(
                time_step=time_step,
                delivered_insulin=delivered_insulin,
                carb_intake=scheduled_carbs,
                fat_intake=scheduled_fat,
                protein_intake=scheduled_protein,
                current_time=float(minute),
            )
        )
        ffa, ketones, beta_mass = _advanced_state(patient)
        sensor_value = None
        if sensor is not None:
            sensor_value = float(sensor.read(glucose, float(minute)).value)
        rows.append(
            TraceRow(
                time_min=float(minute),
                glucose=glucose,
                ffa=ffa,
                ketones=ketones,
                beta_mass=beta_mass,
                iob=float(patient.insulin_on_board),
                cob=float(patient.carbs_on_board),
                delivered_insulin=delivered_insulin,
                carbs=scheduled_carbs,
                fat=scheduled_fat,
                protein=scheduled_protein,
                sensor=sensor_value,
            )
        )
    return ScenarioSummary(name=name, rows=rows, metadata={"minutes": minutes, "time_step": time_step})


def check_no_negative_states(seed: int) -> TheoryCheckResult:
    scenarios = [
        _simulate_patient(
            name="mixed_extreme",
            minutes=12 * 60,
            time_step=5.0,
            seed=seed,
            initial_glucose=160.0,
            insulin_schedule={60: 1.0, 120: 2.0, 240: 0.5, 420: 2.5},
            carb_schedule={30: 90.0, 300: 120.0, 540: 60.0},
            fat_schedule={30: 35.0, 300: 25.0},
            protein_schedule={30: 25.0, 300: 40.0},
            exercise_windows=[(180, 300, 0.8)],
            illness_windows=[(360, 720, 0.9)],
        )
    ]
    min_values: Dict[str, float] = {}
    bad: List[str] = []
    for scenario in scenarios:
        for key in ["glucose", "ffa", "ketones", "beta_mass", "iob", "cob"]:
            values = scenario.values(key)
            minimum = min(values)
            min_values[f"{scenario.name}.{key}"] = round(minimum, 5)
            if not math.isfinite(minimum) or minimum < -1e-6:
                bad.append(f"{scenario.name}.{key}={minimum:.3f}")
    score = 1.0 if not bad else 0.0
    status = _status_from_score(score)
    return TheoryCheckResult(
        code="no_negative_states",
        title="No negative or non-finite physiology states",
        status=status,
        score=score,
        severity=_severity(status),
        detail="All tracked states stayed non-negative and finite." if not bad else "Invalid states detected: " + ", ".join(bad[:6]),
        metrics=min_values,
        weak_point=None if not bad else "Numerical guard or ODE clamp failed under mixed extreme stress.",
    )


def check_hypo_blocks_insulin(seed: int) -> TheoryCheckResult:
    _ = seed
    supervisor = IndependentSupervisor()
    result = supervisor.evaluate_safety(
        current_glucose=62.0,
        proposed_insulin=2.0,
        current_time=0.0,
        current_iob=0.2,
        predicted_glucose_30min=58.0,
    )
    final = float(result.get("safe_insulin", result.get("final_dose", 0.0)))
    triggered = bool(result.get("safety_triggered", result.get("triggered", False)))
    score = 1.0 if final <= 0.1 and triggered else 0.0
    status = _status_from_score(score)
    return TheoryCheckResult(
        code="hypo_blocks_insulin",
        title="Hypoglycemia blocks insulin",
        status=status,
        score=score,
        severity=_severity(status),
        detail=f"Supervisor returned {final:.3f}U for glucose 62 mg/dL and a 2.0U proposal.",
        metrics={"final_insulin_units": final, "triggered": triggered, "raw_result": result},
        weak_point=None if score == 1.0 else "Supervisor allowed insulin during hypoglycemia.",
    )


def check_iob_limits_bolus(seed: int) -> TheoryCheckResult:
    _ = seed
    supervisor = IndependentSupervisor(max_iob=4.0)
    result = supervisor.evaluate_safety(
        current_glucose=170.0,
        proposed_insulin=3.0,
        current_time=0.0,
        current_iob=3.7,
    )
    final = float(result.get("safe_insulin", result.get("final_dose", 0.0)))
    expected_cap = 0.3
    score = 1.0 if final <= expected_cap + 1e-6 else max(0.0, expected_cap / max(final, 1e-9))
    status = _status_from_score(score)
    return TheoryCheckResult(
        code="iob_limits_bolus",
        title="IOB mass-balance limits extra bolus",
        status=status,
        score=score,
        severity=_severity(status),
        detail=f"With 3.7U IOB and max_iob 4.0U, proposed 3.0U was reduced to {final:.3f}U.",
        metrics={"final_insulin_units": final, "expected_max_units": expected_cap, "raw_result": result},
        weak_point=None if status == "passed" else "PD mass-balance cap did not constrain the bolus enough.",
    )


def check_pump_failure_raises_ffa_ketones(seed: int) -> TheoryCheckResult:
    scenario = _simulate_patient(
        name="12h_no_insulin",
        minutes=12 * 60,
        time_step=5.0,
        seed=seed,
        initial_glucose=135.0,
        initial_ffa=0.4,
        initial_ketones=0.1,
        insulin_schedule={},
        carb_schedule={60: 45.0, 300: 35.0},
        pump_dropout_windows=[(0, 12 * 60)],
    )
    first = scenario.rows[0]
    last = scenario.rows[-1]
    ffa_ratio = last.ffa / max(first.ffa, 1e-9)
    ketone_ratio = last.ketones / max(first.ketones, 1e-9)
    ffa_ok = ffa_ratio >= 1.35
    ketone_ok = ketone_ratio >= 1.35
    score = (0.5 if ffa_ok else _clip_score((ffa_ratio - 1.0) / 0.35 * 0.5)) + (
        0.5 if ketone_ok else _clip_score((ketone_ratio - 1.0) / 0.35 * 0.5)
    )
    status = _status_from_score(score)
    return TheoryCheckResult(
        code="pump_failure_raises_ffa_ketones",
        title="Pump failure raises FFA and ketones",
        status=status,
        score=score,
        severity=_severity(status),
        detail=(
            f"After 12h without insulin, FFA changed {first.ffa:.3f}->{last.ffa:.3f} "
            f"and ketones {first.ketones:.3f}->{last.ketones:.3f}."
        ),
        metrics={"ffa_ratio": round(ffa_ratio, 4), "ketone_ratio": round(ketone_ratio, 4)},
        weak_point=None if status == "passed" else "Lipotoxicity/ketogenesis response may be too weak during insulin absence.",
    )


def check_sensor_lag_is_bounded(seed: int) -> TheoryCheckResult:
    sensor = SensorModel(noise_std=0.0, bias=0.0, lag_minutes=10, isf_tau_minutes=5.0, seed=seed)
    true_values: List[float] = []
    sensor_values: List[float] = []
    for minute in range(0, 121, 5):
        true = 100.0 if minute < 30 else min(220.0, 100.0 + (minute - 30) * 2.0)
        true_values.append(true)
        sensor_values.append(float(sensor.read(true, float(minute)).value))
    max_jump = max(abs(b - a) for a, b in zip(sensor_values, sensor_values[1:]))
    max_true_jump = max(abs(b - a) for a, b in zip(true_values, true_values[1:]))
    lagged_at_step = sensor_values[8] < true_values[8] - 5.0  # 40 min, after ramp starts.
    bounded = max_jump <= max_true_jump + 5.0
    score = (0.5 if lagged_at_step else 0.0) + (0.5 if bounded else 0.0)
    status = _status_from_score(score)
    return TheoryCheckResult(
        code="sensor_lag_is_bounded",
        title="CGM lag is visible but bounded",
        status=status,
        score=score,
        severity=_severity(status),
        detail=f"Max sensor jump was {max_jump:.2f} mg/dL per 5 min while true max jump was {max_true_jump:.2f}.",
        metrics={"max_sensor_jump_5min": round(max_jump, 4), "max_true_jump_5min": round(max_true_jump, 4), "lagged_at_step": lagged_at_step},
        weak_point=None if status == "passed" else "Sensor model may teleport or fail to show expected interstitial lag.",
    )


def check_exercise_does_not_create_impossible_crash(seed: int) -> TheoryCheckResult:
    scenario = _simulate_patient(
        name="exercise_high_iob",
        minutes=4 * 60,
        time_step=5.0,
        seed=seed,
        initial_glucose=155.0,
        insulin_schedule={0: 2.2, 30: 1.0},
        carb_schedule={120: 10.0},
        exercise_windows=[(45, 180, 0.9)],
    )
    glucose = scenario.values("glucose")
    times = scenario.values("time_min")
    rates = [abs((g1 - g0) / max(t1 - t0, 1e-9)) for g0, g1, t0, t1 in zip(glucose, glucose[1:], times, times[1:])]
    max_rate = max(rates)
    min_glucose = min(glucose)
    rate_ok = max_rate <= 4.0
    floor_ok = min_glucose >= 20.0
    score = (0.6 if rate_ok else _clip_score(4.0 / max(max_rate, 1e-9)) * 0.6) + (0.4 if floor_ok else 0.0)
    status = _status_from_score(score)
    return TheoryCheckResult(
        code="exercise_does_not_create_impossible_crash",
        title="Exercise stress does not create impossible glucose crash",
        status=status,
        score=score,
        severity=_severity(status),
        detail=f"Exercise/high-IOB scenario reached min glucose {min_glucose:.1f} mg/dL and max rate {max_rate:.2f} mg/dL/min.",
        metrics={"min_glucose_mgdl": round(min_glucose, 4), "max_abs_rate_mgdl_min": round(max_rate, 4)},
        weak_point=None if status == "passed" else "Exercise multiplier or insulin action may create too-fast glucose motion.",
    )


def check_meal_response_has_plausible_peak(seed: int) -> TheoryCheckResult:
    scenario = _simulate_patient(
        name="fatty_meal",
        minutes=6 * 60,
        time_step=5.0,
        seed=seed,
        initial_glucose=105.0,
        insulin_schedule={0: 2.0, 90: 1.0},
        carb_schedule={0: 80.0},
        fat_schedule={0: 35.0},
        protein_schedule={0: 25.0},
    )
    baseline = scenario.rows[0].glucose
    post = [row for row in scenario.rows if row.time_min >= 20]
    peak = max(post, key=lambda row: row.glucose)
    rise = peak.glucose - baseline
    lag = peak.time_min
    rise_ok = 20.0 <= rise <= 180.0
    lag_ok = 45.0 <= lag <= 300.0
    score = (0.5 if rise_ok else 0.0) + (0.5 if lag_ok else 0.0)
    status = _status_from_score(score)
    return TheoryCheckResult(
        code="meal_response_has_plausible_peak",
        title="Meal response has plausible peak size and timing",
        status=status,
        score=score,
        severity=_severity(status),
        detail=f"80g carb/fat meal peak rose {rise:.1f} mg/dL at {lag:.0f} minutes.",
        metrics={"rise_mgdl": round(rise, 4), "peak_lag_minutes": round(lag, 4), "peak_glucose_mgdl": round(peak.glucose, 4)},
        weak_point=None if status == "passed" else "Meal absorption/fat-delay model produced implausible post-prandial shape.",
    )


def check_illness_increases_insulin_need_without_exploding(seed: int) -> TheoryCheckResult:
    healthy = _simulate_patient(
        name="healthy_control",
        minutes=6 * 60,
        time_step=5.0,
        seed=seed,
        initial_glucose=120.0,
        insulin_schedule={minute: 0.08 for minute in range(0, 6 * 60 + 1, 30)},
        carb_schedule={60: 60.0, 240: 40.0},
    )
    ill = _simulate_patient(
        name="illness",
        minutes=6 * 60,
        time_step=5.0,
        seed=seed + 1,
        initial_glucose=120.0,
        insulin_schedule={minute: 0.08 for minute in range(0, 6 * 60 + 1, 30)},
        carb_schedule={60: 60.0, 240: 40.0},
        illness_windows=[(0, 6 * 60, 0.8)],
    )
    healthy_end = healthy.rows[-1].glucose
    ill_end = ill.rows[-1].glucose
    ill_values = ill.values("glucose")
    max_ill = max(ill_values)
    finite = all(math.isfinite(v) for v in ill_values)
    increased = ill_end > healthy_end + 5.0
    not_exploded = max_ill < 500.0
    score = (0.45 if increased else 0.0) + (0.45 if not_exploded else 0.0) + (0.10 if finite else 0.0)
    status = _status_from_score(score)
    return TheoryCheckResult(
        code="illness_increases_insulin_need_without_exploding",
        title="Illness raises glucose pressure without numerical explosion",
        status=status,
        score=score,
        severity=_severity(status),
        detail=f"Healthy end {healthy_end:.1f} mg/dL vs illness end {ill_end:.1f} mg/dL; illness max {max_ill:.1f}.",
        metrics={"healthy_end_mgdl": round(healthy_end, 4), "illness_end_mgdl": round(ill_end, 4), "illness_max_mgdl": round(max_ill, 4)},
        weak_point=None if status == "passed" else "Illness multiplier may be too weak or numerically unstable.",
    )


CHECKS = [
    check_no_negative_states,
    check_hypo_blocks_insulin,
    check_iob_limits_bolus,
    check_pump_failure_raises_ffa_ketones,
    check_sensor_lag_is_bounded,
    check_exercise_does_not_create_impossible_crash,
    check_meal_response_has_plausible_peak,
    check_illness_increases_insulin_need_without_exploding,
]


def run_theory_stress_lab(
    *,
    output_dir: Optional[Path] = None,
    profile: str = "jetson",
    seed: int = 42,
    repeats: int = 1,
    duration_minutes: float = 30.0,
) -> TheoryStressReport:
    """Run deterministic scientific invariant checks against SDK physiology/safety theories."""
    start = time.time()
    all_results: List[TheoryCheckResult] = []
    for repeat in range(max(1, int(repeats))):
        repeat_seed = int(seed) + repeat * 1009
        for check in CHECKS:
            result = check(repeat_seed)
            if repeats > 1:
                result = TheoryCheckResult(
                    code=f"{result.code}#r{repeat + 1}",
                    title=result.title,
                    status=result.status,
                    score=result.score,
                    severity=result.severity,
                    detail=result.detail,
                    metrics={**result.metrics, "repeat": repeat + 1, "seed": repeat_seed},
                    weak_point=result.weak_point,
                )
            all_results.append(result)

    overall = float(np.mean([result.score for result in all_results])) if all_results else 0.0
    failed = sum(1 for result in all_results if result.status == "failed")
    warnings = sum(1 for result in all_results if result.status == "warning")
    if failed:
        verdict = "weak_points_found"
    elif warnings:
        verdict = "needs_review"
    else:
        verdict = "stable_under_configured_checks"

    report = TheoryStressReport(
        profile=profile,
        seed=seed,
        duration_minutes=duration_minutes,
        generated_at_unix=start,
        overall_score=overall,
        verdict=verdict,
        checks=all_results,
        output_dir=str(output_dir) if output_dir else None,
    )
    if output_dir is not None:
        write_theory_stress_outputs(report, output_dir)
    return report


def _weakness_rows(report: TheoryStressReport) -> List[Dict[str, Any]]:
    return [
        {
            "rank": rank,
            "code": check.code,
            "status": check.status,
            "score": round(check.score, 4),
            "severity": check.severity,
            "weak_point": check.weak_point or "",
            "detail": check.detail,
        }
        for rank, check in enumerate(sorted(report.checks, key=lambda item: item.score), start=1)
    ]


def write_theory_stress_outputs(report: TheoryStressReport, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "checks.json"
    summary_json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    ranking_csv = output_dir / "weakness_rankings.csv"
    rows = _weakness_rows(report)
    with ranking_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "code", "status", "score", "severity", "weak_point", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    summary_md = output_dir / "summary.md"
    lines = [
        "# IINTS-AF Theory Stress Lab",
        "",
        f"**Verdict:** `{report.verdict}`",
        f"**Overall score:** `{report.overall_score:.3f}`",
        f"**Profile:** `{report.profile}`",
        f"**Seed:** `{report.seed}`",
        "",
        "## What This Tests",
        "",
        "This is a deterministic scientific red-team pass over SDK physiology and safety assumptions. It is pre-clinical simulation QA, not medical validation.",
        "",
        "## Weakness Ranking",
        "",
        "| Rank | Check | Status | Score | Weak Point |",
        "|---:|---|---|---:|---|",
    ]
    for row in rows:
        weak_point = row["weak_point"] or "-"
        lines.append(f"| {row['rank']} | `{row['code']}` | `{row['status']}` | {row['score']:.3f} | {weak_point} |")
    lines.extend(["", "## Check Details", ""])
    for check in report.checks:
        lines.extend([
            f"### `{check.code}`",
            "",
            f"- Status: `{check.status}`",
            f"- Score: `{check.score:.3f}`",
            f"- Detail: {check.detail}",
        ])
        if check.weak_point:
            lines.append(f"- Weak point: {check.weak_point}")
        lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return {"summary_md": summary_md, "checks_json": summary_json, "weakness_csv": ranking_csv}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the IINTS-AF Theory Stress Lab.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/theory_stress_lab"))
    parser.add_argument("--profile", default="jetson", choices=["jetson", "ci", "deep"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--fail-on-weakness", action="store_true", help="Exit 1 when a configured invariant fails.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = run_theory_stress_lab(
        output_dir=args.output_dir,
        profile=args.profile,
        seed=args.seed,
        repeats=args.repeats,
        duration_minutes=args.duration_minutes,
    )
    print(f"Theory Stress Lab verdict: {report.verdict}")
    print(f"Overall score: {report.overall_score:.3f}")
    print(f"Artifacts written to: {args.output_dir}")
    return 1 if args.fail_on_weakness and report.verdict == "weak_points_found" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
