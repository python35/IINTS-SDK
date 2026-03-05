from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from iints.core.safety.config import SafetyConfig
from iints.core.supervisor import IndependentSupervisor


@dataclass(frozen=True)
class SafetyContractSpec:
    contract_enabled: bool
    contract_glucose_threshold: float
    contract_trend_threshold_mgdl_min: float
    contract_max_iob_units: float
    contract_max_bolus_units: float
    contract_hypo_cutoff_mgdl: float


@dataclass(frozen=True)
class SafetyContractViolation:
    glucose_mgdl: float
    trend_mgdl_min: float
    proposed_insulin_units: float
    approved_insulin_units: float
    reason: str


@dataclass(frozen=True)
class SafetyContractVerificationReport:
    spec: SafetyContractSpec
    total_cases: int
    expected_block_cases: int
    blocked_cases: int
    violations: List[SafetyContractViolation]

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "total_cases": self.total_cases,
            "expected_block_cases": self.expected_block_cases,
            "blocked_cases": self.blocked_cases,
            "passed": self.passed,
            "violations": [asdict(v) for v in self.violations],
        }


def load_contract_spec(path: Optional[Path] = None) -> SafetyContractSpec:
    if path is None:
        import sys

        if sys.version_info >= (3, 9):
            from importlib.resources import files

            content = files("iints.presets").joinpath("safety_contract_default.yaml").read_text()
        else:
            from importlib import resources

            content = resources.read_text("iints.presets", "safety_contract_default.yaml")
    else:
        content = path.read_text()

    payload = yaml.safe_load(content) or {}
    if not isinstance(payload, dict):
        raise ValueError("Contract file must be a YAML mapping.")

    required = [
        "contract_enabled",
        "contract_glucose_threshold",
        "contract_trend_threshold_mgdl_min",
        "contract_max_iob_units",
        "contract_max_bolus_units",
        "contract_hypo_cutoff_mgdl",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Contract file missing keys: {missing}")

    return SafetyContractSpec(
        contract_enabled=bool(payload["contract_enabled"]),
        contract_glucose_threshold=float(payload["contract_glucose_threshold"]),
        contract_trend_threshold_mgdl_min=float(payload["contract_trend_threshold_mgdl_min"]),
        contract_max_iob_units=float(payload["contract_max_iob_units"]),
        contract_max_bolus_units=float(payload["contract_max_bolus_units"]),
        contract_hypo_cutoff_mgdl=float(payload["contract_hypo_cutoff_mgdl"]),
    )


def apply_contract_to_config(spec: SafetyContractSpec, base: Optional[SafetyConfig] = None) -> SafetyConfig:
    cfg = SafetyConfig() if base is None else SafetyConfig(**asdict(base))
    cfg.contract_enabled = spec.contract_enabled
    cfg.contract_glucose_threshold = spec.contract_glucose_threshold
    cfg.contract_trend_threshold_mgdl_min = spec.contract_trend_threshold_mgdl_min
    cfg.max_iob = spec.contract_max_iob_units
    cfg.max_insulin_per_bolus = spec.contract_max_bolus_units
    cfg.hypo_cutoff = spec.contract_hypo_cutoff_mgdl
    return cfg


def verify_safety_contract(
    spec: SafetyContractSpec,
    *,
    glucose_values: List[float],
    trend_values: List[float],
    proposed_doses: List[float],
    iob_values: Optional[List[float]] = None,
) -> SafetyContractVerificationReport:
    cfg = apply_contract_to_config(spec)
    supervisor = IndependentSupervisor(safety_config=cfg)

    violations: List[SafetyContractViolation] = []
    total_cases = 0
    expected_block_cases = 0
    blocked_cases = 0

    iob_grid = iob_values or [0.0, spec.contract_max_iob_units * 0.5, spec.contract_max_iob_units + 0.5]

    for glucose in glucose_values:
        for trend in trend_values:
            for proposed in proposed_doses:
                for current_iob in iob_grid:
                    total_cases += 1
                    should_block_by_glucose_trend = (
                        spec.contract_enabled
                        and glucose < spec.contract_glucose_threshold
                        and trend <= spec.contract_trend_threshold_mgdl_min
                        and proposed > 0.0
                    )
                    should_block_by_hypo = glucose < spec.contract_hypo_cutoff_mgdl and proposed > 0.0
                    should_block_by_iob = current_iob > spec.contract_max_iob_units and proposed > 0.0
                    should_block = should_block_by_glucose_trend or should_block_by_hypo or should_block_by_iob
                    should_cap_bolus = proposed > spec.contract_max_bolus_units
                    if should_block:
                        expected_block_cases += 1

                    g0 = glucose - (trend * 10.0)
                    g5 = glucose - (trend * 5.0)
                    supervisor.glucose_history = [(0.0, g0), (5.0, g5)]
                    result = supervisor.evaluate_safety(
                        current_glucose=glucose,
                        proposed_insulin=proposed,
                        current_time=10.0,
                        current_iob=current_iob,
                        predicted_glucose_30min=glucose + (trend * 30.0),
                    )
                    approved = float(result["approved_insulin"])
                    if approved <= 0.0 and proposed > 0.0:
                        blocked_cases += 1
                    if should_block and approved > 0.0:
                        violations.append(
                            SafetyContractViolation(
                                glucose_mgdl=float(glucose),
                                trend_mgdl_min=float(trend),
                                proposed_insulin_units=float(proposed),
                                approved_insulin_units=approved,
                                reason=f"Expected block (iob={current_iob:.2f}) but insulin was approved",
                            )
                        )
                    if should_cap_bolus and approved > spec.contract_max_bolus_units:
                        violations.append(
                            SafetyContractViolation(
                                glucose_mgdl=float(glucose),
                                trend_mgdl_min=float(trend),
                                proposed_insulin_units=float(proposed),
                                approved_insulin_units=approved,
                                reason="Bolus cap violation: approved insulin exceeds contract_max_bolus_units",
                            )
                        )

    return SafetyContractVerificationReport(
        spec=spec,
        total_cases=total_cases,
        expected_block_cases=expected_block_cases,
        blocked_cases=blocked_cases,
        violations=violations,
    )
