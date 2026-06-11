from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from iints.core.patient.advanced_metabolic_model import AdvancedMetabolicModel


def run_jetson_endurance(
    *,
    days: float = 14.0,
    step_minutes: float = 5.0,
    output: Path = Path("results/red_team/endurance_data.csv"),
    seed: int = 42,
    inject_demo_glitch: bool = False,
) -> Path:
    """Generate an AdvancedMetabolicModel endurance CSV for red-team auditing.

    This is an educational stress generator, not a clinical trial simulator. The
    optional demo glitch writes one deliberately corrupted row so the auditor can
    prove that it catches impossible biology.
    """

    total_minutes = int(days * 24 * 60)
    steps = int(total_minutes / step_minutes)
    rng = np.random.default_rng(seed=seed)
    model = AdvancedMetabolicModel(initial_glucose=120.0, basal_insulin_rate=0.8, initial_beta_mass=0.0)
    records: list[dict[str, float | str]] = []
    pump_failed_until = 0.0
    start = time.time()

    print(f"IINTS-AF AdvancedMetabolicModel endurance stress test")
    print(f"Simulating {days:g} days, {steps} steps, {step_minutes:g} min/step")

    for step in range(steps):
        current_time = step * step_minutes
        delivered_insulin = 0.8
        carb_intake = 0.0
        event = ""

        if current_time > pump_failed_until and rng.random() < 0.005:
            pump_failed_until = current_time + 180.0
            event = "pump_failure_start"

        if current_time < pump_failed_until:
            delivered_insulin = 0.0
            event = event or "pump_failure_active"

        hour_of_day = (current_time / 60.0) % 24.0
        meal_window = (7 < hour_of_day < 9) or (12 < hour_of_day < 14) or (18 < hour_of_day < 20)
        if meal_window and rng.random() < 0.02:
            carb_intake = float(rng.uniform(30.0, 100.0))
            event = (event + ";" if event else "") + "meal"
            if rng.random() > 0.1 and delivered_insulin > 0.0:
                bolus = (carb_intake / 10.0) * float(rng.uniform(0.8, 1.5))
                delivered_insulin += bolus
                event += ";meal_bolus"
            else:
                event += ";missed_bolus"

        if delivered_insulin > 0.0 and rng.random() < 0.001:
            delivered_insulin += float(rng.uniform(5.0, 15.0))
            event = (event + ";" if event else "") + "accidental_overdose"

        glucose = model.update(
            step_minutes,
            delivered_insulin=delivered_insulin,
            carb_intake=carb_intake,
            current_time=current_time,
        )

        record = {
            "time_min": float(current_time),
            "day": float(current_time / 1440.0),
            "glucose": float(glucose),
            "insulin_delivered": float(delivered_insulin),
            "carbs": float(carb_intake),
            "active_insulin": float(model._state[2]),
            "ffa": float(model._state[13]),
            "ketones": float(model._state[14]),
            "event": event,
        }

        if inject_demo_glitch and step == steps // 2:
            record["glucose"] = -50.0
            record["ketones"] = 20.0
            record["event"] = (str(record["event"]) + ";" if record["event"] else "") + "injected_corrupt_row_for_auditor_demo"

        records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output, index=False)
    elapsed = time.time() - start
    print(f"Done. Wrote {len(records)} rows to {output} in {elapsed:.2f}s")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an AdvancedMetabolicModel endurance CSV for the AI Realism Auditor.")
    parser.add_argument("--days", type=float, default=14.0, help="Simulated study duration in days.")
    parser.add_argument("--step-minutes", type=float, default=5.0, help="Simulation step size in minutes.")
    parser.add_argument("--output", type=Path, default=Path("results/red_team/endurance_data.csv"), help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--inject-demo-glitch", action="store_true", help="Write one impossible row so the auditor demo always flags a bug.")
    args = parser.parse_args()
    run_jetson_endurance(
        days=args.days,
        step_minutes=args.step_minutes,
        output=args.output,
        seed=args.seed,
        inject_demo_glitch=args.inject_demo_glitch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
