from __future__ import annotations

import json
from pathlib import Path

import iints
from iints.core.algorithms.pid_controller import PIDController
from iints.data.importer import (
    export_demo_csv,
    export_standard_csv,
    import_cgm_dataframe,
    load_demo_dataframe,
    scenario_from_dataframe,
)


def main() -> None:
    output_root = Path("results/demo_flow")
    output_root.mkdir(parents=True, exist_ok=True)

    raw_csv_path = output_root / "demo_cgm.csv"
    export_demo_csv(raw_csv_path)

    demo_df = load_demo_dataframe()
    standard_df = import_cgm_dataframe(demo_df, data_format="generic", source="demo")
    standard_csv_path = output_root / "cgm_standard.csv"
    export_standard_csv(standard_df, standard_csv_path)

    scenario = scenario_from_dataframe(standard_df, scenario_name="Demo CGM Scenario")
    scenario_path = output_root / "scenario.json"
    scenario_path.write_text(json.dumps(scenario, indent=2))

    run_dir = output_root / "run_full"
    outputs = iints.run_full(
        algorithm=PIDController(),
        scenario=scenario,
        patient_config="default_patient",
        duration_minutes=1440,
        time_step=5,
        seed=42,
        output_dir=run_dir,
    )

    print("Demo flow complete.")
    print("Outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
