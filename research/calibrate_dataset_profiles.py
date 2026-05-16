from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from calibrate_simulator_realism import build_report


REFERENCE_TO_PROFILE = {
    "free_living_t1d": "reference_free_living_t1d",
    "azt1d_daily": "reference_azt1d_t1d",
    "hupa_ucm_daily": "reference_hupa_ucm_t1d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate packaged reference patient profiles for each real-data envelope."
    )
    parser.add_argument("--preset", default="realistic_reference_day")
    parser.add_argument("--seeds", default="1,2,3,42,99")
    parser.add_argument(
        "--references",
        default="free_living_t1d,azt1d_daily,hupa_ucm_daily",
        help="Comma-separated realism reference ids.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/dataset_profile_calibration"),
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=Path("src/iints/data/virtual_patients"),
    )
    return parser.parse_args()


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_text(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.profiles_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, object]] = {}
    for reference in _parse_csv_text(args.references):
        if reference not in REFERENCE_TO_PROFILE:
            raise KeyError(f"No packaged profile mapping configured for reference '{reference}'.")
        report = build_report(
            preset_name=args.preset,
            reference=reference,
            seeds=_parse_csv_ints(args.seeds),
            initial_glucose_values=[125.0, 130.0, 135.0, 140.0, 145.0],
            dawn_strength_values=[0.0, 4.0, 8.0],
            meal_mismatch_values=[0.95, 1.0],
            glucose_decay_values=[0.02, 0.03],
            top_k=10,
        )
        report_path = args.out_dir / f"{reference}.json"
        report_path.write_text(json.dumps(report, indent=2))
        profile_name = REFERENCE_TO_PROFILE[reference]
        profile_path = args.profiles_dir / f"{profile_name}.yaml"
        profile_path.write_text(
            yaml.safe_dump(report["best_candidate"]["patient_profile"], sort_keys=False)
        )
        summaries[reference] = {
            "profile": profile_name,
            "report": str(report_path),
            "best_candidate": report["best_candidate"],
        }
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
