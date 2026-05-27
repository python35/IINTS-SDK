from __future__ import annotations

from typing import Any, Dict, List

from .registry import load_dataset_registry


OPEN_ACCESS_MARKERS = ("public", "cc by", "zenodo", "mendeley", "jaeb")
CONTROLLED_ACCESS_MARKERS = ("request", "controlled", "restricted")


def rank_real_data_sources() -> List[Dict[str, Any]]:
    """Rank registered datasets by usefulness for realism calibration and AI evidence."""

    ranked: list[dict[str, Any]] = []
    for dataset in load_dataset_registry():
        access = str(dataset.get("access", "")).lower()
        license_text = str(dataset.get("license", "")).lower()
        has_reference = bool(dataset.get("realism_reference_profile"))
        has_url = bool(dataset.get("landing_page") or dataset.get("download_url"))
        is_bundled = access == "bundled" or dataset.get("id") == "sample"
        is_openish = any(marker in access or marker in license_text for marker in OPEN_ACCESS_MARKERS)
        is_controlled = any(marker in access or marker in license_text for marker in CONTROLLED_ACCESS_MARKERS)

        if has_reference and not is_bundled:
            tier = "tier_1_calibration_reference"
            use_case = "Primary simulator calibration and strict realism envelope checks."
        elif is_openish and has_url and not is_bundled:
            tier = "tier_2_external_training_candidate"
            use_case = "External predictor/controller data preparation after import and MDMP certification."
        elif is_controlled:
            tier = "tier_3_controlled_validation_candidate"
            use_case = "Strong independent validation once access is approved."
        elif is_bundled:
            tier = "demo_only"
            use_case = "Quickstarts and offline demos, not final realism evidence."
        else:
            tier = "registry_reference"
            use_case = "Track as a potential evidence source; review access and schema first."

        ranked.append(
            {
                "id": dataset.get("id"),
                "name": dataset.get("name"),
                "tier": tier,
                "access": dataset.get("access"),
                "license": dataset.get("license"),
                "realism_reference_profile": dataset.get("realism_reference_profile"),
                "use_case": use_case,
                "has_download_or_landing_page": has_url,
                "citation": (dataset.get("citation") or {}).get("text"),
            }
        )

    order = {
        "tier_1_calibration_reference": 0,
        "tier_2_external_training_candidate": 1,
        "tier_3_controlled_validation_candidate": 2,
        "registry_reference": 3,
        "demo_only": 4,
    }
    return sorted(ranked, key=lambda row: (order.get(str(row["tier"]), 99), str(row["id"])))
