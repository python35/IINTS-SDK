from __future__ import annotations

import json
from pathlib import Path

import pytest

from iints.research.academic_bundle import build_academic_bundle


def _write_complete_run(run_dir: Path, *, csv_header: str = "time_minutes,glucose_actual_mgdl") -> None:
    run_dir.mkdir()
    (run_dir / "results.csv").write_text(f"{csv_header}\n0,110\n5,120\n", encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "git_sha": "0123456789abcdef",
                "sdk_version": "1.5.30",
                "python_version": "3.11.15",
                "dependencies": [{"name": "numpy", "version": "2.0.0"}],
                "config": {"patient_model_type": "hovorka", "duration_minutes": 60},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"files": {"results_csv": {"path": "results.csv"}}}),
        encoding="utf-8",
    )


def test_academic_bundle_writes_reviewable_ro_crate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-42"
    _write_complete_run(run_dir)

    result = build_academic_bundle(
        run_dir,
        title="Held-out Hovorka run",
        creator_name="Researcher Example",
        creator_orcid="https://orcid.org/0000-0002-1825-0097",
        license_id="CC-BY-4.0",
        source_ids=["hovorka_2004_nmpc_t1d", "attd_2019_time_in_range"],
    )

    assert result.readiness_status == "ready"
    assert result.source_count == 2
    assert result.ro_crate_metadata.is_file()
    assert result.audit_json.is_file()
    assert result.sources_json.is_file()
    assert result.readme_md.is_file()

    crate = json.loads(result.ro_crate_metadata.read_text(encoding="utf-8"))
    assert crate["@context"] == "https://w3id.org/ro/crate/1.2/context"
    graph = crate["@graph"]
    root = next(entity for entity in graph if entity.get("@id") == "./")
    assert root["creator"] == {"@id": "#creator"}
    assert {item["@id"] for item in root["hasPart"]} >= {
        "results.csv",
        "run_metadata.json",
        "run_manifest.json",
        "academic_audit.json",
        "academic_sources.json",
        "ACADEMIC_BUNDLE.md",
    }
    assert "ro-crate-metadata.json" not in {item["@id"] for item in root["hasPart"]}
    results_entity = next(entity for entity in graph if entity.get("@id") == "results.csv")
    assert len(results_entity["sha256"]) == 64
    assert str(tmp_path) not in result.ro_crate_metadata.read_text(encoding="utf-8")


def test_academic_bundle_auto_associates_conservative_core_sources(tmp_path: Path) -> None:
    run_dir = tmp_path / "auto-run"
    _write_complete_run(run_dir)

    result = build_academic_bundle(run_dir, creator_name="Researcher Example")
    sources = json.loads(result.sources_json.read_text(encoding="utf-8"))
    source_ids = {entry["id"] for entry in sources["sources"]}

    assert sources["selection_method"] == "conservative_auto_association"
    assert "hovorka_2004_nmpc_t1d" in source_ids
    assert "attd_2019_time_in_range" in source_ids


def test_academic_bundle_associates_mechanistic_engine_sources(tmp_path: Path) -> None:
    run_dir = tmp_path / "mechanistic-run"
    _write_complete_run(run_dir)
    (run_dir / "mechanistic_run_manifest.json").write_text("{}\n", encoding="utf-8")

    result = build_academic_bundle(run_dir, creator_name="Researcher Example")
    sources = json.loads(result.sources_json.read_text(encoding="utf-8"))
    source_ids = {entry["id"] for entry in sources["sources"]}

    assert {"sbml_2019_l3v2_core", "libroadrunner_2015"}.issubset(source_ids)


def test_academic_bundle_flags_direct_identifier_headers(tmp_path: Path) -> None:
    run_dir = tmp_path / "privacy-review"
    _write_complete_run(run_dir, csv_header="time_minutes,patient_name")

    result = build_academic_bundle(run_dir, creator_name="Researcher Example")
    audit = json.loads(result.audit_json.read_text(encoding="utf-8"))

    assert result.readiness_status == "needs_review"
    assert any("patient_name" in finding for finding in audit["privacy_findings"])


def test_academic_bundle_rejects_unknown_source_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "unknown-source"
    _write_complete_run(run_dir)

    with pytest.raises(ValueError, match="Unknown evidence source ID"):
        build_academic_bundle(run_dir, source_ids=["not-a-real-source"])


def test_academic_bundle_rejects_orcid_with_invalid_checksum(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad-orcid"
    _write_complete_run(run_dir)

    with pytest.raises(ValueError, match="valid canonical"):
        build_academic_bundle(run_dir, creator_orcid="https://orcid.org/0000-0002-1825-0098")


def test_academic_bundle_does_not_assume_a_data_license(tmp_path: Path) -> None:
    run_dir = tmp_path / "unknown-license"
    _write_complete_run(run_dir)

    result = build_academic_bundle(run_dir, creator_name="Researcher Example")
    audit = json.loads(result.audit_json.read_text(encoding="utf-8"))

    assert result.readiness_status == "needs_review"
    license_check = next(check for check in audit["checks"] if check["id"] == "data_license_identified")
    assert license_check["passed"] is False


def test_repository_exposes_machine_readable_software_citation() -> None:
    citation = Path("CITATION.cff").read_text(encoding="utf-8")

    assert "cff-version: 1.2.0" in citation
    assert 'title: "IINTS-AF SDK"' in citation
    assert 'version: "1.5.32"' in citation
    assert "https://github.com/python35/IINTS-SDK" in citation
