from __future__ import annotations

from iints.research import anatomy, genetics, pharmacology


def test_clinvar_fetch_parses_public_summary(monkeypatch):
    def fake_download(url: str):
        if "esearch.fcgi" in url:
            return {"esearchresult": {"idlist": ["123"]}}
        return {
            "result": {
                "123": {
                    "title": "NM_000208.4(INSR):c.1A>G",
                    "clinical_significance": {"description": "Pathogenic"},
                    "trait_set": [{"trait_name": "Donohue syndrome"}],
                }
            }
        }

    monkeypatch.setattr(genetics, "_download_json", fake_download)

    variants = genetics.fetch_clinvar_pathogenic("INSR")

    assert variants[0].uid == "123"
    assert variants[0].clinical_significance == "Pathogenic"
    assert variants[0].trait == "Donohue syndrome"


def test_simulate_mutation_uses_deterministic_curated_mapping(monkeypatch):
    monkeypatch.setattr(
        genetics,
        "fetch_clinvar_pathogenic",
        lambda gene: [
            genetics.ClinVarVariant(
                uid="123",
                title="INSR variant",
                clinical_significance="Pathogenic",
                trait="severe insulin resistance",
            )
        ],
    )

    variants = genetics.simulate_mutation("INSR")

    assert variants
    assert "Insulin receptor disruption" in genetics.GENE_EFFECTS["INSR"][0]


def test_insulin_pk_profiles_are_fixed_not_ai_generated():
    assert pharmacology.sdk_pk_profile("lispro").tmax_minutes == 55
    assert pharmacology.sdk_pk_profile("glargine").tmax_minutes == 1440
    assert pharmacology.sdk_pk_profile("unknown analogue").key == "rapid-default"


def test_analyze_insulin_continues_when_chembl_is_unavailable(monkeypatch):
    monkeypatch.setattr(pharmacology, "fetch_chembl_drug", lambda drug_name: None)

    molecule, profile = pharmacology.analyze_insulin("degludec")

    assert molecule is None
    assert profile.key == "degludec"
    assert profile.tmax_minutes == 2500


def test_gtex_gene_alias_and_expression_parser(monkeypatch):
    monkeypatch.setattr(anatomy, "resolve_gtex_gencode_id", lambda gene: "ENSG00000181856.11")
    monkeypatch.setattr(
        anatomy,
        "_download_json",
        lambda url: {
            "data": [
                {"tissueSiteDetailId": "Muscle_Skeletal", "median": 24.0},
                {"tissueSiteDetailId": "Whole_Blood", "median": 0.1},
            ]
        },
    )

    tissues = anatomy.fetch_gtex_expression("GLUT4")

    assert anatomy.official_gene_symbol("GLUT4") == "SLC2A4"
    assert tissues[0].tissue == "Muscle_Skeletal"
    assert tissues[0].median_tpm == 24.0
