from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from iints.research.proteomics_importer import (
    import_and_validate_proteomics,
    import_diann_report,
    import_maxquant_protein_groups,
    import_wide_proteomics_matrix,
    load_sample_metadata,
)
from iints.research.regenerative_islet import compare_regenerative_islet_proteomics


@pytest.fixture
def sample_metadata_csv(tmp_path: Path) -> Path:
    meta_path = tmp_path / "sample_metadata.csv"
    meta_path.write_text(
        "sample_id,group,batch_id,source_id\n"
        "Sample_SC_1,sc_islet,batch_A,PXD001539\n"
        "Sample_SC_2,sc_islet,batch_A,PXD001539\n"
        "Sample_SC_3,sc_islet,batch_B,PXD001539\n"
        "Sample_Primary_1,primary_islet,batch_A,PXD001539\n"
        "Sample_Primary_2,primary_islet,batch_A,PXD001539\n"
        "Sample_Primary_3,primary_islet,batch_B,PXD001539\n",
        encoding="utf-8",
    )
    return meta_path


def test_load_sample_metadata_csv_tsv_json(tmp_path: Path) -> None:
    csv_file = tmp_path / "meta.csv"
    csv_file.write_text("sample_id,group,batch\nS1,sc_islet,B1\nS2,primary_islet,B1\n", encoding="utf-8")
    meta_csv = load_sample_metadata(csv_file)
    assert meta_csv["S1"]["group"] == "sc_islet"
    assert meta_csv["S1"]["batch_id"] == "B1"

    tsv_file = tmp_path / "meta.tsv"
    tsv_file.write_text("sample\tcondition\treplicate\nS1\tsc_islet\tRep1\n", encoding="utf-8")
    meta_tsv = load_sample_metadata(tsv_file)
    assert meta_tsv["S1"]["group"] == "sc_islet"

    json_file = tmp_path / "meta.json"
    json_file.write_text(json.dumps({"S1": {"group": "sc_islet", "batch_id": "B1"}}), encoding="utf-8")
    meta_json = load_sample_metadata(json_file)
    assert meta_json["S1"]["group"] == "sc_islet"


def test_import_maxquant_protein_groups(tmp_path: Path, sample_metadata_csv: Path) -> None:
    maxquant_file = tmp_path / "proteinGroups.txt"
    maxquant_file.write_text(
        "Protein IDs\tMajority protein IDs\tGene names\tPotential contaminant\tReverse\tLFQ intensity Sample_SC_1\tLFQ intensity Sample_SC_2\tLFQ intensity Sample_SC_3\tLFQ intensity Sample_Primary_1\tLFQ intensity Sample_Primary_2\tLFQ intensity Sample_Primary_3\n"
        "P01308\tP01308\tINS;INS-IGF2\t\t\t1500000\t1600000\t1550000\t2000000\t2100000\t2050000\n"
        "P52945\tP52945\tPDX1\t\t\t800000\t820000\t810000\t900000\t920000\t910000\n"
        "Q9NZQ7\tQ9NZQ7\tCD274\t\t\t300000\t310000\t305000\t100000\t110000\t105000\n"
        "CONTAM\tCONTAM\tTRYP\t+\t\t500000\t500000\t500000\t500000\t500000\t500000\n"
        "REV_P01308\tREV_P01308\tINS\t\t+\t500000\t500000\t500000\t500000\t500000\t500000\n",
        encoding="utf-8",
    )

    df = import_maxquant_protein_groups(maxquant_file, sample_metadata_csv)
    assert set(df["gene_symbol"]) == {"INS", "PDX1", "CD274"}
    assert "TRYP" not in set(df["gene_symbol"])  # Filtered contaminant
    assert len(df) == 3 * 6  # 3 genes * 6 samples


def test_import_diann_report(tmp_path: Path, sample_metadata_csv: Path) -> None:
    diann_file = tmp_path / "report.tsv"
    diann_file.write_text(
        "Run\tProtein.Group\tGenes\tPG.MaxLFQ\tPG.Q.Value\n"
        "Sample_SC_1\tP01308\tINS\t1500000\t0.001\n"
        "Sample_SC_2\tP01308\tINS\t1600000\t0.001\n"
        "Sample_SC_3\tP01308\tINS\t1550000\t0.001\n"
        "Sample_Primary_1\tP01308\tINS\t2000000\t0.001\n"
        "Sample_Primary_2\tP01308\tINS\t2100000\t0.001\n"
        "Sample_Primary_3\tP01308\tINS\t2050000\t0.001\n"
        "Sample_SC_1\tP35557\tGCK\t400000\t0.001\n"
        "Sample_SC_2\tP35557\tGCK\t420000\t0.001\n"
        "Sample_SC_3\tP35557\tGCK\t410000\t0.001\n"
        "Sample_Primary_1\tP35557\tGCK\t500000\t0.001\n"
        "Sample_Primary_2\tP35557\tGCK\t520000\t0.001\n"
        "Sample_Primary_3\tP35557\tGCK\t510000\t0.001\n"
        "Sample_SC_1\tBAD\tBADGENE\t100000\t0.05\n",  # High Q-value, should be filtered
        encoding="utf-8",
    )

    df = import_diann_report(diann_file, sample_metadata_csv)
    assert set(df["gene_symbol"]) == {"INS", "GCK"}
    assert "BADGENE" not in set(df["gene_symbol"])
    assert len(df) == 2 * 6


def test_import_wide_proteomics_matrix(tmp_path: Path, sample_metadata_csv: Path) -> None:
    matrix_file = tmp_path / "matrix.tsv"
    matrix_file.write_text(
        "gene_symbol\tSample_SC_1\tSample_SC_2\tSample_SC_3\tSample_Primary_1\tSample_Primary_2\tSample_Primary_3\n"
        "INS\t1500\t1600\t1550\t2000\t2100\t2050\n"
        "PDX1\t800\t820\t810\t900\t920\t910\n"
        "NKX6-1\t700\t710\t705\t850\t860\t855\n",
        encoding="utf-8",
    )

    df = import_wide_proteomics_matrix(matrix_file, sample_metadata_csv)
    assert set(df["gene_symbol"]) == {"INS", "PDX1", "NKX6-1"}
    assert len(df) == 3 * 6


def test_end_to_end_import_and_compare_roundtrip(tmp_path: Path, sample_metadata_csv: Path) -> None:
    matrix_file = tmp_path / "wide_proteomics_raw.tsv"
    matrix_file.write_text(
        "gene_symbol\tSample_SC_1\tSample_SC_2\tSample_SC_3\tSample_Primary_1\tSample_Primary_2\tSample_Primary_3\n"
        "INS\t1500\t1600\t1550\t2000\t2100\t2050\n"
        "PDX1\t800\t820\t810\t900\t920\t910\n"
        "NKX6-1\t700\t710\t705\t850\t860\t855\n"
        "MAFA\t300\t310\t305\t800\t820\t810\n"
        "GCK\t400\t420\t410\t500\t520\t510\n"
        "ABCC8\t200\t210\t205\t300\t310\t305\n"
        "KCNJ11\t180\t190\t185\t250\t260\t255\n"
        "PCSK1\t120\t130\t125\t150\t160\t155\n"
        "PCSK2\t110\t115\t112\t140\t145\t142\n"
        "SLC30A8\t90\t95\t92\t110\t115\t112\n",
        encoding="utf-8",
    )

    standardized_csv = tmp_path / "standardized_data.csv"
    import_result = import_and_validate_proteomics(
        data_path=matrix_file,
        sample_metadata=sample_metadata_csv,
        output_path=standardized_csv,
        input_format="wide_matrix",
        default_source_id="PXD001539",
    )

    assert import_result.row_count == 10 * 6
    assert import_result.gene_count == 10
    assert import_result.sample_count == 6
    assert import_result.target_panel_coverage["beta_cell_identity_and_function"][0] == 10

    # Run regenerative islet comparison on the output
    comp_output_dir = tmp_path / "comparison_results"
    comp_result = compare_regenerative_islet_proteomics(
        data_path=standardized_csv,
        output_dir=comp_output_dir,
        test_group="sc_islet",
        reference_group="primary_islet",
        panel_keys=["beta_cell_identity_and_function"],
        normalization_note="Joint mock batch normalization",
    )

    assert comp_result.status == "ready_for_descriptive_review"
    assert comp_result.observed_target_count == 10
    assert comp_result.comparison_csv.is_file()
    assert comp_result.report_json.is_file()
    assert comp_result.report_md.is_file()
