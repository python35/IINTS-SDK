from __future__ import annotations

from iints.data.importer import import_cgm_dataframe, load_demo_dataframe


def test_bundled_demo_data_looks_like_a_full_day_trace() -> None:
    demo_df = load_demo_dataframe()

    assert len(demo_df) == 288
    assert demo_df["timestamp"].diff().dropna().eq(5).all()
    assert demo_df["carbs"].gt(0).sum() >= 3
    assert demo_df["insulin"].gt(0).sum() >= 3

    standard_df = import_cgm_dataframe(demo_df, data_format="generic", source="demo")
    glucose = standard_df["glucose"]

    assert 80.0 <= float(glucose.min()) <= 110.0
    assert 150.0 <= float(glucose.max()) <= 220.0
    assert 10.0 <= float(glucose.std()) <= 40.0
