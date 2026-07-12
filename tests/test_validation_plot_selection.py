from pathlib import Path

import numpy as np
import xarray as xr

from trident.workflows.series import (
    discover_validation_runs,
    validation_series,
)


def _write_comparison(path: Path, replay_value: float, osu_value: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "trident_replay_npp": (
                ("lat", "lon"),
                np.full((2, 2), replay_value, dtype="float32"),
            ),
            "osu_archived_npp": (
                ("lat", "lon"),
                np.full((2, 2), osu_value, dtype="float32"),
            ),
            "difference": (
                ("lat", "lon"),
                np.full(
                    (2, 2),
                    replay_value - osu_value,
                    dtype="float32",
                ),
            ),
        },
        coords={"lat": [30.0, 31.0], "lon": [-125.0, -124.0]},
    )
    ds.to_netcdf(path)


def test_selected_date_range_excludes_old_runs(tmp_path):
    processed = tmp_path / "data" / "processed"
    _write_comparison(
        processed
        / "trident_osu_replay_vs_archived_comparison_202301_stride10.nc",
        100.0,
        90.0,
    )
    _write_comparison(
        processed
        / "trident_osu_replay_vs_archived_comparison_202401_stride10.nc",
        300.0,
        290.0,
    )

    df = validation_series(
        tmp_path,
        (-132, 28, -116, 45),
        stride=10,
        start="2024-01-01",
        end="2024-01-31",
    )

    assert list(df["tag"]) == ["202401"]
    assert df.iloc[0]["trident_replay_mean"] == 300.0
    assert df.iloc[0]["osu_archived_mean"] == 290.0


def test_explicit_file_selection_reads_only_requested_run(tmp_path):
    processed = tmp_path / "data" / "processed"
    old_path = (
        processed
        / "trident_osu_replay_vs_archived_comparison_202301_stride10.nc"
    )
    new_path = (
        processed
        / "trident_osu_replay_vs_archived_comparison_202401_stride10.nc"
    )
    _write_comparison(old_path, 100.0, 90.0)
    _write_comparison(new_path, 300.0, 290.0)

    df = validation_series(
        tmp_path,
        (-132, 28, -116, 45),
        stride=10,
        selected_files=[new_path],
    )

    assert list(df["tag"]) == ["202401"]


def test_catalog_uses_only_validation_comparison_files(tmp_path):
    processed = tmp_path / "data" / "processed"
    _write_comparison(
        processed
        / "trident_osu_replay_vs_archived_comparison_202401_stride10.nc",
        300.0,
        290.0,
    )
    _write_comparison(
        processed
        / "trident_osu_input_replay_cafe_npp_202301_stride10.nc",
        999.0,
        999.0,
    )

    catalog = discover_validation_runs(tmp_path, stride=10)

    assert list(catalog["tag"]) == ["202401"]
