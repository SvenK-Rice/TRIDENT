from pathlib import Path

import numpy as np
import xarray as xr

from trident.data_manager.catalog import (
    discover_managed_files,
    stale_candidates,
    summary_table,
)


def test_catalog_discovers_downloads_and_analyses(tmp_path):
    reference = tmp_path / "data" / "reference" / "nasa"
    processed = tmp_path / "data" / "processed"
    reference.mkdir(parents=True)
    processed.mkdir(parents=True)

    (reference / "PACE_CHL_202401.dat").write_bytes(b"123")

    xr.Dataset(
        {
            "cafe_npp": (
                ("lat", "lon"),
                np.ones((2, 2), dtype="float32"),
            )
        },
        coords={"lat": [1, 2], "lon": [3, 4]},
    ).to_netcdf(
        processed / "trident_osu_replay_202401.nc"
    )

    catalog = discover_managed_files(tmp_path)

    assert set(catalog["category"]) == {"download", "analysis"}
    assert "NASA Earthdata" in set(catalog["source"])
    assert "OSU Archive" in set(catalog["source"])


def test_tracked_output_is_not_stale(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    xr.Dataset(
        {
            "cafe_npp": (
                ("lat", "lon"),
                np.ones((2, 2), dtype="float32"),
            )
        },
        coords={"lat": [1, 2], "lon": [3, 4]},
        attrs={
            "trident_code_version": "4.1.0",
            "trident_git_commit": "abc123",
            "trident_git_dirty": "false",
        },
    ).to_netcdf(processed / "native_cafe_202401.nc")

    catalog = discover_managed_files(tmp_path)

    assert catalog.iloc[0]["provenance_status"] == "tracked"
    assert stale_candidates(catalog).empty


def test_summary_groups_files(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    for month in ("202401", "202402"):
        xr.Dataset(
            {"cafe_npp": (("x",), np.ones(2))}
        ).to_netcdf(processed / f"native_cafe_{month}.nc")

    catalog = discover_managed_files(tmp_path)
    summary = summary_table(catalog)

    assert summary["files"].sum() == 2
