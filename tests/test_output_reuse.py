from pathlib import Path

import numpy as np
import xarray as xr

from trident.workflows.osu_replay import _existing_output_is_usable


BBOX = (-132.0, 28.0, -116.0, 45.0)


def _coords():
    return {
        "lat": np.linspace(28.083333, 44.916667, 102),
        "lon": np.linspace(-131.916667, -116.083333, 96),
    }


def test_legacy_replay_file_is_reused(tmp_path):
    path = (
        tmp_path
        / "trident_osu_input_replay_cafe_npp_202401_stride1.nc"
    )
    coords = _coords()
    ds = xr.Dataset(
        {
            "cafe_npp": (
                ("lat", "lon"),
                np.ones((102, 96), dtype="float32"),
            ),
            "zeu": (
                ("lat", "lon"),
                np.ones((102, 96), dtype="float32"),
            ),
            "kdpar": (
                ("lat", "lon"),
                np.ones((102, 96), dtype="float32"),
            ),
        },
        coords=coords,
        attrs={
            "title": "legacy file without cache metadata",
        },
    )
    ds.to_netcdf(path)

    assert _existing_output_is_usable(
        path,
        tag="202401",
        stride=1,
        bbox=BBOX,
        role="trident_replay",
    )


def test_legacy_comparison_file_is_reused(tmp_path):
    path = (
        tmp_path
        / "trident_osu_replay_vs_archived_comparison_202401_stride1.nc"
    )
    coords = _coords()
    values = np.ones((102, 96), dtype="float32")
    ds = xr.Dataset(
        {
            "trident_replay_npp": (("lat", "lon"), values),
            "osu_archived_npp": (("lat", "lon"), values),
            "difference": (("lat", "lon"), values * 0),
        },
        coords=coords,
        attrs={
            "title": "legacy comparison without cache metadata",
        },
    )
    ds.to_netcdf(path)

    assert _existing_output_is_usable(
        path,
        tag="202401",
        stride=1,
        bbox=BBOX,
        role="validation_comparison",
    )


def test_wrong_stride_is_not_reused(tmp_path):
    path = (
        tmp_path
        / "trident_osu_input_replay_cafe_npp_202401_stride10.nc"
    )
    coords = _coords()
    values = np.ones((102, 96), dtype="float32")
    xr.Dataset(
        {
            "cafe_npp": (("lat", "lon"), values),
            "zeu": (("lat", "lon"), values),
            "kdpar": (("lat", "lon"), values),
        },
        coords=coords,
    ).to_netcdf(path)

    assert not _existing_output_is_usable(
        path,
        tag="202401",
        stride=1,
        bbox=BBOX,
        role="trident_replay",
    )


def test_wrong_region_is_not_reused(tmp_path):
    path = (
        tmp_path
        / "trident_osu_input_replay_cafe_npp_202401_stride1.nc"
    )
    values = np.ones((10, 10), dtype="float32")
    xr.Dataset(
        {
            "cafe_npp": (("lat", "lon"), values),
            "zeu": (("lat", "lon"), values),
            "kdpar": (("lat", "lon"), values),
        },
        coords={
            "lat": np.linspace(0, 9, 10),
            "lon": np.linspace(0, 9, 10),
        },
    ).to_netcdf(path)

    assert not _existing_output_is_usable(
        path,
        tag="202401",
        stride=1,
        bbox=BBOX,
        role="trident_replay",
    )
