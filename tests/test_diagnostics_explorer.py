import numpy as np
import xarray as xr

from trident.analysis.explorer import (
    classify_variables,
    extract_depth_profile,
    extract_pixel_table,
    integrated_profile,
)


def test_variable_classification():
    dataset = xr.Dataset(
        {
            "map": (("lat", "lon"), np.ones((2, 2))),
            "profile": (("depth",), np.ones(3)),
            "time_depth": (("time", "depth"), np.ones((2, 3))),
        },
        coords={
            "lat": [1, 2],
            "lon": [3, 4],
            "depth": [0, 10, 20],
            "time": [0, 1],
        },
    )

    classes = classify_variables(dataset)

    assert "map" in classes["maps_2d"]
    assert "profile" in classes["profiles_1d"]
    assert "time_depth" in classes["time_depth"]


def test_nearest_pixel_table():
    dataset = xr.Dataset(
        {
            "cafe_npp": (
                ("lat", "lon"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            )
        },
        coords={
            "lat": [30.0, 31.0],
            "lon": [-125.0, -124.0],
        },
    )

    table = extract_pixel_table(
        dataset,
        latitude=30.9,
        longitude=-124.1,
    )

    assert table.iloc[0]["value"] == 4.0


def test_profile_integral():
    dataset = xr.Dataset(
        {
            "npp_z": (
                ("depth",),
                np.array([2.0, 2.0, 2.0]),
            )
        },
        coords={"depth": [0.0, 10.0, 20.0]},
    )

    profile = extract_depth_profile(dataset, "npp_z")

    assert integrated_profile(profile) == 40.0
