from __future__ import annotations

import numpy as np
import xarray as xr

from trident.analysis.satellite_pixel import (
    extract_nearest_valid_pixel,
    find_coordinate_name,
    horizontal_field,
    infer_variable_mapping,
)


def _dataset() -> xr.Dataset:
    lat = np.array([30.0, 31.0, 32.0])
    lon = np.array([-125.0, -124.0, -123.0, -122.0])
    shape = (1, 3, 4)
    base = np.ones(shape)
    ds = xr.Dataset(
        {
            "PAR": (("time", "lat", "lon"), base * 40.0),
            "chlor_a": (("time", "lat", "lon"), base * 0.2),
            "mlotst": (("time", "lat", "lon"), base * 30.0),
            "aph_443": (("time", "lat", "lon"), base * 0.01),
            "adg_443": (("time", "lat", "lon"), base * 0.005),
            "bbp_443": (("time", "lat", "lon"), base * 0.0015),
            "eta": (("time", "lat", "lon"), base * 1.0),
            "sst": (("time", "lat", "lon"), base * 20.0),
            "cafe_npp": (("time", "lat", "lon"), base * 600.0),
        },
        coords={"time": [np.datetime64("2026-07-01")], "lat": lat, "lon": lon},
    )
    return ds


def test_coordinate_and_variable_inference() -> None:
    ds = _dataset()
    assert find_coordinate_name(ds, ("latitude", "lat")) == "lat"
    mapping = infer_variable_mapping(ds)
    assert mapping == {
        "par": "PAR", "chl": "chlor_a", "mld": "mlotst",
        "aph443": "aph_443", "adg443": "adg_443",
        "bbp443": "bbp_443", "bbp_s": "eta", "sst": "sst",
    }


def test_horizontal_field_selects_time() -> None:
    field = horizontal_field(_dataset(), "cafe_npp", lat_name="lat", lon_name="lon", time_name="time", time_index=0)
    assert field.dims == ("lat", "lon")
    assert field.shape == (3, 4)


def test_extract_nearest_valid_pixel() -> None:
    ds = _dataset()
    mapping = {key: value for key, value in infer_variable_mapping(ds).items() if value}
    result = extract_nearest_valid_pixel(ds, 31.1, -123.2, mapping, time_name="time")
    assert result.selected_lat == 31.0
    assert result.selected_lon == -123.0
    assert result.values["par"] == 40.0
    assert result.day_of_year == 182


def test_extract_searches_around_nan_pixel() -> None:
    ds = _dataset()
    ds["aph_443"].values[0, 1, 2] = np.nan
    mapping = {key: value for key, value in infer_variable_mapping(ds).items() if value}
    result = extract_nearest_valid_pixel(ds, 31.0, -123.0, mapping, time_name="time", max_search_radius_cells=2)
    assert (result.selected_lat, result.selected_lon) != (31.0, -123.0)
    assert np.isfinite(result.values["aph443"])
