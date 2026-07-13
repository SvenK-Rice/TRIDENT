"""Utilities for extracting CAFE inputs from gridded satellite datasets."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import xarray as xr

LAT_NAMES = ("lat", "latitude", "nav_lat", "y")
LON_NAMES = ("lon", "longitude", "nav_lon", "x")
TIME_NAMES = ("time", "date", "datetime")

CAFE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "par": ("par", "PAR", "surface_par", "daily_par"),
    "chl": ("chl", "chlor_a", "chlorophyll", "CHL"),
    "mld": ("mld", "mlotst", "mixed_layer_depth", "MLD"),
    "aph443": ("aph443", "aph_443", "aph_443_giop", "aph_443_qaa"),
    "adg443": ("adg443", "adg_443", "adg_443_giop", "adg_443_qaa"),
    "bbp443": ("bbp443", "bbp_443", "bbp_443_giop", "bbp_443_qaa"),
    "bbp_s": ("bbp_s", "bbp_slope", "bbp_spectral_slope", "eta"),
    "sst": ("sst", "SST", "sea_surface_temperature", "analysed_sst", "thetao"),
}


@dataclass(frozen=True)
class PixelSelection:
    requested_lat: float
    requested_lon: float
    selected_lat: float
    selected_lon: float
    values: dict[str, float]
    source_variables: dict[str, str]
    day_of_year: int
    time_value: str | None


def find_coordinate_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    """Return the first matching coordinate/dimension name."""
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    lowered = {name.lower(): name for name in list(ds.coords) + list(ds.dims)}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise KeyError(f"Could not find any of these coordinates: {tuple(candidates)}")


def infer_variable_mapping(ds: xr.Dataset) -> dict[str, str | None]:
    """Infer common CAFE variable names in a dataset."""
    lowered = {name.lower(): name for name in ds.data_vars}
    result: dict[str, str | None] = {}
    for field, aliases in CAFE_ALIASES.items():
        match = None
        for alias in aliases:
            if alias in ds.data_vars:
                match = alias
                break
            if alias.lower() in lowered:
                match = lowered[alias.lower()]
                break
        result[field] = match
    return result


def numeric_data_variables(ds: xr.Dataset) -> list[str]:
    """List numeric variables suitable for mapping or CAFE inputs."""
    return sorted(
        name for name, da in ds.data_vars.items() if np.issubdtype(da.dtype, np.number)
    )


def _select_time(da: xr.DataArray, time_name: str | None, time_index: int) -> xr.DataArray:
    if time_name and time_name in da.dims:
        if da.sizes[time_name] == 0:
            raise ValueError(f"Variable {da.name!r} has an empty time dimension")
        da = da.isel({time_name: min(max(time_index, 0), da.sizes[time_name] - 1)})
    return da


def _reduce_extra_dimensions(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
) -> xr.DataArray:
    """Reduce non-horizontal dimensions deterministically to the first/surface value."""
    for dim in list(da.dims):
        if dim not in {lat_name, lon_name}:
            da = da.isel({dim: 0})
    return da.squeeze(drop=True)


def horizontal_field(
    ds: xr.Dataset,
    variable: str,
    *,
    lat_name: str,
    lon_name: str,
    time_name: str | None = None,
    time_index: int = 0,
) -> xr.DataArray:
    """Return one two-dimensional horizontal field for display and selection."""
    da = _select_time(ds[variable], time_name, time_index)
    da = _reduce_extra_dimensions(da, lat_name, lon_name)
    if lat_name not in da.dims or lon_name not in da.dims:
        raise ValueError(
            f"Variable {variable!r} does not contain both {lat_name!r} and {lon_name!r} dimensions"
        )
    return da.transpose(lat_name, lon_name)


def _day_of_year(ds: xr.Dataset, time_name: str | None, time_index: int) -> tuple[int, str | None]:
    """Return the one-based calendar day of year for the selected time."""
    if not time_name or time_name not in ds.coords or ds[time_name].size == 0:
        return 180, None
    value = ds[time_name].values[min(max(time_index, 0), ds[time_name].size - 1)]
    try:
        day = np.datetime64(value, "D")
        year_start = day.astype("datetime64[Y]").astype("datetime64[D]")
        day_of_year = int((day - year_start) / np.timedelta64(1, "D")) + 1
        return day_of_year, str(day)
    except (TypeError, ValueError, OverflowError):
        return 180, str(value)


def extract_nearest_valid_pixel(
    ds: xr.Dataset,
    requested_lat: float,
    requested_lon: float,
    variable_mapping: Mapping[str, str],
    *,
    lat_name: str | None = None,
    lon_name: str | None = None,
    time_name: str | None = None,
    time_index: int = 0,
    max_search_radius_cells: int = 12,
) -> PixelSelection:
    """Extract the nearest pixel where every mapped CAFE input is finite.

    The initial nearest coordinate is expanded in square index rings until a
    fully valid pixel is found. This prevents coastal/cloud NaNs from causing
    the profile calculation to fail while keeping the selected location close
    to the user's click.
    """
    lat_name = lat_name or find_coordinate_name(ds, LAT_NAMES)
    lon_name = lon_name or find_coordinate_name(ds, LON_NAMES)
    if time_name is None:
        try:
            time_name = find_coordinate_name(ds, TIME_NAMES)
        except KeyError:
            time_name = None

    missing = [field for field in CAFE_ALIASES if field not in variable_mapping]
    if missing:
        raise KeyError(f"Missing variable mappings for: {', '.join(missing)}")

    fields = {
        field: horizontal_field(
            ds,
            variable,
            lat_name=lat_name,
            lon_name=lon_name,
            time_name=time_name,
            time_index=time_index,
        )
        for field, variable in variable_mapping.items()
    }

    lat_values = np.asarray(ds[lat_name].values, dtype=float)
    lon_values = np.asarray(ds[lon_name].values, dtype=float)
    if lat_values.ndim != 1 or lon_values.ndim != 1:
        raise ValueError("This first implementation supports one-dimensional latitude/longitude grids")

    i0 = int(np.nanargmin(np.abs(lat_values - requested_lat)))
    j0 = int(np.nanargmin(np.abs(lon_values - requested_lon)))

    best: tuple[float, int, int, dict[str, float]] | None = None
    for radius in range(max_search_radius_cells + 1):
        i_min, i_max = max(0, i0 - radius), min(len(lat_values) - 1, i0 + radius)
        j_min, j_max = max(0, j0 - radius), min(len(lon_values) - 1, j0 + radius)
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                if radius and i not in {i_min, i_max} and j not in {j_min, j_max}:
                    continue
                values = {field: float(da.values[i, j]) for field, da in fields.items()}
                if not all(np.isfinite(value) for value in values.values()):
                    continue
                distance2 = (lat_values[i] - requested_lat) ** 2 + (lon_values[j] - requested_lon) ** 2
                if best is None or distance2 < best[0]:
                    best = (distance2, i, j, values)
        if best is not None:
            break

    if best is None:
        raise ValueError(
            "No pixel with all required CAFE inputs was found near the selected location. "
            "Try another location, time step, or a larger search radius."
        )

    _, i, j, values = best
    yd, time_value = _day_of_year(ds, time_name, time_index)
    return PixelSelection(
        requested_lat=float(requested_lat),
        requested_lon=float(requested_lon),
        selected_lat=float(lat_values[i]),
        selected_lon=float(lon_values[j]),
        values=values,
        source_variables=dict(variable_mapping),
        day_of_year=yd,
        time_value=time_value,
    )
