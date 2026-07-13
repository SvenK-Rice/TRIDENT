from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from trident.analysis.analysis_profiles import (
    CAFE_INPUTS,
    default_map_variable,
    discover_analysis_datasets,
    open_analysis_dataset,
)


def _inputs(path: Path) -> None:
    shape = (2, 3)
    ds = xr.Dataset(
        {name: (("lat", "lon"), np.ones(shape, dtype="float32")) for name in CAFE_INPUTS},
        coords={"lat": [30.0, 31.0], "lon": [-124.0, -123.0, -122.0]},
    )
    ds.to_netcdf(path)


def test_discover_pairs_output_using_report(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    reports = tmp_path / "reports" / "trident"
    processed.mkdir(parents=True)
    reports.mkdir(parents=True)
    input_path = processed / "prepared_inputs.nc"
    output_path = processed / "cafe_output.nc"
    _inputs(input_path)
    xr.Dataset({"cafe_npp": (("lat", "lon"), np.ones((2, 3)))},
               coords={"lat": [30.0, 31.0], "lon": [-124.0, -123.0, -122.0]}).to_netcdf(output_path)
    (reports / "run_report.json").write_text(json.dumps({"input_file": str(input_path), "output_file": str(output_path)}))
    entries = discover_analysis_datasets(tmp_path)
    output = next(entry for entry in entries if entry.path == output_path.resolve())
    assert output.companion_input_path == input_path.resolve()
    merged = open_analysis_dataset(output)
    assert "cafe_npp" in merged
    assert all(name in merged for name in CAFE_INPUTS)


def test_embedded_inputs_need_no_companion(tmp_path: Path) -> None:
    path = tmp_path / "embedded.nc"
    _inputs(path)
    entries = discover_analysis_datasets(tmp_path)
    entry = next(item for item in entries if item.path == path.resolve())
    assert entry.has_embedded_cafe_inputs
    assert entry.companion_input_path is None


def test_discovery_does_not_recurse_through_unmanaged_data_root(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    unrelated = tmp_path / "large_unrelated_archive" / "nested"
    processed.mkdir(parents=True)
    unrelated.mkdir(parents=True)
    managed = processed / "native_npp.nc"
    ignored = unrelated / "unrelated.nc"
    dataset = xr.Dataset(
        {"cafe_npp": (("lat", "lon"), np.ones((2, 3)))},
        coords={"lat": [30.0, 31.0], "lon": [-124.0, -123.0, -122.0]},
    )
    dataset.to_netcdf(managed)
    dataset.to_netcdf(ignored)

    paths = {entry.path for entry in discover_analysis_datasets(tmp_path)}

    assert managed.resolve() in paths
    assert ignored.resolve() not in paths


def test_analysis_catalog_classifies_family_cadence_period_and_stride(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    monthly = processed / "trident_native_cafe_npp_202301_stride10.nc"
    eight_day = processed / "trident_native_cafe_npp_2023001_8day_stride5.nc"
    dataset = xr.Dataset(
        {"cafe_npp": (("lat", "lon"), np.ones((2, 3)))},
        coords={"lat": [30.0, 31.0], "lon": [-124.0, -123.0, -122.0]},
    )
    dataset.to_netcdf(monthly)
    dataset.to_netcdf(eight_day)

    entries = {entry.path.name: entry for entry in discover_analysis_datasets(tmp_path)}

    monthly_entry = entries[monthly.name]
    assert monthly_entry.product_family == "TRIDENT Native"
    assert monthly_entry.cadence == "Monthly"
    assert monthly_entry.period == "2023-01"
    assert monthly_entry.stride == 10
    assert monthly_entry.role == "Analysis result"

    eight_day_entry = entries[eight_day.name]
    assert eight_day_entry.cadence == "8-day"
    assert eight_day_entry.period == "2023001"
    assert eight_day_entry.stride == 5


def test_default_map_variable_prefers_npp() -> None:
    assert default_map_variable(["chl", "cafe_npp", "sst"]) == "cafe_npp"


def test_discover_pairs_output_using_list_report(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    reports = tmp_path / "reports" / "trident"
    processed.mkdir(parents=True)
    reports.mkdir(parents=True)
    input_path = processed / "native_inputs.nc"
    output_path = processed / "native_npp.nc"
    _inputs(input_path)
    xr.Dataset(
        {"cafe_npp": (("lat", "lon"), np.ones((2, 3)))},
        coords={"lat": [30.0, 31.0], "lon": [-124.0, -123.0, -122.0]},
    ).to_netcdf(output_path)
    (reports / "native_report.json").write_text(json.dumps([
        {"tag": "202601", "input_file": str(input_path), "output_file": str(output_path)}
    ]))
    entries = discover_analysis_datasets(tmp_path)
    output = next(entry for entry in entries if entry.path == output_path.resolve())
    assert output.companion_input_path == input_path.resolve()
