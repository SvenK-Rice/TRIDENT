from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from trident.analysis.analysis_profiles import CAFE_INPUTS, discover_analysis_datasets, open_analysis_dataset


def _grid(values: dict[str, float]) -> xr.Dataset:
    shape = (2, 2)
    return xr.Dataset(
        {name: (("lat", "lon"), np.full(shape, value, dtype="float32")) for name, value in values.items()},
        coords={"lat": [30.0, 31.0], "lon": [-124.0, -123.0]},
    )


def _inputs(path: Path, value: float = 2.0) -> None:
    _grid({name: value for name in CAFE_INPUTS}).to_netcdf(path)


def _output(path: Path, **attrs) -> None:
    ds = _grid({"cafe_npp": 700.0})
    ds.attrs.update(attrs)
    ds.to_netcdf(path)


def test_attribute_resolves_canonical_parent(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    parent = processed / "trident_native_cafe_inputs_202601.nc"
    output = processed / "trident_native_cafe_npp_202601_stride10.nc"
    _inputs(parent, 3.0)
    _output(output, source_input_file=str(parent), tag="202601")
    entry = next(e for e in discover_analysis_datasets(tmp_path) if e.path == output.resolve())
    assert entry.companion_input_path == parent.resolve()
    merged = open_analysis_dataset(entry)
    assert merged.attrs["profile_input_source"] == "canonical_prepared_dataset"
    assert float(merged["par"][0, 0]) == 3.0


def test_report_resolves_canonical_parent(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    reports = tmp_path / "reports" / "trident"
    processed.mkdir(parents=True)
    reports.mkdir(parents=True)
    parent = processed / "prepared_202602.nc"
    output = processed / "cafe_npp_202602.nc"
    _inputs(parent)
    _output(output)
    (reports / "run.json").write_text(json.dumps([{
        "input_file": str(parent), "output_file": str(output)
    }]))
    entry = next(e for e in discover_analysis_datasets(tmp_path) if e.path == output.resolve())
    assert entry.companion_input_path == parent.resolve()


def test_filename_fallback_resolves_matching_month(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    parent = processed / "trident_native_cafe_inputs_202603.nc"
    output = processed / "trident_native_cafe_npp_202603_stride10.nc"
    _inputs(parent)
    _output(output, tag="202603")
    entry = next(e for e in discover_analysis_datasets(tmp_path) if e.path == output.resolve())
    assert entry.companion_input_path == parent.resolve()


def test_canonical_parent_overrides_duplicated_output_inputs(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    parent = processed / "trident_native_cafe_inputs_202604.nc"
    output = processed / "trident_native_cafe_npp_202604_stride10.nc"
    _inputs(parent, 4.0)
    ds = _grid({"cafe_npp": 700.0, **{name: 99.0 for name in CAFE_INPUTS}})
    ds.attrs.update({"source_input_file": str(parent), "tag": "202604"})
    ds.to_netcdf(output)
    entry = next(e for e in discover_analysis_datasets(tmp_path) if e.path == output.resolve())
    merged = open_analysis_dataset(entry)
    assert float(merged["par"][0, 0]) == 4.0
