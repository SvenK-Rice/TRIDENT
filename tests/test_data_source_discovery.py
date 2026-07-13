from pathlib import Path

import numpy as np
import xarray as xr

from trident.data_sources import evaluate_month, scan_data_root


def _write(path: Path, variables: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {name: (("lat", "lon"), np.ones((2, 3), dtype="float32")) for name in variables}
    xr.Dataset(data, coords={"lat": [1.0, 2.0], "lon": [3.0, 4.0, 5.0]}).to_netcdf(path)


def test_prepared_input_file_makes_month_ready(tmp_path: Path) -> None:
    required = ("par", "chl", "mld", "aph443", "adg443", "bbp443", "bbp_s", "sst")
    _write(tmp_path / "data/processed/trident_native_cafe_inputs_202501.nc", required)
    inv = scan_data_root(tmp_path)
    report = evaluate_month(inv, "202501")
    assert report.ready
    assert report.prepared_files


def test_separate_source_files_require_harmonized_preparation(
    tmp_path: Path,
) -> None:
    """Separate source files are available but are not yet CAFE-ready."""
    for name in (
        "par",
        "chl",
        "mld",
        "aph443",
        "adg443",
        "bbp443",
        "bbp_s",
        "sst",
    ):
        _write(
            tmp_path / f"data/reference/nasa/{name}_202301.nc",
            (name,),
        )

    report = evaluate_month(scan_data_root(tmp_path), "202301")

    assert not report.ready
    assert report.prepared_files == ()
    assert report.joint_valid_pixels == 0
    assert all(item.status == "available" for item in report.inputs)


def test_missing_input_is_reported(tmp_path: Path) -> None:
    for name in ("par", "chl", "mld"):
        _write(tmp_path / f"data/reference/nasa/{name}_202301.nc", (name,))
    report = evaluate_month(scan_data_root(tmp_path), "202301")
    assert not report.ready
    missing = {item.name for item in report.inputs if item.status == "missing"}
    assert "aph443" in missing
