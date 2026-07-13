from pathlib import Path

import numpy as np
import xarray as xr

from trident.data_sources.discovery import scan_data_root
from trident.data_sources.registry import CAFE_REQUIRED
from trident.data_sources.resolver import evaluate_month


def _write(path: Path, valid: bool):
    values = np.ones((2, 2), dtype="float32")
    data = {name: (("lat", "lon"), values.copy()) for name in CAFE_REQUIRED}
    if not valid:
        for name in CAFE_REQUIRED:
            data[name][1][:] = -9999.0
    xr.Dataset(data, coords={"lat": [1, 2], "lon": [3, 4]}).to_netcdf(path)


def test_prepared_file_needs_joint_valid_pixel(tmp_path):
    bad = tmp_path / "trident_native_cafe_inputs_202501.nc"
    _write(bad, False)
    inventory = scan_data_root(tmp_path)
    readiness = evaluate_month(inventory, "202501")
    assert not readiness.ready
    assert readiness.joint_valid_pixels == 0


def test_prepared_file_with_overlap_is_ready(tmp_path):
    good = tmp_path / "trident_native_cafe_inputs_202501.nc"
    _write(good, True)
    inventory = scan_data_root(tmp_path)
    readiness = evaluate_month(inventory, "202501")
    assert readiness.ready
    assert readiness.joint_valid_pixels == 4
