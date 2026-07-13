import numpy as np
import xarray as xr

from trident.data_sources.qc import clean_dataarray, joint_valid_mask, summarize


def test_fill_values_and_physical_ranges_are_masked():
    da = xr.DataArray([[1.0, -9999.0], [-1.0, 2.0]], dims=("lat", "lon"), name="chl")
    da.attrs["missing_value"] = -9999.0
    cleaned = clean_dataarray(da, name="chl")
    assert np.isfinite(cleaned.values).sum() == 2
    assert np.isnan(cleaned.values[0, 1])
    assert np.isnan(cleaned.values[1, 0])
    summary = summarize(da, name="chl")
    assert summary.valid == 2
    assert summary.minimum == 1.0


def test_joint_valid_requires_overlap():
    ds = xr.Dataset({
        "par": (("lat", "lon"), [[10.0, -9999.0], [20.0, 30.0]]),
        "chl": (("lat", "lon"), [[1.0, 2.0], [-9999.0, 3.0]]),
    })
    mask = joint_valid_mask(ds, ("par", "chl"))
    assert mask.tolist() == [[True, False], [False, True]]
