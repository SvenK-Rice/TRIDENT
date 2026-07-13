import numpy as np
import xarray as xr

from trident.workflows.native_prepare import _align_to_reference, _find_name, ALIASES


def test_pace_science_variables_win_over_uncertainties() -> None:
    names = (
        "aph_unc_442",
        "aph_442",
        "adg_unc_442",
        "adg_442",
        "bbp_unc_442",
        "bbp_442",
        "bbp_s",
    )
    assert _find_name(names, ALIASES["aph443"]) == "aph_442"
    assert _find_name(names, ALIASES["adg443"]) == "adg_442"
    assert _find_name(names, ALIASES["bbp443"]) == "bbp_442"
    assert _find_name(names, ALIASES["bbp_s"]) == "bbp_s"


def test_pace_par_uses_daily_planar_above_surface() -> None:
    names = (
        "par_day_scalar_below",
        "par_day_planar_above",
        "par_day_planar_below",
    )
    assert _find_name(names, ALIASES["par"]) == "par_day_planar_above"


def test_coordinate_alignment_uses_lat_lon_not_array_shape() -> None:
    source = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 2.0], "lon": [0.0, 2.0]},
    )
    reference = xr.DataArray(
        np.zeros((3, 3), dtype="float32"),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0, 2.0]},
    )
    aligned = _align_to_reference(source, reference)
    assert aligned.shape == (3, 3)
    assert np.array_equal(aligned.lat, reference.lat)
    assert np.array_equal(aligned.lon, reference.lon)
    assert float(aligned.sel(lat=2.0, lon=2.0)) == 4.0
