from types import SimpleNamespace

import numpy as np
import xarray as xr

from trident.analysis.profile_summary import profile_interpretation, selected_map_value
from trident.app.integrated_explorer import _profile_frame


def test_selected_map_value_uses_nearest_coordinate() -> None:
    field = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=("lat", "lon"),
        coords={"lat": [30.0, 31.0], "lon": [-124.0, -123.0]},
    )
    assert selected_map_value(field, "lat", "lon", 30.8, -123.2) == 4.0


def test_profile_interpretation_reports_peak_and_mixed_layer_fraction() -> None:
    profile = SimpleNamespace(
        depth_m=np.array([0.0, 10.0, 20.0, 30.0]),
        npp_z=np.array([4.0, 3.0, 2.0, 1.0]),
        delz_m=10.0,
    )
    result = profile_interpretation(profile, mld=10.0)
    assert result["peak_depth_m"] == 0.0
    assert result["median_production_depth_m"] == 10.0
    assert result["fraction_above_mld"] == 0.7
    assert "Most production" in result["mld_interpretation"]


def test_profile_export_uses_physiological_units_and_includes_rpur() -> None:
    from trident.models.cafe import cafe_profile

    profile = cafe_profile(
        40.0, 0.15, 35.0, 30.0, 180,
        0.010, 0.005, 0.0015, 1.0, 22.0,
    )
    frame = _profile_frame(profile)

    np.testing.assert_allclose(
        frame["ek_umol_photons_m2_s"],
        profile.ek_z / 0.0864,
    )
    np.testing.assert_allclose(
        frame["rpur_spectral_match_dimensionless"],
        1.3 * profile.ek_z / profile.kpur_z,
    )
