from __future__ import annotations

import numpy as np

from trident.models.cafe import cafe_profile
from trident.visualization.cafe_profiles import (
    build_cafe_diagnostics_figures,
    make_physiology_depth_figure,
)


def _profile():
    return cafe_profile(40.0, 0.15, 35.0, 30.0, 180, 0.010, 0.005, 0.0015, 1.0, 22.0)


def test_diagnostics_collection_contains_expected_figures() -> None:
    figures = build_cafe_diagnostics_figures(_profile())
    assert set(figures) == {
        "npp_depth",
        "par_depth",
        "physiology_depth",
        "npp_time_depth",
        "spectral_optics",
    }


def test_npp_depth_plot_uses_preserved_profile() -> None:
    profile = _profile()
    fig = build_cafe_diagnostics_figures(profile)["npp_depth"]
    np.testing.assert_allclose(np.asarray(fig.data[0].x, dtype=float), profile.npp_z)
    np.testing.assert_allclose(np.asarray(fig.data[0].y, dtype=float), profile.depth_m)
    assert fig.layout.yaxis.autorange == "reversed"


def test_par_depth_plot_uses_noon_profile() -> None:
    profile = _profile()
    fig = build_cafe_diagnostics_figures(profile)["par_depth"]
    np.testing.assert_allclose(np.asarray(fig.data[0].x, dtype=float), profile.par_z_noon)
    np.testing.assert_allclose(np.asarray(fig.data[0].y, dtype=float), profile.depth_m)


def test_time_depth_heatmap_has_correct_orientation() -> None:
    profile = _profile()
    fig = build_cafe_diagnostics_figures(profile)["npp_time_depth"]
    heatmap = fig.data[0]
    assert np.asarray(heatmap.z).shape == (101, 51)
    np.testing.assert_allclose(np.asarray(heatmap.z, dtype=float), profile.npp_tz.T)
    np.testing.assert_allclose(np.asarray(heatmap.x, dtype=float), profile.time_fraction)
    np.testing.assert_allclose(np.asarray(heatmap.y, dtype=float), profile.depth_m)


def test_physiology_plot_uses_conventional_units_and_spectral_match() -> None:
    profile = _profile()
    fig = build_cafe_diagnostics_figures(profile)["physiology_depth"]
    assert len(fig.data) == 4
    np.testing.assert_allclose(np.asarray(fig.data[0].x, dtype=float), profile.ek_z / 0.0864)
    np.testing.assert_allclose(np.asarray(fig.data[1].x, dtype=float), profile.kpur_z / 0.0864)
    np.testing.assert_allclose(
        np.asarray(fig.data[2].x, dtype=float),
        1.3 * profile.ek_z / profile.kpur_z,
    )
    np.testing.assert_allclose(np.asarray(fig.data[3].x, dtype=float), profile.phimax_z)


def test_physiology_plot_marks_mld_and_zeu() -> None:
    profile = _profile()
    fig = make_physiology_depth_figure(profile, mld_m=35.0)
    horizontal_levels = [
        float(shape.y0)
        for shape in fig.layout.shapes
        if shape.y0 == shape.y1
    ]
    assert horizontal_levels.count(35.0) == 4
    assert horizontal_levels.count(float(profile.zeu)) == 4


def test_spectral_plot_uses_wavelength_grid() -> None:
    profile = _profile()
    fig = build_cafe_diagnostics_figures(profile)["spectral_optics"]
    assert len(fig.data) == 4
    for trace in fig.data:
        np.testing.assert_allclose(np.asarray(trace.x, dtype=float), profile.wavelength_nm)
