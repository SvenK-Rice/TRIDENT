"""Interactive Plotly figures for depth-resolved CAFE diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trident.models.cafe import CafeProfileResult


def _depth_axis(fig: go.Figure) -> None:
    fig.update_yaxes(
        title_text="Depth (m)",
        autorange="reversed",
        rangemode="tozero",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=70, r=30, t=65, b=60),
        hovermode="closest",
    )


def make_npp_depth_figure(profile: CafeProfileResult) -> go.Figure:
    """Daily volumetric NPP as a function of depth."""
    fig = go.Figure(
        go.Scatter(
            x=np.asarray(profile.npp_z, dtype=float),
            y=np.asarray(profile.depth_m, dtype=float),
            mode="lines",
            name="Daily NPP",
            hovertemplate="Depth: %{y:.2f} m<br>NPP: %{x:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        title="CAFE daily NPP profile",
        xaxis_title="NPP(z) (mg C m⁻³ day⁻¹)",
    )
    _depth_axis(fig)
    return fig


def make_par_depth_figure(profile: CafeProfileResult) -> go.Figure:
    """Local-noon scalar irradiance as a function of depth."""
    fig = go.Figure(
        go.Scatter(
            x=np.asarray(profile.par_z_noon, dtype=float),
            y=np.asarray(profile.depth_m, dtype=float),
            mode="lines",
            name="Noon PAR",
            hovertemplate="Depth: %{y:.2f} m<br>PAR: %{x:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        title="CAFE local-noon irradiance profile",
        xaxis_title="Scalar irradiance (model units)",
    )
    _depth_axis(fig)
    return fig


def make_physiology_depth_figure(
    profile: CafeProfileResult,
    mld_m: float | None = None,
) -> go.Figure:
    """Physiologist-facing light-acclimation and spectral-match profiles."""
    depth = np.asarray(profile.depth_m, dtype=float)
    ek_daily = np.asarray(profile.ek_z, dtype=float)
    kpur_daily = np.asarray(profile.kpur_z, dtype=float)
    ek_umol = ek_daily / 0.0864
    kpur_umol = kpur_daily / 0.0864
    spectral_match = np.divide(
        1.3 * ek_daily,
        kpur_daily,
        out=np.full_like(ek_daily, np.nan),
        where=np.isfinite(kpur_daily) & (kpur_daily != 0),
    )
    fig = make_subplots(
        rows=1,
        cols=4,
        shared_yaxes=True,
        horizontal_spacing=0.055,
        subplot_titles=("Ek", "KPUR", "Spectral match", "φmax"),
    )
    fig.add_trace(
        go.Scatter(
            x=ek_umol,
            y=depth,
            mode="lines",
            name="Ek",
            line={"color": "#636EFA"},
            customdata=ek_daily,
            hovertemplate=(
                "Depth: %{y:.2f} m<br>"
                "Ek: %{x:.2f} µmol photons m⁻² s⁻¹<br>"
                "Daily-equivalent: %{customdata:.4g} mol photons m⁻² d⁻¹"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=kpur_umol,
            y=depth,
            mode="lines",
            name="KPUR",
            line={"color": "#EF553B"},
            customdata=kpur_daily,
            hovertemplate=(
                "Depth: %{y:.2f} m<br>"
                "KPUR: %{x:.2f} µmol photons m⁻² s⁻¹<br>"
                "Daily-equivalent: %{customdata:.4g} mol photons m⁻² d⁻¹"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=spectral_match,
            y=depth,
            mode="lines",
            name="RPUR",
            line={"color": "#AB63FA"},
            hovertemplate=(
                "Depth: %{y:.2f} m<br>"
                "RPUR = 1.3 Ek/KPUR: %{x:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=np.asarray(profile.phimax_z, dtype=float),
            y=depth,
            mode="lines",
            name="φmax",
            line={"color": "#00CC96"},
            hovertemplate=(
                "Depth: %{y:.2f} m<br>"
                "φmax: %{x:.5f} mol C (mol absorbed photons)⁻¹"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=4,
    )
    reference_labels = []
    if mld_m is not None and np.isfinite(mld_m):
        for column in range(1, 5):
            fig.add_hline(
                y=float(mld_m), line_dash="dot", line_color="#555", line_width=1,
                row=1, col=column,
            )
        reference_labels.append(f"dotted: MLD {float(mld_m):.1f} m")
    if np.isfinite(profile.zeu):
        for column in range(1, 5):
            fig.add_hline(
                y=float(profile.zeu), line_dash="dash", line_color="#2A9D8F", line_width=1,
                row=1, col=column,
            )
        reference_labels.append(f"dashed: Zeu {float(profile.zeu):.1f} m")
    fig.update_layout(
        title="CAFE physiological profiles" + (
            " — " + "; ".join(reference_labels) if reference_labels else ""
        ),
        showlegend=False,
    )
    fig.update_xaxes(title_text="µmol photons m⁻² s⁻¹", row=1, col=1)
    fig.update_xaxes(title_text="µmol photons m⁻² s⁻¹", row=1, col=2)
    fig.update_xaxes(title_text="RPUR (dimensionless)", row=1, col=3)
    fig.update_xaxes(
        title_text="mol C (mol absorbed photons)⁻¹", row=1, col=4
    )
    _depth_axis(fig)
    fig.update_layout(margin=dict(l=70, r=30, t=100, b=75))
    return fig


def make_npp_time_depth_figure(profile: CafeProfileResult) -> go.Figure:
    """Daylight-time by depth heat map of instantaneous modeled NPP."""
    daylight = np.asarray(profile.time_fraction, dtype=float)
    depth = np.asarray(profile.depth_m, dtype=float)
    npp_tz = np.asarray(profile.npp_tz, dtype=float)
    fig = go.Figure(
        go.Heatmap(
            x=daylight,
            y=depth,
            z=npp_tz.T,
            colorbar=dict(title="NPP"),
            hovertemplate=(
                "Daylight fraction: %{x:.2f}<br>"
                "Depth: %{y:.2f} m<br>"
                "NPP: %{z:.4g}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="CAFE production through the daylight cycle",
        xaxis_title="Normalized daylight time (0 = sunrise, 1 = sunset)",
    )
    _depth_axis(fig)
    return fig


def make_spectral_optics_figure(profile: CafeProfileResult) -> go.Figure:
    """Spectral absorption, backscattering, and attenuation diagnostics."""
    wavelength = np.asarray(profile.wavelength_nm, dtype=float)
    fields: Mapping[str, np.ndarray] = {
        "Total absorption": profile.absorption_total,
        "Phytoplankton absorption": profile.absorption_phytoplankton,
        "Backscattering": profile.backscattering,
        "Kd": profile.kd_spectral,
    }
    fig = go.Figure()
    for label, values in fields.items():
        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=np.asarray(values, dtype=float),
                mode="lines",
                name=label,
                hovertemplate=f"Wavelength: %{{x:.0f}} nm<br>{label}: %{{y:.4g}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="CAFE spectral optics",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Optical coefficient (m⁻¹)",
        template="plotly_white",
        margin=dict(l=70, r=30, t=65, b=60),
        hovermode="x unified",
    )
    return fig


def build_cafe_diagnostics_figures(profile: CafeProfileResult) -> dict[str, go.Figure]:
    """Build the initial CAFE Diagnostics Explorer figure collection."""
    return {
        "npp_depth": make_npp_depth_figure(profile),
        "par_depth": make_par_depth_figure(profile),
        "physiology_depth": make_physiology_depth_figure(profile),
        "npp_time_depth": make_npp_time_depth_figure(profile),
        "spectral_optics": make_spectral_optics_figure(profile),
    }
