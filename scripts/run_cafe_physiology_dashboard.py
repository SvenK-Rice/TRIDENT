#!/usr/bin/env python3
"""Standalone Streamlit preview for TRIDENT CAFE physiology diagnostics."""

from __future__ import annotations

import io
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from trident.models.cafe import cafe_profile, integrate_npp_profile
from trident.visualization.cafe_profiles import (
    make_npp_depth_figure,
    make_npp_time_depth_figure,
    make_par_depth_figure,
    make_physiology_depth_figure,
    make_spectral_optics_figure,
)

PRESETS = {
    "Open ocean": dict(par=40.0, chl=0.15, mld=35.0, lat=30.0, yd=180, aph443=0.010, adg443=0.005, bbp443=0.0015, bbp_s=1.0, sst=22.0),
    "Productive coastal": dict(par=48.0, chl=2.0, mld=15.0, lat=34.0, yd=120, aph443=0.055, adg443=0.018, bbp443=0.006, bbp_s=0.8, sst=16.0),
    "Cold high latitude": dict(par=25.0, chl=0.8, mld=60.0, lat=55.0, yd=200, aph443=0.030, adg443=0.010, bbp443=0.003, bbp_s=1.2, sst=7.0),
}


def _profile_dataframe(profile) -> pd.DataFrame:
    return pd.DataFrame({
        "depth_m": profile.depth_m,
        "npp_mg_C_m3_day": profile.npp_z,
        "par_noon_model_units": profile.par_z_noon,
        "ek": profile.ek_z,
        "kpur": profile.kpur_z,
        "phimax": profile.phimax_z,
        "absorbed_photons_daily": profile.absorbed_photons_z,
    })


def _spectral_dataframe(profile) -> pd.DataFrame:
    return pd.DataFrame({
        "wavelength_nm": profile.wavelength_nm,
        "absorption_total_m-1": profile.absorption_total,
        "absorption_phytoplankton_m-1": profile.absorption_phytoplankton,
        "backscattering_m-1": profile.backscattering,
        "kd_m-1": profile.kd_spectral,
    })


def _time_depth_dataframe(profile) -> pd.DataFrame:
    rows = []
    for time_idx, daylight_fraction in enumerate(profile.time_fraction):
        for depth_idx, depth_m in enumerate(profile.depth_m):
            rows.append({
                "daylight_fraction": daylight_fraction,
                "depth_m": depth_m,
                "npp_instantaneous": profile.npp_tz[time_idx, depth_idx],
                "irradiance": profile.irradiance_tz[time_idx, depth_idx],
                "absorbed_photons": profile.absorbed_photons_tz_scaled[time_idx, depth_idx],
            })
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="TRIDENT CAFE Physiology Explorer", page_icon="🌊", layout="wide")
    st.title("TRIDENT CAFE Physiology Explorer")
    st.caption("Phase 2B preview — depth-, time-, and wavelength-resolved diagnostics from the validated CAFE engine")

    with st.sidebar:
        st.header("Pixel inputs")
        preset_name = st.selectbox("Preset", list(PRESETS), index=0)
        p = PRESETS[preset_name]

        par = st.number_input("PAR (mol photons m⁻² day⁻¹)", min_value=0.001, value=float(p["par"]), step=1.0)
        chl = st.number_input("Chlorophyll (mg m⁻³)", min_value=0.0001, value=float(p["chl"]), format="%.4f")
        mld = st.number_input("Mixed-layer depth (m)", min_value=0.0, value=float(p["mld"]), step=1.0)
        lat = st.number_input("Latitude (°)", min_value=-89.9, max_value=89.9, value=float(p["lat"]), step=0.5)
        yd = st.number_input("Day of year", min_value=1, max_value=366, value=int(p["yd"]), step=1)
        sst = st.number_input("SST (°C)", value=float(p["sst"]), step=0.5)

        with st.expander("Optical inputs", expanded=False):
            aph443 = st.number_input("aph(443) (m⁻¹)", min_value=0.0, value=float(p["aph443"]), format="%.5f")
            adg443 = st.number_input("adg(443) (m⁻¹)", min_value=0.0, value=float(p["adg443"]), format="%.5f")
            bbp443 = st.number_input("bbp(443) (m⁻¹)", min_value=0.0, value=float(p["bbp443"]), format="%.5f")
            bbp_s = st.number_input("bbp spectral slope", min_value=0.0, value=float(p["bbp_s"]), format="%.3f")

    try:
        profile = cafe_profile(par, chl, mld, lat, int(yd), aph443, adg443, bbp443, bbp_s, sst)
    except Exception as exc:
        st.error(f"CAFE profile could not be calculated: {exc}")
        st.stop()

    reintegrated = integrate_npp_profile(profile.depth_m, profile.npp_z, delz_m=profile.delz_m)
    difference = reintegrated - profile.npp
    peak_index = int(np.nanargmax(profile.npp_z))
    peak_depth = float(profile.depth_m[peak_index])
    peak_npp = float(profile.npp_z[peak_index])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Integrated NPP", f"{profile.npp:,.1f}", help="mg C m⁻² day⁻¹")
    c2.metric("Euphotic depth", f"{profile.zeu:,.1f} m")
    c3.metric("KdPAR", f"{profile.kdpar:.4f} m⁻¹")
    c4.metric("Peak production depth", f"{peak_depth:.1f} m")
    c5.metric("Profile parity", f"{abs(difference):.1e}", help="Absolute difference between stored NPP and reintegrated NPP(z)")

    overview, physiology, daylight, optics, data = st.tabs([
        "Overview", "Physiology", "Daylight cycle", "Spectral optics", "Data & export"
    ])

    with overview:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(make_npp_depth_figure(profile), use_container_width=True)
        with right:
            st.plotly_chart(make_par_depth_figure(profile), use_container_width=True)
        st.info(
            f"This pixel produces {profile.npp:,.1f} mg C m⁻² day⁻¹. "
            f"Maximum daily volumetric production is {peak_npp:.3g} mg C m⁻³ day⁻¹ at {peak_depth:.1f} m. "
            f"Reintegrating the displayed NPP(z) profile reproduces the legacy CAFE output with an absolute difference of {abs(difference):.3e}."
        )

    with physiology:
        st.plotly_chart(make_physiology_depth_figure(profile), use_container_width=True)
        st.caption("Ek, KPUR, and φmax are calculated by the same numerical pathway used for integrated CAFE NPP.")

    with daylight:
        st.plotly_chart(make_npp_time_depth_figure(profile), use_container_width=True)
        st.caption("The heat map preserves all 51 normalized daylight time steps and 101 model depth levels.")

    with optics:
        st.plotly_chart(make_spectral_optics_figure(profile), use_container_width=True)

    with data:
        profile_df = _profile_dataframe(profile)
        spectral_df = _spectral_dataframe(profile)
        time_depth_df = _time_depth_dataframe(profile)

        st.subheader("Depth profile")
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download depth-profile CSV",
            profile_df.to_csv(index=False).encode("utf-8"),
            file_name="trident_cafe_depth_profile.csv",
            mime="text/csv",
        )

        st.subheader("Spectral optics")
        st.dataframe(spectral_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download spectral-optics CSV",
            spectral_df.to_csv(index=False).encode("utf-8"),
            file_name="trident_cafe_spectral_optics.csv",
            mime="text/csv",
        )

        st.subheader("Time–depth fields")
        st.dataframe(time_depth_df.head(1000), use_container_width=True, hide_index=True)
        st.download_button(
            "Download full time–depth CSV",
            time_depth_df.to_csv(index=False).encode("utf-8"),
            file_name="trident_cafe_time_depth.csv",
            mime="text/csv",
        )

    st.divider()
    st.caption(
        "This preview is intentionally separate from the existing area-NPP and environmental-data pages. "
        "The next integration step will pass the currently selected map pixel or station into this same dashboard."
    )


if __name__ == "__main__":
    main()
