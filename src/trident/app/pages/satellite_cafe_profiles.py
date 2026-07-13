#!/usr/bin/env python3
"""Interactive satellite-pixel selection and CAFE depth diagnostics."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from trident.analysis.satellite_pixel import (
    CAFE_ALIASES,
    LAT_NAMES,
    LON_NAMES,
    TIME_NAMES,
    extract_nearest_valid_pixel,
    find_coordinate_name,
    horizontal_field,
    infer_variable_mapping,
    numeric_data_variables,
)
from trident.models.cafe import cafe_profile, integrate_npp_profile
from trident.visualization.cafe_profiles import (
    make_npp_depth_figure,
    make_npp_time_depth_figure,
    make_par_depth_figure,
    make_physiology_depth_figure,
    make_spectral_optics_figure,
)


def _open_dataset(uploaded, path_text: str) -> tuple[xr.Dataset, str]:
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".nc"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getbuffer())
        tmp.close()
        return xr.open_dataset(tmp.name), uploaded.name
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return xr.open_dataset(path), str(path)


def _map_points(field: xr.DataArray, lat_name: str, lon_name: str, max_points: int = 18000) -> pd.DataFrame:
    lat = np.asarray(field[lat_name].values, dtype=float)
    lon = np.asarray(field[lon_name].values, dtype=float)
    values = np.asarray(field.values, dtype=float)
    stride = max(1, int(np.ceil(np.sqrt(values.size / max_points))))
    lat_idx = np.arange(0, len(lat), stride)
    lon_idx = np.arange(0, len(lon), stride)
    sub = values[np.ix_(lat_idx, lon_idx)]
    lon_grid, lat_grid = np.meshgrid(lon[lon_idx], lat[lat_idx])
    valid = np.isfinite(sub)
    return pd.DataFrame({
        "lat": lat_grid[valid],
        "lon": lon_grid[valid],
        "value": sub[valid],
    })


def _make_clickable_map(points: pd.DataFrame, variable: str, selected: tuple[float, float] | None) -> go.Figure:
    fig = go.Figure(
        go.Scattergl(
            x=points["lon"],
            y=points["lat"],
            mode="markers",
            marker={"size": 6, "color": points["value"], "colorscale": "Viridis", "showscale": True,
                    "colorbar": {"title": variable}},
            customdata=np.column_stack([points["lat"], points["lon"], points["value"]]),
            hovertemplate="Lat %{customdata[0]:.4f}<br>Lon %{customdata[1]:.4f}<br>Value %{customdata[2]:.4g}<extra></extra>",
            name=variable,
        )
    )
    if selected is not None:
        fig.add_trace(go.Scatter(
            x=[selected[1]], y=[selected[0]], mode="markers",
            marker={"size": 15, "symbol": "x"}, name="Selected pixel",
            hovertemplate="Selected<br>Lat %{y:.4f}<br>Lon %{x:.4f}<extra></extra>",
        ))
    fig.update_layout(
        title=f"Click a satellite pixel — {variable}",
        xaxis_title="Longitude", yaxis_title="Latitude",
        template="plotly_white", dragmode="pan", height=590,
        margin={"l": 55, "r": 25, "t": 65, "b": 50},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _extract_click(event) -> tuple[float, float] | None:
    try:
        points = event.selection.points
    except Exception:
        try:
            points = event["selection"]["points"]
        except Exception:
            return None
    if not points:
        return None
    point = points[0]
    custom = point.get("customdata") if isinstance(point, dict) else getattr(point, "customdata", None)
    if custom is not None and len(custom) >= 2:
        return float(custom[0]), float(custom[1])
    x = point.get("x") if isinstance(point, dict) else getattr(point, "x", None)
    y = point.get("y") if isinstance(point, dict) else getattr(point, "y", None)
    return (float(y), float(x)) if x is not None and y is not None else None


def main() -> None:
    st.set_page_config(page_title="TRIDENT Satellite Pixel Physiology", page_icon="🌊", layout="wide")
    st.title("Satellite Pixel → CAFE Physiology")
    st.caption("Select a real gridded satellite pixel and inspect the depth-, daylight-, and wavelength-resolved CAFE calculations behind its integrated NPP.")

    with st.sidebar:
        st.header("Satellite dataset")
        uploaded = st.file_uploader("Upload a NetCDF file", type=["nc", "nc4", "cdf"])
        path_text = st.text_input("Or enter a local NetCDF path", value="")
        st.caption("Use a saved TRIDENT native-input or OSU-replay dataset containing the CAFE input fields.")

    if uploaded is None and not path_text.strip():
        st.info("Upload or select a saved satellite dataset to begin.")
        st.stop()

    try:
        ds, source_name = _open_dataset(uploaded, path_text.strip())
        lat_name = find_coordinate_name(ds, LAT_NAMES)
        lon_name = find_coordinate_name(ds, LON_NAMES)
        try:
            time_name = find_coordinate_name(ds, TIME_NAMES)
        except KeyError:
            time_name = None
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    variables = numeric_data_variables(ds)
    inferred = infer_variable_mapping(ds)

    with st.sidebar:
        st.success(f"Opened: {Path(source_name).name}")
        time_index = 0
        if time_name and ds[time_name].size > 1:
            time_index = st.slider("Time index", 0, int(ds[time_name].size - 1), 0)
            st.caption(str(ds[time_name].values[time_index]))

        st.header("CAFE variable mapping")
        mapping: dict[str, str] = {}
        missing: list[str] = []
        choices = ["— not available —"] + variables
        for field in CAFE_ALIASES:
            guessed = inferred[field]
            default_index = choices.index(guessed) if guessed in choices else 0
            selected_var = st.selectbox(field, choices, index=default_index, key=f"map_{field}")
            if selected_var == "— not available —":
                missing.append(field)
            else:
                mapping[field] = selected_var

    if missing:
        st.warning(
            "The dataset is missing mappings for: " + ", ".join(missing) + ". "
            "A full CAFE profile requires PAR, chlorophyll, MLD, SST, aph443, adg443, bbp443, and bbp slope."
        )

    map_default = "cafe_npp" if "cafe_npp" in variables else (mapping.get("chl") or variables[0])
    map_variable = st.selectbox("Environmental/NPP layer shown on map", variables, index=variables.index(map_default))

    try:
        map_field = horizontal_field(ds, map_variable, lat_name=lat_name, lon_name=lon_name, time_name=time_name, time_index=time_index)
        points = _map_points(map_field, lat_name, lon_name)
    except Exception as exc:
        st.error(f"Cannot display {map_variable}: {exc}")
        st.stop()

    if "clicked_latlon" not in st.session_state:
        st.session_state.clicked_latlon = (float(points.lat.median()), float(points.lon.median()))

    map_event = st.plotly_chart(
        _make_clickable_map(points, map_variable, st.session_state.clicked_latlon),
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="satellite_pixel_map",
    )
    clicked = _extract_click(map_event)
    if clicked is not None:
        st.session_state.clicked_latlon = clicked

    manual1, manual2, manual3 = st.columns([1, 1, 2])
    with manual1:
        selected_lat = st.number_input("Selected latitude", value=float(st.session_state.clicked_latlon[0]), format="%.5f")
    with manual2:
        selected_lon = st.number_input("Selected longitude", value=float(st.session_state.clicked_latlon[1]), format="%.5f")
    with manual3:
        search_radius = st.slider("Nearest-valid-pixel search radius (grid cells)", 0, 30, 12)
    st.session_state.clicked_latlon = (selected_lat, selected_lon)

    if missing:
        st.stop()

    try:
        selection = extract_nearest_valid_pixel(
            ds, selected_lat, selected_lon, mapping,
            lat_name=lat_name, lon_name=lon_name, time_name=time_name,
            time_index=time_index, max_search_radius_cells=search_radius,
        )
        v = selection.values
        profile = cafe_profile(
            v["par"], v["chl"], v["mld"], selection.selected_lat,
            selection.day_of_year, v["aph443"], v["adg443"],
            v["bbp443"], v["bbp_s"], v["sst"],
        )
    except Exception as exc:
        st.error(f"Profile calculation failed: {exc}")
        st.stop()

    if abs(selection.selected_lat - selected_lat) > 1e-9 or abs(selection.selected_lon - selected_lon) > 1e-9:
        st.info(
            f"The clicked pixel contained missing inputs. Using the nearest complete pixel at "
            f"{selection.selected_lat:.4f}°, {selection.selected_lon:.4f}°."
        )

    reintegrated = integrate_npp_profile(profile.depth_m, profile.npp_z, delz_m=profile.delz_m)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latitude", f"{selection.selected_lat:.4f}°")
    c2.metric("Longitude", f"{selection.selected_lon:.4f}°")
    c3.metric("Integrated CAFE NPP", f"{profile.npp:,.1f}", help="mg C m⁻² day⁻¹")
    c4.metric("Euphotic depth", f"{profile.zeu:.1f} m")
    c5.metric("Profile parity", f"{abs(reintegrated-profile.npp):.1e}")

    input_df = pd.DataFrame({
        "CAFE field": list(v),
        "Satellite value": list(v.values()),
        "Source variable": [selection.source_variables[key] for key in v],
    })
    with st.expander("Satellite inputs used for this pixel", expanded=True):
        st.dataframe(input_df, hide_index=True, use_container_width=True)
        st.write({"day_of_year": selection.day_of_year, "time": selection.time_value})

    overview, physiology, daylight, optics = st.tabs(["NPP & light", "Physiology", "Daylight cycle", "Spectral optics"])
    with overview:
        left, right = st.columns(2)
        left.plotly_chart(make_npp_depth_figure(profile), use_container_width=True)
        right.plotly_chart(make_par_depth_figure(profile), use_container_width=True)
    with physiology:
        st.plotly_chart(make_physiology_depth_figure(profile), use_container_width=True)
    with daylight:
        st.plotly_chart(make_npp_time_depth_figure(profile), use_container_width=True)
    with optics:
        st.plotly_chart(make_spectral_optics_figure(profile), use_container_width=True)


if __name__ == "__main__":
    main()
