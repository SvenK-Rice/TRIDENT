#!/usr/bin/env python3
"""Catalog-first regional map and click-to-CAFE-profile explorer."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from trident.analysis.analysis_profiles import (
    CAFE_INPUTS,
    default_map_variable,
    discover_analysis_datasets,
    open_analysis_dataset,
)
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


def _map_points(field: xr.DataArray, lat_name: str, lon_name: str, max_points: int = 20000) -> pd.DataFrame:
    lat = np.asarray(field[lat_name].values, dtype=float)
    lon = np.asarray(field[lon_name].values, dtype=float)
    values = np.asarray(field.values, dtype=float)
    stride = max(1, int(np.ceil(np.sqrt(values.size / max_points))))
    lat_idx = np.arange(0, len(lat), stride)
    lon_idx = np.arange(0, len(lon), stride)
    sub = values[np.ix_(lat_idx, lon_idx)]
    lon_grid, lat_grid = np.meshgrid(lon[lon_idx], lat[lat_idx])
    valid = np.isfinite(sub)
    return pd.DataFrame({"lat": lat_grid[valid], "lon": lon_grid[valid], "value": sub[valid]})


def _clickable_map(points: pd.DataFrame, variable: str, selected: tuple[float, float] | None) -> go.Figure:
    fig = go.Figure(go.Scattergl(
        x=points.lon, y=points.lat, mode="markers",
        marker={"size": 6, "color": points.value, "colorscale": "Viridis", "showscale": True,
                "colorbar": {"title": variable}},
        customdata=np.column_stack([points.lat, points.lon, points.value]),
        hovertemplate="Lat %{customdata[0]:.4f}<br>Lon %{customdata[1]:.4f}<br>Value %{customdata[2]:.4g}<extra></extra>",
        name=variable,
    ))
    if selected:
        fig.add_trace(go.Scatter(x=[selected[1]], y=[selected[0]], mode="markers",
                                 marker={"size": 15, "symbol": "x"}, name="Selected pixel"))
    fig.update_layout(title=f"Click a pixel — {variable}", xaxis_title="Longitude", yaxis_title="Latitude",
                      template="plotly_white", dragmode="pan", height=580,
                      margin={"l": 55, "r": 25, "t": 65, "b": 50})
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


def _upload_dataset(uploaded) -> xr.Dataset:
    suffix = Path(uploaded.name).suffix or ".nc"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    with xr.open_dataset(tmp.name) as opened:
        return opened.load()


def main() -> None:
    st.set_page_config(page_title="TRIDENT Regional NPP & Physiology", page_icon="🌊", layout="wide")
    st.title("Regional NPP, Environment & CAFE Physiology")
    st.caption("Use an existing TRIDENT analysis, inspect its regional map, then click a real satellite pixel to reveal the depth-resolved physiology behind its NPP.")

    with st.sidebar:
        st.header("TRIDENT data")
        default_root = "/Volumes/400GB_SvenK/Satellite_data"
        data_root = st.text_input("TRIDENT data root", value=default_root)
        refresh = st.button("Refresh saved analyses")
        if refresh:
            st.cache_data.clear()

    entries = discover_analysis_datasets(data_root)
    source_mode = "Saved TRIDENT analysis"
    uploaded = None
    if entries:
        with st.sidebar:
            labels = [entry.display_name for entry in entries]
            selected_label = st.selectbox("Saved analysis", labels)
            entry = entries[labels.index(selected_label)]
            st.caption(str(entry.path))
            if entry.companion_input_path:
                st.caption(f"CAFE inputs: {entry.companion_input_path.name}")
            elif not entry.has_embedded_cafe_inputs:
                st.warning("This result has no recoverable CAFE input file; its map remains available, but the depth profile may not be.")
    else:
        entry = None
        st.info(f"No NetCDF analyses were found under {Path(data_root).expanduser()}. You can still upload an external file below.")

    with st.sidebar:
        with st.expander("External dataset (optional)"):
            uploaded = st.file_uploader("Upload NetCDF", type=["nc", "nc4", "cdf"])
            if uploaded is not None:
                source_mode = "Uploaded dataset"

    try:
        ds = _upload_dataset(uploaded) if uploaded is not None else open_analysis_dataset(entry)  # type: ignore[arg-type]
    except Exception as exc:
        st.error(f"Could not open dataset: {exc}")
        st.stop()

    lat_name = find_coordinate_name(ds, LAT_NAMES)
    lon_name = find_coordinate_name(ds, LON_NAMES)
    try:
        time_name = find_coordinate_name(ds, TIME_NAMES)
    except KeyError:
        time_name = None
    variables = numeric_data_variables(ds)
    inferred = infer_variable_mapping(ds)

    with st.sidebar:
        st.success(source_mode)
        time_index = 0
        if time_name and ds[time_name].size > 1:
            time_index = st.slider("Time index", 0, int(ds[time_name].size - 1), 0)
        st.header("CAFE inputs")
        mapping: dict[str, str] = {}
        missing: list[str] = []
        choices = ["— unavailable —"] + variables
        for field in CAFE_ALIASES:
            guessed = inferred[field]
            selection = st.selectbox(field, choices, index=choices.index(guessed) if guessed in choices else 0,
                                     key=f"integrated_map_{field}")
            if selection == "— unavailable —":
                missing.append(field)
            else:
                mapping[field] = selection

    map_default = default_map_variable(variables)
    map_variable = st.selectbox("Regional environmental/NPP layer", variables, index=variables.index(map_default))
    try:
        field = horizontal_field(ds, map_variable, lat_name=lat_name, lon_name=lon_name,
                                 time_name=time_name, time_index=time_index)
        points = _map_points(field, lat_name, lon_name)
    except Exception as exc:
        st.error(f"Cannot display {map_variable}: {exc}")
        st.stop()
    if points.empty:
        st.warning("The selected layer contains no finite pixels.")
        st.stop()

    state_key = "integrated_cafe_clicked_latlon"
    if state_key not in st.session_state:
        st.session_state[state_key] = (float(points.lat.median()), float(points.lon.median()))
    event = st.plotly_chart(_clickable_map(points, map_variable, st.session_state[state_key]),
                            use_container_width=True, on_select="rerun", selection_mode="points",
                            key="integrated_regional_map")
    clicked = _extract_click(event)
    if clicked is not None:
        st.session_state[state_key] = clicked

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        selected_lat = st.number_input("Latitude", value=float(st.session_state[state_key][0]), format="%.5f")
    with c2:
        selected_lon = st.number_input("Longitude", value=float(st.session_state[state_key][1]), format="%.5f")
    with c3:
        radius = st.slider("Nearest complete-pixel radius", 0, 30, 12)
    st.session_state[state_key] = (selected_lat, selected_lon)

    if missing:
        st.warning("Map shown, but a CAFE profile cannot be calculated because these inputs are unavailable: " + ", ".join(missing))
        st.stop()

    try:
        pixel = extract_nearest_valid_pixel(ds, selected_lat, selected_lon, mapping,
                                            lat_name=lat_name, lon_name=lon_name, time_name=time_name,
                                            time_index=time_index, max_search_radius_cells=radius)
        v = pixel.values
        profile = cafe_profile(v["par"], v["chl"], v["mld"], pixel.selected_lat, pixel.day_of_year,
                               v["aph443"], v["adg443"], v["bbp443"], v["bbp_s"], v["sst"])
    except Exception as exc:
        st.error(f"Profile calculation failed: {exc}")
        st.stop()

    reintegrated = integrate_npp_profile(profile.depth_m, profile.npp_z, delz_m=profile.delz_m)
    metrics = st.columns(6)
    metrics[0].metric("Latitude", f"{pixel.selected_lat:.4f}°")
    metrics[1].metric("Longitude", f"{pixel.selected_lon:.4f}°")
    metrics[2].metric("CAFE NPP", f"{profile.npp:,.1f}", help="mg C m⁻² d⁻¹")
    metrics[3].metric("Euphotic depth", f"{profile.zeu:.1f} m")
    metrics[4].metric("KdPAR", f"{profile.kdpar:.4f} m⁻¹")
    metrics[5].metric("Profile parity", f"{abs(reintegrated-profile.npp):.1e}")

    with st.expander("Environmental and optical inputs at selected pixel", expanded=True):
        st.dataframe(pd.DataFrame({"CAFE field": list(v), "Value": list(v.values()),
                                   "Dataset variable": [pixel.source_variables[key] for key in v]}),
                     hide_index=True, use_container_width=True)
        st.write({"day_of_year": pixel.day_of_year, "time": pixel.time_value})

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
