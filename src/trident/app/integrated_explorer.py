"""Integrated regional map and click-to-CAFE physiology explorer."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from trident.analysis.analysis_profiles import (
    CAFE_INPUTS,
    AnalysisDataset,
    discover_analysis_datasets,
    open_analysis_dataset,
)
from trident.analysis.satellite_pixel import (
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
from trident.analysis.profile_summary import profile_interpretation, selected_map_value
from trident.visualization.cafe_profiles import (
    make_npp_depth_figure,
    make_npp_time_depth_figure,
    make_par_depth_figure,
    make_physiology_depth_figure,
    make_spectral_optics_figure,
)

_MAP_PRIORITY = (
    "cafe_npp", "trident_replay_npp", "osu_archived_npp", "difference",
    "chl", "par", "sst", "mld", "zeu", "kdpar",
)


def _default_variable(names: list[str]) -> str:
    for name in _MAP_PRIORITY:
        if name in names:
            return name
    return names[0]


def _map_points(field: xr.DataArray, lat_name: str, lon_name: str, max_points: int = 30000) -> pd.DataFrame:
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


def _clickable_map(points: pd.DataFrame, variable: str, selected: tuple[float, float] | None) -> go.Figure:
    fig = go.Figure(go.Scattergl(
        x=points["lon"], y=points["lat"], mode="markers",
        marker={
            "size": 6,
            "color": points["value"],
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": variable},
        },
        customdata=np.column_stack([points["lat"], points["lon"], points["value"]]),
        hovertemplate=(
            "Longitude %{customdata[1]:.4f}<br>"
            "Latitude %{customdata[0]:.4f}<br>"
            f"{variable} %{{customdata[2]:.4g}}<extra></extra>"
        ),
        name=variable,
    ))
    if selected is not None:
        fig.add_trace(go.Scatter(
            x=[selected[1]], y=[selected[0]], mode="markers",
            marker={"size": 16, "symbol": "x", "line": {"width": 2}},
            name="Selected pixel",
        ))
    fig.update_layout(
        title=f"{variable}: click a point for depth-resolved physiology",
        xaxis_title="Longitude", yaxis_title="Latitude",
        template="plotly_white", dragmode="pan", height=620,
        margin={"l": 55, "r": 25, "t": 65, "b": 50},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _coverage_figure(entries: list[AnalysisDataset]) -> go.Figure:
    """Show the world regions already represented by compatible analysis shards."""
    fig = go.Figure()
    seen: set[str] = set()
    for entry in entries:
        if entry.bbox is None or entry.region_id in seen:
            continue
        seen.add(entry.region_id)
        west, south, east, north = entry.bbox
        fig.add_trace(go.Scattergeo(
            lon=[west, east, east, west, west],
            lat=[south, south, north, north, south],
            mode="lines",
            fill="toself",
            opacity=0.35,
            name=entry.region_label,
            hovertemplate=(
                f"{entry.region_label}<br>"
                f"BBox: {west:g}, {south:g}, {east:g}, {north:g}"
                "<extra></extra>"
            ),
        ))
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#E5E7EB",
        showcountries=True,
        showocean=True,
        oceancolor="#F3F8FC",
    )
    fig.update_layout(
        title="Processed regional coverage",
        template="plotly_white",
        height=430,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h"},
    )
    return fig


def _extract_click(event):
    """Extract latitude and longitude from a Streamlit/Plotly selection event.

    Plotly may return coordinates through customdata, lat/lon, or x/y,
    depending on the trace type and Streamlit version.
    """
    if not event:
        return None

    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection", event)

    if not selection:
        return None

    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")

    if not points:
        return None

    point = points[0]

    if not isinstance(point, dict):
        try:
            point = dict(point)
        except (TypeError, ValueError):
            return None

    custom = point.get("customdata")

    if custom is not None:
        if isinstance(custom, dict):
            lat = custom.get("lat", custom.get("latitude"))
            lon = custom.get("lon", custom.get("longitude"))
            if lat is not None and lon is not None:
                return float(lat), float(lon)

        elif isinstance(custom, (list, tuple)):
            if len(custom) >= 2:
                return float(custom[0]), float(custom[1])

        else:
            try:
                values = list(custom)
            except TypeError:
                values = []
            if len(values) >= 2:
                return float(values[0]), float(values[1])

    lat = point.get("lat", point.get("latitude"))
    lon = point.get("lon", point.get("longitude"))
    if lat is not None and lon is not None:
        return float(lat), float(lon)

    x = point.get("x")
    y = point.get("y")
    if x is not None and y is not None:
        # For the current map traces, x is longitude and y is latitude.
        return float(y), float(x)

    return None

def _profile_frame(profile) -> pd.DataFrame:
    ek_daily = np.asarray(profile.ek_z, dtype=float)
    kpur_daily = np.asarray(profile.kpur_z, dtype=float)
    spectral_match = np.divide(
        1.3 * ek_daily,
        kpur_daily,
        out=np.full_like(ek_daily, np.nan),
        where=np.isfinite(kpur_daily) & (kpur_daily != 0),
    )
    return pd.DataFrame({
        "depth_m": profile.depth_m,
        "npp_mg_C_m3_day": profile.npp_z,
        "par_noon_model_units": profile.par_z_noon,
        "ek_umol_photons_m2_s": ek_daily / 0.0864,
        "kpur_umol_photons_m2_s": kpur_daily / 0.0864,
        "ek_daily_equivalent_mol_photons_m2_day": ek_daily,
        "kpur_daily_equivalent_mol_photons_m2_day": kpur_daily,
        "rpur_spectral_match_dimensionless": spectral_match,
        "phimax_mol_C_per_mol_absorbed_photons": profile.phimax_z,
        "absorbed_photons_daily_model_units": profile.absorbed_photons_z,
    })


def _spectral_frame(profile) -> pd.DataFrame:
    return pd.DataFrame({
        "wavelength_nm": profile.wavelength_nm,
        "absorption_total_m-1": profile.absorption_total,
        "absorption_phytoplankton_m-1": profile.absorption_phytoplankton,
        "backscattering_m-1": profile.backscattering,
        "kd_m-1": profile.kd_spectral,
    })


def _entry_label(entry: AnalysisDataset) -> str:
    stride = f"stride {entry.stride}" if entry.stride is not None else "native grid"
    return (
        f"{entry.region_label} | {entry.period} | {stride} | {entry.role} | {entry.path.name} | "
        + (
        "profile ready" if entry.has_embedded_cafe_inputs or entry.companion_input_path else "map only"
        )
    )



def render_integrated_explorer(root) -> None:
    """Render the existing regional products plus click-to-profile diagnostics."""
    st.subheader("Explore regional products and pixel physiology")
    st.caption(
        "Select an analysis already produced by TRIDENT. Regional NPP and environmental maps remain intact; "
        "clicking a valid ocean pixel adds the depth-, time-, and wavelength-resolved CAFE diagnostics."
    )

    discovered = discover_analysis_datasets(root)
    if not discovered:
        st.info("No processed NetCDF analyses were found. Use OSU Validation or TRIDENT Native first.")
        return

    st.markdown("#### Analysis catalog")
    show_inputs = st.checkbox(
        "Include prepared-input datasets",
        value=False,
        help="Prepared inputs remain discoverable but are hidden by default to keep the analysis list concise.",
    )
    entries = [
        entry for entry in discovered
        if show_inputs or entry.role != "Prepared inputs"
    ]
    if not entries:
        st.info("Only prepared-input datasets were found. Enable the checkbox above to inspect them.")
        return
    region_col, family_col, cadence_col, stride_col, period_col = st.columns(5)
    regions = ["All regions", *sorted({entry.region_label for entry in entries})]
    region_label = region_col.selectbox("Region", regions)
    if region_label != "All regions":
        entries = [entry for entry in entries if entry.region_label == region_label]

    families = ["All families", *sorted({entry.product_family for entry in entries})]
    family = family_col.selectbox("Product family", families)
    if family != "All families":
        entries = [entry for entry in entries if entry.product_family == family]

    cadences = ["All cadences", *sorted({entry.cadence for entry in entries})]
    cadence = cadence_col.selectbox("Temporal cadence", cadences)
    if cadence != "All cadences":
        entries = [entry for entry in entries if entry.cadence == cadence]

    stride_values = sorted({entry.stride for entry in entries if entry.stride is not None})
    stride_labels = ["All strides", *[str(value) for value in stride_values]]
    stride_label = stride_col.selectbox("Computation stride", stride_labels)
    if stride_label != "All strides":
        entries = [entry for entry in entries if entry.stride == int(stride_label)]

    periods = ["All periods", *sorted({entry.period for entry in entries}, reverse=True)]
    period = period_col.selectbox("Period", periods)
    if period != "All periods":
        entries = [entry for entry in entries if entry.period == period]

    if not entries:
        st.warning("No analyses match the selected catalog filters.")
        return

    with st.expander("World coverage for the selected catalog filters", expanded=False):
        st.plotly_chart(_coverage_figure(entries), use_container_width=True)
        st.caption(
            "Each shaded region is an immutable analysis shard. Compatible shards share cadence, "
            "period, stride, inputs, and model version; stride 1 is the production coverage layer."
        )

    entries = sorted(
        entries,
        key=lambda entry: (
            entry.period,
            entry.product_family,
            entry.stride if entry.stride is not None else -1,
            entry.path.name,
        ),
    )

    labels = [_entry_label(entry) for entry in entries]
    selected_label = st.selectbox("Saved TRIDENT analysis", labels, index=len(labels) - 1)
    entry = entries[labels.index(selected_label)]
    ds = open_analysis_dataset(entry)

    try:
        lat_name = find_coordinate_name(ds, LAT_NAMES)
        lon_name = find_coordinate_name(ds, LON_NAMES)
    except KeyError as exc:
        st.error(str(exc))
        ds.close()
        return
    try:
        time_name = find_coordinate_name(ds, TIME_NAMES)
    except KeyError:
        time_name = None

    time_index = 0
    if time_name and ds.sizes.get(time_name, 0) > 1:
        time_index = st.slider("Time index", 0, ds.sizes[time_name] - 1, 0)

    variables = []
    for name in numeric_data_variables(ds):
        try:
            horizontal_field(
                ds, name, lat_name=lat_name, lon_name=lon_name,
                time_name=time_name, time_index=time_index,
            )
            variables.append(name)
        except ValueError:
            continue
    if not variables:
        st.error("This analysis contains no two-dimensional numeric map variables.")
        ds.close()
        return

    preferred = _default_variable(variables)
    variable = st.selectbox("Map variable", variables, index=variables.index(preferred))
    field = horizontal_field(
        ds, variable, lat_name=lat_name, lon_name=lon_name,
        time_name=time_name, time_index=time_index,
    )
    points = _map_points(field, lat_name, lon_name)
    if points.empty:
        st.warning("The selected map variable contains no finite pixels at this time step.")
        ds.close()
        return

    selection_key = f"trident_map_{entry.path.name}_{variable}_{time_index}"
    state_key = f"trident_selected_pixel::{entry.path}"
    stored = st.session_state.get(state_key)
    fig = _clickable_map(points, variable, stored)

    map_col, selection_col = st.columns([2.25, 1.0], gap="large")
    with map_col:
        try:
            event = st.plotly_chart(
                fig, use_container_width=True, key=selection_key,
                on_select="rerun", selection_mode="points",
            )
            clicked = _extract_click(event)
        except TypeError:
            st.plotly_chart(fig, use_container_width=True, key=selection_key)
            clicked = None
            st.caption(
                "This Streamlit version does not expose Plotly selection events; "
                "use the coordinate controls to the right."
            )

    if clicked is not None:
        st.session_state[state_key] = clicked
        stored = clicked

    default_lat = float(stored[0]) if stored else float(points.iloc[len(points) // 2]["lat"])
    default_lon = float(stored[1]) if stored else float(points.iloc[len(points) // 2]["lon"])
    with selection_col:
        st.markdown("#### Selected location")
        requested_lat = st.number_input(
            "Latitude", value=default_lat, format="%.5f", key=f"lat::{entry.path}"
        )
        requested_lon = st.number_input(
            "Longitude", value=default_lon, format="%.5f", key=f"lon::{entry.path}"
        )
        radius = st.slider(
            "Nearby valid-pixel radius", 0, 30, 12, key=f"radius::{entry.path}"
        )
        map_value = selected_map_value(
            field, lat_name, lon_name, requested_lat, requested_lon
        )
        st.metric(
            f"Nearest {variable}",
            "No data" if not np.isfinite(map_value) else f"{map_value:,.4g}",
        )
        st.caption(
            "Map selection is retained while you switch environmental or NPP layers."
        )

    mapping = infer_variable_mapping(ds)
    missing = [name for name in CAFE_INPUTS if not mapping.get(name)]
    if missing:
        st.warning(
            "The regional map is available, but this analysis cannot yet produce a CAFE profile because these "
            "inputs are absent: " + ", ".join(missing)
        )
        with st.expander("Dataset details"):
            st.write(entry.path)
            st.write(ds)
        ds.close()
        return

    try:
        pixel = extract_nearest_valid_pixel(
            ds, requested_lat, requested_lon,
            {name: mapping[name] for name in CAFE_INPUTS},
            lat_name=lat_name, lon_name=lon_name,
            time_name=time_name, time_index=time_index,
            max_search_radius_cells=radius,
        )
        v = pixel.values
        profile = cafe_profile(
            v["par"], v["chl"], v["mld"], pixel.selected_lat,
            pixel.day_of_year, v["aph443"], v["adg443"],
            v["bbp443"], v["bbp_s"], v["sst"],
        )
    except Exception as exc:
        st.error(f"Could not calculate the selected pixel profile: {exc}")
        ds.close()
        return

    reintegrated = integrate_npp_profile(profile.depth_m, profile.npp_z, delz_m=profile.delz_m)
    interpretation = profile_interpretation(profile, v["mld"])
    peak_depth = float(interpretation["peak_depth_m"])
    selected_distance = float(np.hypot(
        pixel.selected_lat - requested_lat, pixel.selected_lon - requested_lon
    ))

    st.markdown("### Selected-pixel physiology")
    metrics = st.columns(7)
    metrics[0].metric("Latitude", f"{pixel.selected_lat:.4f}°")
    metrics[1].metric("Longitude", f"{pixel.selected_lon:.4f}°")
    metrics[2].metric("CAFE NPP", f"{profile.npp:,.1f}", help="mg C m⁻² d⁻¹")
    metrics[3].metric("Euphotic depth", f"{profile.zeu:.1f} m")
    metrics[4].metric("Peak NPP depth", f"{peak_depth:.1f} m")
    metrics[5].metric("KdPAR", f"{profile.kdpar:.4f} m⁻¹")
    metrics[6].metric("Profile parity", f"{abs(reintegrated-profile.npp):.1e}")

    regional_npp_name = next(
        (name for name in ("cafe_npp", "trident_replay_npp") if name in ds.data_vars),
        None,
    )
    regional_npp = np.nan
    if regional_npp_name is not None:
        try:
            regional_npp_field = horizontal_field(
                ds,
                regional_npp_name,
                lat_name=lat_name,
                lon_name=lon_name,
                time_name=time_name,
                time_index=time_index,
            )
            regional_npp = selected_map_value(
                regional_npp_field,
                lat_name,
                lon_name,
                pixel.selected_lat,
                pixel.selected_lon,
            )
        except (KeyError, ValueError):
            regional_npp = np.nan

    integration_difference = abs(float(reintegrated) - float(profile.npp))
    storage_difference = (
        abs(float(regional_npp) - float(profile.npp))
        if np.isfinite(regional_npp) else np.nan
    )
    integration_pass = bool(np.isclose(reintegrated, profile.npp, rtol=0.0, atol=1e-9))
    storage_pass = bool(
        not np.isfinite(regional_npp)
        or np.isclose(regional_npp, profile.npp, rtol=1e-6, atol=1e-4)
    )
    validation_pass = integration_pass and storage_pass

    st.markdown("#### Numerical validation")
    validation_columns = st.columns(5)
    validation_columns[0].metric(
        "Stored regional NPP",
        f"{regional_npp:,.4f}" if np.isfinite(regional_npp) else "Not sampled",
        help="mg C m⁻² d⁻¹; sparse stride products may not contain this pixel",
    )
    validation_columns[1].metric("Profile NPP", f"{profile.npp:,.4f}")
    validation_columns[2].metric("Reintegrated NPP", f"{reintegrated:,.4f}")
    validation_columns[3].metric(
        "Stored/profile difference",
        f"{storage_difference:.2e}" if np.isfinite(storage_difference) else "—",
    )
    validation_columns[4].metric(
        "Validation",
        "PASS" if validation_pass else "CHECK",
        help="Depth integration uses 1e-9 absolute tolerance; stored float32 output uses 1e-6 relative tolerance.",
    )

    if selected_distance > 0:
        st.caption(
            f"Requested ({requested_lat:.4f}, {requested_lon:.4f}); nearest complete CAFE pixel "
            f"({pixel.selected_lat:.4f}, {pixel.selected_lon:.4f}), coordinate distance {selected_distance:.4f}°."
        )

    with st.expander("Explain this pixel", expanded=True):
        fraction = float(interpretation["fraction_above_mld"])
        st.markdown(
            f"- Integrated production is **{profile.npp:,.1f} mg C m⁻² d⁻¹**.\n"
            f"- Maximum volumetric production occurs near **{peak_depth:.1f} m**.\n"
            f"- Half of integrated production is accumulated above approximately "
            f"**{float(interpretation['median_production_depth_m']):.1f} m**.\n"
            f"- **{fraction:.0%}** of modeled production occurs above the "
            f"**{v['mld']:.1f} m** mixed-layer depth. "
            f"{interpretation['mld_interpretation']}\n"
            f"- The modeled euphotic depth is **{profile.zeu:.1f} m**, with "
            f"KdPAR = **{profile.kdpar:.4f} m⁻¹**."
        )

    with st.expander("Environmental and optical inputs at selected pixel", expanded=False):
        input_units = {
            "par": "mol photons m⁻² d⁻¹",
            "chl": "mg chlorophyll-a m⁻³",
            "mld": "m",
            "aph443": "m⁻¹",
            "adg443": "m⁻¹",
            "bbp443": "m⁻¹",
            "bbp_s": "dimensionless exponent",
            "sst": "°C",
        }
        st.dataframe(pd.DataFrame({
            "CAFE field": list(v),
            "Value": list(v.values()),
            "Unit": [input_units.get(key, "") for key in v],
            "Dataset variable": [pixel.source_variables[key] for key in v],
        }), hide_index=True, use_container_width=True)
        st.write({"day_of_year": pixel.day_of_year, "time": pixel.time_value})

    overview, physiology, daylight, optics, export = st.tabs([
        "NPP & light", "Physiology", "Daylight cycle", "Spectral optics", "Export",
    ])
    with overview:
        left, right = st.columns(2)
        left.plotly_chart(make_npp_depth_figure(profile), use_container_width=True)
        right.plotly_chart(make_par_depth_figure(profile), use_container_width=True)
    with physiology:
        st.caption(
            "Ek and KPUR use conventional physiological irradiance units. "
            "RPUR = 1.3 Ek/KPUR measures the depth-dependent spectral match; "
            "larger RPUR means the available spectrum is better matched to phytoplankton absorption."
        )
        st.plotly_chart(
            make_physiology_depth_figure(profile, mld_m=v["mld"]),
            use_container_width=True,
        )
    with daylight:
        st.plotly_chart(make_npp_time_depth_figure(profile), use_container_width=True)
    with optics:
        st.plotly_chart(make_spectral_optics_figure(profile), use_container_width=True)
    with export:
        profile_df = _profile_frame(profile)
        spectral_df = _spectral_frame(profile)
        export_metadata = {
            "analysis_file": str(entry.path),
            "companion_input_file": (
                str(entry.companion_input_path) if entry.companion_input_path else None
            ),
            "product_family": entry.product_family,
            "temporal_cadence": entry.cadence,
            "period": entry.period,
            "stride": entry.stride,
            "selected_latitude": pixel.selected_lat,
            "selected_longitude": pixel.selected_lon,
            "day_of_year": pixel.day_of_year,
            "time": pixel.time_value,
            "cafe_integrated_npp_mg_C_m2_day": profile.npp,
            "reintegrated_npp_mg_C_m2_day": float(reintegrated),
            "stored_regional_npp_mg_C_m2_day": (
                float(regional_npp) if np.isfinite(regional_npp) else None
            ),
            "validation_pass": validation_pass,
            "mld_m": v["mld"],
            "zeu_m": profile.zeu,
            "kdpar_m-1": profile.kdpar,
            "ek_primary_unit": "umol photons m-2 s-1",
            "kpur_primary_unit": "umol photons m-2 s-1",
            "rpur_definition": "1.3 * Ek / KPUR",
        }
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download depth-profile CSV", profile_df.to_csv(index=False).encode("utf-8"),
            file_name=f"trident_cafe_profile_{pixel.selected_lat:.4f}_{pixel.selected_lon:.4f}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download spectral-optics CSV", spectral_df.to_csv(index=False).encode("utf-8"),
            file_name=f"trident_cafe_optics_{pixel.selected_lat:.4f}_{pixel.selected_lon:.4f}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download analysis metadata JSON",
            json.dumps(export_metadata, indent=2).encode("utf-8"),
            file_name=f"trident_cafe_metadata_{pixel.selected_lat:.4f}_{pixel.selected_lon:.4f}.json",
            mime="application/json",
        )

    with st.expander("Dataset and provenance"):
        st.write({
            "analysis_file": str(entry.path),
            "companion_input_file": str(entry.companion_input_path) if entry.companion_input_path else None,
            "inputs_embedded": entry.has_embedded_cafe_inputs,
            "product_family": entry.product_family,
            "temporal_cadence": entry.cadence,
            "period": entry.period,
            "stride": entry.stride,
            "role": entry.role,
            "region_label": entry.region_label,
            "region_id": entry.region_id,
            "bbox": list(entry.bbox) if entry.bbox is not None else None,
        })
        st.write(ds)
    ds.close()
