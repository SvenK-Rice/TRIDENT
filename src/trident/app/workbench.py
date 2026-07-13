from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from trident.utils.config import get_data_root, set_data_root, ensure_tree
from trident.utils.dates import month_starts, month_code
from trident.data.osu.archive import load_inventory
from trident.data.esa.occi import ESA_PRODUCTS, esa_status, inspect_copernicus_dataset
from trident.workflows.osu_replay import run_range
from trident.workflows.series import (
    discover_validation_runs,
    validation_series,
    standard_series,
)
from trident.workflows.standard import production_status, standard_cache_status, DEFAULT_NASA_COLLECTIONS, DEFAULT_CMEMS_DATASET, DEFAULT_CMEMS_VARIABLES
from trident.workflows.native import NativeWorkflowRequest, run_native_download
from trident.workflows.native_prepare import prepare_native_inputs, run_native_cafe
from trident.workflows.catalog import FAMILIES, group_by_family
from trident.workflows.merge import build_source_manifest
from trident.grids.align import mask_feasible
from trident.viz.maps import make_geo_heatmap as make_geo_heatmap_v2
from trident.app.integrated_explorer import render_integrated_explorer
from trident.app.native_scientist import render_acquisition, render_processing

PRESETS = {
    "California Current": (-132, 28, -116, 45),
    "CCE-LTER / CalCOFI broad": (-130, 28, -116, 45),
    "Gulf of Mexico": (-98, 18, -80, 31),
    "North Atlantic": (-80, 20, -10, 60),
    "Equatorial Pacific": (-170, -10, -80, 10),
    "Southern Ocean": (-180, -75, 180, -45),
    "Global": (-180, -90, 180, 90),
    "Custom": (-132, 28, -116, 45),
}

WEST_COAST = [
    (-124.8,48.7),(-124.3,47.9),(-124.1,47.1),(-123.8,46.3),(-124.0,45.6),
    (-123.9,45.0),(-123.8,44.5),(-124.1,43.9),(-124.3,43.2),(-124.4,42.5),
    (-124.2,41.9),(-124.1,41.3),(-124.0,40.7),(-123.8,40.0),(-123.5,39.4),
    (-123.1,38.9),(-122.7,38.3),(-122.5,37.8),(-122.3,37.4),(-122.0,37.0),
    (-121.8,36.6),(-121.6,36.0),(-121.2,35.5),(-120.8,35.1),(-120.4,34.7),
    (-119.9,34.4),(-119.5,34.1),(-119.0,33.8),(-118.5,33.5),(-118.0,33.0),
    (-117.6,32.7),(-117.1,32.5),(-116.6,32.2),(-116.0,31.9),(-115.4,31.5),
    (-114.8,31.0),(-114.5,30.5),(-114.2,29.9),(-114.0,29.4),
]


def _fmt_lon(x):
    return f"{abs(float(x)):g}°" + ("W" if x < 0 else "E")


def _fmt_lat(y):
    return f"{abs(float(y)):g}°" + ("S" if y < 0 else "N")


def _limits(var, arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    v = var.lower()
    if "npp" in v or "cafe" in v or v == "pp":
        return 0, max(500, min(5000, float(np.nanpercentile(finite, 99))))
    if v in {"chl", "chlor_a"}:
        return 0, max(1, min(50, float(np.nanpercentile(finite, 99))))
    if v == "sst":
        return max(-2, float(np.nanpercentile(finite, 1))), min(35, float(np.nanpercentile(finite, 99)))
    if v == "par":
        return 0, max(10, min(80, float(np.nanpercentile(finite, 99))))
    if v == "mld":
        return 0, max(50, min(500, float(np.nanpercentile(finite, 99))))
    return float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))


def _overlay_land(fig, lon, lat):
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    if lon.size == 0 or lat.size == 0:
        return
    west, east = float(np.nanmin(lon)), float(np.nanmax(lon))
    south, north = float(np.nanmin(lat)), float(np.nanmax(lat))
    if west < -113 and east > -126 and south < 49 and north > 28:
        coast = [(x, y) for x, y in WEST_COAST if south - 3 <= y <= north + 3]
        if len(coast) >= 2:
            cx, cy = zip(*coast)
            fig.add_trace(go.Scatter(
                x=list(cx) + [east, east, cx[0]], y=list(cy) + [cy[-1], cy[0], cy[0]],
                mode="lines", fill="toself", fillcolor="rgba(150,105,55,0.88)",
                line=dict(color="rgba(80,50,25,0.5)", width=0.5), hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(x=list(cx), y=list(cy), mode="lines", line=dict(color="black", width=2.5), hoverinfo="skip", showlegend=False))


def make_geo_heatmap(ds, var, title=None):
    arr = mask_feasible(var, ds[var].values)
    lat = ds["lat"].values.astype(float) if "lat" in ds.coords else np.arange(arr.shape[0], dtype=float)
    lon = ds["lon"].values.astype(float) if "lon" in ds.coords else np.arange(arr.shape[1], dtype=float)
    zmin, zmax = _limits(var, arr)
    fig = go.Figure(go.Heatmap(
        x=lon, y=lat, z=arr, colorscale="Viridis", zmin=zmin, zmax=zmax,
        connectgaps=False,
        colorbar=dict(title="mg C m⁻² d⁻¹" if ("npp" in var.lower() or "cafe" in var.lower()) else var),
        hovertemplate="Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>" + var + ": %{z:.3g}<extra></extra>",
    ))
    _overlay_land(fig, lon, lat)
    xt = np.linspace(float(np.nanmin(lon)), float(np.nanmax(lon)), 7)
    yt = np.linspace(float(np.nanmin(lat)), float(np.nanmax(lat)), 7)
    fig.update_xaxes(title="Longitude", tickmode="array", tickvals=xt, ticktext=[_fmt_lon(x) for x in xt], showgrid=True)
    fig.update_yaxes(title="Latitude", tickmode="array", tickvals=yt, ticktext=[_fmt_lat(y) for y in yt], showgrid=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(title=title or var, height=650, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white")
    return fig


def cache_status(root, start, end):
    inv = load_inventory(root)
    rows = []
    for d in month_starts(start, end):
        code = month_code(d)
        prods = sorted({r.get("product") for r in inv if r.get("code") == code and r.get("local_path") and Path(r["local_path"]).exists()})
        rows.append({"month": f"{d.year}-{d.month:02d}", "code": code, "cached_products": len(prods), "products": ", ".join(prods)})
    return pd.DataFrame(rows)


def _plot_validation(df, native=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["osu_archived_mean"], mode="lines+markers", name="OSU archived CAFE NPP"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["trident_replay_mean"], mode="lines+markers", name="TRIDENT replay from OSU inputs"))
    if native is not None and len(native):
        fig.add_trace(go.Scatter(x=native["date"], y=native["trident_standard_mean"], mode="lines+markers", name="TRIDENT Native"))
    fig.update_layout(title="Regional NPP seasonal cycle", xaxis_title="Date", yaxis_title="NPP (mg C m⁻² d⁻¹)", height=430, legend=dict(orientation="h", y=1.05))
    return fig


def _region_preview(bbox):
    west, south, east, north = bbox
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=[west, west, east, east, west], lat=[south, north, north, south, south],
        mode="lines", line=dict(color="red", width=3), fill="toself", fillcolor="rgba(255,0,0,0.10)", name="Selected region",
    ))
    fig.update_geos(showland=True, landcolor="rgb(170,125,75)", showocean=True, oceancolor="rgb(215,235,245)", showcoastlines=True, coastlinecolor="black", projection_type="natural earth")
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
    return fig


def _month_label(tag: str) -> str:
    try:
        return pd.Timestamp(f"{str(tag)[:4]}-{str(tag)[4:6]}-01").strftime("%B %Y")
    except Exception:
        return str(tag)


def _service_state(value) -> tuple[str, str]:
    if isinstance(value, dict):
        available = value.get("available")
        authenticated = value.get("authenticated")
        if authenticated is True:
            return "Connected", "success"
        if available is True:
            return "Available", "success"
        message = value.get("message") or value.get("error")
        return str(message or "Unavailable"), "error"
    if value:
        return "Available", "success"
    return "Unavailable", "error"


def _status_badge(label: str, value: str, state: str = "neutral") -> None:
    icons = {"success": "✓", "warning": "!", "error": "×", "neutral": "•"}
    colors = {
        "success": "#15803d",
        "warning": "#a16207",
        "error": "#b91c1c",
        "neutral": "#475569",
    }
    color = colors.get(state, colors["neutral"])
    icon = icons.get(state, icons["neutral"])
    st.markdown(
        f"<div style='padding:0.6rem 0.75rem;border:1px solid #e2e8f0;"
        f"border-radius:0.55rem;margin-bottom:0.35rem'>"
        f"<span style='color:{color};font-weight:700'>{icon}</span> "
        f"<strong>{label}</strong><br>"
        f"<span style='color:#64748b'>{value}</span></div>",
        unsafe_allow_html=True,
    )


def _render_acquisition_summary(reports: list[dict]) -> None:
    if not reports:
        return
    st.markdown("#### Acquisition summary")
    for report in reports:
        month = _month_label(report.get("month") or report.get("tag") or "")
        items = report.get("items") or report.get("results") or []
        if isinstance(items, dict):
            items = [dict(name=k, **(v if isinstance(v, dict) else {"status": v})) for k, v in items.items()]
        ok = 0
        total = 0
        for item in items:
            total += 1
            if str(item.get("status", "")).lower() in {"ok", "reused", "available", "downloaded"}:
                ok += 1
        if total:
            state = "success" if ok == total else "warning"
            _status_badge(month, f"{ok} of {total} required source groups available", state)
        else:
            manifest = report.get("manifest") or report.get("manifest_file")
            _status_badge(month, "Acquisition completed" + (f" · {manifest}" if manifest else ""), "success")


def _render_prepare_summary(reports: list[dict]) -> None:
    if not reports:
        st.info("No preparation report is available yet.")
        return
    st.markdown("#### Prepared-input status")
    for report in reports:
        tag = str(report.get("tag", ""))
        missing = list(report.get("missing") or [])
        output = report.get("output_file")
        valid = int(report.get("valid_pixels") or report.get("joint_valid_pixels") or 0)
        shape = report.get("shape")
        if output and not missing and valid > 0:
            details = f"Ready · {valid:,} jointly valid pixels"
            if shape:
                details += f" · grid {shape[0]} × {shape[1]}"
            _status_badge(_month_label(tag), details, "success")
            st.caption(Path(output).name)
        elif missing:
            _status_badge(
                _month_label(tag),
                "Missing: " + ", ".join(missing),
                "warning",
            )
        else:
            _status_badge(_month_label(tag), "Preparation required", "warning")


def _render_cafe_summary(reports: list[dict]) -> None:
    if not reports:
        return
    st.markdown("#### Native CAFE results")
    for report in reports:
        tag = str(report.get("tag", ""))
        status = str(report.get("status", "")).lower()
        output = report.get("output_file")
        success = report.get("success")
        if output or status in {"calculated", "complete", "ok", "reused"}:
            detail = "Regional CAFE product ready"
            if success is not None:
                detail += f" · {int(success):,} calculated pixels"
            _status_badge(_month_label(tag), detail, "success")
            if output:
                st.caption(Path(output).name)
        else:
            _status_badge(_month_label(tag), status.replace("_", " ") or "Not run", "warning")


def _phase3_acquire(root, start, end, bbox, force: bool = False) -> list[dict]:
    """Use the unified Phase 3 acquisition engine when installed."""
    try:
        from trident.data_sources.acquisition import acquire_month
    except ImportError as exc:
        raise RuntimeError(
            "The unified Phase 3 acquisition engine is not installed. "
            "Install the Phase 3 pipeline update before acquiring data."
        ) from exc

    reports: list[dict] = []
    for month in month_starts(start, end):
        tag = pd.Timestamp(month).strftime("%Y%m")
        result = acquire_month(
            root,
            tag,
            tuple(map(float, bbox)),
            nasa_login_strategy="netrc",
            force=bool(force),
        )
        if isinstance(result, dict):
            reports.append(result)
        else:
            reports.append({"month": tag, "status": "complete", "result": str(result)})
    return reports



def _native_download_controls(root, start, end, bbox, temporal):
    """Render the scientist-facing Native acquisition workflow."""
    render_acquisition(root, start, end, bbox, temporal)

def _native_processing_controls(root, start, end, bbox, stride, region_name, cadence):
    """Render prepared-input status and regional CAFE controls."""
    render_processing(
        root, start, end, bbox, stride,
        region_name=region_name,
        cadence=cadence,
    )

def _explore(root):
    """Explore existing regional products and drill into pixel physiology."""
    render_integrated_explorer(root)


def app():
    st.set_page_config(page_title="TRIDENT NPP Workbench", layout="wide")
    st.title("TRIDENT NPP Workbench")
    st.caption("Validated OSU CAFE replay plus independent NASA, ESA, and Copernicus workflows.")

    root = get_data_root()
    ensure_tree(root)

    with st.sidebar:
        st.header("Project")
        new_root = st.text_input("Data folder / external drive", value=str(root))
        if st.button("Use this data folder"):
            root = set_data_root(new_root)
            ensure_tree(root)
            st.success(f"Data root set to {root}")
        st.divider()
        region = st.selectbox("Region preset", list(PRESETS))
        west, south, east, north = PRESETS[region]
        c1, c2 = st.columns(2)
        with c1:
            west = st.number_input("West", value=float(west))
            south = st.number_input("South", value=float(south))
        with c2:
            east = st.number_input("East", value=float(east))
            north = st.number_input("North", value=float(north))
        bbox = (west, south, east, north)
        start = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
        end = st.date_input("End date", value=pd.to_datetime("2023-03-31"))
        temporal = st.selectbox("Temporal resolution", ["Monthly", "8-day"])
        stride = st.select_slider("Computation stride", [1, 2, 5, 10, 20], value=10)

    overview, validation, native, explore, reports_tab = st.tabs(["Overview", "OSU Validation", "TRIDENT Native", "Explore", "Reports"])

    with overview:
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.plotly_chart(_region_preview(bbox), use_container_width=True)
        with c2:
            st.subheader("Selected analysis")
            st.json({"data_root": str(root), "bbox": list(map(float, bbox)), "start": str(start), "end": str(end), "temporal_resolution": temporal, "stride": int(stride)})
            st.markdown("**OSU Validation** reproduces CAFE from archived OSU inputs and compares it with the direct OSU NPP product.\n\n**TRIDENT Native** downloads and harmonizes NASA/ESA ocean colour with Copernicus SST and MLD, then runs the same validated CAFE model.")

    with validation:
        st.subheader("OSU replay / validation")
        st.dataframe(cache_status(root, start, end), use_container_width=True)

        if st.button(
            "Run OSU replay and compare with archived OSU NPP",
            type="primary",
        ):
            with st.status(
                "Running OSU replay/validation...",
                expanded=True,
            ) as status:
                result = run_range(
                    root,
                    start,
                    end,
                    bbox,
                    stride=int(stride),
                )
                st.json(result)
                status.update(
                    label="OSU validation complete",
                    state="complete",
                )

            # Ensure the next render discovers only files that now exist.
            st.cache_data.clear()
            st.session_state["validation_plot_scope"] = (
                "Selected date range"
            )
            st.rerun()

        st.markdown("### Regional NPP comparison")
        st.caption(
            "The plot reads only completed comparison NetCDF files. "
            "It never combines unrelated replay outputs or old report tables."
        )

        completed = discover_validation_runs(
            root,
            stride=int(stride),
        )

        scope_options = [
            "Selected date range",
            "Choose specific completed runs",
            "All completed runs at this stride",
        ]
        current_scope = st.session_state.get(
            "validation_plot_scope",
            "Selected date range",
        )
        if current_scope not in scope_options:
            current_scope = "Selected date range"

        plot_scope = st.radio(
            "Data shown in the plot",
            scope_options,
            index=scope_options.index(current_scope),
            horizontal=True,
            key="validation_plot_scope",
        )

        selected_paths = None
        plot_start = None
        plot_end = None

        if plot_scope == "Selected date range":
            plot_start = start
            plot_end = end
            filtered = completed[
                (completed["date"] >= pd.Timestamp(start))
                & (
                    completed["date"]
                    <= pd.Timestamp(end) + pd.offsets.MonthEnd(0)
                )
            ]
            st.caption(
                f"Showing completed stride-{int(stride)} runs from "
                f"{start} through {end}: {len(filtered)} dataset(s)."
            )

        elif plot_scope == "Choose specific completed runs":
            labels = {
                (
                    f"{row.date:%Y-%m} | stride {row.stride} | "
                    f"{row.file}"
                ): row.path
                for row in completed.itertuples(index=False)
            }
            default_labels = list(labels)[-1:] if labels else []
            chosen_labels = st.multiselect(
                "Completed validation datasets",
                options=list(labels),
                default=default_labels,
            )
            selected_paths = [
                labels[label] for label in chosen_labels
            ]

        else:
            st.caption(
                f"Showing all {len(completed)} completed "
                f"stride-{int(stride)} comparison dataset(s)."
            )

        if st.button("Refresh completed-run list"):
            st.cache_data.clear()
            st.rerun()

        vdf = validation_series(
            root,
            bbox,
            stride=int(stride),
            selected_files=selected_paths,
            start=plot_start,
            end=plot_end,
        )
        sdf = standard_series(
            root,
            bbox,
            start=plot_start,
            end=plot_end,
        )

        if len(vdf):
            st.plotly_chart(
                _plot_validation(vdf, sdf),
                use_container_width=True,
            )

            display_columns = [
                "tag",
                "stride",
                "trident_replay_mean",
                "osu_archived_mean",
                "bias",
                "percent_bias",
                "rmse",
                "valid_pixels",
                "file",
            ]
            st.dataframe(
                vdf[display_columns],
                use_container_width=True,
            )

            with st.expander("Exact datasets used for this plot"):
                for path in vdf["path"]:
                    st.code(path)
        else:
            st.info(
                "No completed validation comparison file matches the "
                "selected plot scope. Run the requested month(s), or choose "
                "another completed dataset."
            )

    with native:
        _native_download_controls(root, start, end, bbox, temporal)
        st.divider()
        # The current Native preparation pipeline is monthly; the cadence is
        # explicit in storage so future 8-day shards remain isolated.
        native_cadence = "monthly"
        _native_processing_controls(
            root, start, end, bbox, stride, region, native_cadence
        )

    with explore:
        _explore(root)

    with reports_tab:
        report_files = sorted((Path(root) / "reports/trident").glob("*.json"))
        if not report_files:
            st.info("No reports yet.")
        else:
            rp = st.selectbox("Report", [r.name for r in report_files], index=len(report_files)-1)
            st.json(json.loads((Path(root) / "reports/trident" / rp).read_text()))


def main():
    app()


if __name__ == "__main__":
    main()
