import pandas as pd
import plotly.express as px
import streamlit as st

from trident.analysis.explorer import (
    classify_variables,
    discover_processed_datasets,
    extract_depth_profile,
    extract_pixel_table,
    integrated_profile,
    monthly_series,
    open_loaded,
)
from trident.utils.config import ensure_tree, get_data_root
from trident.viz.maps import make_geo_heatmap


st.set_page_config(
    page_title="TRIDENT Diagnostics Explorer",
    layout="wide",
)
st.title("Diagnostics Explorer")
st.caption(
    "Inspect saved maps, point values, monthly time series, and depth profiles."
)

data_root = get_data_root()
ensure_tree(data_root)
entries = discover_processed_datasets(data_root)

if not entries:
    st.info("No processed NetCDF datasets were found.")
    st.stop()

labels = {
    f"{entry.product_type} | {entry.tag or 'no date'} | {entry.path.name}": entry
    for entry in entries
}
entry = labels[st.selectbox("Dataset", list(labels))]
dataset = open_loaded(entry.path)
classes = classify_variables(dataset)

st.code(str(entry.path))

map_tab, point_tab, time_tab, profile_tab, variables_tab = st.tabs(
    ["Map", "Point", "Time series", "Vertical profile", "Variables"]
)

with map_tab:
    if classes["maps_2d"]:
        variable = st.selectbox("Map variable", classes["maps_2d"])
        mode = st.radio(
            "Display",
            ["Raw calculated pixels", "Continuous display interpolation"],
            horizontal=True,
        )

        st.plotly_chart(
            make_geo_heatmap(
                dataset,
                variable,
                f"{variable} — {entry.tag or entry.path.name}",
                display_mode=(
                    "nearest_display"
                    if mode == "Continuous display interpolation"
                    else "raw"
                ),
            ),
            use_container_width=True,
        )
    else:
        st.info("No 2-D latitude/longitude variables are present.")

with point_tab:
    if "lat" in dataset.coords and "lon" in dataset.coords:
        c1, c2 = st.columns(2)
        latitude = c1.number_input("Latitude", value=36.0)
        longitude = c2.number_input("Longitude", value=-124.0)

        table = extract_pixel_table(
            dataset,
            latitude,
            longitude,
        )
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("This dataset has no latitude/longitude grid.")

with time_tab:
    variables = sorted(
        {variable for item in entries for variable in item.variables}
    )
    variable = st.selectbox("Variable across saved files", variables)

    use_point = st.checkbox(
        "Use a selected point instead of the regional mean"
    )
    latitude = longitude = None

    if use_point:
        c1, c2 = st.columns(2)
        latitude = c1.number_input(
            "Time-series latitude",
            value=36.0,
        )
        longitude = c2.number_input(
            "Time-series longitude",
            value=-124.0,
        )

    series = monthly_series(
        entries,
        variable,
        latitude=latitude,
        longitude=longitude,
    )

    if series.empty:
        st.info("No compatible dated files contain this variable.")
    else:
        figure = px.line(
            series,
            x="date",
            y="value",
            color="product_type",
            markers=True,
            hover_data=["file"],
            title=f"{variable} across saved analyses",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(
            series,
            use_container_width=True,
            hide_index=True,
        )

with profile_tab:
    profile_variables = (
        classes["profiles_1d"] + classes["time_depth"]
    )

    if not profile_variables:
        st.info(
            "No depth-resolved variables are present yet. "
            "When NPP(z), PAR(z), or other profiles are saved, "
            "they will appear here automatically."
        )
    else:
        variable = st.selectbox(
            "Depth-resolved variable",
            profile_variables,
        )

        latitude = longitude = None
        if {"lat", "lon"}.issubset(dataset[variable].dims):
            c1, c2 = st.columns(2)
            latitude = c1.number_input(
                "Profile latitude",
                value=36.0,
            )
            longitude = c2.number_input(
                "Profile longitude",
                value=-124.0,
            )

        time_index = 0
        time_dim = next(
            (
                name
                for name in ("time", "date")
                if name in dataset[variable].dims
            ),
            None,
        )
        if time_dim is not None:
            time_index = st.slider(
                "Time index",
                min_value=0,
                max_value=int(dataset.sizes[time_dim]) - 1,
                value=0,
            )

        try:
            profile = extract_depth_profile(
                dataset,
                variable,
                latitude=latitude,
                longitude=longitude,
                time_index=time_index,
            )
            figure = px.line(
                profile,
                x="value",
                y="depth",
                title=f"{variable} vertical profile",
            )
            figure.update_yaxes(autorange="reversed")
            st.plotly_chart(figure, use_container_width=True)
            st.metric(
                "Depth integral",
                f"{integrated_profile(profile):.4g}",
            )
            st.dataframe(
                profile,
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

with variables_tab:
    st.subheader("Variables present in this file")
    st.json(classes)

    expected = [
        "chl",
        "par",
        "sst",
        "mld",
        "aph443",
        "adg443",
        "bbp443",
        "bbp_s",
        "cafe_npp",
        "zeu",
        "kdpar",
        "npp_z",
        "par_z",
        "alpha",
        "pmax",
        "ek",
        "quantum_yield",
        "light_utilization",
    ]

    st.dataframe(
        pd.DataFrame(
            {
                "variable": expected,
                "available": [
                    name in dataset.data_vars
                    for name in expected
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
