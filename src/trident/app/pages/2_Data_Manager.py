from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from trident.data_manager.catalog import (
    delete_files,
    discover_managed_files,
    stale_candidates,
    summary_table,
    write_review_plan,
)
from trident.utils.config import ensure_tree, get_data_root


st.set_page_config(page_title="TRIDENT Data Manager", layout="wide")
st.title("TRIDENT Data Manager")
st.caption(
    "Browse downloaded source data, processed analyses, and reports in one place."
)

data_root = get_data_root()
ensure_tree(data_root)

with st.sidebar:
    st.header("Data root")
    st.code(str(data_root))
    st.warning(
        "Delete actions permanently remove selected local files. "
        "They do not affect GitHub."
    )

if st.button("Refresh catalog", type="primary"):
    st.session_state["data_manager_catalog"] = discover_managed_files(data_root)

catalog = st.session_state.get("data_manager_catalog")
if catalog is None:
    catalog = discover_managed_files(data_root)
    st.session_state["data_manager_catalog"] = catalog

if catalog.empty:
    st.info("No managed TRIDENT files were found.")
    st.stop()

summary_tab, files_tab, review_tab = st.tabs(
    ["Summary", "Files", "Review old analyses"]
)

with summary_tab:
    st.subheader("Storage summary")
    summary = summary_table(catalog)
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "size_mb": st.column_config.NumberColumn(format="%.2f MB"),
        },
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Files", f"{len(catalog):,}")
    c2.metric("Storage", f"{catalog['size_mb'].sum():.1f} MB")
    c3.metric(
        "Needs review",
        f"{len(stale_candidates(catalog)):,}",
    )

with files_tab:
    st.subheader("Managed files")

    categories = sorted(catalog["category"].dropna().unique())
    sources = sorted(catalog["source"].dropna().unique())
    products = sorted(catalog["product"].dropna().unique())

    c1, c2, c3 = st.columns(3)
    selected_categories = c1.multiselect(
        "Category",
        categories,
        default=categories,
    )
    selected_sources = c2.multiselect(
        "Source",
        sources,
        default=sources,
    )
    selected_products = c3.multiselect(
        "Product",
        products,
        default=products,
    )

    filtered = catalog[
        catalog["category"].isin(selected_categories)
        & catalog["source"].isin(selected_sources)
        & catalog["product"].isin(selected_products)
    ].copy()

    display_columns = [
        "category",
        "source",
        "product",
        "tag",
        "name",
        "size_mb",
        "modified_utc",
        "readable",
        "provenance_status",
        "recommendation",
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "size_mb": st.column_config.NumberColumn(format="%.2f MB"),
        },
    )

    labels = {
        f"{row.source} | {row.product} | {row.name}": row.path
        for row in filtered.itertuples(index=False)
    }
    selected_labels = st.multiselect(
        "Select files for an action",
        options=list(labels),
    )
    selected_paths = [labels[label] for label in selected_labels]

    if selected_paths:
        action = st.radio(
            "Action",
            ["Create review plan", "Delete selected files"],
            horizontal=True,
        )

        if action == "Create review plan":
            if st.button("Create review plan"):
                target = (
                    Path(data_root)
                    / "reports"
                    / "trident"
                    / "data_manager_review_plan.json"
                )
                write_review_plan(catalog, target, selected_paths)
                st.success(f"Saved: {target}")
                st.download_button(
                    "Download review plan",
                    target.read_bytes(),
                    file_name=target.name,
                    mime="application/json",
                )

        else:
            confirm = st.checkbox(
                "I understand these local files will be permanently deleted."
            )
            if st.button(
                "Delete selected files",
                type="primary",
                disabled=not confirm,
            ):
                result = delete_files(selected_paths)
                st.json(result)
                st.session_state["data_manager_catalog"] = (
                    discover_managed_files(data_root)
                )
                st.rerun()

with review_tab:
    st.subheader("Analyses that may need review")
    review = stale_candidates(catalog)

    if review.empty:
        st.success("No legacy, development, or unreadable analyses were found.")
    else:
        st.dataframe(
            review[
                [
                    "provenance_status",
                    "recommendation",
                    "source",
                    "product",
                    "tag",
                    "name",
                    "reason",
                    "path",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
- **legacy_unknown**: created before provenance metadata was embedded.
- **development**: created while the Git working tree had uncommitted changes.
- **unreadable**: NetCDF could not be opened.
"""
        )
