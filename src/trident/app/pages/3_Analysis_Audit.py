from pathlib import Path

import streamlit as st

from trident.provenance.audit import (
    audit_saved_analyses,
    current_identity,
    write_rerun_plan,
)
from trident.utils.config import ensure_tree, get_data_root


st.set_page_config(
    page_title="TRIDENT Analysis Audit",
    layout="wide",
)
st.title("Saved Analysis Audit")
st.caption(
    "Identify outputs created with older, unknown, or development versions "
    "of TRIDENT before deciding whether to rerun them."
)

data_root = get_data_root()
ensure_tree(data_root)
repo_root = Path(__file__).resolve().parents[4]
identity = current_identity(repo_root)

with st.sidebar:
    st.header("Current code")
    st.code(f"Version: {identity.version}")
    st.code(
        "Commit: "
        + (
            identity.commit[:12]
            if identity.commit
            else "unavailable"
        )
    )
    st.code(f"Branch: {identity.branch or 'unavailable'}")
    if identity.dirty:
        st.warning("The Git working tree currently has uncommitted changes.")

if st.button("Scan saved analyses", type="primary"):
    with st.spinner("Reading NetCDF metadata..."):
        st.session_state["analysis_audit"] = audit_saved_analyses(
            data_root,
            repo_root,
        )

audit = st.session_state.get("analysis_audit")

if audit is None:
    st.info("Click “Scan saved analyses” to begin.")
    st.stop()

if audit.empty:
    st.info("No processed NetCDF files were found.")
    st.stop()

summary = (
    audit.groupby(["status", "recommendation"])
    .size()
    .reset_index(name="files")
)

st.subheader("Summary")
st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
)

statuses = sorted(audit["status"].dropna().unique())
selected_statuses = st.multiselect(
    "Show statuses",
    statuses,
    default=statuses,
)

filtered = audit[audit["status"].isin(selected_statuses)]

st.subheader("Saved outputs")
st.dataframe(
    filtered[
        [
            "status",
            "recommendation",
            "product_type",
            "filename",
            "reason",
            "saved_version",
            "saved_commit",
            "workflow",
            "model_version",
            "modified_utc",
            "size_mb",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "size_mb": st.column_config.NumberColumn(format="%.2f MB"),
    },
)

st.markdown(
    """
- **current**: provenance matches the current checkout.
- **stale**: an older version or Git commit was recorded.
- **legacy_unknown**: the file predates provenance tracking.
- **development**: the file was produced with uncommitted code changes.
- **unreadable**: the file could not be opened.
"""
)

if st.button("Create rerun review plan"):
    target = (
        Path(data_root)
        / "reports"
        / "trident"
        / "analysis_rerun_plan.json"
    )
    write_rerun_plan(audit, target)
    st.success(f"Saved: {target}")
    st.download_button(
        "Download rerun plan",
        target.read_bytes(),
        file_name=target.name,
        mime="application/json",
    )

with st.expander("Exact paths and complete metadata"):
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )
