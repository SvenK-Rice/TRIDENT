from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
import json
from typing import Any, Iterable

import pandas as pd
import streamlit as st
import xarray as xr

from trident.analysis.storage_layout import (
    bbox_matches_dataset,
    region_identity,
    resolve_native_input_path,
)
from trident.utils.dates import month_starts
from trident.workflows.native_prepare import prepare_native_inputs, run_native_cafe


REQUIRED_INPUTS = (
    "par",
    "chl",
    "mld",
    "aph443",
    "adg443",
    "bbp443",
    "bbp_s",
    "sst",
)

DISPLAY_NAMES = {
    "par": "PAR",
    "chl": "Chlorophyll",
    "mld": "Mixed-layer depth",
    "aph443": "Phytoplankton absorption (442/443 nm)",
    "adg443": "CDOM + detrital absorption (442/443 nm)",
    "bbp443": "Particle backscatter (442/443 nm)",
    "bbp_s": "Backscatter spectral slope",
    "sst": "Sea-surface temperature",
}


def _normalise_report(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalise_report(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_report(item) for item in value]
    return value


def _status_badge(ok: bool, label: str) -> str:
    icon = "✓" if ok else "—"
    return f"{icon} {label}"


def _service_status() -> dict[str, Any]:
    status: dict[str, Any] = {}
    try:
        import earthaccess  # noqa: F401
        status["Earthdata"] = {"available": True, "message": "earthaccess installed"}
    except Exception as exc:  # pragma: no cover - environment dependent
        status["Earthdata"] = {"available": False, "message": str(exc)}

    try:
        import copernicusmarine  # noqa: F401
        status["Copernicus Marine"] = {
            "available": True,
            "message": "copernicusmarine installed",
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        status["Copernicus Marine"] = {"available": False, "message": str(exc)}
    return status


def _month_tags(start: date, end: date) -> list[str]:
    """Return unique calendar-month tags in strict YYYYMM format."""
    return [
        f"{item.year:04d}{item.month:02d}"
        for item in month_starts(start, end)
    ]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _prepared_rows(
    root: Path,
    start: date,
    end: date,
    bbox: tuple[float, float, float, float] | None = None,
    region_name: str | None = None,
    cadence: str = "monthly",
) -> list[dict[str, Any]]:
    report_path = Path(root) / "reports/trident/native_prepare_report.json"
    raw = _read_json(report_path)
    if not isinstance(raw, list):
        raw = []

    wanted = set(_month_tags(start, end))
    region = region_identity(region_name, bbox) if bbox is not None else None
    rows = []
    for item in raw:
        if not isinstance(item, dict) or str(item.get("tag")) not in wanted:
            continue
        if region is not None:
            item_region = item.get("region_id")
            output = item.get("output_file")
            if item_region and str(item_region) != region.region_id:
                continue
            if not item_region and (
                not output or not bbox_matches_dataset(Path(output), region.bbox)
            ):
                continue
        rows.append(item)

    by_tag = {str(item.get("tag")): item for item in rows}
    for tag in sorted(wanted):
        if tag not in by_tag:
            resolved = None
            if region is not None:
                candidate, _legacy = resolve_native_input_path(root, cadence, tag, region)
                if candidate.exists():
                    resolved = candidate
            if resolved is not None:
                with xr.open_dataset(resolved) as ds:
                    present = {name: name for name in REQUIRED_INPUTS if name in ds.data_vars}
                    valid = int(ds.attrs.get("joint_valid_pixels", 0))
                by_tag[tag] = {
                    "tag": tag,
                    "output_file": str(resolved),
                    "variables": present,
                    "missing": [name for name in REQUIRED_INPUTS if name not in present],
                    "valid_pixels": valid,
                    "region_id": region.region_id,
                    "region_label": region.label,
                }
            else:
                by_tag[tag] = {
                    "tag": tag,
                    "output_file": None,
                    "variables": {},
                    "missing": list(REQUIRED_INPUTS),
                    "valid_pixels": 0,
                }
    return [by_tag[tag] for tag in sorted(by_tag)]


def _summary_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    summary = []
    for row in rows:
        missing = [str(item) for item in row.get("missing", [])]
        output = row.get("output_file")
        valid = int(row.get("joint_valid_pixels", row.get("valid_pixels", 0)) or 0)
        summary.append(
            {
                "Month": str(row.get("tag", "")),
                "Status": "Ready" if output and not missing and valid > 0 else "Needs preparation",
                "Joint-valid pixels": valid,
                "Output": Path(output).name if output else "—",
                "Missing": ", ".join(missing) if missing else "None",
            }
        )
    return pd.DataFrame(summary)


def _acquire_month(root: Path, tag: str, bbox: tuple[float, float, float, float]) -> Any:
    try:
        from trident.data_sources.acquisition import acquire_month
    except ImportError as exc:  # pragma: no cover - legacy fallback
        raise RuntimeError(
            "The unified acquisition module is not installed. Install the current Phase 3 data layer."
        ) from exc

    return acquire_month(
        root,
        tag,
        bbox,
        nasa_login_strategy="netrc",
        force=False,
    )


def render_acquisition(
    root: Path,
    start: date,
    end: date,
    bbox: tuple[float, float, float, float],
    temporal: str,
) -> None:
    st.markdown("### 1. Acquire native satellite data")
    st.caption(
        "TRIDENT selects the verified NASA and Copernicus products automatically for the chosen date."
    )

    services = _service_status()
    cols = st.columns(len(services))
    for col, (name, info) in zip(cols, services.items()):
        col.metric(name, "Connected" if info["available"] else "Unavailable")

    tags = _month_tags(start, end)
    if temporal.lower().startswith("8"):
        st.info("The unified Native pipeline currently prepares monthly CAFE inputs. Monthly acquisition will be used.")

    st.write("**Months selected:** " + ", ".join(tags))
    st.write(
        "**Region:** "
        f"{bbox[0]:g} to {bbox[2]:g}° longitude; {bbox[1]:g} to {bbox[3]:g}° latitude"
    )

    ready_to_acquire = all(info["available"] for info in services.values())
    if st.button(
        "Acquire native satellite data",
        type="primary",
        disabled=not ready_to_acquire,
        use_container_width=True,
    ):
        reports = []
        with st.status("Acquiring verified source products…", expanded=True) as status:
            for tag in tags:
                st.write(f"Acquiring {tag}…")
                try:
                    result = _acquire_month(Path(root), tag, tuple(map(float, bbox)))
                    reports.append({"month": tag, "result": _normalise_report(result)})
                    st.success(f"{tag}: acquisition complete")
                except Exception as exc:
                    reports.append({"month": tag, "error": f"{type(exc).__name__}: {exc}"})
                    st.error(f"{tag}: {exc}")
            st.session_state["native_acquisition_report"] = reports
            if all("error" not in item for item in reports):
                status.update(label="Native acquisition complete", state="complete")
            else:
                status.update(label="Native acquisition completed with errors", state="error")

    report = st.session_state.get("native_acquisition_report")
    if isinstance(report, list) and report:
        rows = []
        for item in report:
            result = item.get("result", {})
            rows.append(
                {
                    "Month": item.get("month"),
                    "Status": "Error" if "error" in item else "Complete",
                    "Details": item.get("error", "Source products acquired or reused"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Developer details", expanded=False):
        st.caption("Internal service state, manifests, and acquisition responses.")
        st.json({"services": services, "last_acquisition": report or []})


def _input_checklist(rows: list[dict[str, Any]]) -> dict[str, bool]:
    availability = {name: False for name in REQUIRED_INPUTS}
    for row in rows:
        missing = set(str(item) for item in row.get("missing", []))
        variables = row.get("variables", {})
        for name in REQUIRED_INPUTS:
            if name not in missing and (
                name in variables
                or row.get("output_file")
                or int(row.get("joint_valid_pixels", row.get("valid_pixels", 0)) or 0) > 0
            ):
                availability[name] = True
    return availability


def render_processing(
    root: Path,
    start: date,
    end: date,
    bbox: tuple[float, float, float, float],
    stride: int,
    region_name: str | None = None,
    cadence: str = "monthly",
) -> None:
    st.markdown("### 2. Prepare and run TRIDENT CAFE")
    st.caption(
        "TRIDENT validates all eight CAFE inputs, masks invalid values, and aligns them to one geographic grid before NPP computation."
    )

    prep_col, run_col = st.columns(2)
    if prep_col.button("Prepare Native CAFE inputs", type="primary", use_container_width=True):
        with st.status("Preparing and validating Native CAFE inputs…", expanded=True) as status:
            try:
                report = prepare_native_inputs(
                    root, start, end, bbox,
                    region_name=region_name,
                    cadence=cadence,
                )
                st.session_state["native_prepare_report_current"] = _normalise_report(report)
                status.update(label="Native input preparation complete", state="complete")
            except Exception as exc:
                import traceback

                st.session_state["native_prepare_error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                status.update(label="Native input preparation failed", state="error")
                st.exception(exc)

    rows = _prepared_rows(Path(root), start, end, bbox, region_name, cadence)
    frame = _summary_frame(rows)
    all_ready = bool(len(frame)) and bool((frame["Status"] == "Ready").all())

    if run_col.button(
        "Run TRIDENT Native CAFE",
        disabled=not all_ready,
        use_container_width=True,
    ):
        with st.status("Running regional CAFE…", expanded=True) as status:
            try:
                progress_bar = st.progress(0.0, text="Checking prepared inputs…")
                detail = st.empty()
                metric_columns = st.columns(4)
                month_metric = metric_columns[0].empty()
                pixel_metric = metric_columns[1].empty()
                speed_metric = metric_columns[2].empty()
                eta_metric = metric_columns[3].empty()

                def format_seconds(value: float | None) -> str:
                    if value is None:
                        return "Calculating…"
                    seconds = max(0, int(value))
                    hours, remainder = divmod(seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                def update_progress(update: dict) -> None:
                    fraction = max(0.0, min(1.0, float(update.get("fraction", 0.0))))
                    tag = str(update.get("tag", ""))
                    stage = str(update.get("stage", ""))
                    mode = str(update.get("execution_mode", "starting"))
                    worker_count = int(update.get("workers", 0) or 0)
                    pixel_number = int(update.get("pixel_number", 0) or 0)
                    total_pixels = int(update.get("total_pixels", 0) or 0)
                    failures = int(update.get("failures", 0) or 0)
                    speed = float(update.get("pixels_per_second", 0.0) or 0.0)
                    eta = update.get("eta_seconds")
                    month_number = int(update.get("month_number", 0) or 0)
                    total_months = int(update.get("total_months", 0) or 0)

                    label = "Running validated CAFE"
                    if stage == "serial_fallback":
                        label = "Parallel run unavailable; continuing safely in serial"
                    elif stage == "month_complete" and update.get("resumed"):
                        label = "Existing complete output reused"
                    elif stage == "month_complete":
                        label = "Month complete"

                    progress_bar.progress(fraction, text=f"{label}: {tag}")
                    month_metric.metric("Month", f"{month_number} / {total_months}")
                    pixel_metric.metric(
                        "Pixels",
                        f"{pixel_number:,} / {total_pixels:,}" if total_pixels else "Checking",
                    )
                    speed_metric.metric("Speed", f"{speed:,.1f} px/s" if speed else "—")
                    eta_metric.metric("ETA (month)", format_seconds(eta))
                    detail.caption(
                        f"Mode: {mode}"
                        + (f" ({worker_count} workers)" if worker_count else "")
                        + f"  |  Failures: {failures:,}"
                    )

                report = run_native_cafe(
                    root,
                    start,
                    end,
                    int(stride),
                    progress_callback=update_progress,
                    bbox=bbox,
                    region_name=region_name,
                    cadence=cadence,
                )
                progress_bar.progress(1.0, text="Native CAFE run complete")
                st.session_state["native_cafe_report_current"] = _normalise_report(report)
                status.update(label="Native CAFE complete", state="complete")
            except Exception as exc:
                status.update(label="Native CAFE failed", state="error")
                st.exception(exc)

    availability = _input_checklist(rows)
    st.markdown("#### Input status")
    grid = st.columns(4)
    for index, name in enumerate(REQUIRED_INPUTS):
        grid[index % 4].write(_status_badge(availability[name], DISPLAY_NAMES[name]))

    if len(frame):
        st.markdown("#### Prepared datasets")
        st.dataframe(frame, use_container_width=True, hide_index=True)

    if all_ready:
        total_valid = int(frame["Joint-valid pixels"].sum())
        st.success(
            f"All selected months are ready for CAFE. Joint-valid pixels reported: {total_valid:,}."
        )
    else:
        missing = sorted(
            {
                item
                for row in rows
                for item in row.get("missing", [])
            }
        )
        if missing:
            st.warning(
                "Preparation is still required. Missing inputs: " + ", ".join(missing)
            )
        else:
            st.info("Acquire and prepare the selected months before running CAFE.")

    with st.expander("Developer details", expanded=False):
        st.caption("Raw preparation reports, output paths, source mappings, and model responses.")
        st.json(
            {
                "prepared_report": rows,
                "last_prepare_response": st.session_state.get("native_prepare_report_current"),
                "last_prepare_error": st.session_state.get("native_prepare_error"),
                "last_cafe_response": st.session_state.get("native_cafe_report_current"),
            }
        )
