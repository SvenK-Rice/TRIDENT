from __future__ import annotations

from collections import Counter
from datetime import date
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import xarray as xr
from tqdm import tqdm

from trident.data.osu.archive import ensure_product_month
from trident.data.osu.replay import build_inputs, read_hdf_array, subset_bbox
from trident.grids.align import mask_feasible, resize_nearest_or_block
from trident.models.cafe import cafe_pixel
from trident.utils.dates import as_date, month_starts


MODEL_INPUTS = (
    "par",
    "chl",
    "mld",
    "aph443",
    "adg443",
    "bbp443",
    "bbp_s",
    "sst",
)

_REPLAY_NAME = re.compile(
    r"trident_osu_input_replay_cafe_npp_(?P<tag>\d{6})_stride(?P<stride>\d+)\.nc$"
)
_COMPARISON_NAME = re.compile(
    r"trident_osu_replay_vs_archived_comparison_(?P<tag>\d{6})_stride(?P<stride>\d+)\.nc$"
)


def _tag(date_obj: Any) -> str:
    d = as_date(date_obj)
    return f"{d.year}{d.month:02d}"


def _midmonth_day_of_year(date_obj: Any) -> int:
    d = as_date(date_obj)
    return date(d.year, d.month, 15).timetuple().tm_yday


def _bbox_string(bbox) -> str:
    return ",".join(f"{float(value):.8f}" for value in bbox)


def _atomic_to_netcdf(dataset: xr.Dataset, target: Path) -> None:
    """Write to a temporary file, close it, then atomically replace target."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp.nc",
        dir=str(target.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)

    try:
        dataset.to_netcdf(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _filename_matches(path: Path, tag: str, stride: int, role: str) -> bool:
    pattern = _REPLAY_NAME if role == "trident_replay" else _COMPARISON_NAME
    match = pattern.match(path.name)
    if match is None:
        return False
    return (
        match.group("tag") == str(tag)
        and int(match.group("stride")) == int(stride)
    )


def _coordinate_extent_matches(
    dataset: xr.Dataset,
    bbox,
    *,
    tolerance_factor: float = 1.25,
) -> bool:
    """Accept cell-center coordinates that lie within one grid cell of bbox."""
    if "lat" not in dataset.coords or "lon" not in dataset.coords:
        return False

    lat = np.asarray(dataset["lat"].values, dtype=float)
    lon = np.asarray(dataset["lon"].values, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
        return False

    west, south, east, north = map(float, bbox)
    dlat = float(np.nanmedian(np.abs(np.diff(np.sort(lat)))))
    dlon = float(np.nanmedian(np.abs(np.diff(np.sort(lon)))))
    lat_tol = max(dlat * tolerance_factor, 1e-6)
    lon_tol = max(dlon * tolerance_factor, 1e-6)

    return (
        abs(float(np.nanmin(lon)) - west) <= lon_tol
        and abs(float(np.nanmax(lon)) - east) <= lon_tol
        and abs(float(np.nanmin(lat)) - south) <= lat_tol
        and abs(float(np.nanmax(lat)) - north) <= lat_tol
    )


def _existing_output_is_usable(
    path: Path,
    *,
    tag: str,
    stride: int,
    bbox,
    role: str,
) -> bool:
    """Validate both new and legacy outputs without modifying them.

    Legacy files may lack tag/stride/bbox attributes. In that case TRIDENT
    verifies the filename, required variables, dimensions, and coordinate
    extent. This allows previously completed calculations to be reused.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if not _filename_matches(path, tag, stride, role):
        return False

    required_variables = (
        {"cafe_npp", "zeu", "kdpar"}
        if role == "trident_replay"
        else {"trident_replay_npp", "osu_archived_npp", "difference"}
    )

    try:
        with xr.open_dataset(path) as opened:
            if not required_variables.issubset(set(opened.data_vars)):
                return False
            if not _coordinate_extent_matches(opened, bbox):
                return False

            attrs = dict(opened.attrs)
            attr_tag = attrs.get("tag")
            attr_stride = attrs.get("stride")
            attr_bbox = attrs.get("bbox")
            attr_role = attrs.get("product_role")

            if attr_tag is not None and str(attr_tag) != str(tag):
                return False
            if attr_stride is not None and int(attr_stride) != int(stride):
                return False
            if attr_bbox is not None and str(attr_bbox) != _bbox_string(bbox):
                return False
            if attr_role is not None and str(attr_role) not in {
                role,
                "TRIDENT replay: CAFE recalculated from archived OSU inputs",
                "validation_comparison",
            }:
                return False

        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _candidate_mask(dataset: xr.Dataset) -> np.ndarray:
    missing = [name for name in MODEL_INPUTS if name not in dataset]
    if missing:
        raise KeyError(f"Missing CAFE inputs in prepared dataset: {missing}")

    mask = np.ones(dataset["chl"].shape, dtype=bool)
    for name in MODEL_INPUTS:
        mask &= np.isfinite(np.asarray(dataset[name].values))
    return mask


def run_month(
    root,
    date_obj,
    bbox,
    stride: int = 10,
    preset: str = "monthly1080",
    sensor: str = "modis",
    *,
    force: bool = False,
):
    """Calculate or reuse TRIDENT CAFE replay for one month."""
    root = Path(root).expanduser().resolve()
    date_obj = as_date(date_obj)
    tag = _tag(date_obj)
    step = max(1, int(stride))
    yd = _midmonth_day_of_year(date_obj)

    output_path = (
        root
        / "data"
        / "processed"
        / f"trident_osu_input_replay_cafe_npp_{tag}_stride{step}.nc"
    )
    report_path = (
        root
        / "reports"
        / "trident"
        / f"run_report_{tag}_stride{step}.json"
    )

    if not force and _existing_output_is_usable(
        output_path,
        tag=tag,
        stride=step,
        bbox=bbox,
        role="trident_replay",
    ):
        report = {}
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = {}

        report.update(
            {
                "tag": tag,
                "output_file": str(output_path),
                "stride": step,
                "reused_existing": True,
                "status": "reused",
            }
        )
        return report

    input_path = build_inputs(root, date_obj, bbox, preset, sensor)
    with xr.open_dataset(input_path) as opened:
        dataset = opened.load()

    shape = dataset["chl"].shape
    npp = np.full(shape, np.nan, dtype="float32")
    zeu = np.full(shape, np.nan, dtype="float32")
    kdpar = np.full(shape, np.nan, dtype="float32")

    all_indices = list(zip(*np.where(_candidate_mask(dataset))))
    indices = all_indices[::step]

    failure_counts: Counter[str] = Counter()
    first_failure: dict[str, Any] | None = None

    for i, j in tqdm(indices, desc=f"TRIDENT CAFE replay {tag}"):
        try:
            result = cafe_pixel(
                float(dataset["par"][i, j]),
                float(dataset["chl"][i, j]),
                float(dataset["mld"][i, j]),
                float(dataset["lat"][i]),
                yd,
                float(dataset["aph443"][i, j]),
                float(dataset["adg443"][i, j]),
                float(dataset["bbp443"][i, j]),
                float(dataset["bbp_s"][i, j]),
                float(dataset["sst"][i, j]),
            )
            npp[i, j] = result.npp
            zeu[i, j] = result.zeu
            kdpar[i, j] = result.kdpar
        except Exception as exc:
            key = f"{type(exc).__name__}: {exc}"
            failure_counts[key] += 1
            if first_failure is None:
                first_failure = {
                    "pixel_index": [int(i), int(j)],
                    "exception": key,
                }

    output = xr.Dataset(
        {
            "cafe_npp": (("lat", "lon"), npp),
            "zeu": (("lat", "lon"), zeu),
            "kdpar": (("lat", "lon"), kdpar),
        },
        coords={
            "lat": dataset["lat"].values,
            "lon": dataset["lon"].values,
        },
    )
    output.attrs.update(
        {
            "title": "TRIDENT CAFE replay calculated from archived OSU inputs",
            "product_role": "trident_replay",
            "tag": tag,
            "stride": step,
            "bbox": _bbox_string(bbox),
            "preset": preset,
            "sensor": sensor,
            "year_day_used": int(yd),
            "sampling_note": (
                "For stride > 1, only every nth valid candidate pixel was "
                "calculated. Unsampled cells remain NaN."
            ),
        }
    )
    _atomic_to_netcdf(output, output_path)

    success = int(np.isfinite(npp).sum())
    report = {
        "tag": tag,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "product_role": "TRIDENT replay: CAFE recalculated from archived OSU inputs",
        "stride": step,
        "bbox": list(map(float, bbox)),
        "year_day_used": int(yd),
        "input_valid_pixels": len(all_indices),
        "candidate_pixels": len(indices),
        "success": success,
        "failures": len(indices) - success,
        "failure_counts": dict(failure_counts),
        "first_failure": first_failure,
        "mean_npp": float(np.nanmean(npp)) if success else None,
        "reused_existing": False,
        "status": "calculated",
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compare_month(
    root,
    date_obj,
    bbox,
    stride: int = 10,
    preset: str = "monthly1080",
    *,
    force: bool = False,
):
    """Calculate or reuse comparison with archived OSU CAFE NPP."""
    root = Path(root).expanduser().resolve()
    date_obj = as_date(date_obj)
    tag = _tag(date_obj)
    step = max(1, int(stride))

    replay_path = (
        root
        / "data"
        / "processed"
        / f"trident_osu_input_replay_cafe_npp_{tag}_stride{step}.nc"
    )
    comparison_path = (
        root
        / "data"
        / "processed"
        / f"trident_osu_replay_vs_archived_comparison_{tag}_stride{step}.nc"
    )
    report_path = (
        root
        / "reports"
        / "trident"
        / f"validation_{tag}_stride{step}.json"
    )

    if not force and _existing_output_is_usable(
        comparison_path,
        tag=tag,
        stride=step,
        bbox=bbox,
        role="validation_comparison",
    ):
        report = {}
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = {}

        report.update(
            {
                "tag": tag,
                "comparison_file": str(comparison_path),
                "stride": step,
                "reused_existing": True,
                "status": "reused",
            }
        )
        return report

    if not replay_path.exists():
        raise FileNotFoundError(
            f"Replay output does not exist for {tag}: {replay_path}"
        )

    with xr.open_dataset(replay_path) as opened:
        replay_dataset = opened.load()

    record = ensure_product_month(root, date_obj, "cafe", None, preset)
    archived = mask_feasible(
        "cafe_npp",
        read_hdf_array(record["local_path"]),
    )
    archived_subset, _, _ = subset_bbox(archived, bbox)
    archived_on_grid = resize_nearest_or_block(
        archived_subset,
        replay_dataset["cafe_npp"].shape,
    ).astype("float32")

    replay = replay_dataset["cafe_npp"].values.astype("float32")
    common = np.isfinite(replay) & np.isfinite(archived_on_grid)

    if common.any():
        replay_values = replay[common]
        archived_values = archived_on_grid[common]
        difference = replay_values - archived_values
        correlation = (
            float(np.corrcoef(replay_values, archived_values)[0, 1])
            if common.sum() > 2
            else None
        )
        archived_mean = float(np.mean(archived_values))

        report = {
            "n": int(common.sum()),
            "trident_replay_mean": float(np.mean(replay_values)),
            "osu_archived_mean": archived_mean,
            "bias_trident_minus_osu": float(np.mean(difference)),
            "percent_bias_vs_osu_mean": (
                float(100.0 * np.mean(difference) / archived_mean)
                if archived_mean != 0
                else None
            ),
            "mae": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "correlation_r": correlation,
            "r2": None if correlation is None else correlation**2,
        }
    else:
        report = {"n": 0}

    comparison = xr.Dataset(
        {
            "trident_replay_npp": (("lat", "lon"), replay),
            "osu_archived_npp": (("lat", "lon"), archived_on_grid),
            "difference": (("lat", "lon"), replay - archived_on_grid),
        },
        coords={
            "lat": replay_dataset["lat"].values,
            "lon": replay_dataset["lon"].values,
        },
    )
    comparison.attrs.update(
        {
            "title": "TRIDENT replay compared with direct archived OSU CAFE",
            "product_role": "validation_comparison",
            "tag": tag,
            "stride": step,
            "bbox": _bbox_string(bbox),
            "comparison_mask": "finite intersection of replay and archived product",
        }
    )
    _atomic_to_netcdf(comparison, comparison_path)

    report.update(
        {
            "tag": tag,
            "comparison_file": str(comparison_path),
            "product_role": "validation_comparison",
            "bbox": list(map(float, bbox)),
            "stride": step,
            "reused_existing": False,
            "status": "calculated",
        }
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_range(
    root,
    start,
    end,
    bbox,
    stride: int = 10,
    preset: str = "monthly1080",
    sensor: str = "modis",
    *,
    force: bool = False,
):
    """Run requested months and reuse matching completed outputs by default."""
    reports = []

    for month in month_starts(start, end):
        run_report = run_month(
            root,
            month,
            bbox,
            stride,
            preset,
            sensor,
            force=force,
        )
        validation_report = compare_month(
            root,
            month,
            bbox,
            stride,
            preset,
            force=force,
        )
        reports.append(
            {
                "tag": _tag(month),
                "run": run_report,
                "validation": validation_report,
            }
        )

    return reports
