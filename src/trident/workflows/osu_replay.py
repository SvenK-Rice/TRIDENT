from __future__ import annotations

from collections import Counter
from datetime import date
import json
from pathlib import Path
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


def _tag(date_obj: Any) -> str:
    d = as_date(date_obj)
    return f"{d.year}{d.month:02d}"


def _midmonth_day_of_year(date_obj: Any) -> int:
    """Return day-of-year for the 15th day of the selected month."""
    d = as_date(date_obj)
    return date(d.year, d.month, 15).timetuple().tm_yday


def _candidate_mask(ds: xr.Dataset) -> np.ndarray:
    """Pixels are candidates only when every required CAFE input is finite."""
    missing = [name for name in MODEL_INPUTS if name not in ds]
    if missing:
        raise KeyError(f"Missing CAFE inputs in prepared dataset: {missing}")

    mask = np.ones(ds["chl"].shape, dtype=bool)
    for name in MODEL_INPUTS:
        mask &= np.isfinite(np.asarray(ds[name].values))
    return mask


def run_month(
    root,
    date_obj,
    bbox,
    stride: int = 10,
    preset: str = "monthly1080",
    sensor: str = "modis",
):
    """Calculate TRIDENT CAFE NPP from archived OSU input products."""
    root = Path(root).expanduser().resolve()
    date_obj = as_date(date_obj)
    tag = _tag(date_obj)
    yd = _midmonth_day_of_year(date_obj)

    infile = build_inputs(root, date_obj, bbox, preset, sensor)

    with xr.open_dataset(infile) as opened:
        ds = opened.load()

    shape = ds["chl"].shape
    npp = np.full(shape, np.nan, dtype="float32")
    zeu = np.full(shape, np.nan, dtype="float32")
    kdpar = np.full(shape, np.nan, dtype="float32")

    all_indices = list(zip(*np.where(_candidate_mask(ds))))
    step = max(1, int(stride))
    indices = all_indices[::step]

    failure_counts: Counter[str] = Counter()
    first_failure: dict[str, Any] | None = None

    for i, j in tqdm(indices, desc=f"TRIDENT CAFE replay {tag}"):
        try:
            result = cafe_pixel(
                float(ds["par"][i, j]),
                float(ds["chl"][i, j]),
                float(ds["mld"][i, j]),
                float(ds["lat"][i]),
                yd,
                float(ds["aph443"][i, j]),
                float(ds["adg443"][i, j]),
                float(ds["bbp443"][i, j]),
                float(ds["bbp_s"][i, j]),
                float(ds["sst"][i, j]),
            )
            npp[i, j] = result.npp
            zeu[i, j] = result.zeu
            kdpar[i, j] = result.kdpar
        except Exception as exc:  # record failures rather than silently hiding them
            key = f"{type(exc).__name__}: {exc}"
            failure_counts[key] += 1
            if first_failure is None:
                first_failure = {
                    "pixel_index": [int(i), int(j)],
                    "exception": key,
                }

    out = xr.Dataset(
        {
            "cafe_npp": (("lat", "lon"), npp),
            "zeu": (("lat", "lon"), zeu),
            "kdpar": (("lat", "lon"), kdpar),
        },
        coords={"lat": ds["lat"].values, "lon": ds["lon"].values},
    )
    out.attrs.update(
        {
            "title": "TRIDENT CAFE replay calculated from archived OSU inputs",
            "product_role": "trident_replay",
            "reference": (
                "This is not the direct OSU NPP product. "
                "It is TRIDENT CAFE recalculated from archived OSU inputs."
            ),
            "year_day_used": int(yd),
            "date_basis": "15th day of selected month",
        }
    )

    outpath = (
        root
        / "data"
        / "processed"
        / f"trident_osu_input_replay_cafe_npp_{tag}_stride{step}.nc"
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(outpath)

    success = int(np.isfinite(npp).sum())
    report = {
        "tag": tag,
        "input_file": str(infile),
        "output_file": str(outpath),
        "product_role": "TRIDENT replay: CAFE recalculated from archived OSU inputs",
        "stride": step,
        "year_day_used": int(yd),
        "input_valid_pixels": len(all_indices),
        "candidate_pixels": len(indices),
        "success": success,
        "failures": len(indices) - success,
        "failure_counts": dict(failure_counts),
        "first_failure": first_failure,
        "mean_npp": float(np.nanmean(npp)) if success else None,
    }

    report_path = root / "reports" / "trident" / f"run_report_{tag}_stride{step}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compare_month(
    root,
    date_obj,
    bbox,
    stride: int = 10,
    preset: str = "monthly1080",
):
    """Compare TRIDENT replay NPP with the direct archived OSU CAFE product."""
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
    if not replay_path.exists():
        raise FileNotFoundError(
            f"Replay output does not exist for {tag}: {replay_path}"
        )

    with xr.open_dataset(replay_path) as opened:
        replay_ds = opened.load()

    record = ensure_product_month(root, date_obj, "cafe", None, preset)
    archived = mask_feasible(
        "cafe_npp",
        read_hdf_array(record["local_path"]),
    )
    archived_subset, _, _ = subset_bbox(archived, bbox)
    archived_on_replay_grid = resize_nearest_or_block(
        archived_subset,
        replay_ds["cafe_npp"].shape,
    ).astype("float32")

    replay = replay_ds["cafe_npp"].values.astype("float32")
    archived_values = archived_on_replay_grid

    # The replay is sparse when stride > 1. Using the intersection means both
    # regional means are calculated from exactly the same pixels.
    common = np.isfinite(replay) & np.isfinite(archived_values)

    if common.any():
        replay_common = replay[common]
        archived_common = archived_values[common]
        difference = replay_common - archived_common
        correlation = (
            float(np.corrcoef(replay_common, archived_common)[0, 1])
            if common.sum() > 2
            else None
        )
        archived_mean = float(np.mean(archived_common))
        report = {
            "n": int(common.sum()),
            "trident_replay_mean": float(np.mean(replay_common)),
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
            "osu_archived_npp": (("lat", "lon"), archived_values),
            "difference": (("lat", "lon"), replay - archived_values),
        },
        coords={
            "lat": replay_ds["lat"].values,
            "lon": replay_ds["lon"].values,
        },
    )
    comparison.attrs.update(
        {
            "title": "TRIDENT replay compared with direct archived OSU CAFE",
            "comparison_mask": "finite intersection of replay and archived product",
        }
    )

    comparison_path = (
        root
        / "data"
        / "processed"
        / f"trident_osu_replay_vs_archived_comparison_{tag}_stride{step}.nc"
    )
    comparison.to_netcdf(comparison_path)

    report.update(
        {
            "tag": tag,
            "comparison_file": str(comparison_path),
            "product_role": "validation_comparison",
        }
    )
    report_path = root / "reports" / "trident" / f"validation_{tag}_stride{step}.json"
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
):
    """Run and compare every month in the requested range."""
    reports = []
    for month in month_starts(start, end):
        run_report = run_month(root, month, bbox, stride, preset, sensor)
        validation_report = compare_month(root, month, bbox, stride, preset)
        reports.append(
            {
                "tag": _tag(month),
                "run": run_report,
                "validation": validation_report,
            }
        )
    return reports
