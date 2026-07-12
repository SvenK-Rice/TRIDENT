from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from trident.grids.align import mask_feasible


_VALIDATION_RE = re.compile(
    r"trident_osu_replay_vs_archived_comparison_"
    r"(?P<tag>\d{6})_stride(?P<stride>\d+)\.nc$"
)


def _bbox_subset(ds: xr.Dataset, arr: np.ndarray, bbox) -> np.ndarray:
    """Subset a 2-D array using dataset latitude/longitude coordinates."""
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return arr

    west, south, east, north = map(float, bbox)
    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)

    latmask = (lat >= south) & (lat <= north)
    lonmask = (lon >= west) & (lon <= east)

    if arr.ndim == 2 and latmask.any() and lonmask.any():
        return arr[np.ix_(latmask, lonmask)]
    return arr


def discover_validation_runs(root, stride: int | None = None) -> pd.DataFrame:
    """Return one row for every completed OSU validation comparison file.

    The returned table is the authoritative catalog used by the app. It does
    not infer runs from unrelated replay files or reports.
    """
    processed = Path(root).expanduser() / "data" / "processed"
    rows: list[dict] = []

    for path in processed.glob(
        "trident_osu_replay_vs_archived_comparison_*_stride*.nc"
    ):
        match = _VALIDATION_RE.match(path.name)
        if not match:
            continue

        run_stride = int(match.group("stride"))
        if stride is not None and run_stride != int(stride):
            continue

        tag = match.group("tag")
        rows.append(
            {
                "tag": tag,
                "date": pd.to_datetime(tag + "15", format="%Y%m%d"),
                "stride": run_stride,
                "file": path.name,
                "path": str(path.resolve()),
                "modified": pd.to_datetime(path.stat().st_mtime, unit="s"),
                "size_mb": path.stat().st_size / (1024 * 1024),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "tag",
                "date",
                "stride",
                "file",
                "path",
                "modified",
                "size_mb",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["date", "stride", "modified"])
        .drop_duplicates(["tag", "stride"], keep="last")
        .reset_index(drop=True)
    )


def validation_series(
    root,
    bbox,
    stride: int | None = None,
    selected_files: Sequence[str | Path] | None = None,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Calculate comparable regional means from explicitly selected runs.

    Both NPP products are averaged over the exact same finite-pixel mask.
    This is essential for stride > 1 because replay files are sparse while
    the direct OSU archive product is complete.

    Parameters
    ----------
    selected_files
        Optional explicit file list. When supplied, no other completed run is
        read. This prevents stale files from appearing in a new plot.
    start, end
        Optional date limits applied to the completed-run catalog.
    """
    catalog = discover_validation_runs(root, stride=stride)

    if selected_files is not None:
        selected = {
            str(Path(item).expanduser().resolve()) for item in selected_files
        }
        catalog = catalog[catalog["path"].isin(selected)]

    if start is not None:
        catalog = catalog[catalog["date"] >= pd.Timestamp(start)]
    if end is not None:
        # Include the complete ending month.
        end_ts = pd.Timestamp(end) + pd.offsets.MonthEnd(0)
        catalog = catalog[catalog["date"] <= end_ts]

    rows: list[dict] = []

    for record in catalog.itertuples(index=False):
        path = Path(record.path)
        try:
            # Load data and close the file before any later rerun can replace it.
            with xr.open_dataset(path) as opened:
                ds = opened.load()

            trident = _bbox_subset(
                ds,
                mask_feasible(
                    "cafe_npp",
                    np.asarray(ds["trident_replay_npp"].values),
                ),
                bbox,
            )
            archived = _bbox_subset(
                ds,
                mask_feasible(
                    "cafe_npp",
                    np.asarray(ds["osu_archived_npp"].values),
                ),
                bbox,
            )

            common = np.isfinite(trident) & np.isfinite(archived)
            if not common.any():
                continue

            replay_values = trident[common]
            archived_values = archived[common]
            difference = replay_values - archived_values

            rows.append(
                {
                    "tag": record.tag,
                    "date": record.date,
                    "stride": int(record.stride),
                    "trident_replay_mean": float(
                        np.mean(replay_values)
                    ),
                    "osu_archived_mean": float(
                        np.mean(archived_values)
                    ),
                    "bias": float(np.mean(difference)),
                    "percent_bias": float(
                        100.0
                        * np.mean(difference)
                        / np.mean(archived_values)
                    )
                    if np.mean(archived_values) != 0
                    else np.nan,
                    "rmse": float(
                        np.sqrt(np.mean(difference**2))
                    ),
                    "valid_pixels": int(common.sum()),
                    "file": record.file,
                    "path": record.path,
                }
            )
        except (OSError, ValueError, KeyError):
            # Ignore incomplete/corrupt files, but never silently substitute a
            # different dataset.
            continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "tag",
                "date",
                "stride",
                "trident_replay_mean",
                "osu_archived_mean",
                "bias",
                "percent_bias",
                "rmse",
                "valid_pixels",
                "file",
                "path",
            ]
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def standard_series(root, bbox, start=None, end=None) -> pd.DataFrame:
    """Return TRIDENT Native regional means, optionally date-limited."""
    root = Path(root)
    rows: list[dict] = []

    for path in sorted(
        (root / "data" / "processed").glob(
            "trident_standard_cafe_npp_*.nc"
        )
    ):
        try:
            tags = [
                item
                for item in path.stem.split("_")
                if len(item) == 6 and item.isdigit()
            ]
            if not tags:
                continue
            tag = tags[-1]
            run_date = pd.to_datetime(tag + "15", format="%Y%m%d")

            if start is not None and run_date < pd.Timestamp(start):
                continue
            if end is not None and run_date > (
                pd.Timestamp(end) + pd.offsets.MonthEnd(0)
            ):
                continue

            with xr.open_dataset(path) as opened:
                ds = opened.load()

            arr = _bbox_subset(
                ds,
                mask_feasible(
                    "cafe_npp",
                    np.asarray(ds["cafe_npp"].values),
                ),
                bbox,
            )
            rows.append(
                {
                    "tag": tag,
                    "date": run_date,
                    "trident_standard_mean": float(np.nanmean(arr)),
                    "file": path.name,
                    "path": str(path.resolve()),
                }
            )
        except (OSError, ValueError, KeyError):
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
