from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import xarray as xr

COMMON_SENTINELS = (-9999.0, -999.0, 9999.0, 1.0e20, 9.96921e36)
ATTR_SENTINEL_KEYS = ("_FillValue", "missing_value", "fill_value", "Hole Value")

PHYSICAL_RANGES: dict[str, tuple[float | None, float | None]] = {
    "par": (0.0, 100.0),
    "chl": (0.0, 500.0),
    "mld": (0.0, 5000.0),
    "aph443": (0.0, 20.0),
    "adg443": (0.0, 20.0),
    "bbp443": (0.0, 5.0),
    "bbp_s": (0.0, 5.0),
    "sst": (-5.0, 50.0),
}


@dataclass(frozen=True)
class QualitySummary:
    name: str
    total: int
    valid: int
    invalid: int
    valid_fraction: float
    minimum: float | None
    median: float | None
    maximum: float | None


def _attribute_sentinels(da: xr.DataArray) -> set[float]:
    sentinels: set[float] = set()
    for key in ATTR_SENTINEL_KEYS:
        value = da.attrs.get(key)
        if value is None:
            value = da.encoding.get(key)
        if value is None:
            continue
        values = np.atleast_1d(value)
        for item in values:
            try:
                sentinels.add(float(item))
            except (TypeError, ValueError):
                continue
    return sentinels


def valid_mask(
    da: xr.DataArray,
    *,
    name: str | None = None,
    extra_sentinels: Iterable[float] = (),
) -> np.ndarray:
    values = np.asarray(da.values, dtype=float)
    mask = np.isfinite(values)

    sentinels = set(COMMON_SENTINELS)
    sentinels.update(_attribute_sentinels(da))
    sentinels.update(float(v) for v in extra_sentinels)
    for sentinel in sentinels:
        mask &= ~np.isclose(values, sentinel, rtol=0.0, atol=0.0)

    lower, upper = PHYSICAL_RANGES.get(name or da.name or "", (None, None))
    if lower is not None:
        mask &= values >= lower
    if upper is not None:
        mask &= values <= upper
    return mask


def clean_dataarray(
    da: xr.DataArray,
    *,
    name: str | None = None,
    dtype: str = "float32",
) -> xr.DataArray:
    mask = valid_mask(da, name=name)
    values = np.asarray(da.values, dtype=float)
    cleaned = np.where(mask, values, np.nan).astype(dtype)
    result = xr.DataArray(
        cleaned,
        dims=da.dims,
        coords=da.coords,
        name=name or da.name,
        attrs=dict(da.attrs),
    )
    result.attrs.pop("_FillValue", None)
    result.attrs.pop("missing_value", None)
    result.attrs["trident_qc"] = "nonfinite, fill values, sentinels, and physical range masked"
    return result


def summarize(da: xr.DataArray, *, name: str | None = None) -> QualitySummary:
    mask = valid_mask(da, name=name)
    values = np.asarray(da.values, dtype=float)
    valid = values[mask]
    total = int(values.size)
    count = int(valid.size)
    return QualitySummary(
        name=name or da.name or "unknown",
        total=total,
        valid=count,
        invalid=total - count,
        valid_fraction=(count / total if total else 0.0),
        minimum=(float(np.min(valid)) if count else None),
        median=(float(np.median(valid)) if count else None),
        maximum=(float(np.max(valid)) if count else None),
    )


def joint_valid_mask(dataset: xr.Dataset, names: Iterable[str]) -> np.ndarray:
    names = tuple(names)
    if not names:
        raise ValueError("at least one variable is required")
    first = dataset[names[0]]
    mask = np.ones(first.shape, dtype=bool)
    for name in names:
        if name not in dataset:
            return np.zeros(first.shape, dtype=bool)
        if dataset[name].shape != first.shape:
            raise ValueError(f"{name} shape {dataset[name].shape} != {first.shape}")
        mask &= valid_mask(dataset[name], name=name)
    return mask
