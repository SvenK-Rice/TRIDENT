from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


DATE_RE = re.compile(r"(?P<tag>\d{6})")


@dataclass(frozen=True)
class DatasetEntry:
    path: Path
    tag: str | None
    product_type: str
    variables: tuple[str, ...]


def _product_type(path: Path) -> str:
    name = path.name.lower()
    if "comparison" in name or "validation" in name:
        return "validation"
    if "native" in name or "standard" in name:
        return "native"
    if "replay" in name:
        return "replay"
    if "npp" in name or "cafe" in name:
        return "npp"
    return "other"


def discover_processed_datasets(data_root: Path) -> list[DatasetEntry]:
    processed = Path(data_root).expanduser().resolve() / "data" / "processed"
    entries: list[DatasetEntry] = []

    for path in sorted(processed.glob("**/*.nc")):
        if not path.is_file():
            continue
        try:
            with xr.open_dataset(path) as ds:
                variables = tuple(sorted(ds.data_vars))
            match = DATE_RE.search(path.name)
            entries.append(
                DatasetEntry(
                    path=path,
                    tag=None if match is None else match.group("tag"),
                    product_type=_product_type(path),
                    variables=variables,
                )
            )
        except Exception:
            continue

    return entries


def open_loaded(path: Path) -> xr.Dataset:
    with xr.open_dataset(path) as ds:
        return ds.load()


def classify_variables(dataset: xr.Dataset) -> dict[str, list[str]]:
    classes = {
        "maps_2d": [],
        "profiles_1d": [],
        "time_series": [],
        "time_depth": [],
        "scalars": [],
        "other": [],
    }

    for name, data in dataset.data_vars.items():
        dims = set(data.dims)

        if data.ndim == 0:
            classes["scalars"].append(name)
        elif data.ndim == 2 and {"lat", "lon"}.issubset(dims):
            classes["maps_2d"].append(name)
        elif data.ndim == 1 and dims.intersection({"depth", "z", "lev"}):
            classes["profiles_1d"].append(name)
        elif data.ndim == 1 and dims.intersection({"time", "date"}):
            classes["time_series"].append(name)
        elif (
            data.ndim == 2
            and dims.intersection({"time", "date"})
            and dims.intersection({"depth", "z", "lev"})
        ):
            classes["time_depth"].append(name)
        else:
            classes["other"].append(name)

    return classes


def nearest_pixel(
    dataset: xr.Dataset,
    latitude: float,
    longitude: float,
) -> tuple[int, int, float, float]:
    lat = np.asarray(dataset["lat"].values, dtype=float)
    lon = np.asarray(dataset["lon"].values, dtype=float)

    i = int(np.nanargmin(np.abs(lat - latitude)))
    j = int(np.nanargmin(np.abs(lon - longitude)))

    return i, j, float(lat[i]), float(lon[j])


def extract_pixel_table(
    dataset: xr.Dataset,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    i, j, used_lat, used_lon = nearest_pixel(
        dataset,
        latitude,
        longitude,
    )
    rows = []

    for name, data in dataset.data_vars.items():
        if {"lat", "lon"}.issubset(data.dims):
            selected = data.isel(lat=i, lon=j)

            if selected.ndim == 0:
                try:
                    value = float(selected.values)
                except Exception:
                    continue

                rows.append(
                    {
                        "variable": name,
                        "value": value,
                        "latitude": used_lat,
                        "longitude": used_lon,
                        "units": data.attrs.get("units"),
                        "long_name": data.attrs.get("long_name"),
                    }
                )

    return pd.DataFrame(rows)


def extract_depth_profile(
    dataset: xr.Dataset,
    variable: str,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    time_index: int = 0,
) -> pd.DataFrame:
    data = dataset[variable]
    depth_name = next(
        (name for name in ("depth", "z", "lev") if name in data.dims),
        None,
    )
    if depth_name is None:
        raise ValueError(f"{variable} has no depth dimension.")

    selected = data

    for time_name in ("time", "date"):
        if time_name in selected.dims:
            selected = selected.isel({time_name: time_index})
            break

    if {"lat", "lon"}.issubset(selected.dims):
        if latitude is None or longitude is None:
            raise ValueError("Latitude and longitude are required.")
        i, j, _, _ = nearest_pixel(dataset, latitude, longitude)
        selected = selected.isel(lat=i, lon=j)

    values = np.asarray(selected.values).squeeze()
    depth = np.asarray(dataset[depth_name].values).squeeze()

    if values.ndim != 1:
        raise ValueError(f"{variable} could not be reduced to a 1-D profile.")

    return pd.DataFrame(
        {
            "depth": depth.astype(float),
            "value": values.astype(float),
        }
    )


def integrated_profile(profile: pd.DataFrame) -> float:
    clean = profile.dropna(subset=["depth", "value"]).sort_values("depth")
    if len(clean) < 2:
        return float("nan")
    return float(np.trapezoid(clean["value"], clean["depth"]))


def monthly_series(
    entries: Iterable[DatasetEntry],
    variable: str,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pd.DataFrame:
    rows = []

    for entry in entries:
        if entry.tag is None or variable not in entry.variables:
            continue

        try:
            ds = open_loaded(entry.path)
            data = ds[variable]

            if {"lat", "lon"}.issubset(data.dims):
                if latitude is None or longitude is None:
                    value = float(data.mean(skipna=True).values)
                else:
                    i, j, _, _ = nearest_pixel(ds, latitude, longitude)
                    value = float(data.isel(lat=i, lon=j).values)
            elif data.ndim == 0:
                value = float(data.values)
            else:
                continue

            rows.append(
                {
                    "date": pd.to_datetime(
                        entry.tag + "15",
                        format="%Y%m%d",
                    ),
                    "value": value,
                    "file": entry.path.name,
                    "product_type": entry.product_type,
                }
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(
            columns=["date", "value", "file", "product_type"]
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
