from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from trident.grids.align import resize_nearest_or_block, mask_feasible
from trident.utils.dates import month_starts
from trident.models.cafe import cafe_pixel


REQUIRED = ("chl", "par", "aph443", "adg443", "bbp443", "bbp_s", "sst", "mld")

ALIASES: dict[str, tuple[str, ...]] = {
    "chl": ("chlor_a", "chl", "CHL", "CHL_OC4ME", "CHL1_mean", "chlorophyll"),
    "par": ("par", "PAR", "daily_par", "PAR_mean"),
    "aph443": ("aph_443", "aph443", "APH443", "aph_445", "aph"),
    "adg443": ("adg_443", "adg443", "ADG443", "adg_445", "adg"),
    "bbp443": ("bbp_443", "bbp443", "BBP443", "bbp_445", "bbp"),
    "bbp_s": ("bbp_s", "bbps", "BBP_S", "bbp_slope", "eta"),
    "sst": ("sst", "analysed_sst", "thetao", "sea_surface_temperature"),
    "mld": ("mlotst", "mld", "MLD", "mixed_layer_depth", "mldr10_1"),
}

COORD_ALIASES = {
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "x"),
    "time": ("time", "date", "datetime"),
    "depth": ("depth", "deptht", "lev", "z"),
}


@dataclass
class NativePrepareReport:
    tag: str
    output_file: str | None
    sources: dict[str, str]
    variables: dict[str, str]
    missing: list[str]
    shape: list[int] | None
    valid_pixels: int


def _find_name(names: Iterable[str], aliases: Iterable[str]) -> str | None:
    lookup = {str(n).lower(): str(n) for n in names}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    for alias in aliases:
        a = alias.lower()
        for lower, original in lookup.items():
            if a == lower or a in lower:
                return original
    return None


def _normalize(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    all_names = list(ds.coords) + list(ds.dims)
    for target, aliases in COORD_ALIASES.items():
        found = _find_name(all_names, aliases)
        if found and found != target and target not in ds.coords and target not in ds.dims:
            rename[found] = target
    if rename:
        ds = ds.rename(rename)
    if "lon" in ds.coords:
        lon = ds["lon"]
        if np.nanmax(lon.values) > 180:
            ds = ds.assign_coords(lon=((lon + 180) % 360) - 180).sortby("lon")
    if "lat" in ds.coords and ds["lat"].size > 1 and float(ds["lat"][0]) > float(ds["lat"][-1]):
        ds = ds.sortby("lat")
    return ds


def _open(path: Path) -> xr.Dataset:
    errors = []
    for engine in (None, "netcdf4", "h5netcdf"):
        try:
            kwargs = {} if engine is None else {"engine": engine}
            return _normalize(xr.open_dataset(path, **kwargs))
        except Exception as exc:
            errors.append(f"{engine or 'auto'}: {exc}")
    raise ValueError(f"Could not open {path}: {' | '.join(errors)}")


def _subset_and_reduce(da: xr.DataArray, bbox, month_start: pd.Timestamp) -> xr.DataArray:
    west, south, east, north = map(float, bbox)
    if "time" in da.dims:
        month_end = month_start + pd.offsets.MonthEnd(1)
        da = da.sel(time=slice(np.datetime64(month_start), np.datetime64(month_end)))
        if da.sizes.get("time", 0):
            da = da.mean("time", skipna=True)
    if "depth" in da.dims:
        da = da.isel(depth=0)
    for dim in list(da.dims):
        if dim not in {"lat", "lon"}:
            da = da.isel({dim: 0})
    if "lat" in da.coords and "lon" in da.coords:
        da = da.sel(lat=slice(south, north), lon=slice(west, east))
    return da.squeeze(drop=True)


def _candidate_files(root: Path, month_start: pd.Timestamp) -> list[Path]:
    tag = month_start.strftime("%Y%m")
    year = month_start.strftime("%Y")
    bases = [root / "data/reference/nasa", root / "data/reference/copernicus", root / "data/reference/esa"]
    files: list[Path] = []
    for base in bases:
        if not base.exists():
            continue
        for suffix in ("*.nc", "*.nc4", "*.h5", "*.hdf5"):
            files.extend(base.rglob(suffix))
    # Prefer filenames that mention the month/year, but keep all as fallback.
    return sorted(files, key=lambda p: (0 if tag in p.name else 1 if year in p.name else 2, str(p)))


def _extract_month(root: Path, month_start: pd.Timestamp, bbox) -> tuple[dict[str, xr.DataArray], dict[str, str], dict[str, str]]:
    found: dict[str, xr.DataArray] = {}
    sources: dict[str, str] = {}
    variable_names: dict[str, str] = {}
    for path in _candidate_files(root, month_start):
        if len(found) == len(REQUIRED):
            break
        try:
            ds = _open(path)
        except Exception:
            continue
        try:
            for target in REQUIRED:
                if target in found:
                    continue
                name = _find_name(ds.data_vars, ALIASES[target])
                if not name:
                    continue
                da = _subset_and_reduce(ds[name], bbox, month_start)
                if da.ndim != 2 or "lat" not in da.coords or "lon" not in da.coords:
                    continue
                values = mask_feasible(target, da.values)
                if not np.isfinite(values).any():
                    continue
                found[target] = xr.DataArray(values.astype("float32"), dims=("lat", "lon"), coords={"lat": da.lat.values, "lon": da.lon.values})
                sources[target] = str(path)
                variable_names[target] = name
        finally:
            ds.close()
    return found, sources, variable_names


def prepare_native_inputs(root: Path | str, start, end, bbox) -> list[dict]:
    root = Path(root)
    reports: list[dict] = []
    for month in month_starts(start, end):
        ts = pd.Timestamp(month)
        tag = ts.strftime("%Y%m")
        data, sources, variable_names = _extract_month(root, ts, bbox)
        missing = [v for v in REQUIRED if v not in data]
        report = NativePrepareReport(tag, None, sources, variable_names, missing, None, 0)
        if not missing:
            ref = data["chl"]
            lat = np.asarray(ref.lat.values, dtype="float32")
            lon = np.asarray(ref.lon.values, dtype="float32")
            aligned = {}
            for name, da in data.items():
                arr = resize_nearest_or_block(np.asarray(da.values, dtype="float32"), ref.shape)
                aligned[name] = (("lat", "lon"), arr)
            ds = xr.Dataset(aligned, coords={"lat": lat, "lon": lon})
            valid = np.ones(ref.shape, dtype=bool)
            for name in REQUIRED:
                valid &= np.isfinite(ds[name].values)
            ds["cafe_valid"] = (("lat", "lon"), valid.astype("uint8"))
            ds.attrs.update({"title": "TRIDENT Native CAFE inputs", "tag": tag, "sources_json": json.dumps(sources)})
            out = root / "data/processed" / f"trident_native_cafe_inputs_{tag}.nc"
            out.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(out)
            report.output_file = str(out)
            report.shape = list(ref.shape)
            report.valid_pixels = int(valid.sum())
        reports.append(asdict(report))
    out_report = root / "reports/trident/native_prepare_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(reports, indent=2))
    return reports


def run_native_cafe(root: Path | str, start, end, stride: int = 10) -> list[dict]:
    root = Path(root)
    reports = []
    for month in month_starts(start, end):
        ts = pd.Timestamp(month)
        tag = ts.strftime("%Y%m")
        infile = root / "data/processed" / f"trident_native_cafe_inputs_{tag}.nc"
        if not infile.exists():
            reports.append({"tag": tag, "status": "missing_inputs", "input_file": str(infile)})
            continue
        ds = xr.open_dataset(infile)
        shape = ds["chl"].shape
        npp = np.full(shape, np.nan, dtype="float32")
        zeu = np.full(shape, np.nan, dtype="float32")
        kdpar = np.full(shape, np.nan, dtype="float32")
        mask = ds["cafe_valid"].values.astype(bool) if "cafe_valid" in ds else np.ones(shape, dtype=bool)
        indices = list(zip(*np.where(mask)))[::max(1, int(stride))]
        failures = 0
        yd = int(ts.dayofyear + 14)
        for i, j in indices:
            try:
                res = cafe_pixel(
                    float(ds["par"][i, j]), float(ds["chl"][i, j]), float(ds["mld"][i, j]),
                    float(ds["lat"][i]), yd, float(ds["aph443"][i, j]), float(ds["adg443"][i, j]),
                    float(ds["bbp443"][i, j]), float(ds["bbp_s"][i, j]), float(ds["sst"][i, j]),
                )
                npp[i, j], zeu[i, j], kdpar[i, j] = res.npp, res.zeu, res.kdpar
            except Exception:
                failures += 1
        out = xr.Dataset(
            {"cafe_npp": (("lat", "lon"), npp), "zeu": (("lat", "lon"), zeu), "kdpar": (("lat", "lon"), kdpar)},
            coords={"lat": ds.lat, "lon": ds.lon},
        )
        out.attrs.update({"title": "TRIDENT Native CAFE NPP", "tag": tag, "source": "NASA/ESA ocean colour plus Copernicus physics"})
        outfile = root / "data/processed" / f"trident_native_cafe_npp_{tag}_stride{stride}.nc"
        out.to_netcdf(outfile)
        reports.append({
            "tag": tag, "status": "ok", "input_file": str(infile), "output_file": str(outfile),
            "stride": int(stride), "candidate_pixels": len(indices), "failures": failures,
            "success": len(indices) - failures, "mean_npp": float(np.nanmean(npp)) if np.isfinite(npp).any() else None,
        })
        ds.close()
    out_report = root / "reports/trident/native_cafe_run_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(reports, indent=2))
    return reports
