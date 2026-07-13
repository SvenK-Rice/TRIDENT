from __future__ import annotations

from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
from pathlib import Path
import json
import re
import time
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from trident.grids.align import resize_nearest_or_block, mask_feasible
from trident.data_sources.qc import clean_dataarray, joint_valid_mask
from trident.utils.dates import month_starts
from trident.models.cafe import cafe_pixel
from trident.analysis.storage_layout import (
    bbox_matches_dataset,
    coverage_record,
    legacy_native_output_path,
    native_input_path,
    native_output_path,
    region_attrs,
    region_identity,
    resolve_native_input_path,
    update_coverage_manifest,
)


REQUIRED = ("chl", "par", "aph443", "adg443", "bbp443", "bbp_s", "sst", "mld")

ALIASES: dict[str, tuple[str, ...]] = {
    "chl": ("chlor_a", "chl", "CHL", "CHL_OC4ME", "CHL1_mean", "chlorophyll"),
    # CAFE uses daily planar PAR above the sea surface. PACE V3.2 exposes
    # this explicitly as par_day_planar_above.
    "par": ("par_day_planar_above", "par", "PAR", "daily_par", "PAR_mean"),
    # PACE OCI uses 442 nm. These fields are the direct 1-nm counterparts
    # of the nominal CAFE 443-nm inputs.
    "aph443": ("aph_442", "aph_443", "aph443", "APH443", "aph_445"),
    "adg443": ("adg_442", "adg_443", "adg443", "ADG443", "adg_445"),
    "bbp443": ("bbp_442", "bbp_443", "bbp443", "BBP443", "bbp_445"),
    "bbp_s": ("bbp_s", "bbps", "BBP_S", "bbp_slope", "eta"),
    "sst": ("sst", "analysed_sst", "thetao", "sea_surface_temperature"),
    "mld": ("mlotst", "mld", "MLD", "mixed_layer_depth", "mldr10_1"),
}

REJECT_TOKENS = ("unc", "uncertainty", "palette", "flags")

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


def _write_merged_report(path: Path, records: list[dict], key_fields: tuple[str, ...]) -> None:
    existing: list[dict] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = [item for item in loaded if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            pass

    def key(item: dict) -> tuple:
        values = tuple(str(item.get(field, "")) for field in key_fields)
        if any(values):
            return values
        return (str(item.get("output_file", "")),)

    replacement_keys = {key(item) for item in records}
    merged = [item for item in existing if key(item) not in replacement_keys]
    merged.extend(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2))


def _run_cafe_pixel_task(task: tuple) -> tuple:
    """Run one validated CAFE pixel in a process-safe top-level worker."""
    i, j, par, chl, mld, lat, yd, aph443, adg443, bbp443, bbp_s, sst = task
    try:
        result = cafe_pixel(
            par, chl, mld, lat, yd, aph443, adg443, bbp443, bbp_s, sst
        )
        return i, j, result.npp, result.zeu, result.kdpar, None
    except Exception as exc:
        return i, j, np.nan, np.nan, np.nan, f"{type(exc).__name__}: {exc}"


def _cafe_model_fingerprint() -> str:
    path = Path(cafe_pixel.__code__.co_filename)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _valid_completed_output(
    path: Path,
    *,
    infile: Path,
    tag: str,
    stride: int,
    shape: tuple[int, ...],
) -> bool:
    """Return True only for a complete output matching the current input/run."""
    if not path.exists():
        return False
    try:
        input_stat = infile.stat()
        with xr.open_dataset(path) as existing:
            required = {"cafe_npp", "zeu", "kdpar"}
            return (
                required.issubset(existing.data_vars)
                and tuple(existing["cafe_npp"].shape) == tuple(shape)
                and int(existing.attrs.get("processing_complete", 0)) == 1
                and str(existing.attrs.get("tag", "")) == tag
                and int(existing.attrs.get("stride", -1)) == int(stride)
                and int(existing.attrs.get("source_input_size", -1)) == input_stat.st_size
                and int(existing.attrs.get("source_input_mtime_ns", -1))
                == input_stat.st_mtime_ns
            )
    except Exception:
        return False


def _find_name(names: Iterable[str], aliases: Iterable[str]) -> str | None:
    """Find a science variable while rejecting uncertainty/flag products."""
    original_names = [str(name) for name in names]
    accepted = [
        name for name in original_names
        if not any(token in name.lower() for token in REJECT_TOKENS)
    ]
    lookup = {name.lower(): name for name in accepted}

    # Exact aliases are authoritative and ordered by scientific preference.
    for alias in aliases:
        match = lookup.get(alias.lower())
        if match is not None:
            return match

    # Conservative fallback: only use whole-token-like prefix/suffix matches.
    for alias in aliases:
        token = alias.lower()
        for lower, original in lookup.items():
            if lower.startswith(token + "_") or lower.endswith("_" + token):
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


def _open_hdf4(path: Path) -> xr.Dataset:
    """Open a simple two-dimensional OSU HDF4 grid as xarray."""
    from pyhdf.SD import SD, SDC

    handle = SD(str(path), SDC.READ)
    try:
        variables = {}
        for name in handle.datasets():
            selected = handle.select(name)
            values = np.asarray(selected[:], dtype="float32")
            if values.ndim != 2:
                continue
            attrs = dict(selected.attributes())
            hole = attrs.get("Hole Value", attrs.get("_FillValue", -9999.0))
            try:
                values[np.isclose(values, float(hole), rtol=0.0, atol=0.0)] = np.nan
            except (TypeError, ValueError):
                pass
            ny, nx = values.shape
            # OSU monthly1080 grids are equal-angle global cell centers,
            # stored north-to-south and west-to-east.
            lat = np.linspace(90.0 - 90.0 / ny, -90.0 + 90.0 / ny, ny, dtype="float32")
            lon = np.linspace(-180.0 + 180.0 / nx, 180.0 - 180.0 / nx, nx, dtype="float32")
            variables[name] = xr.DataArray(
                values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, attrs=attrs
            )
        dataset = xr.Dataset(variables)
        dataset.attrs.update(dict(handle.attributes()))
        return _normalize(dataset)
    finally:
        handle.end()


def _open(path: Path) -> xr.Dataset:
    errors = []
    for engine in (None, "netcdf4", "h5netcdf"):
        try:
            kwargs = {} if engine is None else {"engine": engine}
            return _normalize(xr.open_dataset(path, **kwargs))
        except Exception as exc:
            errors.append(f"{engine or 'auto'}: {exc}")
    if path.suffix.lower() == ".hdf":
        try:
            return _open_hdf4(path)
        except Exception as exc:
            errors.append(f"hdf4: {exc}")
    raise ValueError(f"Could not open {path}: {' | '.join(errors)}")

def _subset_and_reduce(
    da: xr.DataArray,
    bbox,
    month_start: pd.Timestamp,
) -> xr.DataArray:
    """Select one month and geographic region without indexing empty dimensions."""

    west, south, east, north = map(float, bbox)

    # Select the requested month.
    if "time" in da.dims:
        month_end = month_start + pd.offsets.MonthEnd(1)
        da = da.sel(
            time=slice(
                np.datetime64(month_start),
                np.datetime64(month_end),
            )
        )

        if da.sizes.get("time", 0) == 0:
            raise ValueError(
                f"No data are available for {month_start:%Y-%m} "
                f"in variable {da.name!r}"
            )

        da = da.mean("time", skipna=True)

    # Use the surface layer only when a depth dimension exists.
    if "depth" in da.dims:
        if da.sizes.get("depth", 0) == 0:
            raise ValueError(
                f"Variable {da.name!r} has an empty depth dimension"
            )
        da = da.isel(depth=0)

    # Reduce any remaining non-spatial dimensions safely.
    for dim in list(da.dims):
        if dim in {"lat", "lon"}:
            continue

        if da.sizes.get(dim, 0) == 0:
            raise ValueError(
                f"Variable {da.name!r} has an empty {dim!r} dimension"
            )

        da = da.isel({dim: 0})

    # Geographic subsetting must account for ascending or descending coordinates.
    if "lat" in da.coords and "lon" in da.coords:
        lat_values = np.asarray(da["lat"].values)
        lon_values = np.asarray(da["lon"].values)

        if lat_values.size == 0 or lon_values.size == 0:
            raise ValueError(
                f"Variable {da.name!r} has empty latitude or longitude coordinates"
            )

        lat_slice = (
            slice(south, north)
            if lat_values[0] <= lat_values[-1]
            else slice(north, south)
        )

        lon_slice = (
            slice(west, east)
            if lon_values[0] <= lon_values[-1]
            else slice(east, west)
        )

        da = da.sel(lat=lat_slice, lon=lon_slice)

        if da.sizes.get("lat", 0) == 0 or da.sizes.get("lon", 0) == 0:
            raise ValueError(
                f"No data from variable {da.name!r} overlap bbox "
                f"{(west, south, east, north)}"
            )

    return da.squeeze(drop=True)


def _candidate_files(root: Path, month_start: pd.Timestamp) -> list[Path]:
    tag = month_start.strftime("%Y%m")
    year = month_start.strftime("%Y")
    doy_tag = month_start.strftime("%Y%j")
    bases = [
        root / "data/reference/nasa",
        root / "data/reference/copernicus",
        root / "data/reference/esa",
        root / "data/reference/osu",
    ]
    files: list[Path] = []
    for base in bases:
        if not base.exists():
            continue
        for suffix in ("*.nc", "*.nc4", "*.h5", "*.hdf5", "*.hdf"):
            files.extend(base.rglob(suffix))

    def rank(path: Path):
        name = path.name.lower()
        exact = tag in name or doy_tag in name
        same_year = year in name
        is_osu_slope = "bbp_s" in name and "/reference/osu/" in str(path).lower()
        # Exact monthly products first; OSU slope ancillary remains a fallback.
        return (0 if exact else 1 if same_year else 2 if is_osu_slope else 3, str(path))

    return sorted(set(files), key=rank)

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
                try:
                    da = _subset_and_reduce(ds[name], bbox, month_start)
                except (ValueError, IndexError):
                    # This file either does not cover the requested month/region
                    # or contains an empty auxiliary dimension. Continue looking
                    # through the remaining local candidate files.
                    continue

                if da.ndim != 2 or "lat" not in da.coords or "lon" not in da.coords:
                    continue
                da = clean_dataarray(da, name=target)
                values = mask_feasible(target, da.values)
                values = np.where(np.isfinite(values), values, np.nan).astype("float32")
                if not np.isfinite(values).any():
                    continue
                found[target] = xr.DataArray(values, dims=("lat", "lon"), coords={"lat": da.lat.values, "lon": da.lon.values})
                sources[target] = str(path)
                variable_names[target] = name
        finally:
            ds.close()
    return found, sources, variable_names


def _align_to_reference(
    source: xr.DataArray,
    reference: xr.DataArray,
) -> xr.DataArray:
    """Align a source field to a reference latitude/longitude grid.

    Uses coordinate-based nearest-neighbor selection without requiring SciPy.
    """
    da = source

    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)

    ref = reference

    ref_rename = {}
    if "latitude" in ref.dims:
        ref_rename["latitude"] = "lat"
    if "longitude" in ref.dims:
        ref_rename["longitude"] = "lon"
    if ref_rename:
        ref = ref.rename(ref_rename)

    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(
            "Source field must contain latitude and longitude coordinates."
        )

    if "lat" not in ref.coords or "lon" not in ref.coords:
        raise ValueError(
            "Reference field must contain latitude and longitude coordinates."
        )

    # xarray.sel(method="nearest") uses coordinate indexes and does not require
    # scipy, unlike DataArray.interp() for multidimensional interpolation.
    aligned = da.sel(
        lat=ref["lat"],
        lon=ref["lon"],
        method="nearest",
    )

    aligned = aligned.assign_coords(
        lat=ref["lat"],
        lon=ref["lon"],
    )

    return aligned

def prepare_native_inputs(
    root: Path | str,
    start,
    end,
    bbox,
    *,
    region_name: str | None = None,
    cadence: str = "monthly",
) -> list[dict]:
    root = Path(root)
    region = region_identity(region_name, tuple(map(float, bbox)))
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
            alignment_methods = {}
            for name, da in data.items():
                aligned_da = _align_to_reference(da, ref)
                arr = np.asarray(aligned_da.values, dtype="float32")
                aligned[name] = (("lat", "lon"), arr)
                alignment_methods[name] = (
                    "native" if da.shape == ref.shape and np.array_equal(da.lat, ref.lat)
                    and np.array_equal(da.lon, ref.lon) else "nearest_coordinate"
                )
            ds = xr.Dataset(aligned, coords={"lat": lat, "lon": lon})
            valid = joint_valid_mask(ds, REQUIRED)
            ds["cafe_valid"] = (("lat", "lon"), valid.astype("uint8"))
            provenance = {
                name: {"file": sources.get(name), "variable": variable_names.get(name)}
                for name in REQUIRED
            }
            ds.attrs.update({
                "title": "TRIDENT Native CAFE inputs",
                "product_family": "TRIDENT Native",
                "temporal_resolution": "monthly",
                "analysis_role": "prepared_inputs",
                "tag": tag,
                **region_attrs(region),
                "sources_json": json.dumps(sources),
                "variable_provenance_json": json.dumps(provenance),
                "alignment_methods_json": json.dumps(alignment_methods),
                "joint_valid_pixels": int(valid.sum()),
                "qc_policy": "mask fill values, common sentinels, and physical-range violations",
                "pace_wavelength_mapping": (
                    "aph_442->aph443; adg_442->adg443; bbp_442->bbp443"
                ),
                "par_definition": "daily planar PAR above sea surface",
            })
            out = native_input_path(root, cadence, tag, region)
            out.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(out)
            report.output_file = str(out)
            report.shape = list(ref.shape)
            report.valid_pixels = int(valid.sum())
        reports.append({
            **asdict(report),
            "region_label": region.label,
            "region_id": region.region_id,
            "bbox": list(region.bbox),
            "cadence": cadence,
        })
    out_report = root / "reports/trident/native_prepare_report.json"
    _write_merged_report(out_report, reports, ("tag", "region_id", "cadence"))
    return reports


def _run_native_cafe_legacy(
    root: Path | str,
    start,
    end,
    stride: int = 10,
    progress_callback=None,
) -> list[dict]:
    import time

    root = Path(root)
    reports = []

    months = list(month_starts(start, end))
    total_months = len(months)

    print(
        f"[TRIDENT] Starting Native CAFE for {total_months} month(s), "
        f"stride={stride}",
        flush=True,
    )

    if total_months == 0:
        print("[TRIDENT] No months were selected.", flush=True)
        return reports

    for month_number, month in enumerate(months, start=1):
        ts = pd.Timestamp(month)
        tag = ts.strftime("%Y%m")
        infile = root / "data/processed" / f"trident_native_cafe_inputs_{tag}.nc"
        outfile = (
            root
            / "data/processed"
            / f"trident_native_cafe_npp_{tag}_stride{stride}.nc"
        )

        month_started = time.monotonic()

        print(
            f"[TRIDENT] Month {month_number}/{total_months}: {tag}",
            flush=True,
        )
        print(f"[TRIDENT] Input:  {infile}", flush=True)
        print(f"[TRIDENT] Output: {outfile}", flush=True)

        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "month_start",
                    "tag": tag,
                    "month_number": month_number,
                    "total_months": total_months,
                    "pixel_number": 0,
                    "total_pixels": 0,
                    "fraction": (month_number - 1) / total_months,
                    "elapsed_seconds": 0.0,
                    "failures": 0,
                }
            )

        if not infile.exists():
            message = f"Missing input file: {infile}"
            print(f"[TRIDENT] ERROR: {message}", flush=True)
            reports.append(
                {
                    "tag": tag,
                    "status": "missing_inputs",
                    "input_file": str(infile),
                    "message": message,
                }
            )
            continue

        # Resume safely when a valid output already exists.
        if outfile.exists():
            try:
                with xr.open_dataset(outfile) as existing:
                    if "cafe_npp" in existing:
                        print(
                            f"[TRIDENT] Existing valid output found; "
                            f"skipping {tag}.",
                            flush=True,
                        )
                        reports.append(
                            {
                                "tag": tag,
                                "status": "already_complete",
                                "input_file": str(infile),
                                "output_file": str(outfile),
                            }
                        )

                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "stage": "month_complete",
                                    "tag": tag,
                                    "month_number": month_number,
                                    "total_months": total_months,
                                    "pixel_number": 0,
                                    "total_pixels": 0,
                                    "fraction": month_number / total_months,
                                    "elapsed_seconds": 0.0,
                                    "failures": 0,
                                }
                            )
                        continue
            except Exception as exc:
                print(
                    f"[TRIDENT] Existing output is unreadable and will "
                    f"be replaced: {exc}",
                    flush=True,
                )
                outfile.unlink(missing_ok=True)

        try:
            ds = xr.open_dataset(infile)
        except Exception as exc:
            message = f"Could not open {infile}: {exc}"
            print(f"[TRIDENT] ERROR: {message}", flush=True)
            reports.append(
                {
                    "tag": tag,
                    "status": "unreadable_inputs",
                    "input_file": str(infile),
                    "message": message,
                }
            )
            continue

        missing_variables = [
            name
            for name in [*REQUIRED, "lat", "lon"]
            if name not in ds.variables and name not in ds.coords
        ]

        if missing_variables:
            message = (
                "Input file is missing required variables: "
                + ", ".join(missing_variables)
            )
            print(f"[TRIDENT] ERROR: {message}", flush=True)
            ds.close()
            reports.append(
                {
                    "tag": tag,
                    "status": "invalid_inputs",
                    "input_file": str(infile),
                    "missing_variables": missing_variables,
                    "message": message,
                }
            )
            continue
        shape = ds["chl"].shape
        npp = np.full(shape, np.nan, dtype="float32")
        zeu = np.full(shape, np.nan, dtype="float32")
        kdpar = np.full(shape, np.nan, dtype="float32")
        mask = ds["cafe_valid"].values.astype(bool) if "cafe_valid" in ds else np.ones(shape, dtype=bool)
        indices = list(zip(*np.where(mask)))[::max(1, int(stride))]
        failures = 0
        yd = int(ts.dayofyear + 14)

        total_pixels = len(indices)

        print(
            f"[TRIDENT] {tag}: {int(mask.sum()):,} joint-valid pixels; "
            f"{total_pixels:,} pixels selected at stride {stride}.",
            flush=True,
        )

        if total_pixels == 0:
            print(
                f"[TRIDENT] WARNING: No valid pixels available for {tag}.",
                flush=True,
            )
            ds.close()
            reports.append(
                {
                    "tag": tag,
                    "status": "no_valid_pixels",
                    "input_file": str(infile),
                    "output_file": None,
                    "failures": 0,
                }
            )
            continue

        update_every = max(1, total_pixels // 200)

        for pixel_number, (i, j) in enumerate(indices, start=1):
            try:
                res = cafe_pixel(
                    float(ds["par"][i, j]), float(ds["chl"][i, j]), float(ds["mld"][i, j]),
                    float(ds["lat"][i]), yd, float(ds["aph443"][i, j]), float(ds["adg443"][i, j]),
                    float(ds["bbp443"][i, j]), float(ds["bbp_s"][i, j]), float(ds["sst"][i, j]),
                )
                npp[i, j], zeu[i, j], kdpar[i, j] = res.npp, res.zeu, res.kdpar
            except Exception:
                failures += 1

            if (
                pixel_number == 1
                or pixel_number == total_pixels
                or pixel_number % update_every == 0
            ):
                elapsed = time.monotonic() - month_started
                month_fraction = pixel_number / total_pixels
                overall_fraction = (
                    (month_number - 1) + month_fraction
                ) / total_months

                rate = pixel_number / elapsed if elapsed > 0 else 0.0
                remaining_pixels = total_pixels - pixel_number
                eta_seconds = (
                    remaining_pixels / rate if rate > 0 else None
                )

                print(
                    f"[TRIDENT] {tag}: "
                    f"{pixel_number:,}/{total_pixels:,} "
                    f"({month_fraction * 100:5.1f}%) | "
                    f"failures={failures} | "
                    f"elapsed={elapsed:,.1f}s | "
                    f"ETA={eta_seconds:,.1f}s"
                    if eta_seconds is not None
                    else
                    f"[TRIDENT] {tag}: "
                    f"{pixel_number:,}/{total_pixels:,} "
                    f"({month_fraction * 100:5.1f}%) | "
                    f"failures={failures}",
                    flush=True,
                )

                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "pixels",
                            "tag": tag,
                            "month_number": month_number,
                            "total_months": total_months,
                            "pixel_number": pixel_number,
                            "total_pixels": total_pixels,
                            "fraction": overall_fraction,
                            "month_fraction": month_fraction,
                            "elapsed_seconds": elapsed,
                            "eta_seconds": eta_seconds,
                            "failures": failures,
                        }
                    )

        out = xr.Dataset(
            {
                "cafe_npp": (("lat", "lon"), npp),
                "zeu": (("lat", "lon"), zeu),
                "kdpar": (("lat", "lon"), kdpar),
                **{name: ds[name].astype("float32") for name in REQUIRED},
            },
            coords={"lat": ds.lat, "lon": ds.lon},
        )
        out.attrs.update({
            "title": "TRIDENT Native CAFE NPP",
            "product_family": "TRIDENT Native",
            "temporal_resolution": "monthly",
            "analysis_role": "analysis_result",
            "tag": tag,
            "source": "NASA/ESA ocean colour plus Copernicus physics",
            "source_input_file": str(infile),
            "profile_inputs_embedded": 1,
            "year_day_used": int(yd),
        })
        out.to_netcdf(outfile)

        month_elapsed = time.monotonic() - month_started

        print(
            f"[TRIDENT] Completed {tag} in {month_elapsed:,.1f} seconds; "
            f"failures={failures}; output={outfile}",
            flush=True,
        )

        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "month_complete",
                    "tag": tag,
                    "month_number": month_number,
                    "total_months": total_months,
                    "pixel_number": total_pixels,
                    "total_pixels": total_pixels,
                    "fraction": month_number / total_months,
                    "month_fraction": 1.0,
                    "elapsed_seconds": month_elapsed,
                    "eta_seconds": 0.0,
                    "failures": failures,
                }
            )
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


def run_native_cafe(
    root: Path | str,
    start,
    end,
    stride: int = 10,
    progress_callback=None,
    *,
    workers: int | None = None,
    parallel: bool = True,
    serial_fallback: bool = True,
    resume: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
    region_name: str | None = None,
    cadence: str = "monthly",
) -> list[dict]:
    """Run Native CAFE, distributing unchanged pixel calls across processes.

    ``workers=1`` or ``parallel=False`` provides the reference serial path.
    Parallel infrastructure failures restart the affected month serially when
    ``serial_fallback`` is enabled. Model-level pixel failures retain the legacy
    behavior: the pixel remains NaN and is counted in the report.
    """
    root = Path(root)
    stride = max(1, int(stride))
    months = list(month_starts(start, end))
    reports: list[dict] = []
    total_months = len(months)
    region = (
        region_identity(region_name, tuple(map(float, bbox)))
        if bbox is not None else None
    )
    region_metadata = (
        {
            "region_label": region.label,
            "region_id": region.region_id,
            "bbox": list(region.bbox),
            "cadence": cadence,
        }
        if region is not None else {}
    )

    requested_workers = workers
    if workers is None:
        workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    workers = max(1, int(workers))

    print(
        f"[TRIDENT] Starting Native CAFE for {total_months} month(s), "
        f"stride={stride}, workers={workers if parallel else 1}",
        flush=True,
    )

    def notify(update: dict) -> None:
        if progress_callback is not None:
            progress_callback(update)

    for month_number, month in enumerate(months, start=1):
        month_started = time.monotonic()
        ts = pd.Timestamp(month)
        tag = ts.strftime("%Y%m")
        if region is not None:
            infile, legacy_input = resolve_native_input_path(root, cadence, tag, region)
            structured_output = native_output_path(root, cadence, tag, stride, region)
            legacy_output = legacy_native_output_path(root, tag, stride)
            outfile = (
                legacy_output
                if not structured_output.exists()
                and legacy_output.exists()
                and bbox_matches_dataset(legacy_output, region.bbox)
                else structured_output
            )
        else:
            infile = root / "data/processed" / f"trident_native_cafe_inputs_{tag}.nc"
            outfile = legacy_native_output_path(root, tag, stride)
            legacy_input = True

        base_update = {
            "tag": tag,
            "month_number": month_number,
            "total_months": total_months,
        }
        notify({
            **base_update,
            "stage": "month_start",
            "pixel_number": 0,
            "total_pixels": 0,
            "fraction": (month_number - 1) / max(1, total_months),
            "elapsed_seconds": 0.0,
            "failures": 0,
        })
        print(f"[TRIDENT] Month {month_number}/{total_months}: {tag}", flush=True)

        if not infile.exists():
            message = f"Missing input file: {infile}"
            print(f"[TRIDENT] ERROR: {message}", flush=True)
            reports.append({
                **region_metadata,
                "tag": tag,
                "status": "missing_inputs",
                "input_file": str(infile),
                "message": message,
            })
            continue

        load_started = time.monotonic()
        try:
            with xr.open_dataset(infile) as source:
                missing_variables = [
                    name for name in [*REQUIRED, "lat", "lon"]
                    if name not in source.variables and name not in source.coords
                ]
                if missing_variables:
                    raise ValueError(
                        "Input file is missing required variables: "
                        + ", ".join(missing_variables)
                    )
                shape = tuple(source["chl"].shape)
                lat = np.asarray(source["lat"].values).copy()
                lon = np.asarray(source["lon"].values).copy()
                fields = {
                    name: np.asarray(source[name].values).copy()
                    for name in REQUIRED
                }
                mask = (
                    np.asarray(source["cafe_valid"].values, dtype=bool).copy()
                    if "cafe_valid" in source
                    else np.ones(shape, dtype=bool)
                )
        except Exception as exc:
            message = f"Could not validate {infile}: {exc}"
            print(f"[TRIDENT] ERROR: {message}", flush=True)
            reports.append({
                **region_metadata,
                "tag": tag,
                "status": "invalid_inputs",
                "input_file": str(infile),
                "message": message,
            })
            continue
        load_seconds = time.monotonic() - load_started

        if resume and _valid_completed_output(
            outfile,
            infile=infile,
            tag=tag,
            stride=stride,
            shape=shape,
        ):
            print(f"[TRIDENT] Existing complete output found; skipping {tag}.", flush=True)
            reports.append({
                **region_metadata,
                "tag": tag,
                "status": "already_complete",
                "input_file": str(infile),
                "output_file": str(outfile),
                "stride": stride,
            })
            if region is not None:
                update_coverage_manifest(
                    root,
                    cadence,
                    tag,
                    stride,
                    coverage_record(
                        region,
                        outfile,
                        status="complete",
                        resumed=True,
                        model_fingerprint=_cafe_model_fingerprint(),
                        input_schema_version="native-cafe-v1",
                    ),
                )
            notify({
                **base_update,
                "stage": "month_complete",
                "pixel_number": 0,
                "total_pixels": 0,
                "fraction": month_number / max(1, total_months),
                "elapsed_seconds": time.monotonic() - month_started,
                "failures": 0,
                "resumed": True,
            })
            continue

        indices = list(zip(*np.where(mask)))[::stride]
        total_pixels = len(indices)
        yd = int(ts.dayofyear + 14)
        tasks = [
            (
                int(i), int(j),
                float(fields["par"][i, j]), float(fields["chl"][i, j]),
                float(fields["mld"][i, j]), float(lat[i]), yd,
                float(fields["aph443"][i, j]), float(fields["adg443"][i, j]),
                float(fields["bbp443"][i, j]), float(fields["bbp_s"][i, j]),
                float(fields["sst"][i, j]),
            )
            for i, j in indices
        ]
        npp = np.full(shape, np.nan, dtype="float32")
        zeu = np.full(shape, np.nan, dtype="float32")
        kdpar = np.full(shape, np.nan, dtype="float32")

        if not total_pixels:
            reports.append({
                **region_metadata,
                "tag": tag,
                "status": "no_valid_pixels",
                "input_file": str(infile),
                "failures": 0,
            })
            continue

        active_workers = min(workers, total_pixels) if parallel else 1
        execution_mode = "parallel" if active_workers > 1 else "serial"
        fallback_reason = None
        failures = 0
        update_every = max(1, total_pixels // 200)
        compute_started = time.monotonic()

        def consume(results) -> None:
            nonlocal failures
            for pixel_number, result in enumerate(results, start=1):
                i, j, pixel_npp, pixel_zeu, pixel_kdpar, error = result
                if error is None:
                    npp[i, j] = pixel_npp
                    zeu[i, j] = pixel_zeu
                    kdpar[i, j] = pixel_kdpar
                else:
                    failures += 1
                if (
                    pixel_number == 1
                    or pixel_number == total_pixels
                    or pixel_number % update_every == 0
                ):
                    elapsed = time.monotonic() - compute_started
                    rate = pixel_number / elapsed if elapsed else 0.0
                    eta = (total_pixels - pixel_number) / rate if rate else None
                    month_fraction = pixel_number / total_pixels
                    overall = ((month_number - 1) + month_fraction) / total_months
                    print(
                        f"[TRIDENT] {tag}: {pixel_number:,}/{total_pixels:,} "
                        f"({month_fraction * 100:5.1f}%) | mode={execution_mode} "
                        f"workers={active_workers} | failures={failures} | "
                        f"rate={rate:,.1f} px/s | ETA={eta:,.1f}s",
                        flush=True,
                    )
                    notify({
                        **base_update,
                        "stage": "pixels",
                        "pixel_number": pixel_number,
                        "total_pixels": total_pixels,
                        "fraction": overall,
                        "month_fraction": month_fraction,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                        "pixels_per_second": rate,
                        "failures": failures,
                        "execution_mode": execution_mode,
                        "workers": active_workers,
                    })

        if execution_mode == "parallel":
            chunksize = max(1, total_pixels // (active_workers * 40))
            try:
                with ProcessPoolExecutor(max_workers=active_workers) as executor:
                    consume(executor.map(_run_cafe_pixel_task, tasks, chunksize=chunksize))
            except Exception as exc:
                if not serial_fallback:
                    raise
                fallback_reason = f"{type(exc).__name__}: {exc}"
                print(
                    f"[TRIDENT] Parallel execution failed ({fallback_reason}); "
                    "restarting this month serially.",
                    flush=True,
                )
                notify({
                    **base_update,
                    "stage": "serial_fallback",
                    "pixel_number": 0,
                    "total_pixels": total_pixels,
                    "fraction": (month_number - 1) / total_months,
                    "elapsed_seconds": time.monotonic() - compute_started,
                    "failures": 0,
                    "message": fallback_reason,
                })
                npp.fill(np.nan)
                zeu.fill(np.nan)
                kdpar.fill(np.nan)
                failures = 0
                active_workers = 1
                execution_mode = "serial_fallback"
                compute_started = time.monotonic()
                consume(map(_run_cafe_pixel_task, tasks))
        else:
            consume(map(_run_cafe_pixel_task, tasks))

        compute_seconds = time.monotonic() - compute_started
        write_started = time.monotonic()
        input_stat = infile.stat()
        out = xr.Dataset(
            {
                "cafe_npp": (("lat", "lon"), npp),
                "zeu": (("lat", "lon"), zeu),
                "kdpar": (("lat", "lon"), kdpar),
                **{
                    name: (("lat", "lon"), fields[name].astype("float32", copy=False))
                    for name in REQUIRED
                },
            },
            coords={"lat": lat, "lon": lon},
        )
        out.attrs.update({
            "title": "TRIDENT Native CAFE NPP",
            "product_family": "TRIDENT Native",
            "temporal_resolution": "monthly",
            "analysis_role": "analysis_result",
            "tag": tag,
            "source": "NASA/ESA ocean colour plus Copernicus physics",
            "source_input_file": str(infile),
            "source_input_size": input_stat.st_size,
            "source_input_mtime_ns": input_stat.st_mtime_ns,
            "profile_inputs_embedded": 1,
            "year_day_used": yd,
            "stride": stride,
            "processing_complete": 1,
            "execution_mode": execution_mode,
            "workers": active_workers,
            "candidate_pixels": total_pixels,
            "failures": failures,
            "legacy_input_reused": int(legacy_input),
            "model_fingerprint": _cafe_model_fingerprint(),
            "input_schema_version": "native-cafe-v1",
            **(region_attrs(region) if region is not None else {}),
        })
        outfile.parent.mkdir(parents=True, exist_ok=True)
        temporary = outfile.with_name(outfile.name + ".tmp.nc")
        temporary.unlink(missing_ok=True)
        out.to_netcdf(temporary)
        temporary.replace(outfile)
        write_seconds = time.monotonic() - write_started
        month_elapsed = time.monotonic() - month_started
        rate = total_pixels / compute_seconds if compute_seconds else 0.0

        report = {
            "tag": tag,
            "status": "ok",
            "input_file": str(infile),
            "output_file": str(outfile),
            "stride": stride,
            "candidate_pixels": total_pixels,
            "failures": failures,
            "success": total_pixels - failures,
            "mean_npp": float(np.nanmean(npp)) if np.isfinite(npp).any() else None,
            "execution_mode": execution_mode,
            "workers": active_workers,
            "requested_workers": requested_workers,
            "fallback_reason": fallback_reason,
            "profile": {
                "input_load_seconds": load_seconds,
                "compute_seconds": compute_seconds,
                "output_write_seconds": write_seconds,
                "total_seconds": month_elapsed,
                "pixels_per_second": rate,
            },
        }
        reports.append(report)
        if region is not None:
            report.update(region_metadata)
            coverage_path = update_coverage_manifest(
                root,
                cadence,
                tag,
                stride,
                coverage_record(
                    region,
                    outfile,
                    status="complete",
                    failures=failures,
                    candidate_pixels=total_pixels,
                    execution_mode=execution_mode,
                    model_fingerprint=_cafe_model_fingerprint(),
                    input_schema_version="native-cafe-v1",
                ),
            )
            report["coverage_manifest"] = str(coverage_path)
        print(
            f"[TRIDENT] Completed {tag}: {rate:,.1f} px/s, "
            f"{month_elapsed:,.1f}s total, failures={failures}",
            flush=True,
        )
        notify({
            **base_update,
            "stage": "month_complete",
            "pixel_number": total_pixels,
            "total_pixels": total_pixels,
            "fraction": month_number / total_months,
            "month_fraction": 1.0,
            "elapsed_seconds": month_elapsed,
            "eta_seconds": 0.0,
            "pixels_per_second": rate,
            "failures": failures,
            "execution_mode": execution_mode,
            "workers": active_workers,
        })

        out_report = root / "reports/trident/native_cafe_run_report.json"
        _write_merged_report(out_report, reports, ("tag", "region_id", "cadence", "stride"))

    out_report = root / "reports/trident/native_cafe_run_report.json"
    _write_merged_report(out_report, reports, ("tag", "region_id", "cadence", "stride"))
    return reports
