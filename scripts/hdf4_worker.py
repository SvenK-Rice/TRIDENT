#!/usr/bin/env python3
"""Isolated OSU HDF4 reader.

This process deliberately imports only NumPy and pyhdf. It never imports
xarray, netCDF4, h5py, Streamlit, requests, or the TRIDENT package. Keeping
HDF4 and NetCDF/HDF5 native libraries in separate processes avoids macOS
OpenSSL/libcurl/native-library conflicts that can terminate Python outright.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pyhdf.SD import SD, SDC


def read_hdf(path: Path) -> tuple[np.ndarray, str, dict]:
    h = SD(str(path), SDC.READ)
    try:
        candidates = []
        for name in h.datasets():
            try:
                sds = h.select(name)
                arr = np.asarray(sds[:])
                attrs = dict(sds.attributes())
                if arr.ndim == 2 and min(arr.shape) > 10:
                    candidates.append((name, arr, attrs))
            except Exception:
                continue
        if not candidates:
            raise ValueError(f"No readable 2-D HDF4 science dataset found in {path}")

        chosen = None
        for name, arr, attrs in candidates:
            low = name.lower()
            if not any(k in low for k in ("lat", "lon", "qual", "mask", "count")):
                chosen = (name, arr, attrs)
                break
        if chosen is None:
            chosen = candidates[0]

        name, arr, attrs = chosen
        arr = arr.astype("float64")
        for key in ("_FillValue", "fill_value", "bad_value", "missing_value"):
            if key in attrs:
                try:
                    arr[arr == float(attrs[key])] = np.nan
                except Exception:
                    pass
        arr[~np.isfinite(arr)] = np.nan

        for key in ("scale_factor", "Slope", "slope"):
            if key in attrs:
                try:
                    arr *= float(attrs[key])
                    break
                except Exception:
                    pass
        for key in ("add_offset", "Intercept", "intercept"):
            if key in attrs:
                try:
                    arr += float(attrs[key])
                    break
                except Exception:
                    pass

        arr[(arr < -1e20) | (arr > 1e20)] = np.nan
        arr[arr <= -9990] = np.nan
        return arr.astype("float32"), name, attrs
    finally:
        try:
            h.end()
        except Exception:
            pass


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: hdf4_worker.py INPUT.hdf OUTPUT.npz", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    arr, name, attrs = read_hdf(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    safe_attrs = {}
    for k, v in attrs.items():
        try:
            json.dumps(v)
            safe_attrs[k] = v
        except Exception:
            safe_attrs[k] = str(v)
    np.savez_compressed(dst, array=arr, dataset_name=np.array(name), attrs_json=np.array(json.dumps(safe_attrs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
