from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from trident.grids.align import mask_feasible


def _bbox_subset(ds, arr, bbox):
    if not ('lat' in ds.coords and 'lon' in ds.coords):
        return arr
    west, south, east, north = map(float, bbox)
    lat = np.asarray(ds['lat'].values)
    lon = np.asarray(ds['lon'].values)
    latmask = (lat >= south) & (lat <= north)
    lonmask = (lon >= west) & (lon <= east)
    if arr.ndim == 2 and latmask.any() and lonmask.any():
        return arr[np.ix_(latmask, lonmask)]
    return arr


def validation_series(root, bbox, stride=None):
    """Return regional time series with both TRIDENT replay and OSU archived NPP.

    This is the key seasonal-cycle comparison: two lines from the same selected
    region, one for TRIDENT recalculation and one for the direct OSU product.
    """
    root = Path(root)
    pattern = 'trident_osu_replay_vs_archived_comparison_*_stride*.nc'
    rows = []
    for f in sorted((root / 'data/processed').glob(pattern)):
        try:
            parts = f.stem.split('_')
            # ... comparison_YYYYMM_strideX
            tag = [p for p in parts if len(p) == 6 and p.isdigit()][-1]
            if stride is not None and f'_stride{stride}' not in f.name:
                continue
            ds = xr.open_dataset(f)
            tr = _bbox_subset(ds, mask_feasible('cafe_npp', ds['trident_replay_npp'].values), bbox)
            os = _bbox_subset(ds, mask_feasible('cafe_npp', ds['osu_archived_npp'].values), bbox)
            # For a fair seasonal-cycle comparison, average both products
            # over exactly the same pixels. This matters when the replay was run
            # with stride > 1: TRIDENT has NaNs between sampled pixels, whereas
            # the direct OSU product is complete. Averaging OSU over all pixels
            # but TRIDENT over sampled pixels makes the two lines incomparable.
            common = np.isfinite(tr) & np.isfinite(os)
            rows.append({
                'tag': tag,
                'date': pd.to_datetime(tag + '15', format='%Y%m%d'),
                'trident_replay_mean': float(np.nanmean(tr[common])) if common.any() else np.nan,
                'osu_archived_mean': float(np.nanmean(os[common])) if common.any() else np.nan,
                'bias': float(np.nanmean(tr[common] - os[common])) if common.any() else np.nan,
                'valid_pixels': int(common.sum()),
                'file': f.name,
            })
        except Exception:
            pass
    return pd.DataFrame(rows).sort_values('date') if rows else pd.DataFrame()


def standard_series(root, bbox):
    root = Path(root)
    rows = []
    for f in sorted((root / 'data/processed').glob('trident_standard_cafe_npp_*.nc')):
        try:
            tag = [p for p in f.stem.split('_') if len(p) == 6 and p.isdigit()][-1]
            ds = xr.open_dataset(f)
            arr = _bbox_subset(ds, mask_feasible('cafe_npp', ds['cafe_npp'].values), bbox)
            rows.append({'tag': tag, 'date': pd.to_datetime(tag + '15', format='%Y%m%d'), 'trident_standard_mean': float(np.nanmean(arr)), 'file': f.name})
        except Exception:
            pass
    return pd.DataFrame(rows).sort_values('date') if rows else pd.DataFrame()
