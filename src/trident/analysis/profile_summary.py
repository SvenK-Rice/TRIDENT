"""Pure helpers for summarizing depth-resolved CAFE profiles."""
from __future__ import annotations

import numpy as np
import xarray as xr


def selected_map_value(
    field: xr.DataArray,
    lat_name: str,
    lon_name: str,
    latitude: float,
    longitude: float,
) -> float:
    """Return the nearest displayed-map value for a selected coordinate."""
    value = field.sel({lat_name: latitude, lon_name: longitude}, method="nearest")
    return float(np.asarray(value.values).squeeze())


def profile_interpretation(profile, mld: float) -> dict[str, float | str]:
    """Create deterministic, model-grounded summary metrics for one profile."""
    depth = np.asarray(profile.depth_m, dtype=float)
    npp = np.asarray(profile.npp_z, dtype=float)
    if depth.size < 2 or not np.isfinite(npp).any():
        raise ValueError("Profile does not contain finite depth-resolved NPP")

    layer = np.full(depth.shape, float(profile.delz_m), dtype=float)
    contribution = np.where(np.isfinite(npp), npp * layer, 0.0)
    total = float(contribution.sum())
    cumulative = np.cumsum(contribution)
    median_index = int(np.searchsorted(cumulative, total * 0.5)) if total > 0 else 0
    peak_index = int(np.nanargmax(npp))
    above_mld = float(contribution[depth <= float(mld)].sum()) if total > 0 else 0.0
    mld_fraction = above_mld / total if total > 0 else np.nan

    if mld_fraction >= 0.8:
        mld_text = "Production is strongly concentrated within the mixed layer."
    elif mld_fraction >= 0.5:
        mld_text = "Most production occurs within the mixed layer."
    else:
        mld_text = "A substantial fraction of production occurs below the mixed layer."

    return {
        "peak_depth_m": float(depth[peak_index]),
        "median_production_depth_m": float(depth[min(median_index, depth.size - 1)]),
        "fraction_above_mld": float(mld_fraction),
        "mld_interpretation": mld_text,
    }
