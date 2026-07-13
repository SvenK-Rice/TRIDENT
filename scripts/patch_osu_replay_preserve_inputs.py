#!/usr/bin/env python3
"""Patch osu_replay.py so future regional CAFE outputs preserve profile inputs."""
from pathlib import Path

path = Path("src/trident/workflows/osu_replay.py")
text = path.read_text(encoding="utf-8")
old = '''    out = xr.Dataset(
        {
            "cafe_npp": (("lat", "lon"), npp),
            "zeu": (("lat", "lon"), zeu),
            "kdpar": (("lat", "lon"), kdpar),
        },
        coords={"lat": ds["lat"].values, "lon": ds["lon"].values},
    )
'''
new = '''    # Preserve the normalized CAFE forcing fields alongside the integrated
    # outputs. This keeps regional files compact while allowing cafe_profile()
    # to reconstruct the full depth/time/spectral diagnostics on a map click.
    out_vars = {
        "cafe_npp": (("lat", "lon"), npp),
        "zeu": (("lat", "lon"), zeu),
        "kdpar": (("lat", "lon"), kdpar),
    }
    for name in MODEL_INPUTS:
        out_vars[name] = (("lat", "lon"), np.asarray(ds[name].values, dtype="float32"))

    out = xr.Dataset(
        out_vars,
        coords={"lat": ds["lat"].values, "lon": ds["lon"].values},
    )
'''
if old not in text:
    raise SystemExit("Expected replay output block not found; no change made.")
text = text.replace(old, new)
needle = '''            "date_basis": "15th day of selected month",
'''
replacement = '''            "date_basis": "15th day of selected month",
            "source_input_file": str(infile),
            "profile_inputs_embedded": "true",
'''
if needle not in text:
    raise SystemExit("Expected attribute block not found; no change made.")
path.write_text(text.replace(needle, replacement), encoding="utf-8")
print(f"Patched {path}")
