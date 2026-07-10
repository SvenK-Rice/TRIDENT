from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path

REQUIRED = [
    "numpy",
    "pandas",
    "xarray",
    "netCDF4",
    "h5netcdf",
    "yaml",
    "tqdm",
    "matplotlib",
    "plotly",
    "streamlit",
    "requests",
]

OPTIONAL = [
    "earthaccess",
    "copernicusmarine",
    "dask",
]

def module_status(name: str) -> dict:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "installed")
        return {"installed": True, "version": str(version)}
    except Exception as exc:
        return {
            "installed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

def main() -> int:
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "required": {name: module_status(name) for name in REQUIRED},
        "optional": {name: module_status(name) for name in OPTIONAL},
    }

    Path("trident_env_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))

    missing = [
        name
        for name, status in report["required"].items()
        if not status["installed"]
    ]
    if missing:
        print("\nMissing required packages: " + ", ".join(missing))
        return 1

    print("\nEnvironment check passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
