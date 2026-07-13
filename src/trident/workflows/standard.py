from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from trident.data.nasa.earthdata import (
    EarthdataRequest,
    auth_status as earthdata_auth_status,
    ensure_request as ensure_earthdata_request,
    is_available as earthdata_available,
    search_collections,
)
from trident.data.copernicus.cmems import (
    CopernicusRequest,
    credentials_status as copernicus_credentials_status,
    ensure_subset as ensure_copernicus_subset,
    is_available as copernicus_available,
)


@dataclass
class ProductionRequest:
    start_date: str
    end_date: str
    bbox: tuple[float, float, float, float]
    temporal: str = "monthly"
    sensor: str = "MODIS Aqua"
    nasa_collections: dict[str, str] | None = None
    copernicus_dataset_id: str | None = None
    copernicus_variables: list[str] | None = None
    earthdata_strategy: str | None = None
    cmems_username: str | None = None
    cmems_password: str | None = None
    force: bool = False


# These remain editable because the exact short_name must be confirmed through
# CMR discovery for the desired mission, processing level, and reprocessing.
DEFAULT_NASA_COLLECTIONS = {
    "chl": "",
    "par": "",
    "aph443": "",
    "adg443": "",
    "bbp443": "",
    "bbp_s": "",
}

# Keep these user-editable. Copernicus dataset IDs and variables evolve.
DEFAULT_CMEMS_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_P1M-m"
DEFAULT_CMEMS_VARIABLES = ["thetao", "mlotst"]


def production_status(root, start, end, bbox):
    earth_ok, earth_message = earthdata_available()
    cm_ok, cm_message = copernicus_available()
    return {
        "mode": "TRIDENT Native acquisition",
        "earthdata_available": earth_ok,
        "earthdata_message": earth_message,
        "earthdata_auth": earthdata_auth_status(),
        "copernicus_available": cm_ok,
        "copernicus_message": cm_message,
        "copernicus_credentials": copernicus_credentials_status(),
        "requested_start": str(start),
        "requested_end": str(end),
        "bbox": list(map(float, bbox)),
        "cache_dirs": {
            "earthdata": str(Path(root) / "data/reference/nasa"),
            "copernicus": str(Path(root) / "data/reference/copernicus"),
        },
        "required_cafe_inputs": [
            "chl",
            "par",
            "aph443",
            "adg443",
            "bbp443",
            "bbp_s",
            "sst",
            "mld",
        ],
    }


def discover_nasa_collections(keyword: str, count: int = 25):
    """UI-friendly wrapper for Earthdata CMR collection discovery."""
    return search_collections(keyword, count=count)


def standard_cache_status(root, start, end):
    root = Path(root)
    dates = pd.date_range(str(start), str(end), freq="MS")
    rows = []

    for month in dates:
        tag = month.strftime("%Y%m")
        nasa = list((root / "data/reference/nasa").glob("**/*"))
        copernicus = list((root / "data/reference/copernicus").glob("**/*"))
        rows.append(
            {
                "month": tag,
                "nasa_cached_files": sum(
                    path.is_file() for path in nasa
                ),
                "copernicus_cached_files": sum(
                    path.is_file() for path in copernicus
                ),
            }
        )
    return pd.DataFrame(rows)


def download_standard_inputs(root, request: ProductionRequest):
    """Acquire and cache NASA ocean color plus Copernicus physics.

    This stage intentionally downloads and catalogs the actual source files.
    It does not yet claim that the variables are CAFE-ready; scaling, units,
    temporal aggregation, and grid harmonization are validated downstream.
    """
    root = Path(root).expanduser().resolve()
    report: dict[str, Any] = {
        "request": {
            **asdict(request),
            "cmems_password": None,
        },
        "earthdata": {},
        "copernicus": {},
        "outputs": [],
    }

    collections = dict(DEFAULT_NASA_COLLECTIONS)
    if request.nasa_collections:
        collections.update(request.nasa_collections)

    for variable, short_name in collections.items():
        if not short_name:
            report["earthdata"][variable] = {
                "status": "skipped",
                "reason": "No NASA collection short_name selected",
            }
            continue

        earth_request = EarthdataRequest(
            short_name=short_name,
            start_date=str(request.start_date),
            end_date=str(request.end_date),
            bbox=request.bbox,
            variable=variable,
            sensor=request.sensor,
            temporal=request.temporal,
        )

        try:
            result = ensure_earthdata_request(
                root,
                earth_request,
                login_strategy=request.earthdata_strategy,
                force=request.force,
            )
            report["earthdata"][variable] = result
            report["outputs"].extend(result.get("files", []))
        except Exception as exc:
            report["earthdata"][variable] = {
                "status": "error",
                "collection": short_name,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if request.copernicus_dataset_id and request.copernicus_variables:
        cm_request = CopernicusRequest(
            dataset_id=request.copernicus_dataset_id,
            variables=tuple(request.copernicus_variables),
            start_datetime=str(request.start_date),
            end_datetime=str(request.end_date),
            bbox=request.bbox,
        )
        try:
            result = ensure_copernicus_subset(
                root,
                cm_request,
                username=request.cmems_username,
                password=request.cmems_password,
                force=request.force,
            )
            report["copernicus"]["subset"] = result
            report["outputs"].append(result["output_file"])
        except Exception as exc:
            report["copernicus"]["subset"] = {
                "status": "error",
                "dataset_id": request.copernicus_dataset_id,
                "variables": request.copernicus_variables,
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        report["copernicus"]["subset"] = {
            "status": "skipped",
            "reason": "No Copernicus dataset ID or variables selected",
        }

    output = root / "reports" / "trident" / "standard_download_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_file"] = str(output)
    return report
