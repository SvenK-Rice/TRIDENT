from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import json
from typing import Any

from trident.data.copernicus.cmems import (
    CopernicusRequest,
    is_available as copernicus_available,
    subset as copernicus_subset,
)

# ESA OC-CCI-derived multi-sensor ocean-colour products distributed through
# Copernicus Marine. Dataset IDs may evolve, so the app keeps them editable.
ESA_PRODUCTS: dict[str, dict[str, Any]] = {
    "ESA OC-CCI / Copernicus Marine multi-year L4 (1997–ongoing)": {
        "dataset_id": "OCEANCOLOUR_GLO_BGC_L4_MY_009_104",
        "temporal": "daily/monthly",
        "resolution": "4 km",
        "variables": ["CHL", "BBP", "CDM", "KD490", "PP"],
        "description": "Gap-filled multi-sensor ocean-colour record derived from SeaWiFS, MODIS, MERIS, VIIRS and OLCI inputs.",
    },
    "ESA/Copernicus GlobColour near-real-time L4 (2023–present)": {
        "dataset_id": "OCEANCOLOUR_GLO_BGC_L4_NRT_009_102",
        "temporal": "daily/monthly",
        "resolution": "4 km",
        "variables": ["CHL", "BBP", "CDM", "KD490", "PP"],
        "description": "Near-real-time, multi-sensor, gap-filled ocean-colour product including OLCI.",
    },
    "ESA OC-CCI / Copernicus Marine multi-year L3 reflectance": {
        "dataset_id": "OCEANCOLOUR_GLO_BGC_L3_MY_009_107",
        "temporal": "daily/monthly",
        "resolution": "4 km",
        "variables": ["CHL", "RRS412", "RRS443", "RRS490", "RRS510", "RRS560", "RRS665"],
        "description": "Multi-year level-3 chlorophyll and remote-sensing reflectance record based on ESA-CCI inputs.",
    },
}


@dataclass
class ESAOceanColourRequest:
    start_date: str
    end_date: str
    bbox: tuple[float, float, float, float]
    product_name: str
    dataset_id: str
    variables: list[str]
    temporal: str = "monthly"
    username: str | None = None
    password: str | None = None


def esa_status() -> dict[str, Any]:
    ok, message = copernicus_available()
    return {
        "available": ok,
        "message": message,
        "products": ESA_PRODUCTS,
        "access_route": "Copernicus Marine Toolbox",
        "note": (
            "TRIDENT accesses ESA OC-CCI-derived and OLCI-containing products "
            "through the Copernicus Marine Toolbox, which supports spatial and "
            "temporal subsetting and avoids downloading global files."
        ),
    }


def _safe_tag(value: str) -> str:
    return str(value)[:10].replace("-", "")


def _cache_path(root: Path, req: ESAOceanColourRequest) -> Path:
    dataset = req.dataset_id.replace("/", "_")
    start = _safe_tag(req.start_date)
    end = _safe_tag(req.end_date)
    west, south, east, north = req.bbox
    bbox_tag = f"W{west:g}_S{south:g}_E{east:g}_N{north:g}".replace("-", "m").replace(".", "p")
    var_tag = "-".join(req.variables)
    return root / "data/reference/esa" / dataset / f"esa_{start}_{end}_{req.temporal}_{bbox_tag}_{var_tag}.nc"


def inspect_copernicus_dataset(dataset_id: str) -> dict[str, Any]:
    """Best-effort metadata discovery without making it a hard dependency.

    Copernicus Marine Toolbox APIs have changed over time. This function tries
    the current describe call and returns a structured error instead of breaking
    the Streamlit app when an older client is installed.
    """
    try:
        import copernicusmarine
    except Exception as exc:
        return {"ok": False, "error": f"copernicusmarine unavailable: {exc}"}

    try:
        describe = getattr(copernicusmarine, "describe", None)
        if describe is None:
            return {"ok": False, "error": "Installed copernicusmarine client has no describe() function."}
        result = describe(dataset_id=dataset_id, disable_progress_bar=True)
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        elif not isinstance(result, (dict, list, str, int, float, bool, type(None))):
            result = str(result)
        return {"ok": True, "dataset_id": dataset_id, "metadata": result}
    except Exception as exc:
        return {"ok": False, "dataset_id": dataset_id, "error": str(exc)}


def download_esa_ocean_colour(root: Path, req: ESAOceanColourRequest) -> dict[str, Any]:
    root = Path(root)
    out_file = _cache_path(root, req)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    report_path = root / "reports/trident/esa_download_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and out_file.stat().st_size > 0:
        report = {
            "status": "cached",
            "path": str(out_file),
            "request": asdict(req),
        }
        report_path.write_text(json.dumps(report, indent=2))
        return report

    ok, message = copernicus_available()
    if not ok:
        report = {"status": "error", "error": message, "request": asdict(req)}
        report_path.write_text(json.dumps(report, indent=2))
        return report

    creq = CopernicusRequest(
        dataset_id=req.dataset_id,
        variables=list(req.variables),
        start_datetime=str(req.start_date),
        end_datetime=str(req.end_date),
        bbox=tuple(map(float, req.bbox)),
    )
    try:
        result = copernicus_subset(
            creq,
            out_file,
            username=req.username,
            password=req.password,
        )
        report = {
            "status": "downloaded",
            "path": str(out_file),
            "request": asdict(req),
            "result": str(result),
        }
    except Exception as exc:
        report = {
            "status": "error",
            "error": str(exc),
            "request": asdict(req),
            "expected_path": str(out_file),
        }
    report_path.write_text(json.dumps(report, indent=2))
    return report
