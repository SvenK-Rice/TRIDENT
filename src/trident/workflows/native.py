from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any

from trident.data.esa.occi import ESAOceanColourRequest, download_esa_ocean_colour
from trident.workflows.standard import ProductionRequest, download_standard_inputs


@dataclass
class NativeWorkflowRequest:
    start_date: str
    end_date: str
    bbox: tuple[float, float, float, float]
    temporal: str
    use_nasa: bool
    use_esa: bool
    use_copernicus_physics: bool
    nasa_sensor: str = "MODIS Aqua"
    nasa_collections: dict[str, str] | None = None
    esa_product_name: str | None = None
    esa_dataset_id: str | None = None
    esa_variables: list[str] | None = None
    physical_dataset_id: str | None = None
    physical_variables: list[str] | None = None
    cmems_username: str | None = None
    cmems_password: str | None = None


def run_native_download(root: Path, req: NativeWorkflowRequest) -> dict[str, Any]:
    """Download/cache only the sources selected in the app.

    This stage deliberately stops before CAFE execution. The CAFE-ready
    harmonizer will be enabled only after every selected variable is inspected
    for units, scale factors, coordinate orientation, and temporal aggregation.
    """
    root = Path(root)
    report: dict[str, Any] = {"request": asdict(req), "nasa_copernicus": None, "esa": None}

    if req.use_nasa or req.use_copernicus_physics:
        prod = ProductionRequest(
            start_date=req.start_date,
            end_date=req.end_date,
            bbox=req.bbox,
            temporal=req.temporal,
            sensor=req.nasa_sensor,
            nasa_collections=req.nasa_collections if req.use_nasa else {},
            copernicus_dataset_id=req.physical_dataset_id if req.use_copernicus_physics else None,
            copernicus_variables=req.physical_variables if req.use_copernicus_physics else [],
            cmems_username=req.cmems_username,
            cmems_password=req.cmems_password,
        )
        report["nasa_copernicus"] = download_standard_inputs(root, prod)

    if req.use_esa and req.esa_dataset_id and req.esa_variables:
        ereq = ESAOceanColourRequest(
            start_date=req.start_date,
            end_date=req.end_date,
            bbox=req.bbox,
            temporal=req.temporal,
            product_name=req.esa_product_name or "ESA ocean colour",
            dataset_id=req.esa_dataset_id,
            variables=list(req.esa_variables),
            username=req.cmems_username,
            password=req.cmems_password,
        )
        report["esa"] = download_esa_ocean_colour(root, ereq)

    path = root / "reports/trident/native_download_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return report
