from __future__ import annotations

from calendar import monthrange
from dataclasses import asdict, dataclass
from datetime import date
import importlib
import json
from pathlib import Path
from typing import Any

from .registry import PACE_START


@dataclass(frozen=True)
class AcquisitionItem:
    variable: str
    source: str
    collection: str
    status: str
    request: dict[str, Any]
    note: str = ""


def _month_parts(month: str) -> tuple[date, str, str, str]:
    if len(month) != 6 or not month.isdigit():
        raise ValueError("month must be YYYYMM")
    year, number = int(month[:4]), int(month[4:])
    last = monthrange(year, number)[1]
    start = date(year, number, 1)
    end = date(year, number, last)
    return start, start.isoformat(), end.isoformat(), f"{year:04d}{number:02d}01_{year:04d}{number:02d}{last:02d}"


def acquisition_plan(month: str, bbox: tuple[float, float, float, float]) -> tuple[AcquisitionItem, ...]:
    when, start, end, span = _month_parts(month)
    items: list[AcquisitionItem] = []

    if when >= PACE_START:
        # PACE V3.2 monthly 4 km products.  The IOP collection is a combined
        # granule containing aph_442, adg_442, bbp_442, and bbp_s.
        pace = {
            "par": (
                "PACE_OCI_L3M_PAR",
                f"*{span}*L3m*MO*PAR*V3_2*4km*",
                "par_day_planar_above",
            ),
            "chl": (
                "PACE_OCI_L3M_CHL",
                f"*{span}*L3m*MO*CHL*V3_2*4km*",
                "chlor_a",
            ),
            "aph443": (
                "PACE_OCI_L3M_IOP",
                f"*{span}*L3m*MO*IOP*V3_2*4km*",
                "aph_442",
            ),
            "adg443": (
                "PACE_OCI_L3M_IOP",
                f"*{span}*L3m*MO*IOP*V3_2*4km*",
                "adg_442",
            ),
            "bbp443": (
                "PACE_OCI_L3M_IOP",
                f"*{span}*L3m*MO*IOP*V3_2*4km*",
                "bbp_442",
            ),
            "bbp_s": (
                "PACE_OCI_L3M_IOP",
                f"*{span}*L3m*MO*IOP*V3_2*4km*",
                "bbp_s",
            ),
        }
        for variable, (collection, pattern, product_variable) in pace.items():
            note = ""
            if variable in {"aph443", "adg443", "bbp443"}:
                note = (
                    f"PACE {product_variable} at 442 nm is mapped to the "
                    "CAFE 443-nm input; the 1-nm offset is recorded in provenance."
                )
            items.append(
                AcquisitionItem(
                    variable,
                    "NASA",
                    collection,
                    "direct",
                    {
                        "short_name": collection,
                        "version": "3.2",
                        "start_date": start,
                        "end_date": end,
                        "bbox": bbox,
                        "variable": variable,
                        "sensor": "PACE_OCI",
                        "granule_name": pattern,
                    },
                    note,
                )
            )
    else:
        modis = {
            "par": ("MODISA_L3m_PAR", "PAR", "par"),
            "aph443": ("MODISA_L3m_IOP", "IOP", "aph_443"),
            "adg443": ("MODISA_L3m_IOP", "IOP", "adg_443"),
            "bbp443": ("MODISA_L3m_IOP", "IOP", "bbp_443"),
            "bbp_s": ("MODISA_L3m_IOP", "IOP", "bbp_s"),
        }
        for variable, (collection, suite, product) in modis.items():
            pattern = f"*{span}*L3m*MO*{suite}*{product}*4km*"
            items.append(AcquisitionItem(variable, "NASA", collection, "direct-search" if variable != "bbp_s" else "direct-or-ancillary", {
                "short_name": collection, "version": "2022.0", "start_date": start,
                "end_date": end, "bbox": bbox, "variable": variable,
                "sensor": "MODIS_Aqua", "granule_name": pattern,
            }, "If monthly bbp_s is unavailable, use an explicitly provenance-tracked OSU ancillary field." if variable == "bbp_s" else ""))
        items.append(AcquisitionItem("chl", "Copernicus", "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M", "direct", {
            "dataset_id": "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M",
            "variables": ("CHL",), "start_datetime": start, "end_datetime": end,
            "bbox": bbox,
        }))

    if when.year <= 2023:
        mld_dataset = "cmems_mod_glo_phy_my_0.083deg_P1M-m"
        sst_dataset = "cmems_mod_glo_phy_my_0.083deg_P1M-m"
    else:
        # The operational GLO12 product is atomized by variable group.
        # Mixed-layer depth remains in the general physics dataset, while
        # temperature is served from the dedicated thetao dataset.
        mld_dataset = "cmems_mod_glo_phy_anfc_0.083deg_P1M-m"
        sst_dataset = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m"

    items.append(
        AcquisitionItem(
            "mld",
            "Copernicus",
            mld_dataset,
            "direct",
            {
                "dataset_id": mld_dataset,
                "variables": ("mlotst",),
                "start_datetime": start,
                "end_datetime": end,
                "bbox": bbox,
            },
        )
    )

    items.append(
        AcquisitionItem(
            "sst",
            "Copernicus",
            sst_dataset,
            "direct",
            {
                "dataset_id": sst_dataset,
                "variables": ("thetao",),
                "start_datetime": start,
                "end_datetime": end,
                "bbox": bbox,
                "minimum_depth": 0.0,
                "maximum_depth": 1.0,
            },
            "Use the shallowest available thetao layer.",
        )
    )
    return tuple(items)


def _import_first(names: tuple[str, ...]):
    errors = []
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError(" | ".join(errors))


def acquire_month(
    root: str | Path,
    month: str,
    bbox: tuple[float, float, float, float],
    *,
    nasa_login_strategy: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Download verified source products for one month without preparing grids."""
    root = Path(root).expanduser().resolve()
    earthdata = _import_first(("trident.data.nasa.earthdata", "trident.nasa.earthdata"))
    cmems = _import_first(("trident.data.copernicus.cmems", "trident.copernicus.cmems"))

    reports: list[dict[str, Any]] = []
    request_cache: dict[str, dict[str, Any]] = {}
    for item in acquisition_plan(month, bbox):
        if item.source == "NASA":
            request_payload = dict(item.request)
            # PACE IOP is one combined V3.2 granule. Download it once and let
            # aph443, adg443, bbp443, and bbp_s share the cached file.
            if item.collection == "PACE_OCI_L3M_IOP":
                request_payload["variable"] = "iop"
            request = earthdata.EarthdataRequest(**request_payload)
            cache_key = f"NASA:{request.cache_key()}"
            if cache_key not in request_cache:
                request_cache[cache_key] = earthdata.ensure_request(
                    root,
                    request,
                    login_strategy=nasa_login_strategy,
                    force=force,
                )
            report = request_cache[cache_key]
        else:
            request = cmems.CopernicusRequest(**item.request)
            cache_key = f"Copernicus:{request.cache_key()}"
            if cache_key not in request_cache:
                request_cache[cache_key] = cmems.ensure_subset(
                    root, request, force=force
                )
            report = request_cache[cache_key]
        reports.append({"variable": item.variable, "plan": asdict(item), "result": report})

    payload = {"month": month, "bbox": list(map(float, bbox)), "reports": reports}
    destination = root / "reports" / "trident" / "acquisition" / f"acquire_{month}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["manifest"] = str(destination)
    return payload
