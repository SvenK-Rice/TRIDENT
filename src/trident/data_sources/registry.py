from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

SourceStatus = Literal["direct", "derived", "ancillary", "missing"]


@dataclass(frozen=True)
class SourceCandidate:
    source: str
    collection: str
    variable_names: tuple[str, ...]
    status: SourceStatus
    start: date | None = None
    end: date | None = None
    note: str = ""

    def supports(self, when: date) -> bool:
        return (self.start is None or when >= self.start) and (
            self.end is None or when <= self.end
        )


@dataclass(frozen=True)
class VariableSpec:
    name: str
    units: str
    aliases: tuple[str, ...]
    candidates: tuple[SourceCandidate, ...]


CAFE_REQUIRED = (
    "par", "chl", "mld", "aph443", "adg443", "bbp443", "bbp_s", "sst"
)
PACE_START = date(2024, 3, 5)

REGISTRY: dict[str, VariableSpec] = {
    "par": VariableSpec("par", "mol photons m-2 d-1", ("par", "PAR", "daily_par", "PAR_mean"), (
        SourceCandidate("NASA", "PACE_OCI_L3M_PAR", ("par", "PAR"), "direct", start=PACE_START),
        SourceCandidate("NASA", "MODISA_L3m_PAR", ("par",), "direct"),
    )),
    "chl": VariableSpec("chl", "mg m-3", ("chl", "chlor_a", "CHL", "chlorophyll"), (
        SourceCandidate("NASA", "PACE_OCI_L3M_CHL", ("chlor_a", "chl"), "direct", start=PACE_START),
        SourceCandidate("Copernicus", "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M", ("CHL",), "direct"),
        SourceCandidate("NASA", "MODISA_L3m_CHL", ("chlor_a",), "direct"),
    )),
    "mld": VariableSpec("mld", "m", ("mld", "mlotst", "MLD", "mixed_layer_depth", "mldr10_1"), (
        SourceCandidate("Copernicus", "cmems_mod_glo_phy_my_0.083deg_P1M-m", ("mlotst",), "direct", end=date(2023, 12, 31)),
        SourceCandidate("Copernicus", "cmems_mod_glo_phy_anfc_0.083deg_P1M-m", ("mlotst",), "direct"),
    )),
    "aph443": VariableSpec("aph443", "m-1", ("aph443", "aph_443", "APH443", "aph_445", "aph"), (
        SourceCandidate("NASA", "PACE_OCI_L3M_IOP", ("aph_443", "aph443"), "direct", start=PACE_START),
        SourceCandidate("NASA", "MODISA_L3m_IOP", ("aph_443",), "direct"),
    )),
    "adg443": VariableSpec("adg443", "m-1", ("adg443", "adg_443", "ADG443", "adg_445", "adg", "CDM"), (
        SourceCandidate("NASA", "PACE_OCI_L3M_IOP", ("adg_443", "adg443"), "direct", start=PACE_START),
        SourceCandidate("NASA", "MODISA_L3m_IOP", ("adg_443",), "direct"),
        SourceCandidate("Copernicus", "cmems_obs-oc_glo_bgc-optics_my_l4-multi-4km_P1M", ("CDM",), "direct", note="Fallback only; verify reference wavelength."),
    )),
    "bbp443": VariableSpec("bbp443", "m-1", ("bbp443", "bbp_443", "BBP443", "bbp_445", "bbp", "BBP"), (
        SourceCandidate("NASA", "PACE_OCI_L3M_IOP", ("bbp_443", "bbp443"), "direct", start=PACE_START),
        SourceCandidate("NASA", "MODISA_L3m_IOP", ("bbp_443",), "direct"),
        SourceCandidate("Copernicus", "cmems_obs-oc_glo_bgc-optics_my_l4-multi-4km_P1M", ("BBP",), "direct"),
    )),
    "bbp_s": VariableSpec("bbp_s", "1", ("bbp_s", "bbps", "BBP_S", "bbp_slope", "eta"), (
        SourceCandidate("NASA", "PACE spectral bbp", ("bbp_s", "eta"), "derived", start=PACE_START, note="Fit power-law slope only when multiple spectral bbp bands exist."),
        SourceCandidate("OSU", "OSU MODIS bbp_s ancillary", ("bbp_s",), "ancillary", note="Validation/continuity ancillary; preserve source date and file provenance."),
    )),
    "sst": VariableSpec("sst", "degrees_C", ("sst", "analysed_sst", "thetao", "sea_surface_temperature"), (
        SourceCandidate("Copernicus", "cmems_mod_glo_phy_my_0.083deg_P1M-m", ("thetao",), "direct", end=date(2023, 12, 31), note="Use shallowest depth."),
        SourceCandidate("Copernicus", "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m", ("thetao",), "direct", note="Use shallowest depth."),
    )),
}


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def source_plan(when: str | date | datetime) -> dict[str, SourceCandidate | None]:
    day = _as_date(when)
    return {
        name: next((candidate for candidate in REGISTRY[name].candidates if candidate.supports(day)), None)
        for name in CAFE_REQUIRED
    }
