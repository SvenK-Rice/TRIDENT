from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .discovery import DataInventory, InventoryRecord
from .registry import CAFE_REQUIRED, source_plan


@dataclass(frozen=True)
class ResolvedInput:
    name: str
    status: str
    preferred_source: str | None
    preferred_collection: str | None
    paths: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class MonthReadiness:
    month: str
    ready: bool
    inputs: tuple[ResolvedInput, ...]
    prepared_files: tuple[str, ...]
    joint_valid_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _best_records(records: list[InventoryRecord], input_name: str) -> list[InventoryRecord]:
    candidates = [record for record in records if input_name in record.matched_inputs]
    rank = {"prepared_inputs": 0, "nasa": 1, "copernicus": 1, "osu_reference": 2, "other": 3, "derived_npp": 9}
    return sorted(candidates, key=lambda record: (rank.get(record.kind, 8), str(record.path)))


def evaluate_month(inventory: DataInventory, month: str) -> MonthReadiness:
    if len(month) != 6 or not month.isdigit():
        raise ValueError("month must be YYYYMM")
    plan = source_plan(f"{month[:4]}-{month[4:]}-15")
    records = inventory.by_month(month)
    valid_prepared = [
        record for record in records
        if record.kind == "prepared_inputs"
        and set(CAFE_REQUIRED).issubset(record.matched_inputs)
        and (record.joint_valid_pixels or 0) > 0
    ]
    prepared = tuple(str(record.path) for record in valid_prepared)
    joint_valid = max((record.joint_valid_pixels or 0 for record in valid_prepared), default=0)

    resolved: list[ResolvedInput] = []
    for name in CAFE_REQUIRED:
        selected = _best_records(records, name)
        preferred = plan[name]
        status = "available" if selected else "missing"
        if selected and selected[0].kind == "osu_reference" and name == "bbp_s":
            status = "ancillary"
        resolved.append(ResolvedInput(
            name=name,
            status=status,
            preferred_source=preferred.source if preferred else None,
            preferred_collection=preferred.collection if preferred else None,
            paths=tuple(str(record.path) for record in selected[:5]),
            note=preferred.note if preferred else "",
        ))

    # A prepared file is ready only when at least one joint-valid pixel exists.
    # Separate source files are merely source-complete; grid harmonization is still required.
    source_complete = all(item.status != "missing" for item in resolved)
    ready = bool(valid_prepared)
    if source_complete and not ready:
        resolved = [
            ResolvedInput(**{**asdict(item), "note": (item.note + " Sources found; build a harmonized prepared-input file.").strip()})
            for item in resolved
        ]
    return MonthReadiness(month, ready, tuple(resolved), prepared, joint_valid)
