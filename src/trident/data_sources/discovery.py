from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable

import xarray as xr

from .qc import joint_valid_mask, summarize
from .registry import CAFE_REQUIRED, REGISTRY

_MONTH_PATTERNS = (
    re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})(\d{3})(?!\d)"),
)


@dataclass(frozen=True)
class InventoryRecord:
    path: Path
    month: str | None
    kind: str
    variables: tuple[str, ...] = ()
    matched_inputs: tuple[str, ...] = ()
    error: str | None = None
    joint_valid_pixels: int | None = None
    variable_valid_pixels: tuple[tuple[str, int], ...] = ()


@dataclass
class DataInventory:
    root: Path
    records: list[InventoryRecord] = field(default_factory=list)

    def by_month(self, month: str) -> list[InventoryRecord]:
        return [record for record in self.records if record.month == month]

    def paths_for_input(self, month: str, input_name: str) -> list[Path]:
        return [record.path for record in self.by_month(month) if input_name in record.matched_inputs]


def _month_from_name(name: str) -> str | None:
    for index, pattern in enumerate(_MONTH_PATTERNS):
        match = pattern.search(name)
        if not match:
            continue
        if index == 0:
            return match.group(1) + match.group(2)
        from datetime import datetime
        return datetime.strptime(f"{int(match.group(1))}{int(match.group(2)):03d}", "%Y%j").strftime("%Y%m")
    return None


def _kind(path: Path) -> str:
    lower = str(path).lower()
    name = path.name.lower()
    if "trident_native_cafe_inputs" in name or "trident_osu_cafe_inputs" in name or "cafe_inputs" in name:
        return "prepared_inputs"
    if "cafe_npp" in name or "replay_cafe_npp" in name:
        return "derived_npp"
    if "/reference/osu/" in lower:
        return "osu_reference"
    if "/reference/nasa/" in lower or "aqua_modis" in name or "pace" in name or "viirs" in name:
        return "nasa"
    if "/reference/copernicus/" in lower or "cmems" in name:
        return "copernicus"
    return "other"


def _matched_from_names(names: Iterable[str]) -> tuple[str, ...]:
    lowered = {str(name).lower() for name in names}
    matched: list[str] = []
    for target in CAFE_REQUIRED:
        aliases = {alias.lower() for alias in REGISTRY[target].aliases}
        if lowered & aliases:
            matched.append(target)
    return tuple(matched)


def _inspect_netcdf(path: Path):
    try:
        with xr.open_dataset(path, decode_times=False) as dataset:
            names = tuple(str(variable) for variable in dataset.data_vars)
            matched = _matched_from_names(names)
            per_variable: list[tuple[str, int]] = []
            for name in matched:
                alias = next((candidate for candidate in dataset.data_vars if candidate.lower() in {a.lower() for a in REGISTRY[name].aliases}), None)
                if alias is not None:
                    per_variable.append((name, summarize(dataset[alias], name=name).valid))
            joint = None
            if set(CAFE_REQUIRED).issubset(dataset.data_vars):
                joint = int(joint_valid_mask(dataset, CAFE_REQUIRED).sum())
            return names, matched, None, joint, tuple(per_variable)
    except Exception as exc:
        return (), (), f"{type(exc).__name__}: {exc}", None, ()


def _inspect_hdf4(path: Path):
    try:
        from pyhdf.SD import SD, SDC
    except Exception as exc:
        return (), (), f"pyhdf unavailable: {exc}", None, ()
    try:
        handle = SD(str(path), SDC.READ)
        try:
            names = tuple(str(name) for name in handle.datasets())
        finally:
            handle.end()
        return names, _matched_from_names(names), None, None, ()
    except Exception as exc:
        return (), (), f"{type(exc).__name__}: {exc}", None, ()


def _infer_from_filename(path: Path) -> tuple[str, ...]:
    lower = path.name.lower()
    return tuple(
        target for target in CAFE_REQUIRED
        if any(alias.lower() in lower for alias in REGISTRY[target].aliases)
    )


def scan_data_root(root: str | Path, inspect_files: bool = True) -> DataInventory:
    root = Path(root).expanduser().resolve()
    inventory = DataInventory(root=root)
    suffixes = {".nc", ".nc4", ".h5", ".hdf5", ".hdf"}
    if not root.exists():
        return inventory

    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and candidate.suffix.lower() in suffixes):
        variables: tuple[str, ...] = ()
        matched = _infer_from_filename(path)
        error = None
        joint = None
        per_variable: tuple[tuple[str, int], ...] = ()
        if inspect_files:
            if path.suffix.lower() in {".nc", ".nc4", ".h5", ".hdf5"}:
                variables, inspected, error, joint, per_variable = _inspect_netcdf(path)
                matched = tuple(sorted(set(matched) | set(inspected)))
            else:
                variables, inspected, error, joint, per_variable = _inspect_hdf4(path)
                matched = tuple(sorted(set(matched) | set(inspected)))
        inventory.records.append(InventoryRecord(
            path=path,
            month=_month_from_name(path.name),
            kind=_kind(path),
            variables=variables,
            matched_inputs=matched,
            error=error,
            joint_valid_pixels=joint,
            variable_valid_pixels=per_variable,
        ))
    return inventory
