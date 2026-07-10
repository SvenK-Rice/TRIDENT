from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


_TAG_RE = re.compile(r"(?<!\d)(20\d{4})(?!\d)")


@dataclass(frozen=True)
class ProductEntry:
    path: Path
    tag: str
    family: str
    label: str


FAMILIES = {
    "osu_comparison": (
        "trident_osu_replay_vs_archived_comparison_*.nc",
        "OSU archived vs TRIDENT replay",
    ),
    "osu_replay": (
        "trident_osu_input_replay_cafe_npp_*.nc",
        "TRIDENT replay from OSU inputs",
    ),
    "native_inputs": (
        "trident_native_cafe_inputs_*.nc",
        "TRIDENT Native harmonized inputs",
    ),
    "native_npp": (
        "trident_native_cafe_npp_*.nc",
        "TRIDENT Native NPP",
    ),
}


def _tag(path: Path) -> str:
    matches = _TAG_RE.findall(path.stem)
    return matches[-1] if matches else "unknown"


def scan_products(root: Path | str, families: Iterable[str] | None = None) -> list[ProductEntry]:
    root = Path(root)
    base = root / "data" / "processed"
    selected = list(families) if families is not None else list(FAMILIES)
    entries: list[ProductEntry] = []
    for family in selected:
        pattern, label = FAMILIES[family]
        for path in sorted(base.glob(pattern)):
            entries.append(ProductEntry(path=path, tag=_tag(path), family=family, label=label))
    return sorted(entries, key=lambda e: (e.tag, e.family, e.path.name))


def group_by_family(root: Path | str) -> dict[str, list[ProductEntry]]:
    grouped: dict[str, list[ProductEntry]] = {k: [] for k in FAMILIES}
    for entry in scan_products(root):
        grouped[entry.family].append(entry)
    return grouped
