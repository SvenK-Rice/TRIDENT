from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any


@dataclass
class MergePolicy:
    ocean_colour_priority: list[str]
    physical_priority: list[str]
    allow_gap_fill: bool = True
    target_grid: str = "highest-common-resolution"


DEFAULT_POLICY = MergePolicy(
    ocean_colour_priority=["NASA", "ESA_OC_CCI", "COPERNICUS_MARINE_OC"],
    physical_priority=["COPERNICUS_MARINE"],
    allow_gap_fill=True,
)


def build_source_manifest(root: Path, policy: MergePolicy = DEFAULT_POLICY) -> dict[str, Any]:
    """Inventory candidate files for a future harmonized TRIDENT input cube.

    This function is intentionally conservative: it records provenance and
    source priority but does not silently combine variables with unverified
    units or retrieval definitions.
    """
    root = Path(root)
    patterns = {
        "NASA": root / "data/reference/nasa",
        "ESA_OC_CCI": root / "data/reference/esa",
        "COPERNICUS_MARINE": root / "data/reference/copernicus",
        "OSU_REFERENCE": root / "data/reference/osu",
    }
    sources = {}
    for name, directory in patterns.items():
        files = sorted(str(p) for p in directory.glob("**/*") if p.is_file()) if directory.exists() else []
        sources[name] = {"directory": str(directory), "n_files": len(files), "files": files[:500]}
    manifest = {
        "policy": asdict(policy),
        "sources": sources,
        "status": "inventory_only",
        "note": (
            "No silent merging is performed until variable names, units, scale "
            "factors, retrieval definitions, and temporal compositing are validated."
        ),
    }
    out = root / "reports/trident/native_source_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    return manifest
