from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
from typing import Iterable

import pandas as pd
import xarray as xr


DATE_RE = re.compile(r"(?P<tag>\d{6}|\d{8})")


@dataclass(frozen=True)
class ManagedFile:
    path: str
    name: str
    category: str
    source: str
    product: str
    tag: str | None
    size_mb: float
    modified_utc: str
    readable: bool
    variables: str
    dimensions: str
    provenance_status: str
    recommendation: str
    reason: str


def _tag(path: Path) -> str | None:
    match = DATE_RE.search(path.name)
    return None if match is None else match.group("tag")[:6]


def _category(path: Path, data_root: Path) -> str:
    relative = path.relative_to(data_root)
    parts = set(relative.parts)

    if "processed" in parts:
        return "analysis"
    if "reference" in parts:
        return "download"
    if "reports" in parts:
        return "report"
    return "other"


def _source(path: Path) -> str:
    lower = str(path).lower()
    if "nasa" in lower or "earthdata" in lower:
        return "NASA Earthdata"
    if "copernicus" in lower or "cmems" in lower:
        return "Copernicus Marine"
    if "esa" in lower or "occi" in lower or "oc_cci" in lower:
        return "ESA OC-CCI"
    if "osu" in lower or "orca.science.oregonstate.edu" in lower:
        return "OSU Archive"
    if "trident" in lower:
        return "TRIDENT"
    return "Unknown"


def _product(path: Path) -> str:
    name = path.name.lower()

    rules = [
        ("validation", ("comparison", "validation")),
        ("native_cafe", ("native", "standard_cafe")),
        ("replay_cafe", ("replay", "input_replay")),
        ("npp", ("npp", "cafe")),
        ("chlorophyll", ("chlor", "chl")),
        ("par", ("par",)),
        ("sst", ("sst", "thetao")),
        ("mld", ("mld", "mlotst")),
        ("optics", ("aph", "adg", "bbp")),
    ]

    for label, tokens in rules:
        if any(token in name for token in tokens):
            return label
    return path.suffix.lower().lstrip(".") or "unknown"


def _provenance_status(attrs: dict) -> tuple[str, str, str]:
    version = attrs.get("trident_code_version")
    commit = attrs.get("trident_git_commit")

    if version is None and commit is None:
        return (
            "legacy_unknown",
            "Review",
            "No embedded TRIDENT provenance metadata.",
        )

    dirty = str(attrs.get("trident_git_dirty", "false")).lower() == "true"
    if dirty:
        return (
            "development",
            "Review before publication use",
            "Output was created from a working tree with uncommitted changes.",
        )

    return (
        "tracked",
        "Keep",
        "Embedded TRIDENT provenance metadata is present.",
    )


def inspect_file(path: Path, data_root: Path) -> ManagedFile:
    stat = path.stat()
    modified = datetime.fromtimestamp(
        stat.st_mtime,
        tz=timezone.utc,
    ).isoformat()

    variables = ""
    dimensions = ""
    readable = True
    status = "not_applicable"
    recommendation = "Keep"
    reason = "Non-NetCDF file."

    if path.suffix.lower() in {".nc", ".nc4", ".netcdf"}:
        try:
            with xr.open_dataset(path) as ds:
                variables = ", ".join(sorted(ds.data_vars))
                dimensions = ", ".join(
                    f"{name}={size}" for name, size in ds.sizes.items()
                )
                status, recommendation, reason = _provenance_status(dict(ds.attrs))
        except Exception as exc:
            readable = False
            status = "unreadable"
            recommendation = "Repair or delete"
            reason = f"{type(exc).__name__}: {exc}"

    return ManagedFile(
        path=str(path.resolve()),
        name=path.name,
        category=_category(path, data_root),
        source=_source(path),
        product=_product(path),
        tag=_tag(path),
        size_mb=stat.st_size / (1024 * 1024),
        modified_utc=modified,
        readable=readable,
        variables=variables,
        dimensions=dimensions,
        provenance_status=status,
        recommendation=recommendation,
        reason=reason,
    )


def discover_managed_files(data_root: Path) -> pd.DataFrame:
    data_root = Path(data_root).expanduser().resolve()

    roots = [
        data_root / "data" / "reference",
        data_root / "data" / "processed",
        data_root / "reports" / "trident",
    ]

    files: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        files.extend(path for path in base.rglob("*") if path.is_file())

    rows = [asdict(inspect_file(path, data_root)) for path in sorted(files)]

    if not rows:
        return pd.DataFrame(columns=[field.name for field in ManagedFile.__dataclass_fields__.values()])

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["category", "source", "product", "tag", "name"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def summary_table(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame(
            columns=["category", "source", "product", "files", "size_mb"]
        )

    return (
        catalog.groupby(["category", "source", "product"], dropna=False)
        .agg(
            files=("path", "count"),
            size_mb=("size_mb", "sum"),
        )
        .reset_index()
        .sort_values(["category", "source", "product"])
    )


def stale_candidates(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return catalog.copy()

    return catalog[
        catalog["provenance_status"].isin(
            ["legacy_unknown", "development", "unreadable"]
        )
    ].copy()


def delete_files(paths: Iterable[str | Path]) -> list[dict]:
    results = []

    for item in paths:
        path = Path(item).expanduser().resolve()
        try:
            size = path.stat().st_size if path.exists() else 0
            path.unlink(missing_ok=True)
            results.append(
                {
                    "path": str(path),
                    "deleted": True,
                    "bytes_removed": size,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "path": str(path),
                    "deleted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return results


def write_review_plan(
    catalog: pd.DataFrame,
    target: Path,
    selected_paths: Iterable[str],
) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    selected = catalog[catalog["path"].isin(set(selected_paths))].copy()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "count": int(len(selected)),
        "files": selected.to_dict(orient="records"),
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
