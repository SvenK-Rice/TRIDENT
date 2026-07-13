"""Region-aware storage identities and coverage manifests for TRIDENT analyses."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import xarray as xr


KNOWN_REGIONS = {
    "California Current": (-132.0, 28.0, -116.0, 45.0),
    "CCE-LTER / CalCOFI broad": (-130.0, 28.0, -116.0, 45.0),
    "Gulf of Mexico": (-98.0, 18.0, -80.0, 31.0),
    "North Atlantic": (-80.0, 20.0, -10.0, 60.0),
    "Equatorial Pacific": (-170.0, -10.0, -80.0, 10.0),
    "Southern Ocean": (-180.0, -75.0, 180.0, -45.0),
    "Global": (-180.0, -90.0, 180.0, 90.0),
}


@dataclass(frozen=True)
class RegionIdentity:
    label: str
    slug: str
    region_id: str
    bbox: tuple[float, float, float, float]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "custom-region"


def region_identity(
    label: str | None,
    bbox: tuple[float, float, float, float],
) -> RegionIdentity:
    normalized = tuple(round(float(value), 6) for value in bbox)
    slug = _slugify(label or "custom-region")
    digest = hashlib.sha1(
        ",".join(f"{value:.6f}" for value in normalized).encode("ascii")
    ).hexdigest()[:10]
    return RegionIdentity(label or "Custom region", slug, f"{slug}__{digest}", normalized)


def native_period_root(root: Path | str, cadence: str, tag: str) -> Path:
    cadence_slug = _slugify(cadence)
    return (
        Path(root)
        / "data/processed/analyses/trident-native"
        / cadence_slug
        / tag[:4]
        / tag
    )


def native_input_path(
    root: Path | str,
    cadence: str,
    tag: str,
    region: RegionIdentity,
) -> Path:
    return native_period_root(root, cadence, tag) / "inputs/regions" / region.region_id / "cafe_inputs.nc"


def native_output_path(
    root: Path | str,
    cadence: str,
    tag: str,
    stride: int,
    region: RegionIdentity,
) -> Path:
    return (
        native_period_root(root, cadence, tag)
        / f"stride_{int(stride):03d}"
        / "regions"
        / region.region_id
        / "cafe_npp.nc"
    )


def legacy_native_input_path(root: Path | str, tag: str) -> Path:
    return Path(root) / "data/processed" / f"trident_native_cafe_inputs_{tag}.nc"


def legacy_native_output_path(root: Path | str, tag: str, stride: int) -> Path:
    return Path(root) / "data/processed" / f"trident_native_cafe_npp_{tag}_stride{int(stride)}.nc"


def dataset_bbox(path: Path | str) -> tuple[float, float, float, float] | None:
    try:
        with xr.open_dataset(path) as ds:
            lon_name = "lon" if "lon" in ds.coords else "longitude"
            lat_name = "lat" if "lat" in ds.coords else "latitude"
            lon = np.asarray(ds[lon_name].values, dtype=float)
            lat = np.asarray(ds[lat_name].values, dtype=float)
            return (
                float(np.nanmin(lon)), float(np.nanmin(lat)),
                float(np.nanmax(lon)), float(np.nanmax(lat)),
            )
    except (OSError, KeyError, ValueError):
        return None


def bbox_matches_dataset(
    path: Path | str,
    bbox: tuple[float, float, float, float],
    tolerance: float = 0.1,
) -> bool:
    observed = dataset_bbox(path)
    if observed is None:
        return False
    return all(abs(left - right) <= tolerance for left, right in zip(observed, bbox))


def infer_region(path: Path | str, attrs: dict | None = None) -> RegionIdentity | None:
    attrs = attrs or {}
    bbox = None
    encoded = attrs.get("bbox_json")
    if encoded:
        try:
            values = json.loads(str(encoded))
            if len(values) == 4:
                bbox = tuple(map(float, values))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    if bbox is None:
        bbox = dataset_bbox(path)
    if bbox is None:
        return None

    label = str(attrs.get("region_label", "")).strip() or None
    if label is None:
        for known_label, known_bbox in KNOWN_REGIONS.items():
            if all(abs(left - right) <= 0.1 for left, right in zip(bbox, known_bbox)):
                label = known_label
                bbox = known_bbox
                break
    identity = region_identity(label, bbox)
    declared_id = str(attrs.get("region_id", "")).strip()
    if declared_id:
        return RegionIdentity(identity.label, identity.slug, declared_id, identity.bbox)
    return identity


def resolve_native_input_path(
    root: Path | str,
    cadence: str,
    tag: str,
    region: RegionIdentity,
) -> tuple[Path, bool]:
    structured = native_input_path(root, cadence, tag, region)
    if structured.exists():
        return structured, False
    legacy = legacy_native_input_path(root, tag)
    if legacy.exists() and bbox_matches_dataset(legacy, region.bbox):
        return legacy, True
    return structured, False


def update_coverage_manifest(
    root: Path | str,
    cadence: str,
    tag: str,
    stride: int,
    record: dict,
) -> Path:
    """Register an immutable regional shard in its compatible coverage group."""
    path = native_period_root(root, cadence, tag) / f"stride_{int(stride):03d}" / "coverage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "cadence": cadence,
        "tag": tag,
        "stride": int(stride),
        "compatibility": {},
        "regions": [],
    }
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload.update(existing)
        except (OSError, json.JSONDecodeError):
            pass
    proposed = {
        key: record[key]
        for key in ("model_fingerprint", "input_schema_version")
        if record.get(key) is not None
    }
    established = payload.get("compatibility", {})
    for key, value in proposed.items():
        if key in established and established[key] != value:
            raise ValueError(
                f"Coverage shard is incompatible: {key}={value!r}, "
                f"expected {established[key]!r}"
            )
    payload["compatibility"] = {**established, **proposed}
    regions = [item for item in payload.get("regions", []) if item.get("region_id") != record.get("region_id")]
    regions.append(record)
    payload["regions"] = sorted(regions, key=lambda item: str(item.get("region_id", "")))
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def region_attrs(region: RegionIdentity) -> dict:
    return {
        "region_label": region.label,
        "region_slug": region.slug,
        "region_id": region.region_id,
        "bbox_json": json.dumps(list(region.bbox)),
    }


def coverage_record(region: RegionIdentity, output_file: Path | str, **extra) -> dict:
    return {
        **asdict(region),
        "bbox": list(region.bbox),
        "output_file": str(output_file),
        **extra,
    }
