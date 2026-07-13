"""Discover TRIDENT analyses and resolve canonical prepared CAFE inputs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import xarray as xr

from trident.analysis.storage_layout import infer_region

CAFE_INPUTS = ("par", "chl", "mld", "aph443", "adg443", "bbp443", "bbp_s", "sst")
MAP_VARIABLE_PRIORITY = ("cafe_npp", "trident_replay_npp", "chl", "par", "sst", "mld")
_TAG_RE = re.compile(r"(?<!\d)(20\d{4})(?!\d)")
_DAY_TAG_RE = re.compile(r"(?<!\d)(20\d{5})(?!\d)")
_STRIDE_RE = re.compile(r"(?:^|[_-])stride[_-]?(\d+)(?:[_\.-]|$)", re.IGNORECASE)
_RESULT_VARIABLES = {
    "cafe_npp", "trident_replay_npp", "osu_archived_npp", "difference",
    "zeu", "kdpar",
}


@dataclass(frozen=True)
class AnalysisDataset:
    path: Path
    title: str
    variables: tuple[str, ...]
    companion_input_path: Path | None
    has_embedded_cafe_inputs: bool
    product_family: str = "Other"
    cadence: str = "Unspecified"
    period: str = "Unknown"
    stride: int | None = None
    role: str = "Regional analysis"
    region_label: str = "Unspecified region"
    region_id: str = "unspecified"
    bbox: tuple[float, float, float, float] | None = None

    @property
    def profile_ready(self) -> bool:
        return self.has_embedded_cafe_inputs or self.companion_input_path is not None

    @property
    def display_name(self) -> str:
        if self.companion_input_path is not None:
            suffix = "canonical prepared inputs linked"
        elif self.has_embedded_cafe_inputs:
            suffix = "profile inputs embedded"
        else:
            suffix = "map only"
        return f"{self.path.name} — {suffix}"

    @property
    def organization_label(self) -> str:
        stride = f"stride {self.stride}" if self.stride is not None else "full/native grid"
        return f"{self.region_label} | {self.period} | {stride} | {self.path.name}"


def _candidate_roots(root: Path) -> list[Path]:
    """Return TRIDENT-managed analysis directories, never the whole data root."""
    candidates = [root / "data" / "processed", root / "processed", root / "analyses"]
    return [candidate for candidate in candidates if candidate.exists()]


def _netcdf_paths(root: Path) -> list[Path]:
    found: set[Path] = set()
    # Preserve support for callers that pass a processed directory directly,
    # and for legacy files stored immediately below the project root. Do not
    # recurse here: a data root may be an external drive containing millions
    # of unrelated files.
    for pattern in ("*.nc", "*.nc4", "*.cdf"):
        found.update(path.resolve() for path in root.glob(pattern) if path.is_file())
    for candidate in _candidate_roots(root):
        for pattern in ("*.nc", "*.nc4", "*.cdf"):
            found.update(path.resolve() for path in candidate.rglob(pattern) if path.is_file())
    return sorted(found)


def _report_paths(root: Path) -> list[Path]:
    """Find reports without recursively walking an external-drive data root."""
    found = {path.resolve() for path in root.glob("*.json") if path.is_file()}
    reports = root / "reports"
    if reports.exists():
        found.update(path.resolve() for path in reports.rglob("*.json") if path.is_file())
    return sorted(found)


def _report_records(payload) -> list[dict]:
    if isinstance(payload, dict):
        records = [payload]
        for value in payload.values():
            if isinstance(value, (dict, list)):
                records.extend(_report_records(value))
        return records
    if isinstance(payload, list):
        records: list[dict] = []
        for value in payload:
            records.extend(_report_records(value))
        return records
    return []


def _resolve_recorded_path(value, *, report: Path, root: Path) -> Path | None:
    if not value:
        return None
    candidate = Path(str(value)).expanduser()
    possibilities = [candidate]
    if not candidate.is_absolute():
        possibilities.extend([report.parent / candidate, root / candidate])
    for possibility in possibilities:
        try:
            resolved = possibility.resolve()
        except OSError:
            continue
        if resolved.exists():
            return resolved
    return None


def _read_report_pairs(root: Path) -> dict[Path, Path]:
    pairs: dict[Path, Path] = {}
    for report in _report_paths(root):
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        for record in _report_records(payload):
            output = _resolve_recorded_path(record.get("output_file"), report=report, root=root)
            source = _resolve_recorded_path(
                record.get("input_file") or record.get("prepared_input_file") or record.get("source_input_file"),
                report=report,
                root=root,
            )
            if output is not None and source is not None:
                pairs[output] = source
    return pairs


def _attr_companion(ds: xr.Dataset, dataset_path: Path, root: Path) -> Path | None:
    for key in (
        "prepared_input_file", "source_input_file", "cafe_input_file", "input_file",
        "parent_dataset", "derived_from",
    ):
        value = ds.attrs.get(key)
        if not value:
            continue
        candidate = Path(str(value)).expanduser()
        possibilities = [candidate]
        if not candidate.is_absolute():
            possibilities.extend([dataset_path.parent / candidate, root / candidate])
        for possibility in possibilities:
            try:
                resolved = possibility.resolve()
            except OSError:
                continue
            if resolved.exists():
                return resolved
    return None


def _contains_cafe_inputs(path: Path) -> bool:
    try:
        with xr.open_dataset(path) as ds:
            return all(name in ds.data_vars for name in CAFE_INPUTS)
    except (OSError, ValueError):
        return False


def _tag(path: Path, attrs: dict | None = None) -> str | None:
    if attrs:
        value = attrs.get("tag")
        if value and re.fullmatch(r"20\d{4}", str(value)):
            return str(value)
    match = _TAG_RE.search(path.name)
    return match.group(1) if match else None


def _analysis_metadata(
    path: Path,
    attrs: dict,
    variables: tuple[str, ...],
) -> tuple[str, str, str, int | None, str]:
    """Classify an analysis for a compact scientist-facing catalog."""
    name = path.name.lower()
    title = str(attrs.get("title", "")).lower()
    combined = f"{name} {title}"

    declared_family = str(attrs.get("product_family", "")).strip()
    if declared_family:
        family = declared_family
    elif "native" in combined:
        family = "TRIDENT Native"
    elif "osu" in combined:
        family = "OSU Validation"
    elif "pace" in combined or "oci" in combined:
        family = "PACE / OCI"
    else:
        family = "Other"

    cadence_value = " ".join(
        str(attrs.get(key, ""))
        for key in ("temporal_resolution", "temporal", "cadence")
    ).lower()
    if "8-day" in combined or "8day" in combined or "8 day" in cadence_value:
        cadence = "8-day"
    elif _tag(path, attrs) or "month" in combined or "month" in cadence_value:
        cadence = "Monthly"
    else:
        cadence = "Unspecified"

    tag = _tag(path, attrs)
    if tag:
        period = f"{tag[:4]}-{tag[4:]}"
    else:
        day_match = _DAY_TAG_RE.search(path.name)
        period = day_match.group(1) if day_match else "Unknown"

    stride_value = attrs.get("stride")
    if stride_value is None:
        match = _STRIDE_RE.search(path.name)
        stride_value = match.group(1) if match else None
    try:
        stride = int(stride_value) if stride_value is not None else None
    except (TypeError, ValueError):
        stride = None

    variable_set = set(variables)
    declared_role = str(attrs.get("analysis_role", "")).strip().lower()
    if declared_role == "prepared_inputs":
        role = "Prepared inputs"
    elif declared_role == "analysis_result":
        role = "Analysis result"
    elif "comparison" in combined or "difference" in variable_set:
        role = "Comparison"
    elif variable_set.intersection(_RESULT_VARIABLES):
        role = "Analysis result"
    elif all(field in variable_set for field in CAFE_INPUTS):
        role = "Prepared inputs"
    else:
        role = "Regional analysis"
    return family, cadence, period, stride, role


def _filename_companion(dataset_path: Path, root: Path, attrs: dict) -> Path | None:
    """Find canonical prepared inputs when legacy provenance is missing.

    Candidates must contain all eight CAFE inputs. Matching month, prepared/input
    terminology, and proximity to the derived output increase the score.
    """
    target_tag = _tag(dataset_path, attrs)
    candidates: list[tuple[int, Path]] = []
    for candidate in _netcdf_paths(root):
        if candidate == dataset_path or not _contains_cafe_inputs(candidate):
            continue
        score = 0
        name = candidate.name.lower()
        if target_tag and _tag(candidate) == target_tag:
            score += 100
        if "input" in name or "prepared" in name:
            score += 40
        if "cafe_inputs" in name:
            score += 30
        if candidate.parent == dataset_path.parent:
            score += 20
        try:
            if candidate.stat().st_mtime <= dataset_path.stat().st_mtime:
                score += 5
        except OSError:
            pass
        if score > 0:
            candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    best_score, best = candidates[0]
    # Require a meaningful relationship; do not link an arbitrary input file.
    return best if best_score >= 60 else None


def _contains_horizontal_data(ds: xr.Dataset) -> bool:
    return bool(ds.data_vars) and any(
        {"lat", "lon"}.issubset(set(da.dims)) or {"latitude", "longitude"}.issubset(set(da.dims))
        for da in ds.data_vars.values()
    )


def discover_analysis_datasets(root: str | Path) -> list[AnalysisDataset]:
    """Discover regional products and resolve their canonical prepared inputs."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return []
    report_pairs = _read_report_pairs(root_path)
    entries: list[AnalysisDataset] = []
    for path in _netcdf_paths(root_path):
        try:
            with xr.open_dataset(path) as opened:
                if not _contains_horizontal_data(opened):
                    continue
                variables = tuple(sorted(opened.data_vars))
                embedded = all(name in opened.data_vars for name in CAFE_INPUTS)
                attrs = dict(opened.attrs)
                # Prefer canonical parent data even if an older output duplicated inputs.
                companion = (
                    _attr_companion(opened, path, root_path)
                    or report_pairs.get(path)
                    or _filename_companion(path, root_path, attrs)
                )
                if companion == path:
                    companion = None
                title = str(opened.attrs.get("title") or path.stem)
        except (OSError, ValueError):
            continue
        family, cadence, period, stride, role = _analysis_metadata(path, attrs, variables)
        region = infer_region(path, attrs)
        entries.append(AnalysisDataset(
            path,
            title,
            variables,
            companion,
            embedded,
            family,
            cadence,
            period,
            stride,
            role,
            region.label if region is not None else "Unspecified region",
            region.region_id if region is not None else "unspecified",
            region.bbox if region is not None else None,
        ))
    return entries


def open_analysis_dataset(entry: AnalysisDataset) -> xr.Dataset:
    """Load a derived map and attach canonical prepared CAFE inputs.

    Derived variables remain authoritative for maps. CAFE forcing variables are
    taken from the prepared-input parent whenever one is available.
    """
    with xr.open_dataset(entry.path) as opened:
        output = opened.load()
    if entry.companion_input_path is None:
        return output
    with xr.open_dataset(entry.companion_input_path) as opened:
        inputs = opened.load()
    additions = {name: inputs[name] for name in CAFE_INPUTS if name in inputs}
    if not additions:
        return output
    # Drop duplicated forcing fields from the derived product so the canonical
    # prepared dataset is the sole source of profile inputs.
    output = output.drop_vars([name for name in CAFE_INPUTS if name in output], errors="ignore")
    merged = xr.merge([output, xr.Dataset(additions)], compat="override", join="left")
    merged.attrs.update(output.attrs)
    merged.attrs["resolved_profile_input_file"] = str(entry.companion_input_path)
    merged.attrs["profile_input_source"] = "canonical_prepared_dataset"
    return merged


def default_map_variable(variables: Iterable[str]) -> str:
    names = list(variables)
    for preferred in MAP_VARIABLE_PRIORITY:
        if preferred in names:
            return preferred
    if not names:
        raise ValueError("Dataset contains no variables")
    return names[0]
