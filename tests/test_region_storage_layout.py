from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from trident.analysis.storage_layout import (
    bbox_matches_dataset,
    coverage_record,
    infer_region,
    native_input_path,
    native_output_path,
    region_identity,
    resolve_native_input_path,
    update_coverage_manifest,
)


CALIFORNIA = (-132.0, 28.0, -116.0, 45.0)
GULF = (-98.0, 18.0, -80.0, 31.0)


def _grid(path: Path, bbox: tuple[float, float, float, float]) -> None:
    west, south, east, north = bbox
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"cafe_npp": (("lat", "lon"), np.ones((2, 2), dtype="float32"))},
        coords={"lat": [south, north], "lon": [west, east]},
    ).to_netcdf(path)


def test_same_period_and_stride_have_distinct_region_paths(tmp_path: Path) -> None:
    california = region_identity("California Current", CALIFORNIA)
    gulf = region_identity("Gulf of Mexico", GULF)

    california_path = native_output_path(tmp_path, "monthly", "202301", 1, california)
    gulf_path = native_output_path(tmp_path, "monthly", "202301", 1, gulf)

    assert california_path != gulf_path
    assert "california-current" in str(california_path)
    assert "gulf-of-mexico" in str(gulf_path)
    assert "stride_001" in str(california_path)


def test_legacy_input_is_reused_only_for_matching_region(tmp_path: Path) -> None:
    legacy = tmp_path / "data/processed/trident_native_cafe_inputs_202301.nc"
    _grid(legacy, CALIFORNIA)
    california = region_identity("California Current", CALIFORNIA)
    gulf = region_identity("Gulf of Mexico", GULF)

    california_path, california_legacy = resolve_native_input_path(
        tmp_path, "monthly", "202301", california
    )
    gulf_path, gulf_legacy = resolve_native_input_path(
        tmp_path, "monthly", "202301", gulf
    )

    assert california_path == legacy
    assert california_legacy
    assert not gulf_legacy
    assert gulf_path == native_input_path(tmp_path, "monthly", "202301", gulf)
    assert bbox_matches_dataset(legacy, CALIFORNIA)
    assert not bbox_matches_dataset(legacy, GULF)


def test_coverage_manifest_preserves_multiple_regions(tmp_path: Path) -> None:
    california = region_identity("California Current", CALIFORNIA)
    gulf = region_identity("Gulf of Mexico", GULF)
    first = native_output_path(tmp_path, "monthly", "202301", 1, california)
    second = native_output_path(tmp_path, "monthly", "202301", 1, gulf)

    manifest = update_coverage_manifest(
        tmp_path, "monthly", "202301", 1,
        coverage_record(california, first, status="complete"),
    )
    update_coverage_manifest(
        tmp_path, "monthly", "202301", 1,
        coverage_record(gulf, second, status="complete"),
    )

    payload = json.loads(manifest.read_text())
    assert payload["stride"] == 1
    assert {item["region_id"] for item in payload["regions"]} == {
        california.region_id,
        gulf.region_id,
    }


def test_legacy_california_grid_is_classified_as_known_region(tmp_path: Path) -> None:
    path = tmp_path / "legacy.nc"
    _grid(path, CALIFORNIA)

    region = infer_region(path)

    assert region is not None
    assert region.label == "California Current"


def test_coverage_manifest_rejects_incompatible_model_shard(tmp_path: Path) -> None:
    california = region_identity("California Current", CALIFORNIA)
    gulf = region_identity("Gulf of Mexico", GULF)
    update_coverage_manifest(
        tmp_path, "monthly", "202301", 1,
        coverage_record(
            california, "california.nc",
            model_fingerprint="model-a",
            input_schema_version="native-cafe-v1",
        ),
    )

    with pytest.raises(ValueError, match="incompatible"):
        update_coverage_manifest(
            tmp_path, "monthly", "202301", 1,
            coverage_record(
                gulf, "gulf.nc",
                model_fingerprint="model-b",
                input_schema_version="native-cafe-v1",
            ),
        )
