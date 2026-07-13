from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import trident.workflows.native_prepare as native_prepare
from trident.analysis.storage_layout import (
    native_input_path,
    native_output_path,
    region_identity,
)


def _write_inputs(root: Path, shape: tuple[int, int] = (3, 4)) -> Path:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    lat = np.linspace(30.0, 32.0, shape[0], dtype="float32")
    lon = np.linspace(-124.0, -121.0, shape[1], dtype="float32")
    values = {
        "par": 45.0,
        "chl": 0.35,
        "mld": 20.0,
        "aph443": 0.025,
        "adg443": 0.015,
        "bbp443": 0.003,
        "bbp_s": 1.0,
        "sst": 20.0,
    }
    ds = xr.Dataset(
        {
            **{
                name: (("lat", "lon"), np.full(shape, value, dtype="float32"))
                for name, value in values.items()
            },
            "cafe_valid": (("lat", "lon"), np.ones(shape, dtype="uint8")),
        },
        coords={"lat": lat, "lon": lon},
    )
    path = processed / "trident_native_cafe_inputs_202301.nc"
    ds.to_netcdf(path)
    return path


def _run(root: Path, **kwargs) -> list[dict]:
    return native_prepare.run_native_cafe(
        root,
        date(2023, 1, 1),
        date(2023, 1, 31),
        stride=1,
        **kwargs,
    )


def _output(root: Path) -> xr.Dataset:
    path = root / "data/processed/trident_native_cafe_npp_202301_stride1.nc"
    return xr.open_dataset(path)


def test_parallel_matches_serial_exactly_and_reports_profile(tmp_path: Path) -> None:
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"
    _write_inputs(serial_root)
    _write_inputs(parallel_root)

    serial_report = _run(serial_root, parallel=False, resume=False)
    parallel_report = _run(parallel_root, workers=2, resume=False)

    with _output(serial_root) as serial, _output(parallel_root) as parallel:
        for name in ("cafe_npp", "zeu", "kdpar"):
            np.testing.assert_array_equal(serial[name].values, parallel[name].values)

    assert serial_report[0]["execution_mode"] == "serial"
    for report in (serial_report[0], parallel_report[0]):
        assert report["failures"] == 0
        assert report["profile"]["compute_seconds"] > 0
        assert report["profile"]["pixels_per_second"] > 0
    if parallel_report[0]["execution_mode"] == "serial_fallback":
        reason = parallel_report[0]["fallback_reason"] or ""
        if "Operation not permitted" in reason:
            pytest.skip("Execution sandbox does not permit process pools")
    assert parallel_report[0]["execution_mode"] == "parallel"


def test_complete_output_is_resumed_without_recomputation(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    first = _run(tmp_path, parallel=False, resume=True)
    output = Path(first[0]["output_file"])
    original_mtime = output.stat().st_mtime_ns

    updates: list[dict] = []
    second = _run(
        tmp_path,
        parallel=False,
        resume=True,
        progress_callback=updates.append,
    )

    assert second[0]["status"] == "already_complete"
    assert output.stat().st_mtime_ns == original_mtime
    assert any(update.get("resumed") for update in updates)


def test_parallel_infrastructure_failure_falls_back_to_exact_serial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected_root = tmp_path / "expected"
    fallback_root = tmp_path / "fallback"
    _write_inputs(expected_root, shape=(2, 2))
    _write_inputs(fallback_root, shape=(2, 2))
    _run(expected_root, parallel=False, resume=False)

    class BrokenExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("simulated process-pool failure")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(native_prepare, "ProcessPoolExecutor", BrokenExecutor)
    updates: list[dict] = []
    report = _run(
        fallback_root,
        workers=2,
        resume=False,
        progress_callback=updates.append,
    )

    with _output(expected_root) as expected, _output(fallback_root) as fallback:
        for name in ("cafe_npp", "zeu", "kdpar"):
            np.testing.assert_array_equal(expected[name].values, fallback[name].values)

    assert report[0]["execution_mode"] == "serial_fallback"
    assert any(update["stage"] == "serial_fallback" for update in updates)


def test_region_aware_run_writes_structured_shard_and_manifest(tmp_path: Path) -> None:
    bbox = (-132.0, 28.0, -116.0, 45.0)
    region = region_identity("California Current", bbox)
    flat_input = _write_inputs(tmp_path, shape=(2, 2))
    structured_input = native_input_path(tmp_path, "monthly", "202301", region)
    structured_input.parent.mkdir(parents=True, exist_ok=True)
    flat_input.replace(structured_input)

    report = _run(
        tmp_path,
        parallel=False,
        resume=False,
        bbox=bbox,
        region_name="California Current",
        cadence="monthly",
    )

    expected = native_output_path(tmp_path, "monthly", "202301", 1, region)
    assert expected.exists()
    assert Path(report[0]["output_file"]) == expected
    assert report[0]["region_id"] == region.region_id
    assert Path(report[0]["coverage_manifest"]).exists()
