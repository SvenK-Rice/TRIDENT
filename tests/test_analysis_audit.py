import numpy as np
import xarray as xr

from trident.provenance.audit import (
    CodeIdentity,
    audit_file,
)


IDENTITY = CodeIdentity(
    version="4.1.0",
    commit="abc123",
    branch="main",
    dirty=False,
)


def test_legacy_file_is_flagged(tmp_path):
    path = tmp_path / "legacy_replay.nc"
    xr.Dataset(
        {
            "cafe_npp": (
                ("lat", "lon"),
                np.ones((2, 2)),
            )
        },
        coords={"lat": [1, 2], "lon": [3, 4]},
    ).to_netcdf(path)

    record = audit_file(path, IDENTITY)

    assert record["status"] == "legacy_unknown"


def test_matching_provenance_is_current(tmp_path):
    path = tmp_path / "current_native.nc"
    xr.Dataset(
        {
            "cafe_npp": (
                ("lat", "lon"),
                np.ones((2, 2)),
            )
        },
        coords={"lat": [1, 2], "lon": [3, 4]},
        attrs={
            "trident_code_version": "4.1.0",
            "trident_git_commit": "abc123",
            "trident_git_dirty": "false",
        },
    ).to_netcdf(path)

    record = audit_file(path, IDENTITY)

    assert record["status"] == "current"
    assert record["recommendation"] == "Keep"


def test_old_commit_is_stale(tmp_path):
    path = tmp_path / "old_validation.nc"
    xr.Dataset(
        {
            "difference": (
                ("lat", "lon"),
                np.ones((2, 2)),
            )
        },
        coords={"lat": [1, 2], "lon": [3, 4]},
        attrs={
            "trident_code_version": "4.0.0",
            "trident_git_commit": "old456",
        },
    ).to_netcdf(path)

    record = audit_file(path, IDENTITY)

    assert record["status"] == "stale"
