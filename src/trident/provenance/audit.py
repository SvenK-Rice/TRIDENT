from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
from typing import Any

import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class CodeIdentity:
    version: str
    commit: str | None
    branch: str | None
    dirty: bool | None


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def current_identity(repo_root: Path) -> CodeIdentity:
    repo_root = Path(repo_root).expanduser().resolve()
    version_file = repo_root / "VERSION"
    version = (
        version_file.read_text(encoding="utf-8").strip()
        if version_file.exists()
        else "unknown"
    )
    commit = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    porcelain = _git(repo_root, "status", "--porcelain")
    dirty = None if porcelain is None else bool(porcelain)
    return CodeIdentity(version, commit, branch, dirty)


def stamp_dataset(
    dataset: xr.Dataset,
    *,
    repo_root: Path,
    workflow: str,
    model_version: str = "unspecified",
) -> xr.Dataset:
    """Attach provenance metadata to future TRIDENT NetCDF outputs."""
    identity = current_identity(repo_root)
    dataset.attrs.update(
        {
            "trident_code_version": identity.version,
            "trident_git_commit": identity.commit or "unknown",
            "trident_git_branch": identity.branch or "unknown",
            "trident_git_dirty": (
                "unknown"
                if identity.dirty is None
                else str(identity.dirty).lower()
            ),
            "trident_workflow": workflow,
            "trident_model_version": model_version,
            "trident_created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return dataset


def _product_type(path: Path) -> str:
    name = path.name.lower()
    if "comparison" in name or "validation" in name:
        return "validation"
    if "native" in name or "standard" in name:
        return "native"
    if "replay" in name:
        return "replay"
    if "npp" in name or "cafe" in name:
        return "npp"
    return "other"


def audit_file(path: Path, identity: CodeIdentity) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()

    base = {
        "path": str(path.resolve()),
        "filename": path.name,
        "product_type": _product_type(path),
        "size_mb": stat.st_size / (1024 * 1024),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }

    try:
        with xr.open_dataset(path) as dataset:
            attrs = dict(dataset.attrs)
            variables = ", ".join(sorted(dataset.data_vars))
            dimensions = ", ".join(
                f"{name}={size}"
                for name, size in dataset.sizes.items()
            )

        saved_version = attrs.get("trident_code_version")
        saved_commit = attrs.get("trident_git_commit")
        saved_dirty = str(
            attrs.get("trident_git_dirty", "false")
        ).lower() == "true"

        if saved_version is None and saved_commit is None:
            status = "legacy_unknown"
            recommendation = "Review; rerun for publication use"
            reason = "No embedded TRIDENT provenance metadata."
        elif saved_dirty:
            status = "development"
            recommendation = "Review before publication use"
            reason = "Created from code with uncommitted changes."
        elif (
            saved_commit
            and saved_commit != "unknown"
            and identity.commit
            and saved_commit != identity.commit
        ):
            status = "stale"
            recommendation = "Rerun recommended"
            reason = (
                f"Saved commit {str(saved_commit)[:12]} differs from "
                f"current {identity.commit[:12]}."
            )
        elif (
            saved_version
            and identity.version != "unknown"
            and saved_version != identity.version
        ):
            status = "stale"
            recommendation = "Rerun recommended"
            reason = (
                f"Saved version {saved_version} differs from "
                f"current {identity.version}."
            )
        else:
            status = "current"
            recommendation = "Keep"
            reason = "Embedded provenance matches the current checkout."

        return {
            **base,
            "status": status,
            "recommendation": recommendation,
            "reason": reason,
            "saved_version": saved_version,
            "saved_commit": saved_commit,
            "workflow": attrs.get("trident_workflow"),
            "model_version": attrs.get("trident_model_version"),
            "created_utc": attrs.get("trident_created_utc"),
            "variables": variables,
            "dimensions": dimensions,
            "readable": True,
        }

    except Exception as exc:
        return {
            **base,
            "status": "unreadable",
            "recommendation": "Repair or delete",
            "reason": f"{type(exc).__name__}: {exc}",
            "saved_version": None,
            "saved_commit": None,
            "workflow": None,
            "model_version": None,
            "created_utc": None,
            "variables": "",
            "dimensions": "",
            "readable": False,
        }


def audit_saved_analyses(
    data_root: Path,
    repo_root: Path,
) -> pd.DataFrame:
    processed = (
        Path(data_root).expanduser().resolve()
        / "data"
        / "processed"
    )
    identity = current_identity(repo_root)

    rows = [
        audit_file(path, identity)
        for path in sorted(processed.glob("**/*.nc"))
        if path.is_file()
    ]

    if not rows:
        return pd.DataFrame(
            columns=[
                "status",
                "recommendation",
                "product_type",
                "filename",
                "reason",
                "saved_version",
                "saved_commit",
                "path",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["status", "product_type", "modified_utc", "filename"],
            ascending=[True, True, False, True],
        )
        .reset_index(drop=True)
    )


def write_rerun_plan(
    audit: pd.DataFrame,
    target: Path,
) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    selected = audit[
        audit["status"].isin(
            ["stale", "legacy_unknown", "development", "unreadable"]
        )
    ].copy()

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "count": int(len(selected)),
        "files": selected.to_dict(orient="records"),
    }
    target.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return target
