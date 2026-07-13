from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CopernicusRequest:
    dataset_id: str
    variables: tuple[str, ...]
    start_datetime: str
    end_datetime: str
    bbox: tuple[float, float, float, float]
    minimum_depth: float | None = None
    maximum_depth: float | None = None
    output_format: str = "netcdf"

    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def is_available() -> tuple[bool, str]:
    try:
        import copernicusmarine
    except Exception as exc:
        return False, f"copernicusmarine unavailable: {type(exc).__name__}: {exc}"

    return (
        True,
        f"copernicusmarine {getattr(copernicusmarine, '__version__', 'installed')}",
    )


def _default_credentials_candidates() -> list[Path]:
    home = Path.home()
    return [
        home / ".copernicusmarine" / ".copernicusmarine-credentials",
        home / ".copernicusmarine" / ".netrc",
        home / ".netrc",
    ]


def _credentials_are_configured(
    *,
    username: str | None = None,
    password: str | None = None,
    credentials_file: Path | None = None,
) -> tuple[bool, str | None]:
    env_user = os.getenv("COPERNICUSMARINE_SERVICE_USERNAME")
    env_password = os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD")

    if username and password:
        return True, "explicit username/password"
    if env_user and env_password:
        return True, "environment variables"

    candidates = (
        [Path(credentials_file).expanduser()]
        if credentials_file
        else _default_credentials_candidates()
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return True, str(candidate)

    return False, None


def login(
    *,
    username: str,
    password: str,
    configuration_file_directory: Path | None = None,
    force_overwrite: bool = False,
    check_credentials_valid: bool = True,
) -> bool:
    import copernicusmarine

    kwargs: dict[str, Any] = {
        "username": username,
        "password": password,
        "force_overwrite": bool(force_overwrite),
    }

    signature = inspect.signature(copernicusmarine.login)
    if "check_credentials_valid" in signature.parameters:
        kwargs["check_credentials_valid"] = bool(check_credentials_valid)
    if configuration_file_directory is not None:
        kwargs["configuration_file_directory"] = str(
            Path(configuration_file_directory).expanduser()
        )

    return bool(copernicusmarine.login(**kwargs))


def credentials_status(
    *,
    username: str | None = None,
    password: str | None = None,
    credentials_file: Path | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    available, message = is_available()
    result: dict[str, Any] = {
        "available": available,
        "message": message,
        "configured": False,
        "valid": False,
        "source": None,
    }
    if not available:
        return result

    configured, source = _credentials_are_configured(
        username=username,
        password=password,
        credentials_file=credentials_file,
    )
    result["configured"] = configured
    result["source"] = source

    if not configured:
        result["message"] = (
            f"{message}; credentials are not configured. "
            "Use the TRIDENT login function or the official "
            "'copernicusmarine login' command."
        )
        return result

    if not validate:
        result["valid"] = None
        return result

    try:
        import copernicusmarine

        signature = inspect.signature(copernicusmarine.login)
        kwargs: dict[str, Any] = {}

        if username is not None:
            kwargs["username"] = username
        if password is not None:
            kwargs["password"] = password
        if credentials_file is not None and "credentials_file" in signature.parameters:
            kwargs["credentials_file"] = str(Path(credentials_file).expanduser())
        if "check_credentials_valid" in signature.parameters:
            kwargs["check_credentials_valid"] = True
        if "force_overwrite" in signature.parameters:
            kwargs["force_overwrite"] = False

        result["valid"] = bool(copernicusmarine.login(**kwargs))
    except Exception as exc:
        result["valid"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def describe_dataset(dataset_id: str) -> Any:
    import copernicusmarine
    return copernicusmarine.describe(
        dataset_id=dataset_id,
        contains=[dataset_id],
    )


def request_output_path(root: Path, request: CopernicusRequest) -> Path:
    return (
        Path(root)
        / "data"
        / "reference"
        / "copernicus"
        / request.dataset_id
        / f"subset_{request.cache_key()}.nc"
    )


def subset(
    request: CopernicusRequest,
    output_file: Path,
    *,
    username: str | None = None,
    password: str | None = None,
    credentials_file: Path | None = None,
    overwrite: bool = False,
):
    import copernicusmarine

    west, south, east, north = map(float, request.bbox)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "dataset_id": request.dataset_id,
        "variables": list(request.variables),
        "minimum_longitude": west,
        "maximum_longitude": east,
        "minimum_latitude": south,
        "maximum_latitude": north,
        "start_datetime": request.start_datetime,
        "end_datetime": request.end_datetime,
        "output_filename": output_file.name,
        "output_directory": str(output_file.parent),
        "overwrite": bool(overwrite),
    }
    if request.minimum_depth is not None:
        kwargs["minimum_depth"] = float(request.minimum_depth)
    if request.maximum_depth is not None:
        kwargs["maximum_depth"] = float(request.maximum_depth)
    if username:
        kwargs["username"] = username
    if password:
        kwargs["password"] = password
    if credentials_file:
        kwargs["credentials_file"] = str(Path(credentials_file).expanduser())

    return copernicusmarine.subset(**kwargs)


def ensure_subset(
    root: Path,
    request: CopernicusRequest,
    *,
    username: str | None = None,
    password: str | None = None,
    credentials_file: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    output_file = request_output_path(root, request)

    reused = output_file.is_file() and output_file.stat().st_size > 0 and not force
    client_result: Any = None

    if not reused:
        client_result = subset(
            request,
            output_file,
            username=username,
            password=password,
            credentials_file=credentials_file,
            overwrite=force,
        )

    report = {
        "source": "Copernicus Marine",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "request": {
            **asdict(request),
            "variables": list(request.variables),
        },
        "cache_key": request.cache_key(),
        "output_file": str(output_file),
        "reused_existing": reused,
        "status": "reused" if reused else "downloaded",
        "client_result": None if client_result is None else str(client_result),
    }

    manifest = (
        root
        / "reports"
        / "trident"
        / "copernicus"
        / f"{request.dataset_id}_{request.cache_key()}.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["manifest"] = str(manifest)
    return report
