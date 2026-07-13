from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EarthdataRequest:
    """A reproducible NASA Earthdata granule request."""

    short_name: str
    start_date: str
    end_date: str
    bbox: tuple[float, float, float, float]
    provider: str | None = None
    count: int = 500
    variable: str | None = None
    sensor: str | None = None
    temporal: str = "monthly"
    version: str | None = None
    granule_name: str | None = None

    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def is_available() -> tuple[bool, str]:
    try:
        import earthaccess  # noqa: F401
    except Exception as exc:
        return False, f"earthaccess unavailable: {type(exc).__name__}: {exc}"
    return True, "earthaccess installed"


def login(
    strategy: str = "netrc",
    *,
    persist: bool = True,
    interactive: bool | None = None,
):
    """Authenticate with NASA Earthdata Login.

    strategy='netrc' uses ~/.netrc when present. For a first interactive login,
    use strategy='interactive'; earthaccess can save the credentials when
    persist=True.
    """
    import earthaccess

    kwargs: dict[str, Any] = {
        "strategy": strategy,
        "persist": persist,
    }
    if interactive is not None:
        kwargs["interactive"] = interactive
    return earthaccess.login(**kwargs)


def auth_status() -> dict[str, Any]:
    """Return a JSON-safe Earthdata authentication status."""
    ok, message = is_available()
    result: dict[str, Any] = {
        "available": ok,
        "message": message,
        "authenticated": False,
    }
    if not ok:
        return result

    try:
        import earthaccess

        status = earthaccess.status()
        result["authenticated"] = bool(status)
        result["raw_status"] = str(status)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def search_collections(
    keyword: str,
    *,
    provider: str | None = None,
    count: int = 25,
) -> list[dict[str, Any]]:
    """Discover NASA collections before committing to a short_name."""
    import earthaccess

    kwargs: dict[str, Any] = {
        "keyword": keyword,
        "count": int(count),
    }
    if provider:
        kwargs["provider"] = provider

    collections = earthaccess.search_datasets(**kwargs)
    rows: list[dict[str, Any]] = []

    for collection in collections:
        summary = {}
        try:
            accessor = collection.summary
            summary = accessor() if callable(accessor) else accessor
        except Exception:
            try:
                summary = dict(collection)
            except Exception:
                summary = {"raw": str(collection)}
        if not isinstance(summary, dict):
            try:
                summary = dict(summary)
            except Exception:
                summary = {"raw": str(summary)}

        rows.append(
            {
                "short_name": summary.get("short-name")
                or summary.get("ShortName")
                or summary.get("short_name"),
                "version": summary.get("version")
                or summary.get("Version"),
                "provider": summary.get("data-center")
                or summary.get("DataCenter")
                or provider,
                "title": summary.get("title")
                or summary.get("EntryTitle")
                or summary.get("dataset-title"),
                "summary": summary,
            }
        )
    return rows


def search_granules(request: EarthdataRequest) -> list[Any]:
    """Search CMR for granules matching the exact request."""
    import earthaccess

    west, south, east, north = map(float, request.bbox)
    kwargs: dict[str, Any] = {
        "short_name": request.short_name,
        "temporal": (str(request.start_date), str(request.end_date)),
        "bounding_box": (west, south, east, north),
        "count": int(request.count),
    }
    if request.provider:
        kwargs["provider"] = request.provider
    if request.version:
        kwargs["version"] = request.version
    if request.granule_name:
        kwargs["granule_name"] = request.granule_name

    return list(earthaccess.search_data(**kwargs))


def _granule_identity(granule: Any) -> dict[str, str | None]:
    """Extract stable metadata without depending on one earthaccess release."""
    concept_id = None
    native_id = None
    filename = None

    try:
        concept_id = granule["meta"].get("concept-id")
        native_id = granule["meta"].get("native-id")
    except Exception:
        pass

    try:
        filename = granule.data_links(access="external")[0].split("/")[-1]
    except Exception:
        try:
            filename = str(granule).split("/")[-1]
        except Exception:
            filename = None

    return {
        "concept_id": concept_id,
        "native_id": native_id,
        "filename": filename,
    }


def request_manifest_path(root: Path, request: EarthdataRequest) -> Path:
    return (
        Path(root)
        / "reports"
        / "trident"
        / "earthdata"
        / f"{request.variable or 'product'}_{request.cache_key()}.json"
    )


def download_granules(
    granules: Iterable[Any],
    output_directory: Path,
    *,
    threads: int = 6,
    force: bool = False,
) -> list[Path]:
    """Download granules using stable external URLs.

    Some earthaccess releases return dictionary-like DataGranule wrappers that
    are accepted by search_data() but rejected when passed back to download().
    Converting each result to its external data URL avoids that API mismatch
    and also deduplicates records that point to the same physical file.
    """
    import earthaccess

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    urls: list[str] = []

    for granule in granules:
        links: list[str] = []

        try:
            links = list(granule.data_links(access="external"))
        except Exception:
            try:
                links = list(granule.data_links())
            except Exception:
                links = []

        for url in links:
            value = str(url)
            if value and value not in urls:
                urls.append(value)

    if not urls:
        return []

    paths = earthaccess.download(
        urls,
        local_path=str(output_directory),
        threads=max(1, int(threads)),
        force=bool(force),
    )

    return [Path(path) for path in paths]


def ensure_request(
    root: Path,
    request: EarthdataRequest,
    *,
    login_strategy: str | None = None,
    threads: int = 6,
    force: bool = False,
) -> dict[str, Any]:
    """Search, download missing data, and save a reproducible manifest."""
    root = Path(root).expanduser().resolve()

    if login_strategy:
        login(strategy=login_strategy)

    granules = search_granules(request)
    identities = [_granule_identity(item) for item in granules]

    output_directory = (
        root
        / "data"
        / "reference"
        / "nasa"
        / (request.sensor or "unspecified_sensor").replace(" ", "_")
        / (request.variable or request.short_name)
        / request.cache_key()
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    expected_names = {
        item["filename"] for item in identities if item.get("filename")
    }
    existing_names = {
        path.name for path in output_directory.iterdir() if path.is_file()
    }
    missing_names = expected_names - existing_names

    if force or missing_names or not expected_names:
        downloaded = download_granules(
            granules,
            output_directory,
            threads=threads,
            force=force,
        )
    else:
        downloaded = sorted(
            path for path in output_directory.iterdir() if path.is_file()
        )

    report = {
        "source": "NASA Earthdata",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "request": asdict(request),
        "cache_key": request.cache_key(),
        "output_directory": str(output_directory),
        "n_granules_found": len(granules),
        "expected_filenames": sorted(expected_names),
        "previously_present": sorted(existing_names & expected_names),
        "missing_before_download": sorted(missing_names),
        "files": [str(path) for path in downloaded],
        "status": "ok" if downloaded or not granules else "no_files_returned",
    }

    manifest = request_manifest_path(root, request)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["manifest"] = str(manifest)
    return report
