from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CopernicusRequest:
    dataset_id: str
    variables: list[str]
    start_datetime: str
    end_datetime: str
    bbox: tuple[float, float, float, float]
    minimum_depth: float | None = None
    maximum_depth: float | None = None


def is_available() -> tuple[bool, str]:
    try:
        import copernicusmarine  # noqa: F401
        return True, 'copernicusmarine installed'
    except Exception as e:
        return False, f'copernicusmarine not installed or unavailable: {e}'


def subset(req: CopernicusRequest, out_file: Path, username: str | None = None, password: str | None = None):
    import copernicusmarine
    west, south, east, north = req.bbox
    kwargs = dict(
        dataset_id=req.dataset_id,
        variables=req.variables,
        minimum_longitude=west,
        maximum_longitude=east,
        minimum_latitude=south,
        maximum_latitude=north,
        start_datetime=req.start_datetime,
        end_datetime=req.end_datetime,
        output_filename=Path(out_file).name,
        output_directory=str(Path(out_file).parent),
        force_download=False,
    )
    if req.minimum_depth is not None:
        kwargs['minimum_depth'] = req.minimum_depth
    if req.maximum_depth is not None:
        kwargs['maximum_depth'] = req.maximum_depth
    if username:
        kwargs['username'] = username
    if password:
        kwargs['password'] = password
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    return copernicusmarine.subset(**kwargs)
