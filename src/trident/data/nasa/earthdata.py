from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

@dataclass
class EarthdataRequest:
    sensor: str
    product: str
    start_date: str
    end_date: str
    bbox: tuple[float, float, float, float]
    temporal: str = 'monthly'


def is_available() -> tuple[bool, str]:
    try:
        import earthaccess  # noqa: F401
        return True, 'earthaccess installed'
    except Exception as e:
        return False, f'earthaccess not installed or unavailable: {e}'


def login(strategy: str = 'netrc'):
    import earthaccess
    return earthaccess.login(strategy=strategy)


def search_granules(req: EarthdataRequest, *, count: int = 200):
    """Search NASA Earthdata granules.

    This is intentionally generic because exact collection short names differ by
    sensor/product/reprocessing. The app exposes this as a production scaffold:
    once the desired collection short names are selected, the same function will
    download and cache the files.
    """
    import earthaccess
    west, south, east, north = req.bbox
    return earthaccess.search_data(
        short_name=req.product,
        temporal=(req.start_date, req.end_date),
        bounding_box=(west, south, east, north),
        count=count,
    )


def download_granules(granules, out_dir: Path):
    import earthaccess
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return earthaccess.download(granules, local_path=str(out_dir))
