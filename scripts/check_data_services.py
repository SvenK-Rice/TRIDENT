#!/usr/bin/env python3
from __future__ import annotations

import json

from trident.data.nasa.earthdata import auth_status as nasa_status
from trident.data.copernicus.cmems import credentials_status as cmems_status


def main() -> int:
    report = {
        "nasa_earthdata": nasa_status(),
        "copernicus_marine": cmems_status(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
