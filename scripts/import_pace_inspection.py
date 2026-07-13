#!/usr/bin/env python3
"""Import already-downloaded PACE inspection files into TRIDENT's data root."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def classify(path: Path) -> str | None:
    upper = path.name.upper()
    if ".IOP." in upper:
        return "iop"
    if ".PAR." in upper:
        return "par"
    if ".CHL." in upper:
        return "chl"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "Downloads" / "TRIDENT_PACE_202501_inspection",
    )
    parser.add_argument("--month", default="202501")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    root = args.data_root.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"PACE inspection directory not found: {source}")

    copied = 0
    for path in sorted(source.glob("*.nc")):
        group = classify(path)
        if group is None or args.month not in path.name:
            continue
        destination = (
            root / "data" / "reference" / "nasa" / "PACE_OCI" /
            group / args.month / path.name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1
        print(f"Imported {group}: {destination}")

    if copied == 0:
        raise SystemExit("No matching PACE IOP/PAR/CHL NetCDF files were found.")
    print(f"\nImported {copied} PACE file(s).")


if __name__ == "__main__":
    main()
