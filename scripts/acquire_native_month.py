#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from trident.data_sources.acquisition import acquire_month, acquisition_plan


def main():
    parser = argparse.ArgumentParser(description="Acquire verified NASA and Copernicus inputs for one TRIDENT month.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--month", required=True, help="YYYYMM")
    parser.add_argument("--bbox", nargs=4, type=float, default=(-132.0, 28.0, -116.0, 45.0), metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--nasa-login", default=None, help="earthaccess login strategy, e.g. netrc or interactive")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    bbox = tuple(args.bbox)

    if args.plan_only:
        for item in acquisition_plan(args.month, bbox):
            print(f"{item.variable:7s} {item.source:11s} {item.collection}")
            if item.note:
                print("         ", item.note)
        return

    result = acquire_month(args.root, args.month, bbox, nasa_login_strategy=args.nasa_login, force=args.force)
    print("Manifest:", result["manifest"])
    for row in result["reports"]:
        status = row["result"].get("status")
        print(f"{row['variable']:7s} {status}")


if __name__ == "__main__":
    main()
