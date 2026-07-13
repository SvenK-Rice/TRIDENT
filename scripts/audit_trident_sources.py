#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trident.data_sources import evaluate_month, scan_data_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit TRIDENT satellite sources and CAFE readiness.")
    parser.add_argument("root", nargs="?", default="/Volumes/400GB_SvenK/Satellite_data")
    parser.add_argument("--month", action="append", help="Month to evaluate in YYYYMM format. Repeatable.")
    parser.add_argument("--fast", action="store_true", help="Use filenames only; do not inspect dataset metadata.")
    parser.add_argument("--json", dest="json_path", help="Write full inventory/readiness report as JSON.")
    args = parser.parse_args()

    inventory = scan_data_root(args.root, inspect_files=not args.fast)
    months = sorted(set(args.month or [r.month for r in inventory.records if r.month]))

    print(f"Data root: {inventory.root}")
    print(f"Files indexed: {len(inventory.records)}")
    print(f"Months detected: {len(months)}")

    reports = []
    for month in months:
        report = evaluate_month(inventory, month)
        reports.append(report.to_dict())
        print("\n" + "=" * 78)
        print(f"{month}: {'CAFE READY' if report.ready else 'NOT READY'}")
        if report.prepared_files:
            print("Prepared input file(s):")
            for path in report.prepared_files:
                print("  ", path)
        for item in report.inputs:
            marker = "OK" if item.status != "missing" else "--"
            print(f"  [{marker}] {item.name:7s} {item.status:10s} preferred={item.preferred_source or '-'}")
            for path in item.paths[:2]:
                print("       ", path)

    if args.json_path:
        target = Path(args.json_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "root": str(inventory.root),
            "files_indexed": len(inventory.records),
            "readiness": reports,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
