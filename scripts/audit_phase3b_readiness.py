#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from trident.data_sources.discovery import scan_data_root
from trident.data_sources.resolver import evaluate_month


def main():
    parser = argparse.ArgumentParser(description="Audit true CAFE readiness with fill-value and overlap QC.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--month", action="append", required=True)
    args = parser.parse_args()

    inventory = scan_data_root(args.root)
    print(f"Files indexed: {len(inventory.records)}")
    for month in args.month:
        result = evaluate_month(inventory, month)
        print("\n" + "=" * 78)
        print(f"{month}: {'CAFE READY' if result.ready else 'NOT READY'}")
        print(f"Joint-valid pixels: {result.joint_valid_pixels}")
        for item in result.inputs:
            print(f"  {'OK' if item.status != 'missing' else '--'} {item.name:7s} {item.status:10s} preferred={item.preferred_source}")
        for path in result.prepared_files:
            print("  prepared:", path)


if __name__ == "__main__":
    main()
