#!/usr/bin/env python3
from pathlib import Path
import argparse

from trident.provenance.audit import (
    audit_saved_analyses,
    write_rerun_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit saved TRIDENT NetCDF analyses."
    )
    parser.add_argument("data_root", type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--write-plan", type=Path)
    args = parser.parse_args()

    audit = audit_saved_analyses(
        args.data_root,
        args.repo_root,
    )

    if audit.empty:
        print("No processed NetCDF files found.")
        return 0

    print(
        audit[
            [
                "status",
                "recommendation",
                "product_type",
                "filename",
                "reason",
            ]
        ].to_string(index=False)
    )

    if args.write_plan:
        write_rerun_plan(audit, args.write_plan)
        print(f"\nWrote rerun plan: {args.write_plan}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
