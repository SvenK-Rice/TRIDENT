from __future__ import annotations

from pathlib import Path
import shutil
import sys


def main() -> None:
    repo = Path.home() / "Documents" / "GitHub" / "TRIDENT"
    if len(sys.argv) > 1:
        repo = Path(sys.argv[1]).expanduser().resolve()

    marker = repo / ".trident_native_ui_v2_backup"
    if not marker.exists():
        raise SystemExit("No Native UI v2 backup marker was found.")

    backup_dir = Path(marker.read_text(encoding="utf-8").strip())
    workbench_backup = backup_dir / "workbench.py"
    if not workbench_backup.exists():
        raise SystemExit(f"Backup workbench is missing: {workbench_backup}")

    shutil.copy2(workbench_backup, repo / "src/trident/app/workbench.py")
    module_backup = backup_dir / "native_scientist.py"
    target_module = repo / "src/trident/app/native_scientist.py"
    if module_backup.exists():
        shutil.copy2(module_backup, target_module)
    elif target_module.exists():
        target_module.unlink()

    marker.unlink()
    print(f"Restored Native interface from {backup_dir}")


if __name__ == "__main__":
    main()
