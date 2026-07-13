from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import sys


def replace_function(text: str, function_name: str, replacement: str) -> str:
    marker = f"def {function_name}("
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not find {function_name} in workbench.py")

    next_start = text.find("\ndef ", start + len(marker))
    if next_start < 0:
        raise RuntimeError(f"Could not locate the end of {function_name}")

    return text[:start] + replacement.rstrip() + "\n\n" + text[next_start + 1 :]


def main() -> None:
    repo = Path.home() / "Documents" / "GitHub" / "TRIDENT"
    if len(sys.argv) > 1:
        repo = Path(sys.argv[1]).expanduser().resolve()

    package_root = Path(__file__).resolve().parents[1]
    source_module = package_root / "src/trident/app/native_scientist.py"
    target_module = repo / "src/trident/app/native_scientist.py"
    workbench = repo / "src/trident/app/workbench.py"

    if not workbench.exists():
        raise SystemExit(f"TRIDENT workbench not found: {workbench}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = repo / ".trident_backups" / f"native_ui_v2_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(workbench, backup_dir / "workbench.py")
    if target_module.exists():
        shutil.copy2(target_module, backup_dir / "native_scientist.py")

    target_module.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_module, target_module)

    text = workbench.read_text(encoding="utf-8")
    import_line = (
        "from trident.app.native_scientist import "
        "render_acquisition, render_processing\n"
    )
    if import_line not in text:
        anchor = "from trident.app.integrated_explorer import render_integrated_explorer\n"
        if anchor not in text:
            raise SystemExit("Expected workbench import anchor was not found; backup preserved.")
        text = text.replace(anchor, anchor + import_line)

    download_replacement = '''def _native_download_controls(root, start, end, bbox, temporal):
    """Render the scientist-facing Native acquisition workflow."""
    render_acquisition(root, start, end, bbox, temporal)'''

    processing_replacement = '''def _native_processing_controls(root, start, end, bbox, stride):
    """Render prepared-input status and regional CAFE controls."""
    render_processing(root, start, end, bbox, stride)'''

    text = replace_function(text, "_native_download_controls", download_replacement)
    text = replace_function(text, "_native_processing_controls", processing_replacement)
    workbench.write_text(text, encoding="utf-8")

    marker = repo / ".trident_native_ui_v2_backup"
    marker.write_text(str(backup_dir), encoding="utf-8")

    print(f"Installed {target_module.relative_to(repo)}")
    print(f"Updated {workbench.relative_to(repo)}")
    print(f"Backup: {backup_dir}")
    print("Rollback: python3 scripts/rollback_native_ui_v2.py")


if __name__ == "__main__":
    main()
