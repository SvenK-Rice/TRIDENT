from __future__ import annotations

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/trident/app/native_scientist.py"
INSTALLER = ROOT / "scripts/install_native_ui_v2.py"
ROLLBACK = ROOT / "scripts/rollback_native_ui_v2.py"


def test_complete_package_files_exist() -> None:
    assert MODULE.exists()
    assert INSTALLER.exists()
    assert ROLLBACK.exists()


def test_scientist_interface_hides_legacy_controls_from_main_view() -> None:
    text = MODULE.read_text(encoding="utf-8")
    assert "Acquire native satellite data" in text
    assert "Prepare Native CAFE inputs" in text
    assert "Run TRIDENT Native CAFE" in text
    assert "Developer details" in text
    assert "Copernicus password" not in text
    assert "collection short_name" not in text
    assert "ESA/Copernicus dataset ID" not in text


def test_installer_contains_backup_and_rollback_support() -> None:
    installer = INSTALLER.read_text(encoding="utf-8")
    rollback = ROLLBACK.read_text(encoding="utf-8")
    assert ".trident_backups" in installer
    assert ".trident_native_ui_v2_backup" in installer
    assert "workbench.py" in rollback


def test_module_is_valid_python() -> None:
    spec = importlib.util.spec_from_file_location("native_scientist_check", MODULE)
    assert spec is not None
    source = MODULE.read_text(encoding="utf-8")
    compile(source, str(MODULE), "exec")


def test_month_tags_use_calendar_yyyymm() -> None:
    from datetime import date

    from trident.app.native_scientist import _month_tags

    assert _month_tags(
        date(2023, 1, 1),
        date(2023, 3, 31),
    ) == ["202301", "202302", "202303"]

    assert _month_tags(
        date(2025, 1, 1),
        date(2025, 1, 31),
    ) == ["202501"]
