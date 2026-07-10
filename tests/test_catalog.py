from pathlib import Path
from trident.workflows.catalog import scan_products


def test_catalog_semantic_entries(tmp_path: Path):
    p = tmp_path / "data" / "processed"
    p.mkdir(parents=True)
    (p / "trident_native_cafe_npp_202301_stride10.nc").touch()
    entries = scan_products(tmp_path)
    assert len(entries) == 1
    assert entries[0].tag == "202301"
    assert entries[0].family == "native_npp"
