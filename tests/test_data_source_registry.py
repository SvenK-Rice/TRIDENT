from datetime import date

from trident.data_sources.registry import CAFE_REQUIRED, source_plan


def test_registry_covers_all_cafe_inputs() -> None:
    assert tuple(source_plan(date(2025, 1, 15)))
    assert set(source_plan(date(2025, 1, 15))) == set(CAFE_REQUIRED)


def test_modis_is_preferred_for_old_par() -> None:
    assert source_plan(date(2023, 1, 15))["par"].collection == "MODISA_L3m_PAR"


def test_pace_is_preferred_for_current_par() -> None:
    assert source_plan(date(2025, 1, 15))["par"].source == "NASA"
    assert "PACE" in source_plan(date(2025, 1, 15))["par"].collection


def test_bbp_s_is_explicitly_derived_or_ancillary() -> None:
    old = source_plan(date(2023, 1, 15))["bbp_s"]
    new = source_plan(date(2025, 1, 15))["bbp_s"]
    assert old.status == "ancillary"
    assert new.status == "derived"
