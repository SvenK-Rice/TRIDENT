from trident.workflows.native_prepare import ALIASES, REQUIRED


def test_required_aliases_present():
    assert set(REQUIRED) == set(ALIASES)
    assert "chlor_a" in ALIASES["chl"]
    assert "mlotst" in ALIASES["mld"]
