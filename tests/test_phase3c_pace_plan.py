from trident.data_sources.acquisition import acquisition_plan


def test_pace_2025_plan_uses_verified_v32_products() -> None:
    plan = acquisition_plan("202501", (-132.0, 28.0, -116.0, 45.0))
    by_name = {item.variable: item for item in plan}

    assert by_name["par"].request["version"] == "3.2"
    assert "V3_2" in by_name["par"].request["granule_name"]
    assert by_name["par"].status == "direct"

    for name in ("aph443", "adg443", "bbp443", "bbp_s"):
        assert by_name[name].collection == "PACE_OCI_L3M_IOP"
        assert by_name[name].request["version"] == "3.2"
        assert "V3_2" in by_name[name].request["granule_name"]
        assert by_name[name].status == "direct"

    assert "ancillary" not in by_name["bbp_s"].note.lower()
