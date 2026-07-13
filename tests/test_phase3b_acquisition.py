from trident.data_sources.acquisition import acquisition_plan


def test_modis_plan_before_pace():
    plan = acquisition_plan("202301", (-132.0, 28.0, -116.0, 45.0))
    by_name = {item.variable: item for item in plan}
    assert by_name["par"].collection == "MODISA_L3m_PAR"
    assert by_name["chl"].source == "Copernicus"
    assert by_name["mld"].collection == "cmems_mod_glo_phy_my_0.083deg_P1M-m"


def test_pace_plan_after_launch():
    plan = acquisition_plan("202501", (-132.0, 28.0, -116.0, 45.0))
    by_name = {item.variable: item for item in plan}
    assert by_name["par"].collection == "PACE_OCI_L3M_PAR"
    assert by_name["chl"].collection == "PACE_OCI_L3M_CHL"
    assert by_name["mld"].collection == "cmems_mod_glo_phy_anfc_0.083deg_P1M-m"
    assert "bbp_s" in by_name
