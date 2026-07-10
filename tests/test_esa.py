from pathlib import Path

from trident.data.esa.occi import ESAOceanColourRequest, ESA_PRODUCTS


def test_esa_presets_have_dataset_ids():
    assert ESA_PRODUCTS
    for cfg in ESA_PRODUCTS.values():
        assert cfg["dataset_id"].startswith("OCEANCOLOUR_")
        assert cfg["variables"]


def test_request_constructs():
    name, cfg = next(iter(ESA_PRODUCTS.items()))
    req = ESAOceanColourRequest(
        start_date="2023-01-01",
        end_date="2023-01-31",
        bbox=(-132, 28, -116, 45),
        product_name=name,
        dataset_id=cfg["dataset_id"],
        variables=cfg["variables"],
    )
    assert req.temporal == "monthly"
