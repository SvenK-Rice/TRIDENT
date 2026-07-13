from trident.data.nasa.earthdata import EarthdataRequest
from trident.data.copernicus.cmems import CopernicusRequest


def test_earthdata_cache_key_is_stable():
    request = EarthdataRequest(
        short_name="TEST_COLLECTION",
        start_date="2024-01-01",
        end_date="2024-01-31",
        bbox=(-132.0, 28.0, -116.0, 45.0),
        variable="chl",
        sensor="MODIS Aqua",
    )
    assert request.cache_key() == request.cache_key()
    assert len(request.cache_key()) == 16


def test_earthdata_cache_key_changes_with_region():
    first = EarthdataRequest(
        short_name="TEST_COLLECTION",
        start_date="2024-01-01",
        end_date="2024-01-31",
        bbox=(-132.0, 28.0, -116.0, 45.0),
    )
    second = EarthdataRequest(
        short_name="TEST_COLLECTION",
        start_date="2024-01-01",
        end_date="2024-01-31",
        bbox=(-130.0, 28.0, -116.0, 45.0),
    )
    assert first.cache_key() != second.cache_key()


def test_copernicus_cache_key_is_stable():
    request = CopernicusRequest(
        dataset_id="test_dataset",
        variables=("thetao", "mlotst"),
        start_datetime="2024-01-01",
        end_datetime="2024-01-31",
        bbox=(-132.0, 28.0, -116.0, 45.0),
    )
    assert request.cache_key() == request.cache_key()
    assert len(request.cache_key()) == 16
