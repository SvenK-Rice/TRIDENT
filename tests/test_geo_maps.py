import numpy as np
import xarray as xr

from trident.viz.maps import Bounds, _clip_polyline, make_geo_heatmap


def test_heatmap_keeps_exact_coordinate_ranges():
    ds = xr.Dataset(
        {"cafe_npp": (("lat", "lon"), np.ones((3, 4), dtype="float32"))},
        coords={
            "lat": [28.0, 36.5, 45.0],
            "lon": [-132.0, -126.0, -120.0, -116.0],
        },
    )

    fig = make_geo_heatmap(ds, "cafe_npp")

    assert list(fig.layout.xaxis.range) == [-132.0, -116.0]
    assert list(fig.layout.yaxis.range) == [28.0, 45.0]


def test_descending_latitude_is_reordered():
    values = np.array([[1, 2], [3, 4]], dtype="float32")
    ds = xr.Dataset(
        {"cafe_npp": (("lat", "lon"), values)},
        coords={"lat": [45.0, 28.0], "lon": [-132.0, -116.0]},
    )

    fig = make_geo_heatmap(ds, "cafe_npp")
    heatmap = fig.data[0]

    assert list(heatmap.y) == [28.0, 45.0]
    assert np.asarray(heatmap.z).tolist() == [[3.0, 4.0], [1.0, 2.0]]


def test_coastline_is_clipped_to_selected_box():
    bounds = Bounds(-132.0, 28.0, -116.0, 45.0)
    line = [(-124.0, 46.0), (-124.0, 44.0), (-114.0, 27.0)]

    clipped = _clip_polyline(line, bounds)

    assert clipped
    assert all(bounds.west <= x <= bounds.east for x, _ in clipped)
    assert all(bounds.south <= y <= bounds.north for _, y in clipped)
