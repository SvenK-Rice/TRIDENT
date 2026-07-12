import numpy as np

from trident.viz.maps import nearest_display_fill


def test_nearest_display_fill_preserves_raw_values():
    raw = np.array(
        [
            [1.0, np.nan, np.nan],
            [np.nan, np.nan, 5.0],
        ]
    )
    filled = nearest_display_fill(raw)

    assert filled[0, 0] == 1.0
    assert filled[1, 2] == 5.0
    assert np.isfinite(filled).all()


def test_nearest_display_fill_can_limit_distance():
    raw = np.full((5, 5), np.nan)
    raw[0, 0] = 2.0

    filled = nearest_display_fill(raw, max_distance_cells=1.1)

    assert filled[0, 1] == 2.0
    assert np.isnan(filled[4, 4])
