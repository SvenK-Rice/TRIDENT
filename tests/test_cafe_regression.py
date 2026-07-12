from __future__ import annotations

import numpy as np
import pytest

from trident.models.cafe import cafe_pixel, opp_cafe_pixel


CASES = {
    "open_ocean": {
        "inputs": (40.0, 0.15, 35.0, 30.0, 180, 0.010, 0.005, 0.0015, 1.0, 22.0),
        "expected": (
            601.990231351131,
            105.96293128747453,
            0.056058955528560356,
            1.426045539520515,
        ),
    },
    "productive_coastal": {
        "inputs": (48.0, 2.0, 15.0, 34.0, 120, 0.055, 0.018, 0.006, 0.8, 16.0),
        "expected": (
            1146.967200223343,
            47.5862154033773,
            0.12866105777934714,
            1.4197465602862678,
        ),
    },
    "cold_high_latitude": {
        "inputs": (25.0, 0.8, 60.0, 55.0, 200, 0.030, 0.010, 0.003, 1.2, 7.0),
        "expected": (
            549.7785983583744,
            56.84071727367601,
            0.09623678035477627,
            1.4850930817342336,
        ),
    },
}


@pytest.mark.parametrize("case_name", CASES)
def test_cafe_pixel_regression(case_name: str) -> None:
    args = CASES[case_name]["inputs"]
    expected_npp, expected_zeu, expected_kdpar, _ = CASES[case_name]["expected"]

    result = cafe_pixel(*args)

    assert result.npp == pytest.approx(expected_npp, rel=1e-12, abs=1e-12)
    assert result.zeu == pytest.approx(expected_zeu, rel=1e-12, abs=1e-12)
    assert result.kdpar == pytest.approx(expected_kdpar, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("case_name", CASES)
def test_opp_cafe_pixel_regression(case_name: str) -> None:
    args = CASES[case_name]["inputs"]
    expected_npp, expected_zeu, expected_kdpar, expected_eu = CASES[case_name]["expected"]

    result = opp_cafe_pixel(*args, return_diagnostics=True)

    assert result.valid
    assert result.reason == "ok"
    assert np.isfinite(result.npp)

    assert result.npp == pytest.approx(expected_npp, rel=1e-12, abs=1e-12)
    assert result.zeu == pytest.approx(expected_zeu, rel=1e-12, abs=1e-12)
    assert result.kdpar == pytest.approx(expected_kdpar, rel=1e-12, abs=1e-12)
    assert result.eu == pytest.approx(expected_eu, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("case_name", CASES)
def test_wrapper_matches_diagnostic_engine(case_name: str) -> None:
    args = CASES[case_name]["inputs"]

    compact = cafe_pixel(*args)
    diagnostic = opp_cafe_pixel(*args, return_diagnostics=True)

    assert compact.npp == diagnostic.npp
    assert compact.zeu == diagnostic.zeu
    assert compact.kdpar == diagnostic.kdpar
