from datetime import date

from trident.workflows.osu_replay import _midmonth_day_of_year


def test_midmonth_day_of_year_non_leap_year():
    assert _midmonth_day_of_year(date(2023, 1, 1)) == 15
    assert _midmonth_day_of_year(date(2023, 6, 1)) == 166
    assert _midmonth_day_of_year(date(2023, 12, 1)) == 349


def test_midmonth_day_of_year_leap_year():
    assert _midmonth_day_of_year(date(2024, 1, 1)) == 15
    assert _midmonth_day_of_year(date(2024, 6, 1)) == 167
    assert _midmonth_day_of_year(date(2024, 12, 1)) == 350
