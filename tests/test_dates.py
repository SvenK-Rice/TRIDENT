from trident.utils.dates import code_to_dates, month_code
from datetime import date

def test_month_code_2023_jan():
    assert month_code(date(2023,1,1)) == '2023001'

def test_month_dates():
    s,e=code_to_dates('2024001','monthly')
    assert str(s)=='2024-01-01' and str(e)=='2024-01-31'

def test_8day_dates():
    s,e=code_to_dates('2024009','8day')
    assert str(s)=='2024-01-09' and str(e)=='2024-01-16'
