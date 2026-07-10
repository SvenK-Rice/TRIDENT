from __future__ import annotations
from datetime import date, datetime, timedelta
import calendar

def as_date(x) -> date:
    if isinstance(x, date):
        return x
    return datetime.fromisoformat(str(x)).date()

def month_starts(start, end):
    s = as_date(start).replace(day=1)
    e = as_date(end).replace(day=1)
    out=[]
    d=s
    while d <= e:
        out.append(d)
        if d.month == 12:
            d = date(d.year+1,1,1)
        else:
            d = date(d.year,d.month+1,1)
    return out

def month_code(d: date) -> str:
    d = as_date(d).replace(day=1)
    return f"{d.year}{d.timetuple().tm_yday:03d}"

def code_to_dates(code: str, product_type: str = 'monthly'):
    year = int(code[:4]); doy = int(code[4:])
    start = date(year,1,1) + timedelta(days=doy-1)
    if product_type == 'monthly':
        last = calendar.monthrange(start.year, start.month)[1]
        end = start.replace(day=last)
    elif product_type == '8day':
        end = start + timedelta(days=7)
    else:
        raise ValueError(f'Unknown product_type: {product_type}')
    return start, end
