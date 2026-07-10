from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import xarray as xr
from tqdm import tqdm
from trident.utils.dates import month_starts, as_date
from trident.data.osu.replay import build_inputs, read_hdf_array, subset_bbox
from trident.data.osu.archive import ensure_product_month
from trident.models.cafe import cafe_pixel
from trident.grids.align import resize_nearest_or_block, mask_feasible


def _tag(date_obj) -> str:
    d = as_date(date_obj)
    return f"{d.year}{d.month:02d}"


def run_month(root, date_obj, bbox, stride=10, preset='monthly1080', sensor='modis'):
    """Calculate TRIDENT CAFE NPP using archived OSU input products."""
    root = Path(root)
    date_obj = as_date(date_obj)
    tag = _tag(date_obj)
    infile = build_inputs(root, date_obj, bbox, preset, sensor)
    ds = xr.open_dataset(infile)
    shape = ds['chl'].shape
    npp = np.full(shape, np.nan, dtype='float32')
    zeu = np.full(shape, np.nan, dtype='float32')
    kd = np.full(shape, np.nan, dtype='float32')
    indices = list(zip(*np.where(np.isfinite(ds['chl'].values))))
    indices = indices[::max(1, int(stride))]
    failures = 0
    yd = 16
    for i, j in tqdm(indices, desc=f'TRIDENT CAFE replay {tag}'):
        try:
            res = cafe_pixel(
                float(ds['par'][i, j]), float(ds['chl'][i, j]), float(ds['mld'][i, j]),
                float(ds['lat'][i]), yd, float(ds['aph443'][i, j]), float(ds['adg443'][i, j]),
                float(ds['bbp443'][i, j]), float(ds['bbp_s'][i, j]), float(ds['sst'][i, j])
            )
            npp[i, j] = res.npp
            zeu[i, j] = res.zeu
            kd[i, j] = res.kdpar
        except Exception:
            failures += 1
    out = xr.Dataset(
        {'cafe_npp': (('lat', 'lon'), npp), 'zeu': (('lat', 'lon'), zeu), 'kdpar': (('lat', 'lon'), kd)},
        coords={'lat': ds.lat, 'lon': ds.lon}
    )
    out.attrs.update({
        'title': 'TRIDENT CAFE replay calculated from archived OSU input products',
        'product_role': 'trident_replay',
        'reference': 'This is NOT the direct OSU NPP product; it is TRIDENT recalculation using OSU inputs.'
    })
    outpath = root / 'data/processed' / f'trident_osu_input_replay_cafe_npp_{tag}_stride{stride}.nc'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(outpath)
    report = {
        'tag': tag,
        'input_file': str(infile),
        'output_file': str(outpath),
        'product_role': 'TRIDENT replay: CAFE recalculated from archived OSU inputs',
        'stride': stride,
        'candidate_pixels': len(indices),
        'failures': failures,
        'success': len(indices) - failures,
        'mean_npp': float(np.nanmean(npp)) if np.isfinite(npp).any() else None
    }
    rp = root / 'reports/trident' / f'run_report_{tag}_stride{stride}.json'
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    return report


def compare_month(root, date_obj, bbox, stride=10, preset='monthly1080'):
    """Compare TRIDENT replay NPP against the direct archived OSU CAFE NPP product."""
    root = Path(root)
    date_obj = as_date(date_obj)
    tag = _tag(date_obj)
    tr_path = root / 'data/processed' / f'trident_osu_input_replay_cafe_npp_{tag}_stride{stride}.nc'
    tr = xr.open_dataset(tr_path)
    rec = ensure_product_month(root, date_obj, 'cafe', None, preset)
    arr = mask_feasible('cafe_npp', read_hdf_array(rec['local_path']))
    sub, lat, lon = subset_bbox(arr, bbox)
    osu = resize_nearest_or_block(sub, tr['cafe_npp'].shape)
    a = tr['cafe_npp'].values
    b = osu.astype('float32')
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum():
        diff = a[mask] - b[mask]
        r = float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 2 else None
        rep = {
            'n': int(mask.sum()),
            'trident_replay_mean': float(np.nanmean(a[mask])),
            'osu_archived_mean': float(np.nanmean(b[mask])),
            'bias_trident_minus_osu': float(np.nanmean(diff)),
            'percent_bias_vs_osu_mean': float(100 * np.nanmean(diff) / np.nanmean(b[mask])) if np.nanmean(b[mask]) else None,
            'mae': float(np.nanmean(np.abs(diff))),
            'rmse': float(np.sqrt(np.nanmean(diff ** 2))),
            'correlation_r': r,
            'r2': None if r is None else r * r,
        }
    else:
        rep = {'n': 0}
    comp = xr.Dataset(
        {'trident_replay_npp': (('lat', 'lon'), a), 'osu_archived_npp': (('lat', 'lon'), b), 'difference': (('lat', 'lon'), a - b)},
        coords={'lat': tr.lat, 'lon': tr.lon}
    )
    comp.attrs.update({
        'title': 'TRIDENT replay compared with direct archived OSU CAFE product',
        'trident_replay_npp': 'TRIDENT calculated CAFE NPP using OSU archived inputs',
        'osu_archived_npp': 'Direct OSU archived CAFE NPP product',
        'difference': 'TRIDENT replay minus OSU archived NPP'
    })
    compath = root / 'data/processed' / f'trident_osu_replay_vs_archived_comparison_{tag}_stride{stride}.nc'
    comp.to_netcdf(compath)
    rep.update({'tag': tag, 'comparison_file': str(compath), 'product_role': 'validation_comparison'})
    rp = root / 'reports/trident' / f'validation_{tag}_stride{stride}.json'
    rp.write_text(json.dumps(rep, indent=2))
    return rep


def run_range(root, start, end, bbox, stride=10, preset='monthly1080', sensor='modis'):
    reports = []
    for d in month_starts(start, end):
        reports.append(run_month(root, d, bbox, stride, preset, sensor))
        reports.append(compare_month(root, d, bbox, stride, preset))
    return reports
