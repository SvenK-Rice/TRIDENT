from __future__ import annotations
from pathlib import Path
from urllib.parse import urljoin
import json, re, gzip
from trident.utils.dates import month_code, code_to_dates
from trident.download import DownloadManager

BASE='https://orca.science.oregonstate.edu/'
PRODUCT_PAGES={
 ('monthly1080','cafe',None): '1080.by.2160.monthly.hdf.cafe.m.php',
 ('monthly1080','chl','modis'): '1080.by.2160.monthly.hdf.chl.modis.php',
 ('monthly1080','par','modis'): '1080.by.2160.monthly.hdf.par.modis.php',
 ('monthly1080','sst','modis'): '1080.by.2160.monthly.hdf.sst.modis.php',
 ('monthly1080','adg443','modis'): '1080.by.2160.monthly.hdf.adg.modis.php',
 ('monthly1080','aph443','modis'): '1080.by.2160.monthly.hdf.aph.modis.php',
 ('monthly1080','bbp443','modis'): '1080.by.2160.monthly.hdf.bbp.m.giop.php',
 ('monthly1080','bbp_s','modis'): '2160.by.4320.monthly.hdf.bbp_s.modis.php',
 ('monthly1080','mld125','modis'): '2160.by.4320.monthly.hdf.mld030.hycom.php',
}
INPUTS=['chl','par','sst','adg443','aph443','bbp443','bbp_s','mld125']

def inv_path(root: Path):
    return Path(root)/'reports/trident/osu_universal_inventory.json'

def load_inventory(root: Path):
    p=inv_path(root)
    if not p.exists():
        return []
    try:
        data=json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        # Preserve corrupted inventory rather than failing the whole app.
        backup=p.with_suffix(p.suffix+'.corrupt')
        try: p.replace(backup)
        except Exception: pass
        return []

def save_inventory(root: Path, inv):
    p=inv_path(root); p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(inv, indent=2))
    tmp.replace(p)

def fetch_text(url: str) -> str:
    return DownloadManager(timeout=90, retries=1).fetch_text(url)

def page_url(preset, product, sensor=None):
    key=(preset, product, sensor if product!='cafe' else None)
    if key not in PRODUCT_PAGES:
        raise KeyError(f'No OSU page mapping for {key}')
    return urljoin(BASE, PRODUCT_PAGES[key])

def probe(preset, product, sensor=None):
    url=page_url(preset, product, sensor)
    html=fetch_text(url)
    hrefs=re.findall(r"href\s*=\s*['\"]([^'\"]+)['\"]", html, flags=re.I)
    records=[]
    for h in hrefs:
        full=urljoin(url,h)
        m=re.search(r'([A-Za-z0-9_]+)\.(\d{7})\.hdf(?:\.gz)?', full)
        if not m: continue
        code=m.group(2)
        start,end=code_to_dates(code,'monthly')
        records.append({
          'source':'OSU','preset':preset,'resolution':'1080x2160' if preset=='monthly1080' else preset,
          'product_type':'monthly','product':product,'sensor':sensor,'code':code,
          'start_date':str(start),'end_date':str(end),'url':full,'filename':Path(full).name,'local_path':None
        })
    seen=set(); out=[]
    for r in records:
        k=(r['product'], r['sensor'], r['code'], r['url'])
        if k not in seen:
            seen.add(k); out.append(r)
    return out

def local_path(root: Path, rec):
    sensor = rec.get('sensor') or 'none'
    fn = rec['filename']
    if fn.endswith('.gz'): fn = fn[:-3]
    return Path(root)/'data/reference/osu/universal'/rec['preset']/sensor/rec['product']/fn

def _same_record(a, product, sensor, code):
    return a.get('product') == product and a.get('sensor') == sensor and a.get('code') == code

def ensure_product_month(root: Path, date_obj, product, sensor='modis', preset='monthly1080'):
    """Ensure one OSU monthly product is present locally and registered in inventory.

    This function is intentionally serial/atomic. Earlier parallel inventory writes
    could corrupt JSON when several products finished at the same time.
    """
    root=Path(root)
    code = month_code(date_obj)
    expected_sensor = None if product == 'cafe' else sensor
    inv = load_inventory(root)

    for r in inv:
        if _same_record(r, product, expected_sensor, code):
            lp = r.get('local_path')
            if lp and Path(lp).exists() and Path(lp).stat().st_size > 0:
                r['cache_status'] = 'hit'
                return r

    candidates = [r for r in probe(preset, product, expected_sensor) if r['code'] == code]
    if not candidates:
        raise FileNotFoundError(f'OSU product not found: product={product} sensor={expected_sensor} code={code}')

    rec = candidates[0]
    out = local_path(root, rec)
    out.parent.mkdir(parents=True, exist_ok=True)
    gz_path = out.with_suffix(out.suffix + '.gz') if not str(out).endswith('.gz') else out
    mgr = DownloadManager(timeout=240, retries=1)

    if out.exists() and out.stat().st_size > 0:
        rec['cache_status'] = 'hit'
    else:
        raw_path = gz_path if rec['filename'].endswith('.gz') else out
        mgr.download_to_file(rec['url'], raw_path, min_bytes=128)
        if rec['filename'].endswith('.gz'):
            out.write_bytes(gzip.decompress(raw_path.read_bytes()))
        rec['cache_status'] = 'downloaded'

    rec['local_path'] = str(out)
    rec['download_method'] = mgr.last_result.method if mgr.last_result else 'cache'
    if mgr.last_result and mgr.last_result.warning:
        rec['download_warning'] = mgr.last_result.warning

    inv = [x for x in inv if not _same_record(x, rec['product'], rec.get('sensor'), rec['code'])]
    inv.append(rec)
    save_inventory(root, inv)
    return rec

def ensure_month(root: Path, date_obj, sensor='modis', preset='monthly1080', include_cafe=True, workers=None):
    """Ensure all OSU products for a month. Uses serial downloads for reliability."""
    products = list(INPUTS)
    if include_cafe:
        products.append('cafe')
    out={}; errors={}
    for p in products:
        try:
            out[p] = ensure_product_month(root, date_obj, p, sensor, preset)
        except Exception as e:
            errors[p] = str(e)
    if errors:
        raise RuntimeError(f'Failed to ensure OSU products for {month_code(date_obj)}: {errors}')
    return out
