from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import pandas as pd
from trident.data.nasa.earthdata import EarthdataRequest, is_available as earthdata_available, search_granules, download_granules
from trident.data.copernicus.cmems import CopernicusRequest, is_available as copernicus_available, subset as copernicus_subset

@dataclass
class ProductionRequest:
    start_date: str
    end_date: str
    bbox: tuple[float,float,float,float]
    temporal: str = 'monthly'
    sensor: str = 'MODIS Aqua'
    nasa_collections: dict | None = None
    copernicus_dataset_id: str | None = None
    copernicus_variables: list[str] | None = None
    earthdata_strategy: str = 'netrc'
    cmems_username: str | None = None
    cmems_password: str | None = None

DEFAULT_NASA_COLLECTIONS = {
    # These are user-editable in the app because NASA collection short names vary
    # with sensor/reprocessing. Empty values are skipped.
    'chl': '',
    'par': '',
    'aph443': '',
    'adg443': '',
    'bbp443': '',
    'bbp_s': '',
}

DEFAULT_CMEMS_DATASET = 'cmems_mod_glo_phy_anfc_0.083deg_P1D-m'
DEFAULT_CMEMS_VARIABLES = ['thetao','mlotst']

def production_status(root, start, end, bbox):
    root = Path(root)
    ea_ok, ea_msg = earthdata_available()
    cm_ok, cm_msg = copernicus_available()
    return {
        'mode': 'TRIDENT Standard: NASA Earthdata + Copernicus',
        'earthdata_available': ea_ok,
        'earthdata_message': ea_msg,
        'copernicus_available': cm_ok,
        'copernicus_message': cm_msg,
        'requested_start': str(start),
        'requested_end': str(end),
        'bbox': list(map(float, bbox)),
        'cache_dirs': {
            'earthdata': str(root / 'data/reference/nasa'),
            'copernicus': str(root / 'data/reference/copernicus'),
            'processed': str(root / 'data/processed'),
        },
        'planned_inputs': ['chl', 'par', 'aph443', 'adg443', 'bbp443', 'bbp_s', 'sst', 'mld'],
        'note': 'Download layer is active. CAFE execution from NASA/Copernicus products will be enabled after collection mappings and variable harmonization are validated.'
    }

def _month_tag(d):
    return f'{d.year}{d.month:02d}'

def standard_cache_status(root, start, end):
    root=Path(root)
    dates=pd.date_range(str(start), str(end), freq='MS')
    rows=[]
    for d in dates:
        tag=f'{d.year}{d.month:02d}'
        nasa=list((root/'data/reference/nasa').glob(f'**/*{tag}*'))
        cm=list((root/'data/reference/copernicus').glob(f'**/*{tag}*'))
        rows.append({'month':tag,'nasa_cached_files':len(nasa),'copernicus_cached_files':len(cm)})
    return pd.DataFrame(rows)

def download_standard_inputs(root, req: ProductionRequest):
    """Download/cache requested NASA Earthdata and Copernicus inputs.

    This is intentionally a data acquisition workflow only. It does not yet
    harmonize all variables into CAFE-ready inputs because the correct NASA
    collection short names and variable names must be selected and validated.
    """
    root=Path(root)
    report={'request':asdict(req),'earthdata':{},'copernicus':{},'outputs':[]}
    root.joinpath('reports/trident').mkdir(parents=True, exist_ok=True)

    collections=dict(DEFAULT_NASA_COLLECTIONS)
    if req.nasa_collections:
        collections.update(req.nasa_collections)

    ea_ok, ea_msg=earthdata_available()
    report['earthdata']['available']=ea_ok
    report['earthdata']['message']=ea_msg
    if ea_ok:
        for var, short_name in collections.items():
            if not short_name:
                report['earthdata'][var]={'status':'skipped','reason':'no collection short_name specified'}
                continue
            out_dir=root/'data/reference/nasa'/req.sensor.replace(' ','_')/var
            try:
                ereq=EarthdataRequest(sensor=req.sensor, product=short_name, start_date=str(req.start_date), end_date=str(req.end_date), bbox=req.bbox, temporal=req.temporal)
                granules=search_granules(ereq, count=500)
                paths=download_granules(granules, out_dir)
                report['earthdata'][var]={'status':'downloaded','collection':short_name,'n_granules':len(granules),'paths':[str(p) for p in paths]}
                report['outputs'].extend([str(p) for p in paths])
            except Exception as e:
                report['earthdata'][var]={'status':'error','collection':short_name,'error':str(e)}

    cm_ok, cm_msg=copernicus_available()
    report['copernicus']['available']=cm_ok
    report['copernicus']['message']=cm_msg
    dsid=req.copernicus_dataset_id
    vars=req.copernicus_variables or []
    if cm_ok and dsid and vars:
        try:
            out_file=root/'data/reference/copernicus'/f'copernicus_{str(req.start_date)[:10]}_{str(req.end_date)[:10]}.nc'
            creq=CopernicusRequest(dataset_id=dsid, variables=list(vars), start_datetime=str(req.start_date), end_datetime=str(req.end_date), bbox=req.bbox)
            result=copernicus_subset(creq, out_file, username=req.cmems_username, password=req.cmems_password)
            report['copernicus']['subset']={'status':'downloaded','dataset_id':dsid,'variables':vars,'path':str(out_file),'result':str(result)}
            report['outputs'].append(str(out_file))
        except Exception as e:
            report['copernicus']['subset']={'status':'error','dataset_id':dsid,'variables':vars,'error':str(e)}
    else:
        report['copernicus']['subset']={'status':'skipped','reason':'copernicus client unavailable or dataset/variables not specified'}

    out=root/'reports/trident/standard_download_report.json'
    out.write_text(json.dumps(report, indent=2))
    return report
