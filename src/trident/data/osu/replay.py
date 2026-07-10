from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
from trident.data.osu.archive import ensure_month, load_inventory, INPUTS
from trident.grids.align import mask_feasible, resize_nearest_or_block
from trident.utils.dates import month_code

def read_hdf_array(path, return_meta=False):
    """Read OSU HDF4 data in an isolated helper process.

    The Streamlit/NetCDF process never imports pyhdf. This prevents the HDF4
    stack (libmfhdf/libcurl/OpenSSL) from being loaded into the same process as
    netCDF4/HDF5, which caused hard macOS interpreter crashes.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    repo_root = Path(__file__).resolve().parents[4]
    worker = repo_root / 'scripts' / 'hdf4_worker.py'
    hdf_python = os.environ.get('TRIDENT_HDF_PYTHON')
    if not hdf_python:
        candidate = repo_root / '.venv-hdf' / 'bin' / 'python'
        hdf_python = str(candidate if candidate.exists() else Path(sys.executable))

    with tempfile.TemporaryDirectory(prefix='trident_hdf4_') as td:
        output = Path(td) / 'result.npz'
        proc = subprocess.run(
            [hdf_python, str(worker), str(source), str(output)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0 or not output.exists():
            detail = (proc.stderr or proc.stdout or f'exit code {proc.returncode}').strip()
            raise RuntimeError(
                'The isolated HDF4 reader failed. The main TRIDENT app remains '
                f'alive. Reader output: {detail[-3000:]}'
            )
        with np.load(output, allow_pickle=False) as z:
            arr = np.asarray(z['array'], dtype='float32')
            name = str(z['dataset_name'].item())
            attrs = json.loads(str(z['attrs_json'].item()))
    if return_meta:
        return arr, name, attrs
    return arr

def global_lat_lon(shape):
    ny,nx=shape
    lat=np.linspace(90-180/(2*ny), -90+180/(2*ny), ny, dtype='float32')
    lon=np.linspace(-180+360/(2*nx), 180-360/(2*nx), nx, dtype='float32')
    return lat, lon

def subset_bbox(arr, bbox):
    # bbox = west, south, east, north. Generate global grid for this array shape.
    lat,lon=global_lat_lon(arr.shape)
    west,south,east,north=map(float,bbox)
    latmask=(lat>=south)&(lat<=north)
    lonmask=(lon>=west)&(lon<=east)
    return arr[np.ix_(latmask,lonmask)], lat[latmask], lon[lonmask]

def build_inputs(root, date_obj, bbox, preset='monthly1080', sensor='modis'):
    recs=ensure_month(root,date_obj,sensor,preset,include_cafe=True)
    tag=f'{date_obj.year}{date_obj.month:02d}'
    raw={}
    coords=None
    # CHL is reference grid for region subset
    for name in INPUTS:
        arr=read_hdf_array(recs[name]['local_path'])
        arr=mask_feasible(name, arr)
        sub,lat,lon=subset_bbox(arr,bbox)
        raw[name]=sub
        if name=='chl': coords=(lat,lon); ref_shape=sub.shape
    if coords is None:
        raise RuntimeError('No CHL reference grid')
    lat,lon=coords
    aligned={}
    for name, arr in raw.items():
        aligned[name]=(('lat','lon'), resize_nearest_or_block(arr, ref_shape))
    # rename mld125 -> mld for model
    aligned['mld'] = aligned.pop('mld125')
    ds=xr.Dataset(aligned, coords={'lat':lat,'lon':lon})
    ds['time']=np.datetime64(f'{date_obj.year}-{date_obj.month:02d}-15')
    ds.attrs['source']='OSU archived inputs aligned to CHL grid'
    out=Path(root)/'data/processed'/f'trident_osu_cafe_inputs_{tag}.nc'
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return out
