from __future__ import annotations
import numpy as np
import xarray as xr

def finite_masked(arr, fill_values=(-9999,-32767,32767,65535)):
    a=np.asarray(arr, dtype='float32')
    a[~np.isfinite(a)] = np.nan
    for fv in fill_values:
        a[np.isclose(a, fv)] = np.nan
    a[np.abs(a)>1e6] = np.nan
    return a

def resize_nearest_or_block(arr, target_shape):
    arr = finite_masked(arr)
    if arr.shape == tuple(target_shape):
        return arr.astype('float32')
    ty, tx = target_shape
    sy, sx = arr.shape
    if sy % ty == 0 and sx % tx == 0:
        fy, fx = sy//ty, sx//tx
        blocks = arr.reshape(ty, fy, tx, fx)
        valid = np.isfinite(blocks)
        count = valid.sum(axis=(1,3))
        total = np.nansum(blocks, axis=(1,3))
        out = np.full((ty, tx), np.nan, dtype='float32')
        np.divide(total, count, out=out, where=count>0)
        return out.astype('float32')
    if ty % sy == 0 and tx % sx == 0:
        fy, fx = ty//sy, tx//sx
        return np.repeat(np.repeat(arr, fy, axis=0), fx, axis=1).astype('float32')
    yi=np.round(np.linspace(0, sy-1, ty)).astype(int)
    xi=np.round(np.linspace(0, sx-1, tx)).astype(int)
    return arr[np.ix_(yi,xi)].astype('float32')

def align_arrays_to_reference(data: dict, ref_name='chl') -> dict:
    if ref_name not in data:
        ref_name = next(iter(data))
    ref = data[ref_name]
    if isinstance(ref, tuple):
        ref_arr = ref[1]
    else:
        ref_arr = ref
    target_shape = np.asarray(ref_arr).shape
    out={}
    for name, val in data.items():
        if isinstance(val, tuple):
            dims, arr = val
            out[name] = (dims, resize_nearest_or_block(arr, target_shape))
        else:
            out[name] = resize_nearest_or_block(val, target_shape)
    return out

def mask_feasible(name, values):
    a=finite_masked(values)
    ranges={
        'sst':(-3,40),'par':(0,80),'chl':(0,100),'mld':(0,1000),'mld125':(0,1000),
        'aph443':(0,10),'adg443':(0,10),'bbp443':(0,1),'bbp_s':(-5,5),
        'cafe_npp':(0,5000),'npp':(0,5000),'zeu':(0,300),'kdpar':(0,10)
    }
    lohi=ranges.get(name)
    if lohi:
        lo,hi=lohi; a[(a<lo)|(a>hi)] = np.nan
    return a
