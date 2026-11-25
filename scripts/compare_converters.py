#!/usr/bin/env python3
"""Full-plane C/Rust conversion compatibility check; run real data only in PBS."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import logging
import re
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import dask
import numpy as np
import xarray as xr
from comparison_notebook import create_notebook, sample_figure

LOG=logging.getLogger('comparison')
RTOL=1e-6
COORD_ATOL=1e-5
_CF_EXPECTED={}

def serialise(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    return str(value)

def json_safe(value):
    if isinstance(value, dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return str(value)
    return value

def write_json(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(json_safe(value),indent=2,default=serialise,allow_nan=False))
    temporary.replace(path)

def fingerprint(path,hash_content=False):
    path=Path(path).resolve();st=path.stat()
    result={'path':str(path),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    if hash_content:
        with path.open('rb') as f:
            h=hashlib.sha256()
            for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        result['sha256']=h.hexdigest()
    return result

def role(array):
    attrs=array.attrs
    if array.dtype.kind=='M' or attrs.get('axis')=='T':return 'time'
    if attrs.get('standard_name')=='grid_latitude' or attrs.get('axis')=='Y':return 'y'
    if attrs.get('standard_name')=='grid_longitude' or attrs.get('axis')=='X':return 'x'
    if attrs.get('axis')=='Z' or attrs.get('positive') or attrs.get('standard_name') in ('air_pressure','altitude','height','depth'):return 'z'
    return 'other'

def canonical_time(coord):
    values=coord.to_numpy()
    if values.dtype.kind!='M':return values
    if np.dtype(coord.encoding.get('dtype','float64'))!=np.dtype('float32'):return values
    # The UM lookup stores whole seconds. C's float32 elapsed-hour encoding
    # introduces subsecond errors; establish a bound from its representable ULP.
    units=coord.encoding.get('units','')
    unit=units.split(' since ')[0]
    seconds={'hours':3600,'minutes':60,'seconds':1,'days':86400}.get(unit)
    if seconds is None:raise ValueError(f'Unsupported reference time units {units}')
    ns=values.astype('datetime64[ns]').astype('int64')
    rounded=((ns+500_000_000)//1_000_000_000)*1_000_000_000
    origin=np.datetime64(units.split(' since ')[1],'ns').astype('int64')
    encoded=(ns-origin)/1e9/seconds
    allowance=np.abs(np.spacing(encoded.astype('float32')).astype(float))*seconds/2+2e-9
    if not np.all(np.abs(ns-rounded)/1e9<=allowance):
        raise ValueError('Time differences exceed float32 encoding rounding error')
    return rounded.astype('datetime64[ns]')

def methods(field):
    return re.sub(r'\btime\d*\s*:', 'time:', field.attrs.get('cell_methods',''))

def match_dimensions(a,b):
    if a.ndim!=b.ndim:raise ValueError(f'Dimension counts differ: {a.dims} versus {b.dims}')
    mapping={};checks=[]
    for dim in a.dims:
        if dim not in a.coords:raise ValueError(f'Missing reference coordinate {dim}')
        candidates=[d for d in b.dims if d not in mapping.values() and d in b.coords
                    and (role(a[dim])==role(b[d]) or (dim.startswith('surface') and d.startswith('pseudo_level')))
                    and a.sizes[dim]==b.sizes[d]]
        matching=[]
        for other in candidates:
            left,right=a[dim].to_numpy(),b[other].to_numpy();note=None
            if left.dtype.kind=='M' or right.dtype.kind=='M':
                equal=left.dtype.kind==right.dtype.kind=='M' and np.array_equal(canonical_time(a[dim]),canonical_time(b[other]))
                if equal and not np.array_equal(left,right):note='C float32 time canonicalised to whole UM lookup seconds; bounded by encoding ULP'
            elif dim.startswith('surface') and other.startswith('pseudo_level'):
                equal=np.array_equal(left,np.arange(len(left))) and np.array_equal(right,left+1)
                if equal:note='C zero-based tile ordinal versus original one-based UM pseudo-level ID'
            else:equal=np.allclose(left,right,rtol=RTOL,atol=COORD_ATOL,equal_nan=True)
            if equal:matching.append((other,note))
        if len(matching)!=1:raise ValueError(f'Cannot match {dim} uniquely: candidates={candidates}, matching={matching}')
        other,note=matching[0];mapping[dim]=other
        delta=None if a[dim].dtype.kind=='M' else float(np.max(np.abs(a[dim].to_numpy().astype(float)-b[other].to_numpy().astype(float))))
        checks.append({'reference':dim,'candidate':other,'role':role(a[dim]),'max_abs':delta,'representation_note':note})
    return mapping,checks

def data_fields(ds):
    return {n:v for n,v in ds.data_vars.items() if v.ndim>0 and ('stash_item' in v.attrs or n=='field')}

def field_metadata(field):
    return {'dims':list(field.dims),'shape':list(field.shape),'dtype':str(field.dtype),
            'attrs':dict(field.attrs),'encoding':{k:field.encoding.get(k) for k in
            ('dtype','_FillValue','scale_factor','add_offset','zlib','complevel','shuffle','fletcher32','chunksizes')}}

def bounds(ds,field,dim):
    if role(field[dim])!='time':return None
    coord=field[dim];name=coord.attrs.get('bounds',coord.encoding.get('bounds'))
    return canonical_time(ds[name]) if name in ds else None

def compare_files(reference,candidate,stream,require_candidate_cf=True):
    results=[];aux_cache={}
    with xr.open_dataset(reference,decode_coords='all') as left,xr.open_dataset(candidate,decode_coords='all') as right:
        aa,bb=data_fields(left),data_fields(right)
        unmatched=set(bb)
        for name,a in aa.items():
            other=name if name in bb else None
            if other is None:
                keys=('stash_model','stash_section','stash_item')
                identity=tuple(a.attrs.get(k) for k in keys)+(methods(a),)
                matches=[n for n in unmatched if identity==tuple(bb[n].attrs.get(k) for k in keys)+(methods(bb[n]),)]
                if len(matches)==1:other=matches[0]
            row={'stream':stream,'field':name,'candidate_field':other,'status':'missing_candidate','reference':field_metadata(a)}
            results.append(row)
            if other is None:continue
            unmatched.discard(other);b=bb[other];row['candidate']=field_metadata(b)
            try:
                mapping,coords=match_dimensions(a,b);row['dimension_mapping']=mapping;row['coordinate_checks']=coords
                row['metadata_differences']={k:{'C':a.attrs.get(k),'Rust':b.attrs.get(k)} for k in ('units','standard_name','long_name','cell_methods') if (methods(a)!=methods(b) if k=='cell_methods' else a.attrs.get(k)!=b.attrs.get(k))}
                for dim,other_dim in mapping.items():
                    av,bv=bounds(left,a,dim),bounds(right,b,other_dim)
                    if (av is None)!=(bv is None) or (av is not None and not np.array_equal(av,bv)):
                        row.setdefault('scientific_issues',[]).append(f'Reporting bounds differ: {dim}/{other_dim}')
                row['auxiliary_checks']=[]
                for standard in ('latitude','longitude'):
                    find=lambda ds,f:[n for n in ds.variables if ds[n].attrs.get('standard_name')==standard and set(ds[n].dims).issubset(set(f.dims)) and ds[n].ndim==2]
                    la,lb=find(left,a),find(right,b)
                    if len(la)==len(lb)==1:
                        key=(la[0],lb[0]);check=aux_cache.get(key)
                        if check is None:
                            x,y=left[la[0]].to_numpy(),right[lb[0]].to_numpy()
                            delta=x.astype(float)-y.astype(float)
                            if standard=='longitude':delta=(delta+180)%360-180
                            check={'standard_name':standard,'max_abs':float(np.nanmax(abs(delta))),
                                   'matched':bool(np.all(abs(delta)<=COORD_ATOL+RTOL*abs(x))),
                                   'longitude_convention':'circular modulo 360' if standard=='longitude' else None}
                            # Verify geographic coordinates against each file's own CF grid.
                            from pyproj import CRS,Transformer
                            projection_checks={}
                            for label,ds,f,actual in [('C',left,a,x),('Rust',right,b,y)]:
                                grid_name=f.attrs.get('grid_mapping',f.encoding.get('grid_mapping'))
                                attrs=dict(ds[grid_name].attrs)
                                xd=next(d for d in f.dims if role(f[d])=='x');yd=next(d for d in f.dims if role(f[d])=='y')
                                signature=(json.dumps(attrs,sort_keys=True,default=serialise),f[xd].to_numpy().tobytes(),f[yd].to_numpy().tobytes())
                                if signature not in _CF_EXPECTED:
                                    transform=Transformer.from_crs(CRS.from_cf(attrs),4326,always_xy=True)
                                    xx,yy=np.meshgrid(f[xd].to_numpy(),f[yd].to_numpy())
                                    _CF_EXPECTED[signature]=transform.transform(xx,yy)
                                expected=_CF_EXPECTED[signature][1 if standard=='latitude' else 0]
                                difference=actual.astype(float)-expected
                                if standard=='longitude':difference=(difference+180)%360-180
                                projection_checks[label]={'max_abs_degrees':float(np.max(abs(difference))),
                                    'consistent':bool(np.all(abs(difference)<=COORD_ATOL+RTOL*abs(expected)))}
                            check['CF_projection_checks']=projection_checks
                            check['explained_reference_difference']=(not check['matched'] and not projection_checks['C']['consistent'] and projection_checks['Rust']['consistent'])
                            aux_cache[key]=check
                        row['auxiliary_checks'].append(check)
                        if require_candidate_cf and not check['CF_projection_checks']['Rust']['consistent']:
                            row.setdefault('scientific_issues',[]).append(f'Candidate {standard} disagrees with its CF projection')
                        if not check['matched']:
                            if check.get('explained_reference_difference'):row.setdefault('metadata_notes',[]).append(f'C {standard} auxiliary disagrees with its own CF projection; Rust agrees. Raw differences retained.')
                            else:row.setdefault('scientific_issues',[]).append(f'{standard} differs')
                    else:
                        row.setdefault('metadata_notes',[]).append(f'Cannot compare unique {standard} arrays: {la}/{lb}')
                        if la and not lb:row.setdefault('scientific_issues',[]).append(f'Candidate missing {standard} auxiliary')
                def grid(ds,f):
                    n=f.attrs.get('grid_mapping',f.encoding.get('grid_mapping'))
                    return dict(ds[n].attrs) if n in ds else None
                ga,gb=grid(left,a),grid(right,b)
                row['grid_mapping']={'C':ga,'Rust':gb}
                if (ga is None)!=(gb is None) or (ga is not None and any(
                    not (np.isclose(ga.get(k,0),gb.get(k,0),rtol=RTOL,atol=COORD_ATOL) if isinstance(ga.get(k,0),(int,float,np.number)) and isinstance(gb.get(k,0),(int,float,np.number)) else ga.get(k)==gb.get(k)) for k in set(ga)|set(gb))):row.setdefault('scientific_issues',[]).append('Grid mapping attributes differ')
                b=b.transpose(*(mapping[d] for d in a.dims))
                # Match all dimensions explicitly before positional plane access.
                spatial=[d for d in a.dims if role(a[d]) in ('x','y')]
                if len(spatial)!=2:raise ValueError(f'Expected two horizontal axes: {spatial}')
                leading=[d for d in a.dims if d not in spatial]
                count=outside=masks=positive=negative=planes=0
                absolute=squared=signed=maximum=relative_max=0.;worst={}
                start=time.perf_counter()
                for index in itertools.product(*(range(a.sizes[d]) for d in leading)):
                    select=dict(zip(leading,index));select_b={mapping[d]:v for d,v in select.items()}
                    x=a.isel(select).transpose(*spatial).to_numpy().astype('float64')
                    y=b.isel(select_b).transpose(*(mapping[d] for d in spatial)).to_numpy().astype('float64')
                    mask_difference=np.isnan(x)!=np.isnan(y)
                    if mask_difference.any():
                        samples=row.setdefault('mask_difference_samples',[])
                        for location in np.argwhere(mask_difference)[:max(0,10-len(samples))]:
                            ij=tuple(int(v) for v in location)
                            samples.append({'leading_indices':select,'spatial_indices':dict(zip(spatial,ij)),'C':float(x[ij]),'Rust':float(y[ij])})
                    masks+=int(np.count_nonzero(mask_difference))
                    positive+=int(np.count_nonzero(np.isposinf(x)!=np.isposinf(y)))
                    negative+=int(np.count_nonzero(np.isneginf(x)!=np.isneginf(y)))
                    finite=np.isfinite(x)&np.isfinite(y);delta=y[finite]-x[finite];base=x[finite]
                    if len(delta):
                        ad=abs(delta);local=float(ad.max())
                        if local>maximum:maximum=local;worst=select
                        nonzero=base!=0
                        if nonzero.any():relative_max=max(relative_max,float(np.max(ad[nonzero]/abs(base[nonzero]))))
                        count+=len(delta);absolute+=float(ad.sum());squared+=float(np.dot(delta,delta));signed+=float(delta.sum())
                        outside+=int(np.count_nonzero(ad>RTOL*abs(base)))
                    planes+=1
                    if planes%100==0:LOG.info('Verify stream=%s field=%s planes=%d',stream,name,planes)
                row.update(planes=planes,finite_pairs=count,nan_mask_mismatches=masks,
                    positive_inf_mismatches=positive,negative_inf_mismatches=negative,outside_tolerance=outside,
                    max_abs=maximum if count else None,mae=absolute/count if count else None,
                    rmse=float(np.sqrt(squared/count)) if count else None,bias=signed/count if count else None,
                    max_relative_nonzero_reference=relative_max if count else None,worst_plane=worst,
                    comparison_seconds=time.perf_counter()-start,rtol=RTOL,atol=0)
                if masks or positive or negative or outside:row['status']='different'
                elif not count:row['status']='matching_masks_no_finite_pairs'
                else:row['status']='exact' if maximum==0 else 'within_tolerance'
                if row.get('scientific_issues') or any(k in row['metadata_differences'] for k in ('units','standard_name','cell_methods')):
                    row['numeric_status']=row['status'];row['status']='scientific_metadata_difference'
                LOG.info('Verified %s/%s status=%s max_abs=%s',stream,name,row['status'],row.get('max_abs'))
            except Exception as error:
                row['status']='comparison_error';row['error']=str(error);LOG.exception('Comparison failed %s/%s',stream,name)
        for name in sorted(unmatched):results.append({'stream':stream,'field':name,'status':'missing_reference','candidate_field':name,'candidate':field_metadata(bb[name])})
    return results

def measured(command,out,label):
    metric=out/f'{label}.time.json';log=out/f'{label}.log'
    fmt='{"user_seconds":%U,"system_seconds":%S,"peak_rss_kib":%M,"exit_status":%x}'
    invocation=['/usr/bin/time','--quiet','-f',fmt,'-o',str(metric),*command]
    LOG.info('Starting %s: %s',label,' '.join(command))
    begin=time.perf_counter()
    with log.open('x') as stream:result=subprocess.run(invocation,stdout=stream,stderr=subprocess.STDOUT)
    LOG.info('Finished %s: exit=%d elapsed=%.1fs',label,result.returncode,time.perf_counter()-begin)
    usage=json.loads(metric.read_text()) if metric.exists() else {}
    return usage|{'returncode':result.returncode,'wall_seconds':time.perf_counter()-begin,'command':command,'log':str(log)}

def synthetic():
    import tempfile
    with tempfile.TemporaryDirectory() as scratch:
        a=Path(scratch)/'a.nc';b=Path(scratch)/'b.nc'
        ds=xr.Dataset({'field':(('time','rlat','rlon'),np.arange(12,dtype=float).reshape(2,2,3))},
            coords={'time':np.array(['2000-01-01','2000-01-02'],dtype='datetime64[ns]'),'rlat':[-1.,1.],'rlon':[2.,3.,4.]})
        ds['time'].attrs['axis']='T';ds['rlat'].attrs['axis']='Y';ds['rlon'].attrs['axis']='X';ds['field'].attrs['units']='K'
        ds.to_netcdf(a);ds.rename({'time':'time1'}).to_netcdf(b)
        assert compare_files(a,b,'tiny')[0]['status']=='exact'
        changed=ds.copy(deep=True);changed['field'][0,0,0]=1.;changed.to_netcdf(b,mode='w')
        assert compare_files(a,b,'tiny')[0]['outside_tolerance']==1
        changed['field'][0,0,0]=np.nan;changed.to_netcdf(b,mode='w')
        assert compare_files(a,b,'tiny')[0]['nan_mask_mismatches']==1
        changed=ds.assign_coords(rlat=[-2.,2.]);changed.to_netcdf(b,mode='w')
        assert compare_files(a,b,'tiny')[0]['status']=='comparison_error'
    print('Synthetic comparison checks passed')

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--fixtures',type=Path);p.add_argument('--xml',type=Path);p.add_argument('--converter',type=Path);p.add_argument('--synthetic',action='store_true');p.add_argument('--references',type=Path);p.add_argument('--workers',type=int,default=4);p.add_argument('--wheel',type=Path);args=p.parse_args()
    dask.config.set(scheduler='synchronous')
    if args.synthetic:synthetic();return
    if not os.environ.get('PBS_JOBID'):raise RuntimeError('Run full-data conversions and comparisons inside an allocated PBS job')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'generated_utc':datetime.now(timezone.utc).isoformat(),'complete':False,'streams':{},'fields':[],
        'rtol':RTOL,'atol':0,'coordinate_rtol':RTOL,'coordinate_atol':COORD_ATOL,
        'note':'All planes are compared. C then Rust conversions run sequentially; filesystem caches/order are uncontrolled, so timings are not a controlled performance benchmark.',
        'candidate_wheel':fingerprint(args.wheel,True) if args.wheel else None,'xml':fingerprint(args.xml,True),'c_converter':fingerprint(args.converter,True),
        'notebook_generator':fingerprint(Path(__file__).with_name('comparison_notebook.py'),True),'runner':fingerprint(Path(__file__),True),'package_version':importlib.metadata.version('xarray-um'),
        'environment':{n:importlib.metadata.version(n) for n in ('numpy','xarray','dask','netCDF4','matplotlib','ipywidgets')},
        'allocation':{'job_id':os.environ.get('PBS_JOBID'),'project':os.environ.get('BENCHMARK_PROJECT'),'cpus':int(os.environ.get('NCPUS','1')),'memory_gb':os.environ.get('BENCHMARK_MEMORY_GB'),'walltime':os.environ.get('BENCHMARK_WALLTIME')}}
    write_json(out/'results.json',report);create_notebook(out)
    fixtures=json.loads(args.fixtures.read_text())
    for stream in ('mlev','sfc','plev'):
        detail={'status':'running','input':fixtures[stream]['path']};report['streams'][stream]=detail
        write_json(out/'results.json',report)
        try:
            source=Path(fixtures[stream]['path']);before=fingerprint(source)
            if before['bytes']!=fixtures[stream]['bytes']:raise ValueError('Staged input size differs from fixture identity')
            detail['input_identity']=before
            from xarray_um import _native
            cat=json.loads(_native.catalogue(str(source)))
            estimate=sum(r['i'][17]*r['i'][18]*4 for r in cat['records'])
            required=int(estimate*2.4)
            detail['estimated_two_output_bytes_with_headroom']=required
            if shutil.disk_usage(out).free<required:raise OSError('Insufficient filesystem free space for both full conversions')
            directory=out/stream;directory.mkdir()
            reference=(args.references/stream/'c.nc') if args.references else directory/'c.nc';candidate=directory/'rust.nc'
            detail.update(reference=str(reference),candidate=str(candidate))
            c=[str(args.converter),str(source),'-c',str(args.xml),'-i','-o',str(reference)]
            rust=[sys.executable,'-m','xarray_um.cli',str(source),'-c',str(args.xml),'-i','-o',str(candidate),'--workers',str(args.workers)]
            if args.references:
                previous=json.loads((args.references/'results.json').read_text())
                if previous['xml']['sha256']!=report['xml']['sha256'] or previous['c_converter']['sha256']!=report['c_converter']['sha256'] or previous['streams'][stream]['input_identity']!=before:raise ValueError('Reference provenance does not match')
                detail['C']=previous['streams'][stream]['C']|{'reused_reference':True}
            else:detail['C']=measured(c,directory,'c')
            write_json(out/'results.json',report)
            detail['Rust']=measured(rust,directory,'rust')
            write_json(out/'results.json',report)
            if any(detail[n]['returncode'] for n in ('C','Rust')):raise RuntimeError('Conversion failed; see per-tool logs')
            if fingerprint(source)!=before:raise ValueError('Input changed during comparison')
            detail['output_bytes']={'C':reference.stat().st_size,'Rust':candidate.stat().st_size}
            fields=compare_files(reference,candidate,stream);report['fields'].extend(fields)
            detail['auxiliary_inventory']={}
            for label,path in [('C',reference),('Rust',candidate)]:
                with xr.open_dataset(path,decode_coords='all') as inventory:
                    detail['auxiliary_inventory'][label]={n:field_metadata(v) for n,v in inventory.variables.items() if 'stash_item' not in v.attrs}
            detail['auxiliary_notes']=['C eta_theta/eta_rho arrays are vertical hybrid coefficients, not missing fields; xarray-um retains physical reference heights and original model-level numbers.']
            counts=defaultdict(int)
            for field in fields:counts[field['status']]+=1
            detail.update(status='compared',field_status_counts=dict(counts))
            write_json(out/'results.json',report)
            try:sample_figure(out,stream,fields)
            except Exception as error:detail['render_error']=str(error);LOG.exception('Sample rendering failed')
        except Exception as error:
            detail['status']='failed';detail['error']=str(error);detail['traceback']=traceback.format_exc();LOG.exception('Stream %s failed',stream)
        write_json(out/'results.json',report)
    report['complete']=True
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    report['success']=all(s.get('status')=='compared' for s in report['streams'].values()) and all(f['status'] in ('exact','within_tolerance','matching_masks_no_finite_pairs') for f in report['fields'])
    write_json(out/'results.json',report)
    print('NOTEBOOK',out/'comparison.ipynb',flush=True)
    print('STATUS',json.dumps({k:v['status'] for k,v in report['streams'].items()}),flush=True)
    if not report['success']:raise SystemExit(1)

if __name__=='__main__':main()
