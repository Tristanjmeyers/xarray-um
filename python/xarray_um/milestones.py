"""Bounded scientific demonstrations. Execute substantial checks inside PBS."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import time
import traceback
from unittest.mock import patch
import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import xarray as xr
from IPython.display import display
from . import _native
from .backend import mapping, UMBackend

CASES = {
    'soil': ('sfc',8208,'sfc_soil_moisture'),
    'pressure': ('plev',16203,'std_lev_temp'),
    'tke': ('mlev',3473,'turb_kinetic_energy'),
    'wind': ('mlev',2,'model_u'),
    'vwind': ('mlev',3,'model_v'),
    'temperature': ('sfc',3236,'sfc_temp'),
}

class Demonstration:
    def __init__(self,stage):
        if not os.environ.get('PBS_JOBID'):
            raise RuntimeError('Execute this demonstration in a PBS allocation, not a login node. See notebooks/README.md.')
        self.stage=stage
        self.repo=Path(os.environ.get('UM2NETCDFR_REPO',Path(__file__).resolve().parents[2]))
        self.root=Path(os.environ.get('UM2NETCDFR_ARTIFACTS',Path.home()/'agent-datasets/xarray_um-milestones'/os.environ['PBS_JOBID']))
        self.root.mkdir(parents=True,exist_ok=True)
        self.inputs=Path(os.environ.get('UM2NETCDFR_INPUTS',Path.home()/'agent-datasets/xarray_um-pilot'))
        self.inputs.mkdir(parents=True,exist_ok=True)
        self.workflow=Path(os.environ.get('NZRA_WORKFLOW',Path.home()/'cylc-src/nzra_extract'))
        self.xml=self.workflow/'config/stash2cf_nzra_lbproc2.xml'
        self.results=[];self.measurements=[];self.sources=[]
        self.provenance={'generated_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,
            'package_version':importlib.metadata.version('xarray_um'),'pbs_job':os.environ['PBS_JOBID'],
            'resources':{'cpus':os.environ.get('NCPUS'),'host':os.uname().nodename},
            'xml':str(self.xml),'xml_sha256':hashlib.sha256(self.xml.read_bytes()).hexdigest(),
            'source_sha256':{str(p.relative_to(self.repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [self.repo/'src/lib.rs',*sorted((self.repo/'python/xarray_um').glob('*.py'))]}}
        print(json.dumps(self.provenance,indent=2))
    def check(self,label,function):
        started=time.monotonic()
        try:
            detail=function()
            row={'check':label,'status':'PASS','detail':str(detail)}
        except NotImplementedError as e:
            row={'check':label,'status':'UNSUPPORTED','detail':str(e)}
        except Exception as e:
            row={'check':label,'status':'FAIL','detail':f'{type(e).__name__}: {e}','traceback':traceback.format_exc()}
        row['seconds']=time.monotonic()-started;self.results.append(row)
        print(row['status'],label,row['detail'],flush=True)
        return row
    def finish(self):
        complete=bool(self.results) and all(r['status']=='PASS' for r in self.results)
        report=self.provenance|{'milestone_complete':complete,'checks':self.results,'measurements':self.measurements,'inputs':self.sources,
            'limitation':'Selected examples are regression evidence, not proof for every NZRA field or date.'}
        path=self.root/(self.stage+'.json');path.write_text(json.dumps(report,indent=2,default=str))
        print('MILESTONE COMPLETE' if complete else 'MILESTONE NOT COMPLETE — see failures/unsupported checks')
        print('Evidence:',path)
        return pd.DataFrame(self.results)[['check','status','seconds','detail']]
    def paths(self,case,prepare=False):
        stream,code,name=CASES[case]
        fixture_path=self.inputs/'fixtures.json'
        if not fixture_path.exists():
            raise FileNotFoundError('Stage existing completed UM files first; no archive hydration is performed')
        fixture=json.loads(fixture_path.read_text())[stream]
        live=Path(fixture['path'])
        if not live.is_file():raise FileNotFoundError(live)
        st=live.stat()
        if st.st_size!=fixture['bytes']:raise ValueError('Working copy size differs from verified fixture')
        self.sources.append(fixture | {'working_mtime_ns':st.st_mtime_ns})
        # Reference caches are keyed by the actual working input and XML, never
        # just a field name: changing dates must not reuse another run's oracle.
        identity={'input':str(live.resolve()),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,
                  'xml_sha256':self.provenance['xml_sha256'],'stash':code,
                  'converter_sha256':hashlib.sha256((self.workflow/'bin/um2netcdf').read_bytes()).hexdigest()}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
        oracle=self.inputs/(case+'_'+key+'_oracle.nc')
        provenance=oracle.with_suffix('.provenance.json')
        if oracle.exists():
            if not provenance.exists() or json.loads(provenance.read_text())!=identity:
                raise ValueError('Reference identity missing/mismatched: '+str(oracle))
        if prepare and not oracle.exists():
            temporary=oracle.with_name(oracle.name+'.partial')
            if temporary.exists():raise FileExistsError('Incomplete oracle requires inspection: '+str(temporary))
            cmd=[str(self.workflow/'bin/um2netcdf'),str(live),'-c',str(self.xml),'-i','-o',str(temporary),'-s',str(code)]
            print('Oracle command:',cmd,flush=True)
            result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            (self.root/(case+'_oracle.log')).write_text(result.stdout)
            result.check_returncode();temporary.replace(oracle)
            provenance.write_text(json.dumps(identity,indent=2))
        return live,oracle,name
    def open(self,case,**kwargs):
        live,_,name=self.paths(case)
        return xr.open_dataset(live,engine='um',variables=[name],stash_xml=self.xml,**kwargs)
    def compare(self,case):
        live,oracle,name=self.paths(case,prepare=True)
        with xr.open_dataset(live,engine='um',stash_xml=self.xml,variables=[name],grid='thermodynamic') as rust, xr.open_dataset(oracle) as reference:
            a,b=rust[name],reference[name]
            assert a.shape==b.shape,(a.shape,b.shape)
            assert a.attrs.get('units')==b.attrs.get('units'),'units differ'
            for x,y in zip(a.dims,b.dims):
                av,bv=a[x].to_numpy(),b[y].to_numpy()
                if av.dtype.kind=='M':np.testing.assert_array_equal(av,bv)
                else:np.testing.assert_allclose(av,bv,rtol=1e-6,atol=1e-5)
            errors=[]
            leading=a.dims[:-2]
            for fraction in (0,.5,1):
                index={d:int((a.sizes[d]-1)*fraction) for d in leading}
                other={e:index[d] for d,e in zip(a.dims,b.dims) if d in index}
                aa=a.isel(index).to_numpy();bb=b.isel(other).to_numpy()
                np.testing.assert_array_equal(np.isnan(aa),np.isnan(bb))
                errors.append(float(np.nanmax(abs(aa-bb))))
                np.testing.assert_array_equal(aa,bb)
            # A bounded map from the final comparison, with common source limits.
            fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
            clim=(min(np.nanmin(aa),np.nanmin(bb)),max(np.nanmax(aa),np.nanmax(bb)))
            for ax,data,title in zip(axes,[bb,aa,aa-bb],['C reference','Rust','Rust − C']):
                options={'vmin':clim[0],'vmax':clim[1]} if title!='Rust − C' else {'cmap':'RdBu_r'}
                im=ax.imshow(data,origin='lower',**options);fig.colorbar(im,ax=ax,label=a.attrs.get('units',''))
                ax.set_title(title);ax.set_xlabel('Native x index');ax.set_ylabel('Native y index')
            selections=', '.join(f'{d}={a[d].to_numpy()[j]}' for d,j in index.items())
            fig.suptitle(f'NZRA {name} | {selections} | {live.name}')
            fig.savefig(self.root/(case+'_comparison.png'),dpi=120)
            plt.show();plt.close(fig)
            return {'shape':a.shape,'max_abs_error':max(errors),'slices_compared':3}
    def selections(self,case,domain,values):
        _,_,name=self.paths(case)
        with self.open(case) as full,self.open(case,levels={domain:values}) as subset:
            a,b=full[name],subset[name]
            assert b.shape[1]==len(values)
            coord=next(c for c in subset.coords if c.endswith('_model_level')) if domain!='PLEVS' else b.dims[1]
            np.testing.assert_array_equal(sorted(subset[coord].to_numpy()),sorted(values))
            selected=b.isel({b.dims[0]:0,b.dims[1]:0}).to_numpy()
            physical=b[b.dims[1]].to_numpy()[0]
            expected=a.sel({a.dims[1]:physical}).isel({a.dims[0]:0}).to_numpy()
            np.testing.assert_array_equal(selected,expected)
            return dict(b.sizes)
    def lazy(self):
        with patch('xarray_um.backend._native.plane',wraps=_native.plane) as calls:
            with self.open('soil') as ds:
                name=CASES['soil'][2];assert calls.call_count==0
                ds[name].isel({d:0 for d in ds[name].dims}).to_numpy()
                assert calls.call_count==1,calls.call_count
        return '0 payload decodes on open; exactly 1 plane for a selected point'
    def precision(self):
        # Real source dtype/packing inventory; float64 inputs must stay float64.
        from .backend import FieldArray
        live,_,_=self.paths('soil');cat=json.loads(_native.catalogue(str(live)))
        packed=sorted({r['i'][20] for r in cat['records']})
        # A focused synthetic call exposes dtype narrowing, without a large read.
        cat['word_size']=8;rec={'i':[0]*45,'r':[0.]*19};rec['i'][20]=0
        array=FieldArray(str(live),cat,{(0,):rec},(1,1,1),1,1,'native')
        assert array.dtype==np.dtype('float64'),'Unpacked float64 preservation is not implemented'
        return {'packing_modes':packed}
    def interpolation(self):
        results={};failures=[]
        for component,case in [('u','wind'),('v','vwind')]:
            try:results[component]=self.compare(case)
            except Exception as error:
                failures.append(f'{component}: {type(error).__name__}: {error}')
        if failures:raise AssertionError('Wind comparison failures: '+ '; '.join(failures))
        return results
    def intervals(self):
        live,oracle,_=self.paths('temperature',prepare=True)
        from .backend import interval,valid_time
        cat=json.loads(_native.catalogue(str(live)))
        with xr.open_dataset(live,engine='um',stash_xml=self.xml,stash_codes=[3236]) as ds, xr.open_dataset(oracle) as reference:
            aggregated=[v for v in ds.data_vars if ds[v].attrs.get('um_lbproc',0)]
            assert aggregated,'No aggregate fixture available'
            details={}
            for name in aggregated:
                field=ds[name];other=reference[name]
                t=next(d for d in field.dims if ds[d].attrs.get('axis')=='T')
                rt=other.dims[0]
                bound_name=ds[t].attrs.get('bounds')
                assert bound_name in ds,'Aggregation interval bounds missing'
                np.testing.assert_array_equal(ds[t],reference[rt])
                records=[r for r in cat['records'] if r['i'][41]==3236 and r['i'][24]==field.attrs['um_lbproc']]
                expected={valid_time(r):interval(r) for r in records}
                actual=ds[bound_name].to_numpy()
                np.testing.assert_array_equal(actual,np.asarray([expected[v] for v in ds[t].to_numpy()]))
                for index in (0,field.shape[0]-1):
                    aa=field.isel({t:index}).to_numpy();bb=other.isel({rt:index}).to_numpy()
                    np.testing.assert_array_equal(np.isnan(aa),np.isnan(bb))
                    np.testing.assert_array_equal(aa,bb)
                details[name]={'first_bounds':actual[0].astype(str).tolist(),'cell_methods':field.attrs['cell_methods']}
            return details
    def xml_snapshot(self):
        import xml.etree.ElementTree as ET
        content=self.xml.read_bytes();root=ET.fromstring(content)
        item=next(i for section in root.iter('section') if section.attrib['section_id']=='8'
                  for i in section.findall('item') if i.findtext('stash_code')=='208')
        name=item.find('varname');old=name.text
        config=self.root/'metadata-demo.toml';xml=self.root/'metadata-demo.xml'
        xml.write_bytes(content);config.write_text('[metadata]\nstash_xml="metadata-demo.xml"\n')
        live,_,_=self.paths('soil')
        with xr.open_dataset(live,engine='um',config=config,variables=[old]) as before:
            name.text='soil_moisture_demo';ET.ElementTree(root).write(xml,encoding='utf-8')
            with xr.open_dataset(live,engine='um',config=config,variables=['soil_moisture_demo']) as after:
                assert old in before and 'soil_moisture_demo' in after
                assert before.attrs['stash_xml_sha256']!=after.attrs['stash_xml_sha256']
        return 'New opens use changed XML; already-open dataset retains its mapping'
    def parallel(self):
        name=CASES['tke'][2]
        timing=[]
        with self.open('tke',chunks={}) as ds:
            a=ds[name];subset=a.isel({a.dims[0]:slice(0,2),a.dims[1]:slice(0,2)})
            reference=None
            for repetition in range(2):
                for workers in (1,4):
                    start=time.monotonic();cpu=time.process_time()
                    result=subset.data.compute(scheduler='threads',num_workers=workers)
                    seconds=time.monotonic()-start
                    if reference is None:reference=result
                    else:np.testing.assert_array_equal(reference,result)
                    row={'workers':workers,'repetition':repetition+1,'wall_seconds':seconds,'cpu_seconds':time.process_time()-cpu,
                         'rss_after_bytes':psutil.Process().memory_info().rss,'decoded_planes':4,'note':'Warm-cache effects; RSS after operation is not peak memory'}
                    timing.append(row)
            process_result=subset.data.compute(scheduler='processes',num_workers=2)
            np.testing.assert_array_equal(reference,process_result)
        self.measurements.extend(timing)
        frame=pd.DataFrame(timing);display(frame)
        ax=frame.groupby('workers').wall_seconds.mean().plot.bar(title='NZRA TKE: four selected planes, mean of two repeats',ylabel='seconds')
        ax.figure.savefig(self.root/'thread_benchmark.png',dpi=120);plt.show();plt.close(ax.figure)
        return 'Thread and process values match; timings are measurements, not a speedup guarantee'
    def compatibility(self):
        import importlib.util
        from unittest.mock import patch
        helper_path=self.workflow/'bin/extract_archived_nzra.py'
        spec=importlib.util.spec_from_file_location('existing_conversion',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        live,oracle,name=self.paths('soil');destination=self.root/'compatibility/sfc/nzra_sfc_1990020100-utc.nc.partial'
        destination.parent.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env['PATH']=str(self.repo/'bin')+os.pathsep+env['PATH']
        with patch.dict(os.environ,env):helper.convert_um_file(live,destination,self.xml,['8208'])
        self.provenance['workflow_helper_sha256']=hashlib.sha256(helper_path.read_bytes()).hexdigest()
        with xr.open_dataset(destination) as rust,xr.open_dataset(oracle) as ref:
            assert rust[name].shape==ref[name].shape
            for t in [0,rust[name].shape[0]-1]:
                np.testing.assert_array_equal(rust[name].isel({rust[name].dims[0]:t}).to_numpy(),ref[name].isel({ref[name].dims[0]:t}).to_numpy())
        # Call the actual unchanged splitter, with its existing packing helpers.
        sp=self.workflow/'bin/split_nzra_netcdf_parallel.py'
        spec=importlib.util.spec_from_file_location('existing_splitter',sp);split=importlib.util.module_from_spec(spec);spec.loader.exec_module(split)
        final=destination.with_suffix('');destination.replace(final)
        with dask.config.set(scheduler='synchronous'):
            outputs=split.split_nzra_netcdf_parallel(final,cpus=1,quantise_func=split.quantise_nzra,compress_func=split.compress_nzra,
                                                   delete_source_netcdf=False)
        assert outputs,'No split files generated'
        with xr.open_dataset(outputs[0]) as packed,xr.open_dataset(oracle) as reference:
            a,b=packed[name],reference[name]
            for t in (0,a.shape[0]-1):
                aa=a.isel({a.dims[0]:t}).to_numpy();bb=b.isel({b.dims[0]:t}).to_numpy()
                np.testing.assert_array_equal(np.isnan(aa),np.isnan(bb))
                assert np.nanmax(abs(aa-bb))<=2**split.FIELD_PREC[name]['packacc']/2+1e-6
        return {'intermediate':str(final),'retained_outputs':list(map(str,outputs))}
    def cli_errors(self):
        live,_,_=self.paths('soil')
        cmd=[str(self.repo/'bin/um2netcdf'),str(live),'-c',str(self.xml),'-o',str(self.root/'invalid.nc.partial'),'-s','999999']
        result=subprocess.run(cmd,capture_output=True,text=True)
        assert result.returncode!=0 and not (self.root/'invalid.nc.partial').exists()
        return 'Invalid selection returns nonzero and does not publish an output'
