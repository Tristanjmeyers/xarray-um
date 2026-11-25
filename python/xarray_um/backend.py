"""Scientific metadata and xarray adapter; binary decoding lives in Rust."""
from pathlib import Path
from importlib.resources import files
import hashlib
import json
import tomllib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import xarray as xr
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.core import indexing
from . import _native


def mapping(config=None, stash_xml=None):
    cfg = Path(config).expanduser() if config else Path.cwd()/'xarray-um.toml'
    if config or cfg.exists():
        values = tomllib.loads(cfg.read_text())
        if stash_xml is None:
            value = values.get('metadata', {}).get('stash_xml')
            if value:
                stash_xml = Path(value).expanduser()
                if not stash_xml.is_absolute(): stash_xml = cfg.parent/stash_xml
    path = Path(stash_xml).expanduser() if stash_xml else Path(str(files('xarray_um').joinpath('data/stash2cf_nzra.xml')))
    content = path.read_bytes()
    root = ET.fromstring(content)
    out = {}
    for section in root.iter('section'):
        for item in section.findall('item'):
            attrs = {node.tag:(node.text or '').strip() for node in item}
            code = int(section.attrib['section_id'])*1000 + int(attrs['stash_code'])
            proc = int(attrs['lbproc_code']) if 'lbproc_code' in attrs else None
            key = code, proc
            if key in out and out[key] != attrs: raise ValueError(f'Ambiguous XML mapping {key}')
            if not attrs.get('varname'): raise ValueError(f'Missing variable name {key}')
            out[key] = attrs
    if not out: raise ValueError(f'No STASH definitions: {path}')
    return out, str(path.resolve()), hashlib.sha256(content).hexdigest()


def metadata(definitions, code, proc):
    exact = definitions.get((code, proc))
    if exact is not None: return dict(exact)
    base = definitions.get((code,None))
    if base is None: return {'varname':f'stash_{code}_proc_{proc}', 'units':''}
    base = dict(base)
    if proc:
        prefix = {64:'mean_',128:'sum_' if int(base.get('accum_field','0')) else 'mean_',4096:'min_',8192:'max_'}.get(proc)
        if prefix is None: raise ValueError(f'Unsupported processing code {proc} for {code}')
        base['varname'] = base['varname'] if base['varname'].startswith(prefix) else prefix+base['varname']
    return base


def timestamp(i, start=0):
    # UM fields-file lookup releases use seconds at index 5 in release 3,
    # but day-of-year at index 5 in release 2.
    return np.datetime64(datetime(*i[start:start+5], second=i[start+5] if i[21]>=3 else 0).isoformat(),'ns')


# LBTIM's middle digit denotes a time interval; supported temporal LBPROC
# variants store interval start/end in the first/second date tuples.
TEMPORAL_PROCESSING = {128, 4096, 8192}

def interval(record):
    i=record['i']
    if i[24] not in TEMPORAL_PROCESSING:
        return None
    if (i[12]//10)%10 != 2:
        raise ValueError(f'Unsupported interval time indicator LBTIM={i[12]}')
    start,end=timestamp(i),timestamp(i,6)
    if end<=start:raise ValueError(f'Non-positive aggregation interval: {start} to {end}')
    return start,end

def valid_time(record):
    bounds=interval(record)
    return bounds[1] if bounds else timestamp(record['i'])


def geographic(x, y, pole_latitude, pole_longitude, component):
    from pyproj import CRS, Transformer
    rotated=CRS.from_cf({'grid_mapping_name':'rotated_latitude_longitude',
                        'grid_north_pole_latitude':pole_latitude,
                        'grid_north_pole_longitude':pole_longitude})
    transform=Transformer.from_crs(rotated,4326,always_xy=True)
    return np.asarray(transform.transform(x,y)[component],dtype='float32')


class FieldArray(BackendArray):
    def __init__(self,path,cat,records,shape,scale,grid_type,grid,reader):
        self.path=path; self.cat=cat; self.records=records; self.shape=shape
        self.scale=scale; self.grid_type=grid_type; self.grid=grid; self.reader=reader
        self.dtype=np.dtype('float64' if cat['word_size']==8 and any(r['i'][20]==0 for r in records.values()) else 'float32')
    def __getstate__(self):
        # File handles are process-local. Dask process schedulers serialise the
        # lazy array, then reopen a positioned handle in the receiving process.
        state=dict(self.__dict__)
        state.pop('reader')
        return state
    def __setstate__(self,state):
        self.__dict__.update(state)
        self.reader=_native.Reader(self.path)
    def __getitem__(self,key):
        return indexing.explicit_indexing_adapter(key,self.shape,indexing.IndexingSupport.BASIC,self._read)
    def _read(self,key):
        leading=len(self.shape)-2
        selections=[np.arange(self.shape[a])[key[a]] for a in range(leading)]
        indices=[np.atleast_1d(s) for s in selections]
        # Bound allocations to the requested leading indices and spatial window.
        chunks=[]
        for where in np.ndindex(*(len(v) for v in indices)):
            index=tuple(int(indices[a][where[a]]) for a in range(leading))
            record=self.records[index];i,r=record['i'],record['r'];w=self.cat['word_size']
            length=(i[29] if i[29]>0 else i[14])*w
            data=_native.plane(self.reader,i[28]*w,length,i[18],i[17],i[20],w,
                               self.cat['big_endian'],r[17],i[38],
                               self.grid_type if self.grid=='thermodynamic' else 0,
                               self.shape[-2],self.scale,self.dtype==np.dtype('float64')).reshape(self.shape[-2:])
            chunks.append(data[key[-2:]])
        spatial=np.empty(self.shape[-2:],dtype=[])[key[-2:]].shape
        if not chunks: return np.empty(tuple(len(v) for a,v in enumerate(indices) if not np.isscalar(selections[a]))+spatial,dtype=self.dtype)
        result=(chunks[0] if len(chunks)==1 else np.stack(chunks)).reshape(*(len(v) for v in indices),*spatial)
        for axis in reversed(range(leading)):
            if np.isscalar(selections[axis]): result=np.take(result,0,axis=axis)
        return result


class UMBackend(BackendEntrypoint):
    description='Lazy NZRA UM fields files using a Rust decoder'
    def guess_can_open(self,filename_or_obj):
        return isinstance(filename_or_obj,(str,Path)) and str(filename_or_obj).endswith('.um')
    def open_dataset(self,filename_or_obj,*,drop_variables=None,variables=None,stash_codes=None,levels=None,grid='native',config=None,stash_xml=None,decode_times=True,mask_and_scale=True,decode_coords=True):
        if variables is not None and stash_codes is not None: raise ValueError('Choose variables OR stash_codes')
        if grid not in ('native','thermodynamic'): raise ValueError('grid must be native or thermodynamic')
        if not mask_and_scale: raise NotImplementedError('Raw WGDOS integers are not a physical data array')
        path=str(Path(filename_or_obj).expanduser().resolve())
        definitions,xml,fingerprint=mapping(config,stash_xml)
        cat=json.loads(_native.catalogue(path))
        groups=defaultdict(list)
        for rec in cat['records']:
            i=rec['i'];groups[(i[41],i[24])].append(rec)
        requested=set(variables or []);wanted_codes=set(map(int,stash_codes or []));dropped=set(drop_variables or [])
        coords={};data={};registry={};used_domains=set();found_codes=set()
        reader=_native.Reader(path)
        def coordinate(base,values,attrs,identity=None):
            values=np.asarray(values)
            signature=(base,values.dtype.str,values.tobytes(),json.dumps({k:v for k,v in attrs.items() if k!='long_name'},sort_keys=True),identity)
            if signature in registry:return registry[signature]
            name=base
            index=0
            while name in coords:
                index+=1
                name=f'time{index}' if base=='time0' else f'{base}_{index}'
            coords[name]=xr.Variable((name,),values,attrs);registry[signature]=name
            return name
        level_constants=np.asarray(cat['level_constants']).reshape(cat['level_shape'])
        for (code,proc),records in sorted(groups.items()):
            meta=metadata(definitions,code,proc);name=meta['varname']
            if (requested and name not in requested) or (wanted_codes and code not in wanted_codes) or name in dropped:continue
            found_codes.add(code)
            if name in data:
                raise ValueError('Variable name collision across STASH/processing groups: '+name)
            i,r=records[0]['i'],records[0]['r']
            if i[15] not in (1,101):raise ValueError(f'{name}: unsupported grid code {i[15]}')
            times=sorted({valid_time(v) for v in records})
            zvalues=sorted({v['i'][32] for v in records})
            has_z=len(zvalues)>1
            # UM lookup word 43 (LBUSER(5)) is an independent STASH
            # pseudo dimension, e.g. surface tiles, not a duplicate level.
            pseudo_values=sorted({v['i'][42] for v in records})
            has_pseudo=len(pseudo_values)>1 or any(pseudo_values)
            domain='single';zattrs={};physical=zvalues;zbase='level'
            if has_z:
                if i[25]==8:domain='PLEVS';physical=[value*100.0 for value in zvalues];zbase=f'pressure{len(zvalues)}';zattrs={'standard_name':'air_pressure','units':'Pa','positive':'down','axis':'Z'}
                elif i[25]==65:
                    rho=int(meta.get('level_type','2'))==1;domain='Rho_1_70' if rho else 'Theta_1_70'
                    physical=[float(np.float32(level_constants[6 if rho else 4][int(z)-1 if rho else int(z)])) for z in zvalues]
                    zbase=f'hybrid{"r" if rho else "t"}{len(zvalues)}';zattrs={'units':'meters','positive':'up','axis':'Z'}
                elif i[25]==6:domain='ATM_SOIL';physical=[float(level_constants[3][int(z)-1]) for z in zvalues];zbase=f'soil_level{len(zvalues)}';zattrs={'units':'meters','positive':'down','axis':'Z'}
                elif i[25]==1:
                    # BLEV (real lookup word 52), rather than integer LBLEV,
                    # holds the physical height. Match the NZRA converter's
                    # geometric-altitude convention, in metres above sea level.
                    domain='HEIGHTS';zbase=f'altitude{len(zvalues)}'
                    heights={}
                    for record in records:
                        level=record['i'][32];height=record['r'][6]
                        if not np.isfinite(height) or not np.isclose(height,heights.get(level,height),rtol=0,atol=1e-6):
                            raise ValueError(f'{name}: inconsistent geometric height for level {level}')
                        heights[level]=height
                    physical=[float(np.float32(heights[z])) for z in zvalues]
                    zattrs={'standard_name':'altitude','long_name':'height above sea level','units':'m','positive':'up','axis':'Z'}
                else:raise ValueError(f'{name}: unsupported vertical coordinate LBVC={i[25]}')
            if domain in (levels or {}):
                used_domains.add(domain);selection=list(levels[domain]);candidates=physical if domain in ('PLEVS','ATM_SOIL','HEIGHTS') else [z-.5 if domain=='Rho_1_70' else z for z in zvalues]
                missing=set(selection)-set(candidates)
                if missing:raise ValueError(f'{name}: unavailable {domain} values {missing}; available={candidates}')
                chosen=[j for j,v in enumerate(candidates) if v in selection]
                zvalues=[zvalues[j] for j in chosen];physical=[physical[j] for j in chosen]
                if not zvalues:raise ValueError('Empty level selection')
            gridtype=int(meta.get('umgrid','1'))
            source_ny=i[17];nx=i[18]
            ny=cat['int_constants'][6] if grid=='thermodynamic' else source_ny
            # PP BZY/BZX describe the point before the first grid coordinate.
            ys=np.asarray(r[13]+(np.arange(source_ny)+1)*r[14],dtype='float32')
            xs=np.asarray(r[15]+(np.arange(nx)+1)*r[16],dtype='float32')
            if grid=='thermodynamic':
                constants=cat['real_constants'];ys=np.asarray(constants[2]+(np.arange(ny)+0.5)*constants[1],dtype='float32');xs=np.asarray(constants[3]+(np.arange(nx)+0.5)*constants[0],dtype='float32')
            y=coordinate('rlat',ys,{'standard_name':'grid_latitude','units':'degrees','axis':'Y'})
            x=coordinate('rlon',xs,{'standard_name':'grid_longitude','units':'degrees','axis':'X'})
            time_attrs={'standard_name':'time','axis':'T','long_name':'end of reporting period' if proc in TEMPORAL_PROCESSING else 'valid time'}
            time_bounds = None
            if proc in TEMPORAL_PROCESSING:
                by_time = {}
                for record in records:
                    bounds = interval(record)
                    stamp = valid_time(record)
                    if stamp in by_time and by_time[stamp] != bounds:
                        raise ValueError(f'{name}: conflicting interval bounds')
                    by_time[stamp] = bounds
                time_bounds = np.asarray([by_time[value] for value in times], dtype='datetime64[ns]')
            t=coordinate('time0',np.asarray(times),time_attrs,
                         None if time_bounds is None else time_bounds.tobytes())
            if proc in TEMPORAL_PROCESSING:
                by_time={}
                for record in records:
                    bounds=interval(record);valid=valid_time(record)
                    if valid in by_time and by_time[valid]!=bounds:raise ValueError(f'{name}: conflicting interval bounds')
                    by_time[valid]=bounds
                bounds_name=t+'_bnds'
                bounds_array=np.asarray([by_time[value] for value in times],dtype='datetime64[ns]')
                if bounds_name in coords:
                    if not np.array_equal(coords[bounds_name].data,bounds_array):raise ValueError('Conflicting shared interval bounds')
                else:coords[bounds_name]=xr.Variable((t,'nv'),bounds_array,{'long_name':'start and end of reporting period'})
                coords[t].attrs['bounds']=bounds_name
            dims=[t];shape=[len(times)]
            if has_z:
                if domain in ('Theta_1_70','Rho_1_70'):
                    zattrs['long_name']='reference height of native model levels'
                native_levels=tuple(v-.5 if domain=='Rho_1_70' else v for v in zvalues)
                z=coordinate(zbase,np.asarray(physical,dtype='float32'),zattrs,
                             native_levels if domain in ('Theta_1_70','Rho_1_70') else None)
                dims.append(z);shape.append(len(zvalues))
                if domain in ('Theta_1_70','Rho_1_70'):
                    coords[z+'_model_level']=xr.Variable((z,),np.asarray([v-.5 if domain=='Rho_1_70' else v for v in zvalues]),{'long_name':'native model level number','units':'1'})
            if has_pseudo:
                pseudo=coordinate('pseudo_level',np.asarray(pseudo_values,dtype='int32'),
                                  {'long_name':'STASH pseudo-level identifier (UM LBUSER(5))','units':'1'})
                dims.append(pseudo);shape.append(len(pseudo_values))
            dims += [y,x];shape += [ny,nx]
            look={};ti={v:j for j,v in enumerate(times)};zi={v:j for j,v in enumerate(zvalues)}
            pi={v:j for j,v in enumerate(pseudo_values)}
            for rec in records:
                if rec['i'][32] not in zi:continue
                key=(ti[valid_time(rec)],)
                if has_z:key+=(zi[rec['i'][32]],)
                if has_pseudo:key+=(pi[rec['i'][42]],)
                if key in look:raise ValueError(f'{name}: duplicate time/level/pseudo-level')
                if (rec['i'][17],rec['i'][18])!=(source_ny,nx):raise ValueError(f'{name}: changing grid')
                look[key]=rec
            if len(look)!=len(times)*len(zvalues)*len(pseudo_values):raise ValueError(f'{name}: incomplete time/level/pseudo-level coverage')
            attrs={k:meta[v] for k,v in [('long_name','longname'),('standard_name','standardname'),('units','units')] if meta.get(v)}
            attrs.update(stash_model=i[44],stash_section=code//1000,stash_item=code%1000,um_lbproc=proc)
            if proc in (64,128,4096,8192):attrs['cell_methods']='time: '+({64:'mean',128:'sum' if int(meta.get('accum_field','0')) else 'mean',4096:'minimum',8192:'maximum'}[proc])
            array=FieldArray(path,cat,look,tuple(shape),float(meta.get('scalefact','1')),gridtype,grid,reader)
            data[name]=xr.Variable(tuple(dims),indexing.LazilyIndexedArray(array),attrs)
            data[name].encoding['source_missing_value']=float(r[17])
            data[name].encoding['preferred_chunks']={d:(n if d in (y,x) else 1) for d,n in zip(dims,shape)}
        if requested-set(data)-dropped:raise ValueError(f'Variables not found: {requested-set(data)-dropped}')
        if wanted_codes-found_codes:raise ValueError(f'STASH codes not found: {wanted_codes-found_codes}')
        if set(levels or {})-used_domains:raise ValueError(f'Unused level domains: {set(levels)-used_domains}')
        if not data:raise ValueError('No fields selected')
        # Geographic auxiliary coordinates are generated lazily for each native grid.
        import dask.array as da
        grids={}
        pole_lat,pole_lon=cat['real_constants'][4:6]
        coords['rotated_pole']=xr.Variable((),0,{'grid_mapping_name':'rotated_latitude_longitude',
            'grid_north_pole_latitude':pole_lat,'grid_north_pole_longitude':pole_lon,'north_pole_grid_longitude':0.})
        for name,variable in data.items():
            y,x=variable.dims[-2:]
            if (y,x) not in grids:
                suffix='' if not grids else '_'+hashlib.sha256(f'{y}/{x}'.encode()).hexdigest()[:8]
                lat,lon='latitude'+suffix,'longitude'+suffix
                xx,yy=da.meshgrid(da.from_array(coords[x].data,chunks=-1),da.from_array(coords[y].data,chunks=-1))
                for cname,component,standard,units in [(lat,1,'latitude','degrees_north'),(lon,0,'longitude','degrees_east')]:
                    values=da.map_blocks(geographic,xx,yy,pole_latitude=pole_lat,pole_longitude=pole_lon,component=component,dtype='float32')
                    coords[cname]=xr.Variable((y,x),values,{'standard_name':standard,'units':units})
                grids[y,x]=(lat,lon)
            variable.attrs['grid_mapping']='rotated_pole'
            variable.encoding['coordinates']=' '.join(grids[y,x])
        dataset=xr.Dataset(data,coords=coords,attrs={'source':path,'stash_xml':xml,'stash_xml_sha256':fingerprint,'grid':grid})
        if not decode_times:
            for k in list(dataset.coords):
                if dataset[k].dtype.kind=='M':dataset[k]=xr.Variable(dataset[k].dims,dataset[k].values.astype('datetime64[s]').astype('int64'),{'units':'seconds since 1970-01-01','calendar':'proleptic_gregorian'})
        dataset.set_close(reader.close)
        return dataset
