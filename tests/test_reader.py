import json,struct
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import xarray as xr
from xarray_um import _native
from xarray_um.backend import mapping,metadata,UMBackend

def fixture_file(tmp_path,word=8,endian='>'):
    # One unpacked 2x3 surface field, synthetic UM fixed header and lookup.
    integer='q' if word==8 else 'i';real='d' if word==8 else 'f'
    h=[0]*256;h[1]=1;h[150]=64;h[151]=1;h[149]=320;h[104]=257;h[99]=263;h[109]=309;h[110]=1;h[111]=7
    b=bytearray(400*word)
    def ints(at,v):b[at*word:(at+len(v))*word]=struct.pack(endian+integer*len(v),*v)
    def reals(at,v):b[at*word:(at+len(v))*word]=struct.pack(endian+real*len(v),*v)
    ints(0,h);reals(256,[1,1,-1,-1,90,0]);constants=[0]*46;constants[5:7]=[3,2];ints(262,constants);reals(308,[0]*7)
    i=[0]*45;i[:12]=[1990,2,1,0,0,0]*2;i[14]=6;i[15]=101;i[17]=2;i[18]=3;i[21]=3;i[28]=390;i[29]=6;i[38]=1;i[41]=8208
    ints(319,i);r=[0.]*19;r[13]=-1;r[14]=1;r[15]=-1;r[16]=1;r[17]=-1e30;reals(319+45,r);reals(390,[1,2,3,4,5,6])
    p=tmp_path/'test.um';p.write_bytes(b);return p

@pytest.mark.parametrize('word,endian',[(8,'>'),(8,'<'),(4,'>'),(4,'<')])
def test_decode_lazy(tmp_path,word,endian):
    p=fixture_file(tmp_path,word,endian)
    with patch('xarray_um.backend._native.plane',wraps=_native.plane) as decode:
        ds=UMBackend().open_dataset(p,variables=['sfc_soil_moisture'])
        assert decode.call_count==0
        np.testing.assert_array_equal(ds.sfc_soil_moisture.isel(time0=0,rlat=1).values,[4,5,6])
        assert decode.call_count==1
        ds.close()

def test_truncated(tmp_path):
    p=tmp_path/'bad.um';p.write_bytes(b'bad')
    with pytest.raises(ValueError):_native.catalogue(str(p))

def test_xml_relative_and_snapshot(tmp_path):
    xml=tmp_path/'fields.xml';xml.write_text('<root><section section_id="0"><item><stash_code>2</stash_code><varname>u</varname></item></section></root>')
    config=tmp_path/'config.toml';config.write_text('[metadata]\nstash_xml="fields.xml"')
    a,_,fingerprint=mapping(config)
    xml.write_text(xml.read_text().replace('>u<','>wind<'))
    b,_,other=mapping(config)
    assert a[(2,None)]['varname']=='u' and b[(2,None)]['varname']=='wind' and fingerprint!=other

def test_selection_errors(tmp_path):
    p=fixture_file(tmp_path)
    with pytest.raises(ValueError,match='Variables not found'):UMBackend().open_dataset(p,variables=['absent'])
    with pytest.raises(ValueError,match='OR'):UMBackend().open_dataset(p,variables=['x'],stash_codes=[2])

def test_float64_precision_preserved(tmp_path):
    p=fixture_file(tmp_path)
    value=1.0+2**-40
    with p.open('r+b') as f:
        f.seek(390*8);f.write(struct.pack('>d',value))
    with UMBackend().open_dataset(p,variables=['sfc_soil_moisture']) as ds:
        field=ds.sfc_soil_moisture
        assert field.dtype==np.dtype('float64')
        assert field.isel(time0=0,rlat=0,rlon=0).item()==value

def test_thermodynamic_cell_centres(tmp_path):
    p=fixture_file(tmp_path)
    with UMBackend().open_dataset(p,variables=['sfc_soil_moisture'],grid='thermodynamic') as ds:
        np.testing.assert_array_equal(ds.rlat,[-.5,.5])
        np.testing.assert_array_equal(ds.rlon,[-.5,.5,1.5])

def test_pressure_uses_pa(tmp_path):
    p=fixture_file(tmp_path)
    cat=json.loads(_native.catalogue(str(p)))
    record=cat['records'][0];record['i'][25]=8;record['i'][32]=500
    other=json.loads(json.dumps(record));other['i'][32]=850
    cat['records']=[record,other]
    with patch('xarray_um.backend._native.catalogue',return_value=json.dumps(cat)):
        with UMBackend().open_dataset(p,variables=['sfc_soil_moisture'],levels={'PLEVS':[85000]}) as ds:
            np.testing.assert_array_equal(ds.pressure2,[85000])

@pytest.mark.parametrize('grid,nyout,expected',[
    (18,3,[[.5,1.5,2.],[3.5,4.5,5.],[6.5,7.5,8.]]),
    (19,2,[[1.5,2.5,3.5],[4.5,5.5,6.5]]),
    (11,2,[[2.,3.,3.5],[5.,6.,6.5]])])
def test_staggered_interpolation(grid,nyout,expected):
    values=np.arange(9,dtype='<f8').reshape(3,3)
    result=np.frombuffer(_native.interpolate(values.tobytes(),3,3,nyout,grid,1.),dtype='<f8').reshape(nyout,3)
    np.testing.assert_array_equal(result,expected)

def test_interpolation_missing_and_scale():
    values=np.array([[1,np.nan,3],[4,5,6]],dtype='<f8')
    result=np.frombuffer(_native.interpolate(values.tobytes(),3,2,2,18,2.),dtype='<f8').reshape(2,3)
    assert np.isnan(result[0,:2]).all()
    assert result[0,2]==6
    np.testing.assert_array_equal(result[1],[9,11,12])

def test_interpolation_rejects_shape():
    with pytest.raises(ValueError):_native.interpolate(b'bad',3,2,2,18,1.)

def test_aggregation_intervals(tmp_path):
    from xarray_um.backend import interval,valid_time
    p=fixture_file(tmp_path)
    cat=json.loads(_native.catalogue(str(p)))
    i=cat['records'][0]['i'];i[41]=3236;i[24]=128;i[12]=121
    i[:12]=[1990,2,1,23,30,0,1990,2,2,0,0,0]
    with patch('xarray_um.backend._native.catalogue',return_value=json.dumps(cat)):
        with UMBackend().open_dataset(p,variables=['mean_sfc_temp']) as ds:
            np.testing.assert_array_equal(ds.time0,[np.datetime64('1990-02-02T00:00:00','ns')])
            b=ds[ds.time0.attrs['bounds']].to_numpy()
            np.testing.assert_array_equal(b,[[np.datetime64('1990-02-01T23:30','ns'),np.datetime64('1990-02-02T00:00','ns')]])
            assert ds.mean_sfc_temp.attrs['cell_methods']=='time: mean'
    i[6:12]=i[:6]
    with pytest.raises(ValueError,match='Non-positive'):interval(cat['records'][0])


def test_installed_engine_and_default_mapping(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert 'um' in xr.backends.list_engines()
    assert mapping()[0]
    path=fixture_file(tmp_path)
    with xr.open_dataset(path,engine='um') as ds:
        assert 'sfc_soil_moisture' in ds


@pytest.mark.parametrize('same_times',[True,False])
def test_readable_time_axes(tmp_path,same_times):
    path=fixture_file(tmp_path)
    cat=json.loads(_native.catalogue(str(path)))
    a=cat['records'][0];b=json.loads(json.dumps(a));b['i'][41]=8209
    if not same_times:b['i'][3]=6
    cat['records']=[a,b]
    with patch('xarray_um.backend._native.catalogue',return_value=json.dumps(cat)):
        ds=UMBackend().open_dataset(path)
    assert ds.sfc_soil_moisture.dims[0]=='time0'
    other=next(v for n,v in ds.data_vars.items() if n!='sfc_soil_moisture' and v.ndim>0)
    assert other.dims[0]==('time0' if same_times else 'time1')


@pytest.mark.parametrize('same_bounds',[True,False])
def test_reporting_bounds_define_axis_identity(tmp_path,same_bounds):
    path=fixture_file(tmp_path);cat=json.loads(_native.catalogue(str(path)))
    a=cat['records'][0];a['i'][41]=3236;a['i'][24]=128;a['i'][12]=121
    a['i'][:12]=[1990,2,1,23,30,0,1990,2,2,0,0,0]
    b=json.loads(json.dumps(a));b['i'][24]=8192
    if not same_bounds:b['i'][4]=0
    cat['records']=[a,b]
    with patch('xarray_um.backend._native.catalogue',return_value=json.dumps(cat)):
        ds=UMBackend().open_dataset(path)
    mean=ds.mean_sfc_temp.dims[0];maximum=ds.max_sfc_temp.dims[0]
    assert mean=='time0'
    assert maximum==('time0' if same_bounds else 'time1')
    for name in {mean,maximum}:assert ds[name].attrs['bounds'] in ds.coords


def test_model_level_mapping_not_deduplicated_by_height_only(tmp_path):
    path=fixture_file(tmp_path);cat=json.loads(_native.catalogue(str(path)))
    template=cat['records'][0];records=[]
    for code,levels in [(2,[1,2]),(3,[2,3])]:
        for level in levels:
            r=json.loads(json.dumps(template));r['i'][41]=code;r['i'][25]=65;r['i'][32]=level
            records.append(r)
    cat['records']=records;cat['level_shape']=[7,4];cat['level_constants']=[100.]*28
    def definition(defs,code,proc):return {'varname':f'field{code}','level_type':'1','umgrid':'1','units':'m s-1'}
    with patch('xarray_um.backend._native.catalogue',return_value=json.dumps(cat)),patch('xarray_um.backend.metadata',side_effect=definition):
        ds=UMBackend().open_dataset(path)
    za=ds.field2.dims[1];zb=ds.field3.dims[1]
    assert za=='hybridr2' and zb=='hybridr2_1'
    np.testing.assert_array_equal(ds[za+'_model_level'],[.5,1.5])
    np.testing.assert_array_equal(ds[zb+'_model_level'],[1.5,2.5])


def test_no_duplicate_aggregation_prefix():
    assert metadata({(4201,None):{'varname':'sum_rain_amount','accum_field':'1'}},4201,128)['varname']=='sum_rain_amount'


def test_reader_close_and_plane_ownership(tmp_path):
    path=fixture_file(tmp_path)
    with UMBackend().open_dataset(path,variables=['sfc_soil_moisture']) as ds:
        a=ds.sfc_soil_moisture.isel(time0=0).to_numpy()
        assert a.flags.c_contiguous
    np.testing.assert_array_equal(a,[[1,2,3],[4,5,6]])
    reader=_native.Reader(str(path));reader.close()
    with pytest.raises(ValueError,match='closed'):
        _native.plane(reader,390*8,48,3,2,0,8,True,-1e30,1,0,2,1.,True)


def test_single_export_thread_equivalence(tmp_path):
    from xarray_um.cli import write_single
    path=fixture_file(tmp_path)
    a,b=tmp_path/'one.nc',tmp_path/'two.nc'
    write_single(path,a,workers=1);write_single(path,b,workers=2)
    with xr.open_dataset(a) as left,xr.open_dataset(b) as right:
        xr.testing.assert_identical(left,right)
        np.testing.assert_array_equal(left.sfc_soil_moisture,[[[1,2,3],[4,5,6]]])
        assert left.sfc_soil_moisture.encoding['chunksizes']==(1,2,3)
    with pytest.raises(FileExistsError):write_single(path,a)
    with pytest.raises(ValueError):write_single(path,path)


def test_stash_export_source_stem_and_manifest(tmp_path):
    from xarray_um.cli import write_stash
    path=fixture_file(tmp_path)
    destination=tmp_path/'groups'
    write_stash(path,destination,workers=2)
    report=json.loads((destination/'manifest.json').read_text())
    assert report['complete']
    assert report['files'][0]['file']=='test_8208.nc'
    with xr.open_dataset(destination/'test_8208.nc') as ds:
        np.testing.assert_array_equal(ds.sfc_soil_moisture,[[[1,2,3],[4,5,6]]])
    with pytest.raises(FileExistsError):write_stash(path,destination)


def test_export_failure_not_published(tmp_path):
    from xarray_um.cli import write_single,write_stash
    path=fixture_file(tmp_path)
    with patch('xarray_um.cli._read_plane',side_effect=ValueError('broken plane')):
        with pytest.raises(ValueError,match='broken'):
            write_single(path,tmp_path/'bad.nc',workers=2)
        assert not (tmp_path/'bad.nc').exists()
        assert not list(tmp_path.glob('*.writing'))
        with pytest.raises(ValueError,match='broken'):
            write_stash(path,tmp_path/'failed',workers=1)
    assert not (tmp_path/'failed'/'manifest.json').exists()
    assert (tmp_path/'failed'/'failure.json').exists()


@pytest.mark.parametrize('width',[1,3,7,12,24,31])
def test_wgdos_crossing_bits(tmp_path,width):
    integers=[0,1,(1<<width)-1,1,0,1,(1<<width)-1]
    packed=''.join(f'{v:0{width}b}' for v in integers)
    packed += '0' * (-len(packed)%32)
    row=int(packed,2).to_bytes(len(packed)//8,'big')
    header=struct.pack('>IiHH',5+len(row)//4,0,len(integers),1)
    content=header+struct.pack('>IHH',0x41100000,width,len(row)//4)+row
    path=tmp_path/'packed.bin';path.write_bytes(content)
    result=np.frombuffer(_native.decode(str(path),0,len(content),7,1,1,8,True,-1e30,1),dtype='float64')
    expected=np.array(integers,dtype='float32')+np.float32(1.)
    np.testing.assert_array_equal(result,expected.astype('float64'))



def test_float32_export_missing_collision(tmp_path):
    from xarray_um.cli import write_single
    path=fixture_file(tmp_path)
    indicator=-1073741824.
    with path.open('r+b') as f:
        f.seek((319+45+17)*8);f.write(struct.pack('>d',indicator))
        f.seek(390*8);f.write(struct.pack('>d',indicator-2**-20))
    with xr.open_dataset(path,engine='um') as ds:
        assert ds.sfc_soil_moisture[0,0,0].item()==indicator-2**-20
    output=tmp_path/'single.nc';write_single(path,output,workers=2)
    with xr.open_dataset(output) as ds:
        assert np.isnan(ds.sfc_soil_moisture[0,0,0].item())
        assert ds.sfc_soil_moisture.encoding['_FillValue']==np.float32(indicator)



def test_wgdos_missing_bitmap(tmp_path):
    mask=bytes([128,0,0,0])
    packed=''.join(f'{v:03b}' for v in range(1,7))
    packed+='0'*(-len(packed)%32)
    row=mask+int(packed,2).to_bytes(len(packed)//8,'big')
    content=struct.pack('>IiHHIHH',5+len(row)//4,0,7,1,0x41100000,35,len(row)//4)+row
    p=tmp_path/'mask.bin';p.write_bytes(content)
    decoded=np.frombuffer(_native.decode(str(p),0,len(content),7,1,1,8,True,-1e30,1),dtype='float64')
    np.testing.assert_allclose(decoded,[np.nan,2,3,4,5,6,7],equal_nan=True)



def test_lazy_array_serialisation_reopens_reader(tmp_path):
    import pickle
    path=fixture_file(tmp_path)
    with xr.open_dataset(path,engine='um') as ds:
        restored=pickle.loads(pickle.dumps(ds.sfc_soil_moisture.variable._data))
        np.testing.assert_array_equal(np.asarray(restored),[[[1,2,3],[4,5,6]]])



def test_truncated_wgdos_bitstream(tmp_path):
    content=struct.pack('>IiHHIHH',5,0,7,1,0x41100000,3,0)
    path=tmp_path/'bits.bin';path.write_bytes(content)
    with pytest.raises(ValueError,match='truncated WGDOS bits'):
        _native.decode(str(path),0,len(content),7,1,1,8,True,-1e30,1)


def test_positioned_reader_detects_truncation(tmp_path):
    path=fixture_file(tmp_path);reader=_native.Reader(str(path))
    path.write_bytes(b'short')
    with pytest.raises(ValueError):
        _native.plane(reader,390*8,48,3,2,0,8,True,-1e30,1,0,2,1.,True)
    reader.close()



def test_ambiguous_variable_names_are_rejected(tmp_path):
    path=fixture_file(tmp_path);cat=json.loads(_native.catalogue(str(path)))
    other=json.loads(json.dumps(cat['records'][0]));other['i'][41]=8209
    cat['records'].append(other)
    with patch('xarray_um.backend._native.catalogue',return_value=json.dumps(cat)),patch('xarray_um.backend.metadata',return_value={'varname':'same','umgrid':'1'}):
        with pytest.raises(ValueError,match='Variable name collision'):
            UMBackend().open_dataset(path)
