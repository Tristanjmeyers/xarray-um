//! UM headers and WGDOS decoding. Derived from Mark Cheeseman's um2netcdf,
//! NIWA (2014), GPL-3.0-or-later. All reads are independently positioned.
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::fs::File;
use std::os::unix::fs::FileExt;
use serde_json::json;
use pyo3::types::PyBytes;
fn read(path:&str, offset:u64, n:usize)->Result<Vec<u8>,String>{
 let f=File::open(path).map_err(|e|e.to_string())?;
 if offset.checked_add(n as u64).ok_or("offset overflow")?>f.metadata().map_err(|e|e.to_string())?.len(){return Err("truncated UM record".into())}
 let mut b=vec![0;n];f.read_exact_at(&mut b,offset).map_err(|e|e.to_string())?;Ok(b)
}
fn integer(b:&[u8],w:usize,be:bool)->i64{if w==8 {let a=b[..8].try_into().unwrap();if be{i64::from_be_bytes(a)}else{i64::from_le_bytes(a)}}else{let a=b[..4].try_into().unwrap();(if be{i32::from_be_bytes(a)}else{i32::from_le_bytes(a)}) as i64}}
fn real(b:&[u8],w:usize,be:bool)->f64{if w==8{let a=b[..8].try_into().unwrap();if be{f64::from_be_bytes(a)}else{f64::from_le_bytes(a)}}else{let a=b[..4].try_into().unwrap();(if be{f32::from_be_bytes(a)}else{f32::from_le_bytes(a)}) as f64}}
fn header(path:&str)->Result<(usize,bool,Vec<i64>),String>{let size=File::open(path).map_err(|e|e.to_string())?.metadata().map_err(|e|e.to_string())?.len();let b=read(path,0,std::cmp::min(size,2048) as usize)?;for w in [8,4]{if b.len()<256*w{continue}for be in [true,false]{let h:Vec<_>=b[..256*w].chunks(w).map(|x|integer(x,w,be)).collect();if h[1]==1&&h[150]==64&&h[149]>0&&h[151]>0{return Ok((w,be,h))}}}Err("unsupported UM header/word size".into())}
#[pyfunction]
fn catalogue(py:Python<'_>,path:String)->PyResult<String>{py.allow_threads(||->Result<String,String>{let(w,be,h)=header(&path)?;
 let constants=|start:i64,n:usize,is_real:bool|->Result<serde_json::Value,String>{if start<1||n>1_000_000{return Err("invalid constant table".into())}let b=read(&path,(start as u64-1)*w as u64,n*w)?;Ok(if is_real{json!(b.chunks(w).map(|v|real(v,w,be)).collect::<Vec<_>>())}else{json!(b.chunks(w).map(|v|integer(v,w,be)).collect::<Vec<_>>())})};
 if h[151]>1_000_000{return Err("too many lookup records".into())}
 let b=read(&path,(h[149] as u64-1)*w as u64,h[151] as usize*64*w)?;let mut records=vec![];
 for (id,row) in b.chunks(64*w).enumerate(){let ints:Vec<_>=row[..45*w].chunks(w).map(|v|integer(v,w,be)).collect();if ints[28]<0||ints[17]<=0||ints[18]<=0{continue}let floats:Vec<_>=row[45*w..].chunks(w).map(|v|real(v,w,be)).collect();records.push(json!({"id":id,"i":ints,"r":floats}));}
 Ok(json!({"word_size":w,"big_endian":be,"header":h,"real_constants":constants(h[104],6,true)?,"int_constants":constants(h[99],46,false)?,"level_constants":constants(h[109],(h[110]*h[111]) as usize,true)?,"level_shape":[h[111],h[110]],"records":records}).to_string())}).map_err(PyValueError::new_err)}
fn bits(b:&[u8],p:usize,n:usize)->Result<u32,String>{if n>32||p.checked_add(n).ok_or("bit offset overflow")?>b.len()*8{return Err("truncated WGDOS bits".into())}let mut v=0;for k in p..p+n{v=(v<<1)|((b[k/8]>>(7-k%8))&1) as u32}Ok(v)}
fn ibm(v:u32)->f32{if v&0xffffff==0{return 0.}let sign=if v>>31==0{1.}else{-1.};(sign*(v&0xffffff) as f64/16777216.*16f64.powi(((v>>24)&127) as i32-64)) as f32}
fn wgdos(b:&[u8],nx:usize,ny:usize,mdi:f64)->Result<Vec<f64>,String>{if b.len()<12{return Err("truncated WGDOS header".into())}let u32at=|n|u32::from_be_bytes(b[n..n+4].try_into().unwrap());let prec=u32at(4) as i32;let scale=2f32.powi(prec);if !scale.is_finite(){return Err("invalid WGDOS precision".into())}let cols=u16::from_be_bytes(b[8..10].try_into().unwrap()) as usize;let rows=u16::from_be_bytes(b[10..12].try_into().unwrap()) as usize;if (cols,rows)!=(nx,ny){return Err("WGDOS shape mismatch".into())}
 let mut out=Vec::with_capacity(nx*ny);let mut at=12;
 for _ in 0..ny{if at+8>b.len(){return Err("truncated WGDOS row header".into())}let base=ibm(u32at(at));let flags=u16::from_be_bytes(b[at+4..at+6].try_into().unwrap());if flags&!255!=0{return Err("unsupported WGDOS flags".into())}let n=u16::from_be_bytes(b[at+6..at+8].try_into().unwrap()) as usize;at+=8;if at+n*4>b.len(){return Err("truncated WGDOS row".into())}let row=&b[at..at+n*4];let mut values=vec![base;nx];let mut marked=vec![false;nx];let mut pos=0;
 for(flag,value,reverse) in [(32,mdi as f32,false),(64,base,false),(128,0.,true)]{if flags&flag!=0{for x in 0..nx{let set=bits(row,pos+x,1)?==1;if set!=reverse{values[x]=value;marked[x]=true}}pos+=nx;}}
 if flags&224!=0{pos=pos.div_ceil(32)*32}let nb=(flags&31) as usize;
 if nb>0{for x in 0..nx{if !marked[x]{values[x]=base+scale*bits(row,pos,nb)? as f32;pos+=nb}}}
 out.extend(values.into_iter().map(|v|if v==mdi as f32{f64::NAN}else{v as f64}));at+=n*4;}
 Ok(out)
}
#[pyfunction]
fn decode(py:Python<'_>,path:String,offset:u64,length:usize,nx:usize,ny:usize,packing:i64,word_size:usize,big_endian:bool,mdi:f64,datatype:i64)->PyResult<Py<PyBytes>>{let values=py.allow_threads(||->Result<Vec<f64>,String>{if ![4,8].contains(&word_size)||nx==0||ny==0||nx.checked_mul(ny).ok_or("shape overflow")?>100_000_000{return Err("invalid field shape".into())}let b=read(&path,offset,length)?;match packing{1=>wgdos(&b,nx,ny,mdi),0=>{if b.len()<nx*ny*word_size{return Err("truncated unpacked field".into())}Ok(b[..nx*ny*word_size].chunks(word_size).map(|v|{let x=if datatype==1{real(v,word_size,big_endian)}else{integer(v,word_size,big_endian) as f64};if x==mdi{f64::NAN}else{x}}).collect())},_=>Err(format!("unsupported packing {packing}"))}}).map_err(PyValueError::new_err)?;let bytes:Vec<u8>=py.allow_threads(||values.into_iter().flat_map(|v|v.to_le_bytes()).collect());Ok(PyBytes::new(py,&bytes).unbind())}
#[pymodule]
fn _native(m:&Bound<'_,PyModule>)->PyResult<()>{m.add_function(wrap_pyfunction!(catalogue,m)?)?;m.add_function(wrap_pyfunction!(decode,m)?)?;m.add_function(wrap_pyfunction!(interpolate,m)?)?;Ok(())}
#[cfg(test)]mod tests{use super::*;#[test]fn bit_bounds(){assert_eq!(bits(&[0b10110000],0,4).unwrap(),11);assert!(bits(&[0],7,2).is_err());}#[test]fn ibm_numbers(){assert_eq!(ibm(0x41100000),1.);assert_eq!(ibm(0xc1200000),-2.);}#[test]fn constant_row(){let mut b=vec![0,0,0,5,0,0,0,0,0,2,0,1,0x41,0x10,0,0,0,0,0,0];assert_eq!(wgdos(&b,2,1,-1.).unwrap(),vec![1.,1.]);b.pop();assert!(wgdos(&b,2,1,-1.).is_err());}}

/// Interpolate staggered C/B grids to P points. Limited-area outer columns
/// retain the nearest native boundary value; no periodic wrap is introduced.
#[pyfunction]
fn interpolate(py:Python<'_>, data:Vec<u8>, nx:usize, ny:usize, target_ny:usize, grid_type:i64, scale:f64)->PyResult<Py<PyBytes>> {
 let bytes=py.allow_threads(||->Result<Vec<u8>,String>{
  if nx<2||ny<2||target_ny==0||data.len()!=nx.checked_mul(ny).and_then(|n|n.checked_mul(8)).ok_or("shape overflow")? {return Err("invalid interpolation shape".into())}
  if ![11,18,19].contains(&grid_type){return Err("unsupported staggered grid".into())}
  let shift_x=grid_type==11||grid_type==18;
  let shift_y=grid_type==11||grid_type==19;
  if (!shift_y&&target_ny!=ny)||(shift_y&&target_ny!=ny&&target_ny+1!=ny){return Err("unsupported target grid extent".into())}
  let value=|j:usize,i:usize| f64::from_le_bytes(data[(j*nx+i)*8..(j*nx+i+1)*8].try_into().unwrap());
  let mut bytes=Vec::with_capacity(target_ny*nx*8);
  for j in 0..target_ny {for i in 0..nx{
   let x=i.min(nx-1);let y=j.min(ny-1);let xp=(x+1).min(nx-1);let yp=(y+1).min(ny-1);
   let v=match (shift_x,shift_y){(true,true)=>(value(y,x)+value(y,xp)+value(yp,x)+value(yp,xp))*0.25,
    (true,false)=>(value(y,x)+value(y,xp))*0.5,
    (false,true)=>(value(y,x)+value(yp,x))*0.5,_=>unreachable!()};
   bytes.extend_from_slice(&(v*scale).to_le_bytes());
  }}Ok(bytes)
 }).map_err(PyValueError::new_err)?;
 Ok(PyBytes::new(py,&bytes).unbind())
}
