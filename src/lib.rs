//! UM headers and WGDOS decoding. Derived from Mark Cheeseman's um2netcdf,
//! NIWA (2014), GPL-3.0-or-later. All reads are independently positioned.
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::json;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::{Arc, Mutex};
fn read(path: &str, offset: u64, n: usize) -> Result<Vec<u8>, String> {
    let f = File::open(path).map_err(|e| e.to_string())?;
    if offset.checked_add(n as u64).ok_or("offset overflow")?
        > f.metadata().map_err(|e| e.to_string())?.len()
    {
        return Err("truncated UM record".into());
    }
    let mut b = vec![0; n];
    f.read_exact_at(&mut b, offset).map_err(|e| e.to_string())?;
    Ok(b)
}
fn integer(b: &[u8], w: usize, be: bool) -> i64 {
    if w == 8 {
        let a = b[..8].try_into().unwrap();
        if be {
            i64::from_be_bytes(a)
        } else {
            i64::from_le_bytes(a)
        }
    } else {
        let a = b[..4].try_into().unwrap();
        (if be {
            i32::from_be_bytes(a)
        } else {
            i32::from_le_bytes(a)
        }) as i64
    }
}
fn real(b: &[u8], w: usize, be: bool) -> f64 {
    if w == 8 {
        let a = b[..8].try_into().unwrap();
        if be {
            f64::from_be_bytes(a)
        } else {
            f64::from_le_bytes(a)
        }
    } else {
        let a = b[..4].try_into().unwrap();
        (if be {
            f32::from_be_bytes(a)
        } else {
            f32::from_le_bytes(a)
        }) as f64
    }
}
fn header(path: &str) -> Result<(usize, bool, Vec<i64>), String> {
    let size = File::open(path)
        .map_err(|e| e.to_string())?
        .metadata()
        .map_err(|e| e.to_string())?
        .len();
    let b = read(path, 0, std::cmp::min(size, 2048) as usize)?;
    for w in [8, 4] {
        if b.len() < 256 * w {
            continue;
        }
        for be in [true, false] {
            let h: Vec<_> = b[..256 * w].chunks(w).map(|x| integer(x, w, be)).collect();
            if h[1] == 1 && h[150] == 64 && h[149] > 0 && h[151] > 0 {
                return Ok((w, be, h));
            }
        }
    }
    Err("unsupported UM header/word size".into())
}
#[pyfunction]
fn catalogue(py: Python<'_>, path: String) -> PyResult<String> {
    py.allow_threads(||->Result<String,String>{let(w,be,h)=header(&path)?;
 let constants=|start:i64,n:usize,is_real:bool|->Result<serde_json::Value,String>{if start<1||n>1_000_000{return Err("invalid constant table".into())}let b=read(&path,(start as u64-1)*w as u64,n*w)?;Ok(if is_real{json!(b.chunks(w).map(|v|real(v,w,be)).collect::<Vec<_>>())}else{json!(b.chunks(w).map(|v|integer(v,w,be)).collect::<Vec<_>>())})};
 if h[151]>1_000_000{return Err("too many lookup records".into())}
 let b=read(&path,(h[149] as u64-1)*w as u64,h[151] as usize*64*w)?;let mut records=vec![];
 for (id,row) in b.chunks(64*w).enumerate(){let ints:Vec<_>=row[..45*w].chunks(w).map(|v|integer(v,w,be)).collect();if ints[28]<0||ints[17]<=0||ints[18]<=0{continue}let floats:Vec<_>=row[45*w..].chunks(w).map(|v|real(v,w,be)).collect();records.push(json!({"id":id,"i":ints,"r":floats}));}
 Ok(json!({"word_size":w,"big_endian":be,"header":h,"real_constants":constants(h[104],6,true)?,"int_constants":constants(h[99],46,false)?,"level_constants":constants(h[109],(h[110]*h[111]) as usize,true)?,"level_shape":[h[111],h[110]],"records":records}).to_string())}).map_err(PyValueError::new_err)
}

/// A read-only positioned handle shared by independent decoder threads.
#[pyclass]
struct Reader {
    file: Mutex<Option<Arc<File>>>,
    size: u64,
}
#[pymethods]
impl Reader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let file = File::open(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let size = file
            .metadata()
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .len();
        Ok(Self {
            file: Mutex::new(Some(Arc::new(file))),
            size,
        })
    }
    fn close(&self) {
        *self.file.lock().unwrap() = None;
    }
}
impl Reader {
    fn handle(&self) -> PyResult<Arc<File>> {
        self.file
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| PyValueError::new_err("UM reader is closed"))
    }
}

/// Extract at most 32 bits with at most five byte operations, including crossings.
fn bits(b: &[u8], p: usize, n: usize) -> Result<u32, String> {
    if n > 32 || p.checked_add(n).ok_or("bit offset overflow")? > b.len() * 8 {
        return Err("truncated WGDOS bits".into());
    }
    if n == 0 {
        return Ok(0);
    }
    let mut value = 0u64;
    let first = p / 8;
    let end = (p + n).div_ceil(8);
    for byte in &b[first..end] {
        value = (value << 8) | u64::from(*byte)
    }
    let shift = end * 8 - p - n;
    Ok(((value >> shift) & ((1u64 << n) - 1)) as u32)
}
fn ibm(v: u32) -> f32 {
    if v & 0xffffff == 0 {
        return 0.;
    }
    let sign = if v >> 31 == 0 { 1. } else { -1. };
    (sign * (v & 0xffffff) as f64 / 16777216. * 16f64.powi(((v >> 24) & 127) as i32 - 64)) as f32
}
fn wgdos(b: &[u8], nx: usize, ny: usize, mdi: f64) -> Result<Vec<f32>, String> {
    if b.len() < 12 {
        return Err("truncated WGDOS header".into());
    }
    let u32at = |n| u32::from_be_bytes(b[n..n + 4].try_into().unwrap());
    let scale = 2f32.powi(u32at(4) as i32);
    if !scale.is_finite() {
        return Err("invalid WGDOS precision".into());
    }
    let cols = u16::from_be_bytes(b[8..10].try_into().unwrap()) as usize;
    let rows = u16::from_be_bytes(b[10..12].try_into().unwrap()) as usize;
    if (cols, rows) != (nx, ny) {
        return Err("WGDOS shape mismatch".into());
    }
    let mut out = vec![0f32; nx * ny];
    let mut marked = vec![false; nx];
    let mut at = 12;
    for j in 0..ny {
        if at + 8 > b.len() {
            return Err("truncated WGDOS row header".into());
        }
        let base = ibm(u32at(at));
        let flags = u16::from_be_bytes(b[at + 4..at + 6].try_into().unwrap());
        if flags & !255 != 0 {
            return Err("unsupported WGDOS flags".into());
        }
        let n = u16::from_be_bytes(b[at + 6..at + 8].try_into().unwrap()) as usize;
        at += 8;
        if at + n * 4 > b.len() {
            return Err("truncated WGDOS row".into());
        }
        let row = &b[at..at + n * 4];
        let values = &mut out[j * nx..(j + 1) * nx];
        values.fill(base);
        marked.fill(false);
        let mut pos = 0;
        for (flag, value, reverse) in [(32, mdi as f32, false), (64, base, false), (128, 0., true)]
        {
            if flags & flag != 0 {
                for x in 0..nx {
                    let set = bits(row, pos + x, 1)? == 1;
                    if set != reverse {
                        values[x] = value;
                        marked[x] = true
                    }
                }
                pos += nx;
            }
        }
        if flags & 224 != 0 {
            pos = pos.div_ceil(32) * 32
        }
        let nb = (flags & 31) as usize;
        if nb > 0 {
            for x in 0..nx {
                if !marked[x] {
                    values[x] = base + scale * bits(row, pos, nb)? as f32;
                    pos += nb
                }
            }
        }
        for v in values {
            if *v == mdi as f32 {
                *v = f32::NAN
            }
        }
        at += n * 4;
    }
    Ok(out)
}
enum Values {
    Single(Vec<f32>),
    Double(Vec<f64>),
}
impl Values {
    fn at(&self, index: usize) -> f64 {
        match self {
            Self::Single(v) => v[index] as f64,
            Self::Double(v) => v[index],
        }
    }
    fn numpy(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Single(v) => v.into_pyarray(py).into_any().unbind(),
            Self::Double(v) => v.into_pyarray(py).into_any().unbind(),
        }
    }
}
fn unpack(
    b: &[u8],
    nx: usize,
    ny: usize,
    packing: i64,
    w: usize,
    be: bool,
    mdi: f64,
    datatype: i64,
) -> Result<Values, String> {
    match packing {
        1 => Ok(Values::Single(wgdos(b, nx, ny, mdi)?)),
        0 => {
            if b.len() < nx * ny * w {
                return Err("truncated unpacked field".into());
            }
            let values: Vec<f64> = b[..nx * ny * w]
                .chunks(w)
                .map(|v| {
                    let x = if datatype == 1 {
                        real(v, w, be)
                    } else {
                        integer(v, w, be) as f64
                    };
                    if x == mdi {
                        f64::NAN
                    } else {
                        x
                    }
                })
                .collect();
            Ok(if w == 8 {
                Values::Double(values)
            } else {
                Values::Single(values.into_iter().map(|x| x as f32).collect())
            })
        }
        _ => Err(format!("unsupported packing {packing}")),
    }
}
fn shape(nx: usize, ny: usize, w: usize) -> Result<(), String> {
    if ![4, 8].contains(&w)
        || nx == 0
        || ny == 0
        || nx.checked_mul(ny).ok_or("shape overflow")? > 100_000_000
    {
        return Err("invalid field shape".into());
    }
    Ok(())
}
fn interpolate_values(
    data: &Values,
    nx: usize,
    ny: usize,
    target_ny: usize,
    grid: i64,
    scale: f64,
    double: bool,
) -> Result<Values, String> {
    let shift_x = grid == 11 || grid == 18;
    let shift_y = grid == 11 || grid == 19;
    if nx < 2
        || ny < 2
        || target_ny == 0
        || ![11, 18, 19].contains(&grid)
        || (!shift_y && target_ny != ny)
        || (shift_y && target_ny != ny && target_ny + 1 != ny)
    {
        return Err("invalid interpolation shape/grid".into());
    }
    let at = |j: usize, i: usize| data.at(j * nx + i);
    let value = |j: usize, i: usize| {
        let xp = (i + 1).min(nx - 1);
        let yp = (j + 1).min(ny - 1);
        let v = match (shift_x, shift_y) {
            (true, true) => (at(j, i) + at(j, xp) + at(yp, i) + at(yp, xp)) * 0.25,
            (true, false) => (at(j, i) + at(j, xp)) * 0.5,
            (false, true) => (at(j, i) + at(yp, i)) * 0.5,
            _ => unreachable!(),
        };
        v * scale
    };
    if double {
        Ok(Values::Double(
            (0..target_ny)
                .flat_map(|j| (0..nx).map(move |i| value(j, i)))
                .collect(),
        ))
    } else {
        Ok(Values::Single(
            (0..target_ny)
                .flat_map(|j| (0..nx).map(move |i| value(j, i) as f32))
                .collect(),
        ))
    }
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn plane(
    py: Python<'_>,
    reader: &Reader,
    offset: u64,
    length: usize,
    nx: usize,
    ny: usize,
    packing: i64,
    word_size: usize,
    big_endian: bool,
    mdi: f64,
    datatype: i64,
    grid_type: i64,
    target_ny: usize,
    scale: f64,
    double: bool,
) -> PyResult<Py<PyAny>> {
    let file = reader.handle()?;
    let size = reader.size;
    let values = py
        .allow_threads(|| -> Result<Values, String> {
            shape(nx, ny, word_size)?;
            if offset.checked_add(length as u64).ok_or("offset overflow")? > size {
                return Err("truncated UM record".into());
            }
            let mut b = vec![0; length];
            file.read_exact_at(&mut b, offset)
                .map_err(|e| e.to_string())?;
            let values = unpack(&b, nx, ny, packing, word_size, big_endian, mdi, datatype)?;
            if [11, 18, 19].contains(&grid_type) {
                return interpolate_values(&values, nx, ny, target_ny, grid_type, scale, double);
            }
            if target_ny != ny {
                return Err("unstaggered source and target extents differ".into());
            }
            Ok(match (double, values) {
                (true, Values::Double(mut v)) => {
                    if scale != 1. {
                        for x in &mut v {
                            *x *= scale
                        }
                    }
                    Values::Double(v)
                }
                (true, Values::Single(v)) => {
                    Values::Double(v.into_iter().map(|x| x as f64 * scale).collect())
                }
                (false, Values::Double(v)) => {
                    Values::Single(v.into_iter().map(|x| x as f32 * scale as f32).collect())
                }
                (false, Values::Single(mut v)) => {
                    if scale != 1. {
                        for x in &mut v {
                            *x *= scale as f32
                        }
                    }
                    Values::Single(v)
                }
            })
        })
        .map_err(PyValueError::new_err)?;
    Ok(values.numpy(py))
}
/// Compatibility helper for small direct decoder checks; the backend uses plane().
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decode(
    py: Python<'_>,
    path: String,
    offset: u64,
    length: usize,
    nx: usize,
    ny: usize,
    packing: i64,
    word_size: usize,
    big_endian: bool,
    mdi: f64,
    datatype: i64,
) -> PyResult<Py<PyAny>> {
    let values = py
        .allow_threads(|| -> Result<Values, String> {
            shape(nx, ny, word_size)?;
            unpack(
                &read(&path, offset, length)?,
                nx,
                ny,
                packing,
                word_size,
                big_endian,
                mdi,
                datatype,
            )
        })
        .map_err(PyValueError::new_err)?;
    let v = match values {
        Values::Double(v) => v,
        Values::Single(v) => v.into_iter().map(f64::from).collect(),
    };
    Ok(v.into_pyarray(py).into_any().unbind())
}
#[pyfunction]
fn interpolate(
    py: Python<'_>,
    data: Vec<u8>,
    nx: usize,
    ny: usize,
    target_ny: usize,
    grid_type: i64,
    scale: f64,
) -> PyResult<Py<PyAny>> {
    if nx.checked_mul(ny).and_then(|n| n.checked_mul(8)) != Some(data.len()) {
        return Err(PyValueError::new_err("invalid interpolation shape"));
    }
    let values = py
        .allow_threads(|| {
            let v = Values::Double(
                data.chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            );
            interpolate_values(&v, nx, ny, target_ny, grid_type, scale, true)
        })
        .map_err(PyValueError::new_err)?;
    Ok(values.numpy(py))
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Reader>()?;
    m.add_function(wrap_pyfunction!(catalogue, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate, m)?)?;
    m.add_function(wrap_pyfunction!(plane, m)?)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bit_bounds() {
        assert_eq!(bits(&[0b10110000], 0, 4).unwrap(), 11);
        assert!(bits(&[0], 7, 2).is_err());
    }
    #[test]
    fn crossings() {
        let b = [0xac, 0xf0, 0x12, 0x34, 0x56];
        for p in 0..8 {
            for n in 0..=32 {
                let mut v = 0;
                for k in p..p + n {
                    v = (v << 1) | ((b[k / 8] >> (7 - k % 8)) & 1) as u32
                }
                assert_eq!(bits(&b, p, n).unwrap(), v)
            }
        }
    }
    #[test]
    fn ibm_numbers() {
        assert_eq!(ibm(0x41100000), 1.);
        assert_eq!(ibm(0xc1200000), -2.);
    }
    #[test]
    fn constant_row() {
        let mut b = vec![
            0, 0, 0, 5, 0, 0, 0, 0, 0, 2, 0, 1, 0x41, 0x10, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(wgdos(&b, 2, 1, -1.).unwrap(), vec![1., 1.]);
        b.pop();
        assert!(wgdos(&b, 2, 1, -1.).is_err());
    }
}
