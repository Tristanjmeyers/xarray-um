"""Bounded parallel UM export with a single writer per NetCDF file."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools
import json
import logging
import multiprocessing
from pathlib import Path
import uuid

import dask
import netCDF4
import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)


def _planes(dataset):
    for name, field in dataset.data_vars.items():
        for index in itertools.product(*(range(n) for n in field.shape[:-2])):
            yield name, index, field


def _read_plane(task):
    name, index, field = task
    data = field.isel(dict(zip(field.dims[:-2], index))).to_numpy()
    return name, index, data


def _bounded_planes(tasks, workers):
    """Retain at most twice the worker count of submitted horizontal planes."""
    if workers == 1:
        for task in tasks:
            yield _read_plane(task)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for task in itertools.islice(tasks, 2 * workers):
            pending.append(pool.submit(_read_plane, task))
        while pending:
            yield pending.popleft().result()
            task = next(tasks, None)
            if task is not None:
                pending.append(pool.submit(_read_plane, task))


def write_single(source, output, *, stash_xml=None, stash_codes=None,
                 interpolate=False, workers=1, encoding_template=None):
    """Write selected fields, with bounded decoding and one NetCDF writer.

    encoding_template is used by the benchmark to match the C reference's
    data-variable storage encoding. Ordinary exports use float32, zlib level 3,
    shuffle and complete horizontal-plane chunks, without further quantisation.
    """
    source, output = Path(source), Path(output)
    if workers < 1:
        raise ValueError('workers must be positive')
    if source.resolve() == output.resolve():
        raise ValueError('Input and output must differ')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.' + uuid.uuid4().hex + '.writing')
    try:
        with dask.config.set(scheduler='synchronous'):
            with xr.open_dataset(source, engine='um', stash_xml=stash_xml,
                                 stash_codes=stash_codes,
                                 grid='thermodynamic' if interpolate else 'native') as ds:
                LOG.info('Writing %s: %d fields; decoder threads=%d', output, len(ds.data_vars), workers)
                # Coordinates are small shared metadata; all field data remain lazy.
                ds.drop_vars(list(ds.data_vars)).to_netcdf(temporary, engine='netcdf4')
                with netCDF4.Dataset(temporary, 'a') as nc:
                    for name, field in ds.data_vars.items():
                        encoding = (encoding_template or {}).get(name, {})
                        dtype = encoding.get('dtype', 'float32')
                        chunks = encoding.get('chunksizes', (1,) * (field.ndim - 2) + field.shape[-2:])
                        kwargs = dict(zlib=encoding.get('zlib', True),
                                      complevel=encoding.get('complevel', 3),
                                      shuffle=encoding.get('shuffle', True),
                                      fletcher32=encoding.get('fletcher32', False),
                                      fill_value=encoding.get('_FillValue', np.float32(field.encoding.get('source_missing_value', np.nan))))
                        if encoding.get('contiguous'):
                            kwargs['contiguous'] = True
                        else:
                            kwargs['chunksizes'] = chunks
                        variable = nc.createVariable(name, dtype, field.dims, **kwargs)
                        attrs = dict(field.attrs)
                        if field.encoding.get('coordinates'):
                            attrs['coordinates'] = field.encoding['coordinates']
                        attrs.pop('_FillValue', None)
                        variable.setncatts(attrs)
                    count = 0
                    previous = None
                    for name, index, data in _bounded_planes(iter(_planes(ds)), workers):
                        if name != previous:
                            LOG.info('Field %s -> %s', name, output.name)
                            previous = name
                        # Encode NaNs with this field's declared fill. Float32 values
                        # rounding onto that fill retain the converter's mask convention.
                        masked = np.ma.array(data, mask=np.isnan(data), copy=False)
                        nc[name][index + (slice(None), slice(None))] = masked
                        count += 1
                        if count % 100 == 0:
                            LOG.info('Written %d planes: %s index=%s', count, name, index)
        # A concurrent writer must not be silently overwritten at publication.
        import os
        os.link(temporary, output)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info('Published %s', output)
    return output


def _write_group(arguments):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    source, output, xml, code, interpolate = arguments
    write_single(source, output, stash_xml=xml, stash_codes=[code],
                 interpolate=interpolate, workers=1)
    return {'stash_code': code, 'file': Path(output).name,
            'bytes': Path(output).stat().st_size}


def write_stash(source, output, *, stash_xml=None, stash_codes=None,
                interpolate=False, workers=1):
    """Export STASH groups to <source_stem>_<code>.nc and a completion manifest."""
    source, output = Path(source), Path(output)
    if workers < 1:
        raise ValueError('workers must be positive')
    if output.exists():
        raise FileExistsError(output)
    with xr.open_dataset(source, engine='um', stash_xml=stash_xml,
                         stash_codes=stash_codes) as ds:
        codes = sorted({int(v.attrs['stash_section']) * 1000 + int(v.attrs['stash_item'])
                        for v in ds.data_vars.values()})
    output.mkdir(parents=True, exist_ok=False)
    arguments = [(str(source), str(output / f'{source.stem}_{code}.nc'),
                  str(stash_xml) if stash_xml else None, code, interpolate)
                 for code in codes]
    files = []
    try:
        if workers == 1:
            files = [_write_group(a) for a in arguments]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(codes)),
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                for item in pool.map(_write_group, arguments):
                    files.append(item)
        manifest = {'complete': True, 'input': str(source.resolve()),
                    'layout': 'stash', 'files': files}
        temporary = output / 'manifest.json.writing'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(output / 'manifest.json')
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps({'complete': False,
            'error': f'{type(error).__name__}: {error}',
            'completed_files': sorted(p.name for p in output.glob('*.nc'))}, indent=2) + '\n')
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('-c', '--stash-xml', type=Path)
    parser.add_argument('-i', dest='interpolate', action='store_true')
    parser.add_argument('-o', '--output', required=True, type=Path)
    parser.add_argument('-s', dest='stash_codes', nargs='+', type=int)
    parser.add_argument('-r', dest='float32', action='store_true',
                        help='32-bit output (also the compatibility default)')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--layout', choices=['single', 'stash'], default='single')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    export = write_single if args.layout == 'single' else write_stash
    export(args.input, args.output, stash_xml=args.stash_xml,
           stash_codes=args.stash_codes, interpolate=args.interpolate, workers=args.workers)


if __name__ == '__main__':
    main()
