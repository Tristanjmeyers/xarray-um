"""Check release contents without installing or decoding scientific datasets."""
import os
from pathlib import Path
import sys
import tarfile
import zipfile

root = Path(sys.argv[1])
for wheel in root.glob('xarray_um-*.whl'):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert 'xarray_um/data/stash2cf_nzra.xml' in names
        assert any(name.endswith('/LICENSE') or name=='LICENSE' for name in names)
        assert any(name.startswith('xarray_um/_native') and name.endswith('.so') for name in names)
        metadata = archive.read(next(n for n in names if n.endswith('.dist-info/METADATA'))).decode()
        assert 'Name: xarray-um\n' in metadata
        entrypoints = archive.read(next(n for n in names if n.endswith('/entry_points.txt'))).decode()
        assert 'um=xarray_um.backend:UMBackend' in entrypoints.replace(' ', '')
        version = next(line.split(': ',1)[1] for line in metadata.splitlines() if line.startswith('Version: '))
        if os.environ.get('REF_TYPE')=='tag':
            assert os.environ['REF_NAME']=='v'+version
        print('Verified wheel:',wheel.name)
for source in root.glob('*.tar.gz'):
    with tarfile.open(source) as archive:
        names=archive.getnames()
        for suffix in ('/Cargo.toml','/src/lib.rs','/python/xarray_um/data/stash2cf_nzra.xml','/LICENSE'):
            assert any(name.endswith(suffix) for name in names),suffix
        print('Verified source distribution:',source.name)
