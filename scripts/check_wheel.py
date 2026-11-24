"""Install a built wheel in an isolated Pixi environment, outside the checkout."""
import json
from pathlib import Path
import subprocess
import sys

artifacts=Path(sys.argv[1]).resolve()
check=Path(sys.argv[2]).resolve()
python=sys.argv[3]
wheel=next(artifacts.glob(f'xarray_um-*-cp{python.replace(".","")}-*.whl'))
check.mkdir(parents=True,exist_ok=False)
manifest=check/'pixi.toml'
manifest.write_text(f'''[workspace]
name = "xarray-um-wheel-check"
channels = ["conda-forge"]
platforms = ["linux-64"]
[dependencies]
python = "{python}.*"
numpy = ">=2"
xarray = ">=2026.9"
dask = "*"
netcdf4 = "*"
pyproj = "*"
pytest = "*"
[pypi-dependencies]
xarray-um = {{ path = {json.dumps(str(wheel))} }}
''')
subprocess.run(['pixi','install','--manifest-path',str(manifest)],check=True,cwd=check)
repo=Path(__file__).resolve().parents[1]
subprocess.run(['pixi','run','--manifest-path',str(manifest),'pytest',str(repo/'tests'),'-q'],check=True,cwd=check)
subprocess.run(['pixi','run','--manifest-path',str(manifest),'python','-c',
    'import xarray as xr; from xarray_um.backend import mapping; assert "um" in xr.backends.list_engines(); assert mapping()[0]; print("Installed wheel engine and XML verified")'],check=True,cwd=check)
