# Working on xarray-um

Read [README.md](README.md) for installation, supported formats and public usage.
This is a general-purpose package repository; do not assume access to any
particular institution, filesystem, scheduler or archive.

## Public contract

- Distribution and repository: `xarray-um`.
- Python import: `xarray_um`.
- Xarray backend: `um`, discovered through the installed `xarray.backends`
  entry point without a prior `import xarray_um`.
- Native extension: `xarray_um._native`, built with PyO3 and Maturin.
- CLI: `xarray-um`; `bin/um2netcdf` is a converter-compatible wrapper.
- Initial supported wheels: Linux x86-64, CPython 3.12 and 3.13. The Rust reader
  currently uses Unix positioned-file APIs; do not claim Windows support.
- Preserve GPL-3.0-or-later licensing and the attribution in LICENSE, NOTICE,
  Rust source and the bundled STASH XML.

## Architecture

| Location | Responsibility |
| --- | --- |
| `src/lib.rs` | Binary headers, lookup catalogue, WGDOS/unpacked decoding and interpolation |
| `python/xarray_um/backend.py` | Scientific metadata, coordinate identity, selection and lazy xarray adapter |
| `python/xarray_um/cli.py` | Explicit NetCDF export and converter-shaped arguments |
| `python/xarray_um/data/` | Bundled NZRA STASH-to-CF definitions |
| `tests/test_reader.py` | Small synthetic reader and coordinate checks |
| `pyproject.toml`, `Cargo.toml` | Python/Rust distribution metadata and native build |
| `pixi.toml`, `pixi.lock` | Development environment and tasks |
| `.github/workflows/wheels.yml` | Portable wheel builds, clean installation checks and tagged releases |

Some local milestone/benchmark helpers are development tools, not prerequisites
for using the installed backend. Keep machine-specific runners and datasets out
of the public package and CI requirements.

## Build and verify

Use Pixi for development. Preserve its configured cache location and lockfile.
Do not introduce another environment manager as part of a routine change.

```bash
pixi install --locked
pixi run test
CARGO_BUILD_JOBS=2 pixi run build
```

Python source is installed editable. After Rust changes, rebuild the installed
extension rather than accidentally testing a stale binary:

```bash
CARGO_BUILD_JOBS=2 pixi reinstall xarray-um
pixi run test
```

Check the actual wheel and source distribution when changing packaging:

```bash
CARGO_BUILD_JOBS=2 pixi run maturin build --release --locked --sdist -o dist
pixi run python scripts/check_artifacts.py dist
```

`scripts/check_wheel.py` installs a matching built wheel into a new Pixi
environment and runs checks outside the source checkout. It takes the artifact
directory, a new scratch environment directory and the Python version:

```bash
pixi run python scripts/check_wheel.py dist /path/to/scratch/clean-env 3.13
```

Ask for scratch locations when not already agreed. The clean environment needs
package-network access and must not overwrite an existing directory. Use the
existing small synthetic checks for ordinary verification. Real archive reads,
large exports and benchmarks require appropriate compute resources.

## Scientific and performance invariants

- Opening a file and inspecting metadata must remain lazy with respect to field
  decoding. Preserve xarray indexing and horizontal-plane chunk preferences.
- Rust releases the GIL around decoding and performs independently positioned
  reads. Parallelism comes from scheduling independent planes with Dask; the
  converter-compatible CLI remains serial.
- Preserve bounds checks on file offsets, lengths, dimensions and packed bit
  access. Unsupported formats and truncated data must fail clearly.
- Keep native grids by default. Thermodynamic-grid interpolation is an explicit
  scientific choice; preserve wind staggering, masks, units and precision.
- Decode the original archive representation without introducing additional
  quantisation. The export CLI intentionally uses float32 and compressed
  NetCDF for compatibility; do not describe that CLI output as uncompressed.
- Selection domains have explicit units: pressure in Pa, geometric heights and
  soil depths in metres, model levels in native level numbers. Missing requested
  values fail instead of silently selecting the nearest level.
- Time-axis identity includes timestamps, relevant units/calendar and reporting
  bounds. Distinct intervals with equal end times must not share incompatible
  bounds. Cosmetic labels alone must not create another axis.
- Use deterministic readable time names (`time0`, `time1`, etc.). These names are
  relative to the selected field set; they are not fixed identifiers for a
  specific scientific time axis across every open operation.
- Vertical reference heights and native `*_model_level` values are different
  coordinates on the same samples. Retain both. Share an axis only when both
  mappings agree; do not collapse theta/rho grids or different level subsets.
- Reference heights are not a terrain-dependent three-dimensional altitude
  field. Do not silently reinterpret them as one.
- Read bundled XML through package resources. Explicit metadata overrides and
  relative TOML paths must work outside the checkout, with provenance retained.
- Scope compatibility claims to supported NZRA samples and synthetic checks.
  Sample agreement with another converter does not validate every UM file.

## General HPC guidance

- Treat login nodes as shared: editing, metadata inspection and tiny synthetic
  checks are appropriate. Move large decoding, exports and benchmarks to an
  allocation provided by the site's scheduler or computing service.
- Ask which project/account to charge and where to store scratch data. Confirm
  archive and output visibility from the execution environment. Never infer
  permissions or allocation limits from a path or host-wide CPU/RAM count.
- Bound Dask workers, compilation jobs and numerical-library threads to the
  allocation. Avoid multiplying Dask concurrency by uncontrolled BLAS/OpenMP
  threads. More file readers can saturate storage before CPUs are exhausted.
- Inputs are read-only. Use explicit new destinations, preserve source files
  and close datasets. Do not overwrite valuable outputs or remove archive data.
- Prefer reproducible batch scripts with an explicit Pixi manifest, resources,
  command, inputs, outputs and logs. Job submission and costly runs require user
  authorization; preparing code does not itself authorize launching work.
- Measure CPU time, elapsed time, memory, bytes read/written and cache effects
  separately. Distinguish decoding speed from compression and filesystem speed.

## Releases and repository hygiene

Check current Git state and preserve unrelated edits. Do not commit environments,
compiled extensions, `target/`, `dist/`, data files or generated notebook outputs.
Keep credentials and site-specific paths out of public documentation and CI.

The wheel workflow builds Linux wheels for each supported Python version and
checks archive contents and clean Pixi installation. A version tag such as
`v0.1.0` publishes verified artifacts to GitHub Releases. Keep version numbers in
Python, Cargo and Pixi metadata consistent, and never move a published tag to
replace a release silently. Publishing tags or releases requires authorization.

Use the actual wheel asset URLs in installation instructions; platform tags can
contain multiple compatibility names in a specific order. GitHub wheel releases
are not a PyPI publication. Verify runtime XML and licensing assets in wheels
and source distributions, and retain installed-engine discovery checks.
