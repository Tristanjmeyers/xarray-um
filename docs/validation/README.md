# Reproduce compatibility and performance results

The release evidence compares the installed C `um2netcdf` with the xarray-um
export pipeline. These are checks against another converter, rather than an
independent validation of NZRA meteorology. Raw UM and NetCDF datasets are not
included in this repository.

Use the development Pixi environment. Substantial conversion, validation and
benchmarking belong in a PBS allocation, with a project/account and scratch
location appropriate to your system. The tools have no site-specific path or
account defaults.

Prepare `fixtures.json` with entries named `mlev`, `sfc` and `plev`. Each entry
contains `path` (a staged, read-only UM file) and `bytes` (its size). Supply your
own C executable and STASH XML. Use the same XML with both converters.

```bash
pixi run python scripts/compare_converters.py \
    --fixtures /path/to/fixtures.json \
    --xml /path/to/stash.xml --converter /path/to/um2netcdf \
    --workers 8 --output /new/path/comparison

pixi run python scripts/benchmark_scaling.py \
    --fixtures /path/to/fixtures.json --references /new/path/comparison \
    --xml /path/to/stash.xml --converter /path/to/um2netcdf \
    --workers 1 2 4 8 16 --repeats 2 --output-dir /new/path/benchmark

pixi run python scripts/publish_results.py \
    --comparison /new/path/comparison --benchmark /new/path/benchmark \
    --destination /new/path/public-evidence
```

The benchmark requires at least 16 allocated logical CPUs and an explicit
`NCPUS` environment variable. The reference directory must contain successful
C conversions under `mlev/c.nc`, `sfc/c.nc` and `plev/c.nc`. Output directories
must be new. The publication command requires both correctness and benchmark
success, plus an improvement surviving the observed ranges of both repeats.

## What is measured

Full correctness covers every field, time and level in the three supplied
files. Finite values use `rtol=1e-6, atol=0`; NaN and signed-infinity masks are
checked separately. Native numeric coordinates use `rtol=1e-6, atol=1e-5`.

The installed C converter encodes time as float32 elapsed hours. Comparison
canonicalises its timestamps to the whole seconds stored in UM lookup records
only when the residual is bounded by half an encoding ULP. Calendar dates and
reporting bounds must then match exactly. C tile ordinals beginning at zero are
explicitly matched to original UM pseudo-level IDs beginning at one.

Geographic coordinates are compared modulo 360 for longitude and independently
checked against each file's CF rotated-grid projection. In the tested C
reference, saved geographic auxiliaries differ from that projection, reaching
about 0.0071 degrees longitude and 0.00072 degrees latitude. xarray-um retains
its CF-consistent coordinates; the discrepancy is reported rather than copied.
C hybrid eta coefficient arrays are inventoried separately from physical fields.
xarray-um retains reference heights and original native model-level numbers.

Representative scaling selections are frozen before timing: the first eight
common MLEV STASH codes, four PLEV codes and twenty SFC codes in ascending
numeric order. Every available time, level and aggregation variant is retained.
The machine-readable summary records the exact selections.

Single-file latency compares serial C with threaded Rust export at each worker
count. Batch throughput compares the same 32 independent STASH-group tasks with
equal process/CPU limits for both implementations. Precision, interpolation,
chunks, fill values and compression parameters match. Each output is validated
outside its timed interval.

Measurements include process startup, decoding, interpolation and NetCDF output.
They exclude archive hydration, scheduler waiting, cache passes and validation.
Trials share one allocation and use CPU affinity limits; physical cores are
preferred before SMT siblings. A cache pass precedes each repeat, but shared
filesystem, server and output-cache effects remain uncontrolled. Two observed
repeat ranges are not confidence intervals. Linked library builds can differ;
results describe the complete installed software stacks.

Both native-field float32-equivalent GB/hour (before interpolation) and compressed output GB/hour are reported
using decimal GB. Peak aggregate process-tree RSS is sampled every 0.2 seconds;
shared pages may be counted more than once. These are not predictions of archive
hydration or an entire extraction workflow.

The benchmark retains representative endpoint outputs. It deletes only its own
validated, disposable trial NetCDFs and keeps all measurements and validation
reports. Inputs and reference outputs remain untouched.
