# Changes

## 0.2.0

- Decode WGDOS packed values with bounded byte/word extraction and hand owned
  NumPy arrays directly from Rust to Python.
- Reuse positioned-read handles and row buffers; preserve lazy selection,
  source float64 precision and process-local reader reconstruction.
- Add `--workers` for bounded threaded decoding into one NetCDF writer.
- Add `--layout stash` for independent process-based exports named
  `<input_stem>_<stash_code>.nc`, retaining aggregation variants together.
- Publish files atomically, refuse existing destinations, and write a completion
  manifest only after all STASH groups succeed.
- Avoid duplicate aggregation prefixes and preserve the C float32 export's
  missing-indicator rounding convention.
- Add reproducible full-field compatibility checks, interactive notebooks,
  equal-budget scaling benchmarks and public visual evidence.
