# GOES ABI and MRMS attribution

## Included derived examples

- `mini_case.npz`: physical GOES ABI C08–C14 brightness temperatures (K), MRMS
  reflectivity (dBZ), masks and derived target arrays for the small execution
  example. Exact period, timestamp, crop, band order and SHA-256 are recorded in
  `mini_case_metadata.json`.
- `fig11_case/`: archived case probabilities, original CI35 core indices,
  projected/native MRMS observations and observed-lineage profiles. Original
  paths and hashes are in its `manifest.json`. The `code/` directory retains
  source evidence; notebook runtime uses the portable functions in `tools/`.

GOES-R ABI observations are produced through the NOAA/NASA GOES-R programme.
MRMS products are produced by NOAA's Multi-Radar/Multi-Sensor system. The
recovered acquisition code uses the NOAA Open Data cloud archives, including
`noaa-mrms-pds`; the miniature metadata identifies the radar product as
`MergedReflectivityQCComposite_00.50`.

Brightness-temperature conversion, parallax correction/fill, projection,
cropping, masks, CI detections and model outputs are research processing. They
must not be represented as official NOAA forecasts or as NOAA endorsement of
the model. Source-data availability does not license third-party software or
grant permission to use NOAA marks.

## Source descriptions and policies

- [NOAA NCEI GOES-R ABI/GLM description](https://www.ncei.noaa.gov/products/goes-terrestrial-weather-abi-glm)
- [NOAA NSSL MRMS project](https://www.nssl.noaa.gov/projects/mrms/)
- [NOAA NWS disclaimer and public-domain information](https://www.weather.gov/disclaimer)
- [NOAA NSSL disclaimer](https://www.nssl.noaa.gov/disclaimer.php)

This note records attribution and source-policy links. It does not assert
institutional approval or relicense all components as Creative Commons. The
miniature sample's original redistribution-review flag is preserved in metadata;
the previous notice is retained in
`provenance/original_sample_redistribution_notice.md`. Clearing that flag requires
an actual rights review of the included derived assets. The research authors
should record the outcome before declaring the data release unrestricted.
