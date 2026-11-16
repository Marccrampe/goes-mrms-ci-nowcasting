# Data availability

## Published catalogue

The radar-derived catalogue of **38,313 canonical CI35 events** over the
northeastern United States during selected periods in 2021–2025 is publicly
available on Zenodo:

**Version 1.0.0: https://doi.org/10.5281/zenodo.24179651**

The catalogue is distributed under **Creative Commons Attribution 4.0
International (CC BY 4.0)**.

The release contains:

- `CI35_events.csv`: canonical event IDs, period IDs, initiation times,
  coordinates, original split assignments and event diagnostics.
- `period_split_assignment.csv`: original assignments for all 548 periods,
  including periods without catalogue events.
- `catalogue_scan_windows.csv`: catalogue-building observation windows.
- `data_dictionary.md`: field definitions and coordinate conventions.
- `catalogue_overview.png`: spatial and monthly event distributions.
- `reproducibility.zip`: catalogue rules, detection-to-event mapping,
  QC records and validation evidence.
- `README.md` and `SHA256SUMS.txt`: documentation and file checksums.

The release preserves the original event identities and period assignments.
Observation periods were selected using NOAA Storm Events reports; the
catalogue is not a continuous, exposure-normalized climatology.

## Scope and limitations

The Zenodo release does not include native birth-footprint arrays, training
target masks, full GOES/MRMS imagery, final model issuance lists or complete
evaluation caches. Catalogue scan windows are not model forecast issuance lists.

Consequently, the catalogue alone is insufficient to reproduce the complete
training dataset or object-based verification. The repository's miniature
inference example and interactive case demonstrate specific workflows; they
do not reproduce the aggregate held-out TEST results.

## Source observations

GOES ABI and MRMS observations are publicly available through NOAA Open Data
archives. Storm Events records are available from the NOAA National Centers
for Environmental Information:

https://www.ncei.noaa.gov/stormevents/

Source attribution and research transformations are documented in
[DATA_ATTRIBUTION.md](../examples/DATA_ATTRIBUTION.md).

## Catalogue export utility

`tools/export_catalogue_release.py` exports existing event tables and period
assignments without rerunning detection or generating new splits. It checks
event identities and split consistency. It does not replace the published
catalogue or infer unresolved QC decisions.

## Citation

Crampe, M. et al. (2026). Radar-derived convection initiation catalogue over
the northeastern United States, 2021–2025 (Version 1.0.0). Zenodo.
https://doi.org/10.5281/zenodo.24179651

This DOI identifies the catalogue release, not the software or trained weights.
