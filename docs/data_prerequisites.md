# Data and reviewed decisions

The previously missing generators are now included:
`workflow/scripts/build_ci_v4_binary20.py`,
`workflow/scripts/pilot_ci_v4_binary_envelopes.py`, and
`workflow/scripts/qc_ci_v4_binary20.py`.

Build or obtain the native MRMS crops, filled GOES stores/grid metadata, canonical
CI35 events, grouped splits, original soft targets, per-issuance masks, repaired
MRMS inputs and common paired indexes. Keep READY/config/index hashes together.
Archived project prefixes inside indexes are relocated at read time without
rewriting hashed parquet inputs.

The historical paired builder consumes binary20 manifests and reviewed QC files;
their generators are included. Original metadata/selected cases are preserved in
`provenance/redacted_dataset_QC_evidence.tar.gz`, not automatically installed into
a newly generated dataset. The original builder and pilot hashes matched the
archived build_config; the recovered manifest matched the QC's recorded hash.

Human exclusions are part of the original protocol. Reproduce them only for
matching original source inputs; review new catalogues independently. Do not
assume that a new dataset's first four examples represent the original exclusions.
The final hard+2 target is based on original soft targets, not binary20 envelopes.

Full native-data retrieval, catalogue reconstruction and GPU training have not
been executed in local release validation. The miniature is an inference example,
not the complete training dataset.
