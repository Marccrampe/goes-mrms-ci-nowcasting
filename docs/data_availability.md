# Data availability and immutable release

The repository includes miniature/case examples and recovered source/QC
evidence. It does not yet include a published immutable event-table release for
the manuscript's 38,313 CI events. Reconstructing the same dataset requires
more than executing the miniature notebook.

The dataset release should contain:

1. Canonical accepted event table, retaining event IDs, period IDs, birth times,
   native/projected footprints or references, coordinates and QC outcomes.
2. Exact period-to-split assignment and issuance/lead lists used by the paper.
3. Frozen exclusion/QC manifests and the final catalogue-building settings.
4. SHA-256 checksums and a schema explaining every field and coordinate system.
5. Dataset/software versions that link the paper, trained checkpoint and event
   release without substituting newly generated split assignments.

Use `tools/export_catalogue_release.py` to export an existing accepted event
table and period split assignment without recomputing events or changing
splits. It preserves all original fields and rejects missing event/period IDs,
duplicate event IDs, ambiguous split assignments and conflicting split columns.
Provide the actual final accepted table; the tool cannot infer human QC choices
from an intermediate catalogue. `--expected-events 38313` is an explicit check,
not a filter that silently forces that count.

Once rights and identity have been verified, archive a versioned dataset and
software release in a suitable persistent repository such as Zenodo. Add the
real DOI only after it exists. Do not put an invented DOI or a claim that an
upload was completed in README/CITATION.cff.

This local preparation does not publish files, create external accounts,
change GitHub visibility or deposit anything in Zenodo.
