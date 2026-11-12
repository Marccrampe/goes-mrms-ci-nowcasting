# Validation scope

`local_validation.txt` records executed checks; `tested_environment.txt` records
local dependency versions. Checks cover real checkpoint inference, model/logit
equivalence after loading identical weights, a finite CPU gradient step,
training-only pre-crop target transformation, optional numerical verification,
and the recovered binary20/QC selftests. Stage help/import and separate-workspace
configuration are checked too.

These tests do not execute full downloads, full catalogue reconstruction, human
review or epoch-20 CUDA training. The portable training driver's full data/GPU
smoke run remains to be executed on the cluster.

IO migration produces new source hashes. Original source fingerprints and redacted public snapshots are
preserved in `provenance/` and `provenance/source_manifest.csv`. Public hashes are
refreshed after redaction; see `public_provenance.md`. Historical fingerprints
must not be mistaken for hashes of the redacted bytes. Dependency ranges and a local snapshot are supplied,
not a cross-platform guarantee of bitwise identical scientific results.
