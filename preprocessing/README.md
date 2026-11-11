# Preprocessing entry points

`goes/run.py` launches the parallax builder; use `pipeline.py goes-fill` for its
separate fill stage. `mrms/run.py` repairs/finishes MRMS inputs, after radar aux
construction. `dataset/run.py` assembles the paired soft/binary dataset.
The full dependency order and arguments are documented in `docs/pipeline.md`.

For full training, paths in `configs/*.json` identify prepared data; they do not
cause automatic downloading or bypass the original READY/cohort checks.
