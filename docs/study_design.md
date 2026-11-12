# Study design and split implementation

The released grouped-split implementation is
`workflow/scripts/isolated_ci_binary_v1/goes_ir_v1/build_ci_period_splits.py`.

- `SPLIT_RATIOS` specifies 80/10/10 target proportions.
- Issuances and CI-object statistics are aggregated by `period_id` and year.
- A randomized search balances unique CI objects, positive issuances, total
  issuances and period counts, globally and by year; seed=20260930.
- `period_to_split` maps every issuance from the same period to one split.
- The program writes `period_split_assignment.parquet` and split-specific
  natural issuance/lead indexes.

The grouping is by period, not by randomly separated pixels, crops or individual
CI events. It is not a future-year holdout. Period grouping prevents the same
period ID from being assigned to multiple splits. That property alone does not
prove that separately constructed periods cannot overlap in time/space; the
original period definitions, split manifests and downstream cohort checks
remain necessary evidence. Do not claim an additional temporal buffer or
year-based holdout that this program does not implement.

Final training is implemented in `models/train.py`, with settings in
`configs/final_model.json`. The training sampler uses a 50/50 positive-issuance
fraction. The supplied sigmoid output is therefore presented as an uncalibrated
model score rather than an observed-event frequency. Applying an isotonic map
would require the actual VAL-fitted calibration artefact and matching output/
reference definition; none is fabricated for the demo.

Hard+2 is a training-only spatial transformation of the base soft-target
support before crop selection. The observation catalogue and strict matching
geometry remain unchanged. The target comparison in the supplied Figure 4
provides the sensitivity experiment. The code release does not add an untested
physical interpretation of the dilation.
