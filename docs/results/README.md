# Archived strict TEST results

`main_TEST_results.csv` contains the exact operating-point results shown in the
README. Values are read from archived source tables, not from image labels or
new threshold sweeps. The standard formulas CSI=TP/(TP+FP+FN), POD=TP/(TP+FN)
and FAR=FP/(TP+FP) are checked against the archived numbers to 1e-12.

## Source files

- `comparison_with_resunet.csv`: CTC fixed, CTC VAL-selected, GOES-only and final
  GOES+MRMS, 0–30 min; the common archived comparison uses 4,712 TEST issuances.
- `joint_specialist_scores.csv`: the `I4 joint head1` row supplies the final
  model's 30–60-min result on 4,712 TEST issuances. Specialist rows are preserved
  as source evidence but are not substituted for the joint final model.
- `sources.json`: original recovered file locations and SHA-256.

Each README row uses the canonical CI35 catalogue, original one-to-one matching,
5-km tolerance, minimum two predicted GOES pixels and frozen operating threshold.
CTC's thresholds have units K/15 min; model thresholds are raw probabilities.
The 0–30-min reference has 10,116 observed objects; 30–60 min has 10,119.

The GOES-only controlled configuration records `training_target_kind=hard`,
`inflation_kind=hard`, `inflation_pixels=2`, and a training-only transform.
Its legacy `target_kind=soft` field refers to the base dataset; it must not be
read as evidence that the controlled model was trained on the untransformed
soft target. The final model likewise uses hard+2. Configuration evidence is
preserved in `GOES_only_training_config.json` and the existing final config.

These tables do not add association/FN-recovery counts, promote weaker growth,
select TEST-optimal thresholds, or replace missing scores with VALIDATION values.
No BSS is inferred from a Brier score alone: its reference forecast and scoring
population would need to be established first. No radar-persistence,
extrapolation or external-publication scores are supplied in this release.

The case explorer is a separate illustrative case. Its Fig11 growth diagnostic
keeps strict FN while promoting eligible strict-FP forecast objects. It is not
the strict aggregate result table or the separately defined S12/competition
diagnostics. Refer to the notebook's explicit accounting before comparing them.
