# Strict verification input contract

The scorer consumes a full-domain NPZ with these keys:

| Key | Contract |
| --- | --- |
| `probability` | Float probabilities, shape N × 2 × H × W |
| `core` | Binary **uninflated observed CI35 birth cores**, same shape |
| `issuance_ns` | N unique issuance times within each period |
| `period_id` | N strings; no object/pickle arrays |
| `split` | Scalar string `test` |
| `dy_km`, `dx_km` | Positive physical grid spacing |

Select the intended common cohort before export and retain exact period/time
keys. Do not supply soft targets, inflated training targets, or a cropped map
to represent full-domain verification. The miniature training target is not
an observed verification-core mask.

```bash
python verification/strict_ci35/score.py \
  --input outputs/I4_common_TEST.npz \
  --thresholds configs/frozen_thresholds.json \
  --model I4_CI35 --output outputs/I4_strict
```

The archived component extraction, original footprint-distance geometry and
maximum-cardinality matcher are reused. Strict outputs report TP/FP/FN and
pooled CSI/POD/FAR. Both windows use their own frozen threshold.
Only the final model's archived thresholds are included: 0.075 for 0–30 min
and 0.015 for 30–60 min. New runs require their own VALIDATION selection;
an identical architecture name does not identify a checkpoint.

Add `--association` to produce separate `association_scores.csv` and an explicit
FN-to-originally-matched-forecast association ledger. It keeps the original
forecast FP and original pairs. CSI_association is H/(H+remaining FN+original FP),
event POD is H/observed events, and forecast FAR remains FP/(original TP+FP).
The original synthetic tests cover empty populations, duplicates, deterministic
ties, tolerance boundaries and 250 independent maximum-cardinality comparisons.

The CLI checks array validity and identifiers. Scientific catalogue provenance,
intended common-cohort membership and checkpoint/threshold identity still require
the original manifests. The exported contract hashes the supplied arrays and
threshold table; this is not a claim that an arbitrary user-supplied NPZ is canonical.

`verification/broader_growth/persistence_numeric.py` retains the historical
persistence primitives for reference. It is not used to promote any events in
the strict scorer. The full manuscript final-QC and threshold-sensitivity
experiments are outside this focused release; they require the original
manuscript evaluation workspace.
