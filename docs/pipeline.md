# Main-model workflow

Run from the checkout root. Append `-- --help` to a pipeline task for its full
argument list; help does not start computation. Use the source checkout rather
than a standalone wheel.

## Workspace and source data

```bash
export CI_WORKSPACE=/absolute/path/to/ci-workspace
python pipeline.py configure
```

The workspace defaults to the checkout. Generated catalogue configurations are
saved under `$CI_WORKSPACE/configs`. Existing configs are preserved. Large raw
and derived data are not included. MRMS decoding uses cfgrib/ecCodes; acquisition
requires network access and adequate storage. This is the original study's
domain, dates, grid and cohort conventions, not a general-domain library.

## 1. Native radar, candidates and periods

```bash
python pipeline.py candidate-catalogue -- --help
python pipeline.py native-periods -- --help
python pipeline.py native-label-index -- --help
```

Build the CIDS candidate catalogue for the chosen dates with an explicit output
root. Build native periods with the generated yearly configurations, then build
their label indexes using the original forecast-window arguments. The candidate
scan is not itself the final strict CI35 event definition.

## 2. GOES retrieval, parallax and fill

```bash
python pipeline.py goes-requests -- --help
python pipeline.py goes-inventory -- --help
python pipeline.py goes-parallax -- --help
python pipeline.py goes-fill -- --help
```

Execute these stages in order: expand the causal history requests, locate source
scans, download/process ABI/ACHA inputs with the original Satpy parallax geometry,
then fill missing pixels separately. Retain grid metadata and manifests.

## 3. Strict catalogue, masks and original soft dataset

```bash
python pipeline.py ci35-catalogue -- --help
python pipeline.py training-index -- --help
python pipeline.py period-splits -- --help
python pipeline.py radar-aux -- --help
python pipeline.py valid-masks -- --help
python pipeline.py dataset-two-windows -- --help
python pipeline.py mrms-complete -- --help
```

Construct strict native events/projected targets, training indexes and grouped
splits, radar auxiliary inputs and validity masks, then the two-window V3 dataset
and repaired MRMS histories. `training-index` and `period-splits` are original
no-argument programs using the workspace defaults; split search can be expensive.

## 4. Dataset QC and common cohort

```bash
python pipeline.py binary20-build -- --selftest
python pipeline.py binary20-qc -- --selftest
python pipeline.py binary20-build -- --help
python pipeline.py binary20-qc -- --help
python pipeline.py dataset-paired -- --help
```

Build binary20 targets, then run the QC generator with `--mode all` to produce
audit and TRAIN review assets. Review the cases before passing explicit
`--ignore-reviewed-cases` identifiers to `dataset-paired`. The paper used cases
1–4; do not silently apply those choices to a different catalogue. Original
review records are in `provenance/original_dataset_QC_evidence.tar.gz`.

The final model selects the paired **soft** dataset and applies hard+2 during
training. Binary20 is a historical cohort-preparation dependency, not the final
training target. The recovered generators close the prior missing-source gap.

## 5. CUDA training

Edit `dataset_root`, `mrms_input_root` and an empty `run_dir` in
`configs/final_model.json`. Relative training paths resolve against that JSON
file, independently of the shell working directory. `CI_WORKSPACE` does not
override explicitly configured training inputs.

```bash
python models/train.py --config configs/final_model.json --check-only
python models/train.py --config configs/final_model.json --smoke
# Select a fresh run_dir for the full experiment.
python models/train.py --config configs/final_model.json
```

The public driver reuses the original sampler, masked loss, full-domain target
transform before crop selection, FP32, AdamW, cosine schedule and gradient
clipping. Settings: width48, epochs20, batch8, 12,456 issuance samples/epoch,
50/50 positive sampling, PW1/BG1, Dice0, seed20260930.

It preserves original leakage/cohort checks and the expected 106 study exclusions.
New data need an explicitly reviewed cohort/configuration. Existing output
directories are rejected; this portable driver does not implement resume.
Use epoch-20 `last.pt`; `best.pt` follows original-soft validation loss.
The pixel CSI printed during training is monitoring, not manuscript object CSI.

## 6. Inference

```bash
python notebooks/run_quickstart.py
python models/inference.py --help
```

Physical NPZ keys are `goes_bt_k` (4×7×H×W), `mrms_reflectivity_dbz` and
`mrms_valid` (4×1×H×W). Use the supplied original normalization. Full-domain
inference averages probabilities from four fixed 128-pixel tiles. The miniature
64-pixel example demonstrates CPU execution directly.
