# GOES–MRMS convective-initiation nowcasting

This repository focuses on the **final GOES+MRMS ResUNet**: build the radar
catalogue, retrieve/preprocess GOES and MRMS, correct GOES parallax and missing
pixels, prepare the training dataset, train the model and run inference.
The paper's comparison experiments and figure-generation histories are excluded.

GOES ABI C08–C14 at four times and MRMS-derived predictors provide 46 channels.
Outputs cover (0,30] and (30,60] minutes. The final training target is the hard
support of the original soft target dilated by two model-grid pixels.

## Real CPU demonstration

Python 3.11 or later, from this source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python -m unittest discover -s tests -v
python notebooks/run_quickstart.py
```

The full archive includes the final 309-MiB checkpoint and a real miniature TEST
sample. Git clones require Git LFS for `weights/goes_mrms_ci35_hard_r2.pt`.
Checkpoint SHA-256:
`9231c9bb9aa540717986c1ae729cdbbf56629ed74b072de10a200be4f9ae507e`.

## Main-model pipeline

```bash
python -m pip install -r requirements.txt
python pipeline.py list
python pipeline.py configure
```

Follow [the ordered workflow](docs/pipeline.md). `CI_WORKSPACE` selects the large
data/output directory; sources resolve independently to this checkout.
The recovered binary20 and QC generators are included. They remain historical
prerequisites of the original paired-cohort preparation, even though the final
model trains on hard+2 of the original soft target, not on binary20 envelopes.

After preparing data and editing paths in `configs/final_model.json`:

```bash
python models/train.py --config configs/final_model.json --check-only
python models/train.py --config configs/final_model.json --smoke
# Choose a fresh run_dir after the smoke test.
python models/train.py --config configs/final_model.json
```

Training requires CUDA and the complete prepared data. Primary weights are
epoch-20 `last.pt`; validation uses original soft targets and four fixed tiles.
Human QC choices must be reviewed explicitly; generating a report does not
automatically approve or exclude events.

## Repository

- `catalogue/`, `preprocessing/`: public stage entry points.
- `models/`: standalone architecture, training driver and physical-input inference.
- `workflow/`: curated scientific implementation shared by the entry points.
- `configs/`: final-model settings and catalogue templates.
- `weights/`, `examples/`, `notebooks/`: released checkpoint and working demo.
- `verification/`: optional scoring utilities, not a prerequisite for training.
- `provenance/`: original sources, final training config and dataset/QC evidence.
- `tests/`, `docs/`: tests, execution reports and instructions.

Internal historical module names are retained for traceability; no comparison
experiment must be executed to use the public main-model training driver.

## Validation and publication

See [validation status](docs/validation.md) and [data prerequisites](docs/data_prerequisites.md).
Real inference and numerical/startup checks are tested locally. Full downloads,
catalogue regeneration, human review and epoch-20 CUDA training have not been
repeated on the Mac. Those executions require validation on a data/GPU-equipped system.

Choose the authors' intended licence/citation and review the miniature-data
notice before publication. Git/LFS commands are in [release instructions](docs/release.md).
