# Interactive case study

Open `notebooks/quickstart.ipynb` in Colab or Jupyter. The notebook begins with
three author-supplied manuscript figures embedded in Markdown, followed by the
case explorer. It works without a GPU or model checkpoint. See `INSTALL.txt` for
uploading the update before the new files are available on GitHub.

## What is reproduced

The exported case is `PERIOD_77b4fa7b1c66677da43a3c0d`, initialized 25 July 2025
at 19:00 UTC. Its actual split is **train**. Only its 0–30-min probability map
is available. All counts use the full 212 × 230 verification grid before zoom.

Original settings: raw threshold 0.05, forecast minimum two pixels, strict CI35
5-km matching; diagnostic ≥25 dBZ, P5, minimum one native birth pixel, 15 km.
Strict counts are TP=7, FP=10, FN=9. Eight strict-FP forecasts are accepted by
the growth diagnostic. Strict CSI=7/26 and diagnostic CSI*=15/26.
The loader verifies source-export hashes, strict matched object IDs, strict
counts and exact accepted growth candidate associations against archived files.

The saved diagnostic decisions specify P5/25 dBZ although the original config
description says P10/30 dBZ. The notebook reports that discrepancy and validates
against the saved decisions; it does not silently reinterpret the figure.

## What is interactive

- Raw probability threshold: new exploratory connected forecast objects.
- ≥25-dBZ lineage persistence: 0/base growth rule, 5, 10, 15 or 20 min.
- Growth-diagnostic radius: 5, 10, 15, 20 or 30 km; strict distance stays 5 km.
- Minimum native birth area: 1, 2, 3 or 5 pixels.
- Display extent and forecast-object identifiers.
- Object-specific probability view and radar evolution.
- Radar playback through the actual forecast window.
- Export of exploratory metrics, forecast-object table and explicit settings.

Growth eligibility retains the original novelty, contiguous-scan, growth,
coverage and topology tests. P10/P15/P20 are filtered from temporal profiles,
not copied from P5 classifications. Only ≥25-dBZ tracks exist in this export.

## Accounting

This is the **Figure 11 growth diagnostic**, not the separate S12 convention or
competition-FN recovery. With G accepted growth forecasts, diagnostic counts
are TP+G, FP−G, FN. Strict FN remain unchanged. Duplicate forecasts are not
promoted. The diagnostic is explicitly labeled CSI*/POD*/FAR*.

Forecast input histories and model results are unchanged when selecting radar
criteria. Changing a probability threshold creates exploratory objects; it
does not select a new published threshold. Restore the source settings with
the reset button. The live tables are single-case results, not TEST aggregates.

## Separate model inference

The optional final notebook section uses the full repository's miniature TEST
sample and released weights to run both forecast heads. This is independent
of the saved full-domain Figure 11 forecast. The single-case archive does not
contain a complete C08–C14 input cube or a 30–60-min probability map.

## Validation

`python -m unittest discover -s tests -p test_case_demo.py` checks original-case
regression, invariants across diagnostic controls, minimum forecast size,
duplicate prevention, empty forecasts and empty truth.

The notebook's scientific cells and widget callbacks were exercised locally.
Actual hosted Colab and GitHub asset availability remain to be checked after
upload. No files were published and no cluster jobs were started.
