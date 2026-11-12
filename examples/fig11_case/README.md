# Figure 11 Colab case

+Immutable export of `PERIOD_77b4fa7b1c66677da43a3c0d`, initialized 2025-07-25 19:00 UTC; actual split: **train**. No inference, matching, catalogue generation, training, or download was performed.
+
+- `predictions/*original_cache*`: full-domain probability (0–30 min) plus saved C13/MRMS model histories. Units: probability [0,1], C13 kelvin per original reader, MRMS dBZ, masks dimensionless.
+- `verification/*`: original decisions and precisely filtered source-table rows. Published counts: TP=7, FP=10, FN=9; diagnostic accepted growth=8; remaining FP=2; competition-FN covered=2, uncovered=7.
+- `growth_reference/`: existing >=25 dBZ, P5 cache, including candidate footprints, native pixel counts, profiles and diagnostic flags. Persistence P5 is published; P10 can be independently explored from the exported native history, not inferred from P5 labels.
+- `radar/mrms_native_T0m35_T0p60.npz`: native MRMS dBZ, exact `time_ns`, validity, coordinates and source indices.
+- `radar/mrms_projected_T0m35_T0p60.npz`: same interval projected to the complete model/verification grid.
+- `grid/verification_grid.npz`: latitude/longitude and projected x/y coordinates. Grid shape 212x230; matching precedes the display zoom.
+- `code/`: Figure 11 renderer and local scientific kernels.
+- `manifest.json`: source-to-export mapping and SHA-256. `MISSING.md`: unavailable items, never synthesized.
+
+Strict settings: probability threshold 0.05, minimum 2 predicted pixels, CI35 distance 5 km. Diagnostic: >=25 dBZ, native birth >=1 pixel, P5, 15 km; strict matching, growth promotion, and competition-FN recovery remain distinct. Interactive changes are exploratory and must not replace saved published results.
+