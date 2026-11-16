# Source and model-weight distribution

## Source code

Source code is publicly available at:

https://github.com/Marccrampe/goes-mrms-ci-nowcasting

Original software is distributed under the [MIT License](../LICENSE).
See [licensing and attribution](licensing.md) for other assets.

## Pretrained model

The final checkpoint is distributed as a separate GitHub Release asset:

[Download goes_mrms_ci35_hard_r2_public.pt](https://github.com/Marccrampe/goes-mrms-ci-nowcasting/releases/latest/download/goes_mrms_ci35_hard_r2_public.pt)

The file is approximately 103 MiB. It is not stored in ordinary Git;
Git LFS is not required to download it.

For local inference, place the file in `weights/`. The demonstration
notebook can download it automatically and verifies its SHA-256:

9bc69822fb18d9b5036bc4e90fa7487aa08aafdaeda364ebb176fc55b01d5b0b

Distribution metadata are recorded in
[release_assets.json](../weights/release_assets.json).
The MIT software licence does not automatically establish reuse terms
for the model weights.

## Demonstration data

The repository includes a miniature inference sample and an archived
case-study example. Sources and research transformations are documented
in [DATA_ATTRIBUTION.md](../examples/DATA_ATTRIBUTION.md).

These examples demonstrate inference and verification workflows;
they do not contain the complete training or TEST evaluation dataset.

## Published catalogue

The CI35 catalogue is published on Zenodo, version 1.0.0, under CC BY 4.0:

https://doi.org/10.5281/zenodo.24179651

See [data availability](data_availability.md) for its contents and limitations.
This DOI identifies the catalogue, not the source code or model weights.
