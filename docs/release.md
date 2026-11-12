# Source and model-weight distribution

This repository distributes source code through Git and the 103-MiB checkpoint
as a separately attached GitHub Release asset.

1. Publish the source files and miniature example in this repository.
2. Create a published Release and attach `goes_mrms_ci35_hard_r2_public.pt`.
3. Keep that exact asset name so the notebook's latest-Release URL resolves.
4. The public Colab demo requires public access to the repository and asset.

The notebook verifies SHA-256 before inference:
`9bc69822fb18d9b5036bc4e90fa7487aa08aafdaeda364ebb176fc55b01d5b0b`.

Users place the downloaded file in `weights/`, or let the notebook download it.
Git LFS is not required for this Release-asset distribution. The existing
`.gitattributes` entry applies only if the weight file is actually committed
through Git LFS; do not add the binary through ordinary Git.

The authors should complete the chosen code licence, paper/preprint URL and
publication metadata when available. No DOI or publication status is invented
by the README or citation file. Keep the miniature-data attribution/review notice.
