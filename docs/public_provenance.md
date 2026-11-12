# Public provenance and path redaction

Public source/configuration snapshots use the symbolic root `/project` and
generic local-environment labels instead of personal account names, cluster
hostnames or home directories. Runtime paths are still selected by CI_WORKSPACE
or resolved relative to the configured files.

The source and dataset/QC archives under provenance/ are **redacted snapshots**,
not byte-identical originals. Each archive includes PUBLIC_REDACTION_MANIFEST.json
with original/private source fingerprints and public snapshot fingerprints.
Original digests remain useful identifiers; they must not be described as the
current hashes of redacted public files. The public source manifest and case
export manifest have refreshed hashes; integrity checks remain enabled.

The inference-only public checkpoint contains the exact original trained model
tensors and a minimal path-free architecture configuration. Optimizer state,
RNG state and private training metadata are omitted. Its file SHA-256 differs
from the training checkpoint because the container changed; the tensors and
miniature-example predictions are exactly equal. See
provenance/public_weights_validation.json.

Public authorship, correspondence addresses and the public GitHub URL are
retained intentionally. Paths in archival records are evidence identifiers,
not directions requiring a reader to access an institutional cluster.

This cleanup does not remove previously published files from Git history,
older Releases or copies downloaded by other users. Updating the current branch
alone does not erase earlier versions. No remote history was changed here.
