# Publication

The folder is intended for manual author review before upload. No remote was
created and no commit/push is performed.

For a new local Git repository:

```bash
git init
git lfs install
git add .
git status --short
git lfs ls-files
```

Confirm the checkpoint appears in `git lfs ls-files` before committing. Uploading
the 309-MiB `.pt` directly through ordinary Git will not work on GitHub.
The final TAR contains the complete weights; Git LFS is only needed for Git upload.

Choose the intended code licence, paper citation, author list and data attribution.
No code licence has been chosen on the authors' behalf. The sample-data review
notice remains explicit. Full-dataset users need their own source data storage.
