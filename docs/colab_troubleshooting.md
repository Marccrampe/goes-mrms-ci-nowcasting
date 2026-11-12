# Colab startup

Open `notebooks/quickstart.ipynb` from the main branch, then run its setup cell.
The setup downloads a fresh public source ZIP directly from GitHub, checks every
case asset (including Git LFS pointer detection), imports the small demo dependency
set, and enables the Colab widget manager. It does not require the trained model,
a GPU, an editable package installation, or a manual data upload.

The following cell checks all case checksums and reproduces the archived original
matching decisions before opening the controls. The optional model section downloads
the public 103-MiB checkpoint and verifies its SHA-256 before loading it.

After updating the repository, reopen the notebook from GitHub and use a fresh
Colab runtime. A runtime restart does not replace an already-open old notebook.
If setup fails, retain the numbered step and the complete traceback. Dependency
installation and GitHub download errors are displayed explicitly; they are not
silently replaced with different data or metrics.

Local notebook execution verifies numerical code and widget construction; it does
not prove that the browser frontend or a live Google Colab session works.
