# Catalogue entry points

Run `pipeline.py` from the repository root; `CI_WORKSPACE` controls the data/output
location. `period_selection/run.py` launches native period construction;
`ci35_detection/run.py` launches the V3 strict catalogue on those native scans;
`quality_control/run.py` launches the original strict catalogue validator.
All accept `--help` before touching data. See `docs/pipeline.md` for stage order.

The numerical source uses the historical folder structure under `workflow/` to
keep dynamic sibling imports consistent. The public tree does not contain
duplicate copies of the same training/dataset source in each experiment folder.
