# Changelog

## 0.1.1 (2026-10-30)

### Changed

- Files that cannot be processed (no solution, e.g. a model stopped by numerical trouble, or not a Calliope 0.7
  result file) are reported in one line (`SKIPPED <file>: no results: the model was not solved to optimality`)
  instead of a Python traceback. The solver's termination condition is added when the file has it.
- The command line skips such files and processes the others: one unsolved file no longer stops a whole folder.
  It lists the skipped files at the end and exits with code 1, so a Slurm job still shows the failure.
- A missing input path or a typo in the configuration gives a one-line `ERROR:` and exit code 2.
- New `tidy_calliope.ResultFileError` for these cases. In Python, `process(...)` still raises it by default;
  `process(..., skip_unprocessable=True)` skips the file as the command line does.

## 0.1.0 (2026-10-30)

First release.

- Post-processing of Calliope 0.7 result files into tidy tables: capacities, flows, costs (per cost class), input
  costs, transmission (net transfer capacity, net imports, costs) and primary energy supply.
- Annual, monthly and hourly tables, selected in a YAML configuration merged on top of the defaults.
- Clustered (typical-day) runs: sums weighted by the real days each typical day stands for, hourly tables rebuilt
  to the full year, storage level rebuilt from Calliope's inter-cluster storage.
- Consistency checks of every run against the model (demand, objective, monthly and hourly totals, storage bounds),
  written to `metadata.json`.
- Command line `tidy-calliope`, Apptainer recipe, tests on synthetic result files.
