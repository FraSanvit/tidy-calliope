# Changelog

## 0.2.0 (2026-10-30)

### Added

- Shadow prices (`outputs.shadow_prices: true`, off by default):
  - `shadow_price_system_balance`: price of energy per node, carrier and timestep (columns `nodes`, `carriers`,
    `unit`, `timesteps`, `value`). Calliope's dual is weight x price: it is divided by the timestep weights, and for
    clustered runs the typical-day prices are rebuilt to every real hour of the year
    (`shadow_prices.full_year: false` keeps the model timesteps). Unit: objective cost unit per energy unit of the
    carrier, e.g. `billion_2015eur_per_twh`.
  - `shadow_price_constraints`: the other tracked shadow prices (e.g. CO2 caps), as stored by Calliope.
- Check `shadow_price_timesteps`: one price per (rebuilt) timestep for every node and carrier.
- SPORES runs, in both Calliope file formats: one file with a `spores` dimension, or per-SPORE files
  (`spore_<n>.nc`, one run per folder; inputs from `spore_0.nc`, also for a single `spore_<n>.nc`). Every table
  gets a `spores` column (0 = cost-optimal); checks are run per SPORE, plus `spores_cost_within_slack`.

### Changed

- The check `hours_in_year` is now `hours_in_period`: it compares the hours represented by the weighted model
  timesteps with the modelled period, so models of part of a year pass (`full_year` tells whether it is a year).
- The command line reports only user errors (missing input paths, unknown configuration keys) as a one-line
  `ERROR:`; any other error keeps its traceback (0.1.1 also turned some internal errors into `ERROR:` lines).

### Fixed

- Input cost parameters defined for the whole model (no `techs` dimension, e.g. `cost_interest_rate` in the Calliope
  example models) made the input-cost tables fail; they are now reported with their unit only.

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
