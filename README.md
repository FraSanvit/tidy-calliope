# tidy-calliope

[![tests](https://github.com/FraSanvit/tidy-calliope/actions/workflows/tests.yml/badge.svg)](https://github.com/FraSanvit/tidy-calliope/actions/workflows/tests.yml)

Post-processing of [Calliope](https://github.com/calliope-project/calliope) **0.7** result files (`.nc`) into tidy
tables: one folder per scenario, one CSV (or parquet) file per table.

It started as a rewrite, for the Calliope 0.7 data model, of
[friendly-calliope](https://github.com/brynpickering/friendly-calliope) by Bryn Pickering (Calliope 0.6), whose
output tables it largely follows. Like friendly-calliope, it is released under the Apache-2.0 license.

- Reads the result files with xarray only: no Calliope installation needed.
- Works with **clustered** runs (tsam / `config.init.time_cluster`): sums and maxima are computed for the real year,
  and hourly tables are **rebuilt to the full year** (8760 h), including the storage level from Calliope's
  inter-day storage (`storage_inter_cluster`).
- Settings such as the time resolution, the clustering and the weights are read from the file itself.
- Every run checks its own results against the model (see [Checks](#checks)) and writes them in `metadata.json`.

## Usage

```bash
# a folder (all .nc inside), files, or glob patterns; default output folder: tidy-results/
tidy-calliope results/ -c tidy.yaml -o tidy-results
python -m tidy_calliope results/2030-1x-test-1h-24d.nc     # same, as a module
tidy-calliope --show-config                                 # all settings, with comments
```

Files without a solution (a model not solved to optimality is saved with empty results) are skipped with a one-line
message; the other files are still processed, and the exit code is 1. Exit code 2: no input files or a configuration
error.

From Python:

```python
from tidy_calliope import process
process(["results/"], output_dir="tidy-results", config={"outputs": {"hourly": True}})
```

After each model run, in the run's sbatch file (or as a dependent job, see `examples/postprocess.sbatch`):

```bash
srun python run-model.py
srun tidy-calliope results/<scenario>.nc -c tidy.yaml
```

### Installation

- **pip**: `pip install tidy-calliope` (`pip install "tidy-calliope[parquet]"` for parquet output).
  It can go into the Calliope environment: it only needs `xarray`, `pandas`, `netcdf4` and `pyyaml`.
- **From source**: `pip install .` in a clone of this repository, or `conda env create -f environment.yml` first.
- **Container** (Apptainer), from a clone:
  ```bash
  apptainer build --fakeroot tidy-calliope.sif tidy-calliope.def
  apptainer run --bind "$PWD" tidy-calliope.sif results/ -c tidy.yaml -o tidy-results
  ```

## Configuration

A YAML file with only the settings to change; it is merged on top of `tidy_calliope/default_config.yaml`
(typos in top-level keys or output names are an error). Example: `examples/tidy.yaml`.

| Setting | Default | |
|---|---|---|
| `outputs.names` / `capacities` / `annual` / `monthly` / `costs` / `input_costs` / `transmission` / `primary_energy_supply` | `true` | which tables to write |
| `outputs.hourly` | `false` | hourly flows, storage level and net imports |
| `outputs.shadow_prices` | `false` | energy balance prices and other shadow prices (see below) |
| `shadow_prices.full_year` | `true` | clustered runs: prices for every real hour; `false`: per model timestep |
| `format`, `hourly_format` | `csv` | `csv` or `parquet` (needs pyarrow) |
| `hourly_layout` | `long` | `wide`: one column per timestep (much smaller for full-year hourly data) |
| `regions` | `subcountries` | `subcountries` (model nodes), `countries` (node prefix before `_`) or `{node: region}` |
| `units` | TWh / TW, billion km, ... | scale and labels (as the v0.6 scripts: model value x 0.1) |
| `cost_classes` | `monetary`, `co2` | table names per cost class |
| `techs.drop`, `techs.rename` | `_converter$`, `wind_onshore_*`, heat storage | regex rules; renamed techs are summed |
| `carriers.rename` | `*_heat -> heat`, `syn_* -> *` | regex rules |
| `transmission.rename` | `ac_*`, `dc_*` | link type, from the link tech name without `<from>_to_<to>_` |
| `primary_energy_supply` | v0.6 categories | producers, exports, net imports |

## Tables

All tables have the columns `scenario`, the table's dimensions, `unit` and a value column named after the table.

| Table | Dimensions | |
|---|---|---|
| `names` | techs | long names |
| `nameplate_capacity` | techs, locs, carriers, unit, flow | `flow_cap` per carrier; `flow` = in / out / both |
| `storage_capacity` | techs, locs, carriers, unit | `storage_cap` |
| `flow_out_sum`, `flow_in_sum` | techs, locs, carriers, unit | energy over the year |
| `flow_out_max`, `flow_in_max` | techs, locs, carriers, unit | highest power (energy per timestep / resolution) |
| `*_sum_1M`, `*_max_1M` | ... , month | per calendar month (`YYYY-MM`) |
| `flow_out`, `flow_in`, `storage_level` | ... , timesteps | hourly (`outputs.hourly`); real timesteps |
| `total_system_cost`, `annualised_investment_cost`, `fixed_operation_cost`, `variable_operation_cost`, `total_investment_cost` | techs, locs, unit | per cost class (`*_emissions` for `co2`) |
| `cost_per_*`, `annual_cost_per_*`, `interest_rate`, `emissions_per_*` | techs[, locs][, carriers], unit | input costs |
| `net_transfer_capacity` | techs (link type), exporting_region, importing_region, carriers, unit | both directions |
| `net_import_sum`, `net_import_max`, `*_1M`, `net_import` | exporting_region, importing_region, carriers, unit[, month / timesteps] | import into `importing_region` minus export to it |
| `total_transmission_costs` | techs, exporting_region, importing_region, unit | per link direction as defined in the model |
| `primary_energy_supply` | locs, carriers (category), unit | annual |
| `shadow_price_system_balance` | nodes, carriers, unit, timesteps, `value` | price of energy (`outputs.shadow_prices`) |
| `shadow_price_constraints` | constraint, index, `value` | other tracked shadow prices, e.g. CO2 caps (raw) |

Transmission links (one tech per link in Calliope 0.7) are only in the transmission tables; links within one region
(e.g. with `regions: countries`) are left out.

## Time handling

- **Not clustered** (e.g. 1 h, 2 h, 730 h): model timesteps are real timesteps. A timestep crossing a month boundary is
  split between the months in proportion to its hours. "Hourly" tables are at the model resolution.
- **Clustered**: each real day is mapped to its representative day (`lookup_datestep_cluster`). Annual and monthly
  sums weight each model timestep by the real days it stands for; maxima are taken over the representative days of
  each month; hourly tables repeat each representative day for every real day it stands for. The storage level is
  `storage_inter_cluster` (level at the start of the real day, decayed by `storage_loss`) plus `storage` (change
  within the representative day).
- Calliope's `cost_operation_variable` already includes the timestep weights: it is summed without weights.

## SPORES

SPORES runs (Calliope `config.init.mode: spores`) are processed in both of Calliope's file formats:

- **one file** (`model.to_netcdf` after a SPORES solve): the results have a `spores` dimension;
- **one file per SPORE** (`config.solve.spores.save_per_spore_path`): `spore_0.nc`, `spore_1.nc`, ... in one
  folder. Only `spore_0.nc` has the inputs; they are used for every SPORE of the folder, also when a single
  `spore_<n>.nc` is given. The folder is one run, named after the folder (e.g. `max_battery`).

Each SPORE is processed on its own (time rebuilding, transmission, checks), and every table gets a `spores`
column after `scenario` (0 = the cost-optimal solution).

## Shadow prices

Written when the model was solved with `config.solve.shadow_prices` (e.g. `[system_balance, ...]`) and
`outputs.shadow_prices: true`.

- `shadow_price_system_balance`: the price of energy per node, carrier and timestep. Calliope weights each model
  timestep in the objective by `timestep_weights`, so its dual is weight x price: it is divided by the weights (for
  non-clustered runs the weights are 1). For clustered runs the prices of each typical day are then repeated for
  every real day it stands for (8760 h); `shadow_prices.full_year: false` keeps one row per model timestep.
  Unit: objective cost unit per unit of energy of the carrier (e.g. `billion_2015eur_per_twh`; x 1000 = EUR/MWh).
  Model nodes and carrier names are kept: prices are not summed over regions or merged carriers.
  Like the hourly tables, it follows `hourly_format` and `hourly_layout`.
- `shadow_price_constraints`: every other tracked constraint (e.g. `group_systemwide_co2_max_*`), as stored by
  Calliope (no unit conversion). Annual constraints such as CO2 caps need no weight correction.

## Checks

Written to `metadata.json` (`checks`, `checks_failed`); `--strict` makes a failed check an error (exit code 1).

| Check | |
|---|---|
| `timestep_weights`, `hours_in_period` | the rebuilt time axis has the model's weights and covers the modelled period (`full_year`: 8760/8784 h) |
| `*_adds_up_to_*` | monthly and hourly tables add up to the annual ones |
| `demand_met` | inflow of demand techs = `sink_use_equals`, both over the real year |
| `costs_sum_to_objective` | the techs' costs (weighted by cost class) = the objective |
| `<cost table>_complete` | reported costs + transmission costs = model total (minus dropped techs, listed) |
| `net_import_antisymmetric` | import of A from B = - import of B from A |
| `storage_level_within_bounds` | rebuilt storage level within [0, `storage_cap`] |
| `shadow_price_timesteps` | one price per hour of the (rebuilt) year for every node and carrier |
| `spores_cost_within_slack` | SPORES: every SPORE costs at most (1 + `spores_slack`) x the cost-optimal one |

For SPORES runs the checks are run per SPORE (`checks` has one entry per `spores=<n>`).

## Development

```bash
pip install -e ".[test]"
pytest            # synthetic clustered and resampled result files, see tests/conftest.py
```

### Releasing

1. Once: on [pypi.org](https://pypi.org/manage/account/publishing/) add a *pending trusted publisher* for project
   `tidy-calliope`: owner `FraSanvit`, repository `tidy-calliope`, workflow `release.yml`, environment `pypi`.
   On GitHub, create the environment `pypi` (Settings > Environments).
2. Set `__version__` in `tidy_calliope/__init__.py`, commit, then tag and push:
   `git tag v0.1.0 && git push origin main v0.1.0`. The `release` workflow tests, builds and publishes to PyPI.

## Not included (v0.6 scripts)

Tables built from external data (final energy consumption, service demands from the Euro-Calliope annual demand
files) are left out: this library only processes model outputs.
