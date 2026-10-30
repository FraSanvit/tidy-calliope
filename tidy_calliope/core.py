"""Processing of result files: one output folder per scenario, one table per file."""

import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from tidy_calliope import __version__
from tidy_calliope.checks import run_checks
from tidy_calliope.config import load_config
from tidy_calliope.outputs import Processor
from tidy_calliope.reader import ResultFileError, Run, find_result_files, group_runs, open_run

LOGGER = logging.getLogger("tidy_calliope")


def compute_tables(model, cfg: dict) -> tuple[Processor, dict]:
    """All configured tables of one model, as {name: Series}."""
    out = cfg["outputs"]
    proc = Processor(model, cfg)
    tables: dict[str, pd.Series] = {}

    def _add(name: str, compute):
        t0 = time.time()
        result = compute()
        tables.update(result)
        LOGGER.info(f"{model.scenario}: {name} ({', '.join(result) or 'nothing'}) in {time.time() - t0:.1f} s")

    if out["names"]:
        _add("names", lambda: proc.names())
    if out["capacities"]:
        _add("capacities", lambda: proc.capacities())
    need_annual = out["annual"] or out["primary_energy_supply"]
    if need_annual or out["monthly"] or out["hourly"]:
        _add("flows", lambda: proc.flows(annual=need_annual, monthly=out["monthly"], hourly=out["hourly"]))
    if out["hourly"]:
        _add("storage level", lambda: proc.storage_level())
    if out["costs"]:
        _add("costs", lambda: proc.costs())
    if out["input_costs"]:
        _add("input costs", lambda: proc.input_costs())
    need_transmission = out["transmission"] or out["primary_energy_supply"]
    if need_transmission:
        _add("transmission", lambda: proc.transmission(annual=need_annual, monthly=out["monthly"] and out["transmission"],
                                               hourly=out["hourly"] and out["transmission"]))
    if out["shadow_prices"]:
        _add("shadow prices", lambda: proc.shadow_prices())
    if out["primary_energy_supply"]:
        _add("primary energy supply", lambda: proc.primary_energy_supply(
            tables.get("flow_out_sum"), tables.get("flow_in_sum"), tables.get("net_import_sum")))

    # tables only computed because another table needs them
    drop = []
    if not out["annual"]:
        drop += [n for n in tables if n.startswith("flow_") and (n.endswith("_sum") or n.endswith("_max"))]
    if not out["transmission"]:
        drop += [n for n in tables if n.startswith("net_") or n.startswith("total_transmission")]
    return proc, {k: v for k, v in tables.items() if k not in drop}


def write_table(series: pd.Series, name: str, folder: Path, scenario: str, scenario_dim: str, fmt: str,
                wide: bool = False, value_name: str | None = None) -> Path:
    """One table to file; the value column is named after the table, or `value_name`."""
    if wide:  # one column per timestep
        df = series.unstack("timesteps")
        df.columns = [str(c) for c in df.columns]
        df = df.reset_index()
    else:
        df = series.rename(value_name or name).reset_index()
    df.insert(0, scenario_dim, scenario)
    if fmt == "parquet":
        path = folder / f"{name}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = folder / f"{name}.csv"
        df.to_csv(path, index=False)
    return path


def _with_spore(series: pd.Series, spore: int) -> pd.Series:
    """Prepend a `spores` index level."""
    return pd.concat({spore: series}, names=["spores"])


def _spores_cost_check(models: list) -> dict | None:
    """SPORES: total monetary cost of every SPORE within (1 + spores_slack) x the cost-optimal one (SPORE 0)."""
    costs = {}
    for m in models:
        r = m.results
        if "cost" in r and "costs" in r.coords and "monetary" in r["costs"].values:
            costs[m.spore] = float(r["cost"].sel(costs="monetary").sum())
    slack = models[0].inputs.get("spores_slack")
    if 0 not in costs or slack is None or len(costs) < 2:
        return None
    limit = (1 + float(slack)) * costs[0]
    above = {s: c for s, c in costs.items() if c > limit * (1 + 1e-6)}
    return {"ok": not above, "cost_optimal": costs[0], "spores_slack": float(slack), "limit": limit,
            "max_cost": max(costs.values()), "above_limit": above}


def process_run(run: Run, output_dir: str | Path, cfg: dict, checks: bool = True) -> dict:
    """Process one run (a result file, or the per-SPORE files of a folder) into `output_dir/<scenario>/`.

    SPORES runs: each SPORE is processed on its own and the tables get a `spores` column (0 = cost-optimal);
    the checks are run per SPORE. Returns the run summary (also written as metadata.json).
    """
    t0 = time.time()
    models = open_run(run)
    first = models[0]
    is_spores = first.spore is not None
    folder = Path(output_dir) / run.scenario
    folder.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"{run.scenario}: {[str(f) for f in run.files]} -> {folder} ({first.time.describe()}"
                + (f", SPORES {[m.spore for m in models]})" if is_spores else ")"))

    parts: dict[str, list[pd.Series]] = {}
    all_checks = {}
    for model in models:
        proc, tables = compute_tables(model, cfg)
        if checks:
            result = run_checks(proc, tables)
            if is_spores:
                all_checks[f"spores={model.spore}"] = result
            else:
                all_checks = result
        for name, series in tables.items():
            parts.setdefault(name, []).append(_with_spore(series, model.spore) if is_spores else series)
    tables = {name: pd.concat(series) if len(series) > 1 else series[0] for name, series in parts.items()}

    hourly_tables = {"flow_out", "flow_in", "storage_level", "net_import", "shadow_price_system_balance"}
    value_column = {"shadow_price_system_balance": "value", "shadow_price_constraints": "value"}
    written = {}
    for name, series in tables.items():
        hourly = name in hourly_tables
        fmt = cfg["hourly_format"] if hourly else cfg["format"]
        wide = hourly and cfg["hourly_layout"] == "wide"
        p = write_table(series, name, folder, run.scenario, cfg["scenario_dim_name"], fmt, wide=wide,
                        value_name=value_column.get(name))
        written[name] = {"file": p.name, "rows": int(len(series))}

    summary = {
        "scenario": run.scenario,
        "source": [str(f.resolve()) for f in run.files] if len(run.files) > 1 else str(run.files[0].resolve()),
        "created": datetime.now().isoformat(timespec="seconds"),
        "tidy_calliope_version": __version__,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "time": first.time.describe(),
        "spores": [m.spore for m in models] if is_spores else None,
        "tables": written,
        "runtime_seconds": None,
        "config": cfg,
    }
    if checks:
        if is_spores:
            spores_check = _spores_cost_check(models)
            if spores_check is not None:
                all_checks["spores_cost_within_slack"] = spores_check
            failed = [f"{s}: {k}" for s, c in all_checks.items() if s.startswith("spores=")
                      for k, v in c.items() if not v["ok"]]
            failed += [k for k, v in all_checks.items() if not k.startswith("spores=") and not v["ok"]]
        else:
            failed = [k for k, v in all_checks.items() if not v["ok"]]
        summary["checks"] = all_checks
        summary["checks_failed"] = failed
        (LOGGER.warning if failed else LOGGER.info)(
            f"{run.scenario}: checks {'FAILED: ' + ', '.join(failed) if failed else 'all passed'}")
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(folder / "metadata.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    for model in {id(m.results): m for m in models}.values():
        model.close()
    return summary


def process_file(path: str | Path, output_dir: str | Path, config: str | Path | dict | None = None,
                 scenario: str | None = None, checks: bool = True, *, loaded_config: dict | None = None) -> dict:
    """Process one result file into `output_dir/<scenario>/`. Returns the run summary (also written as JSON).

    `config`: YAML path or dict with the settings to change; `loaded_config`: an already merged configuration.
    A per-SPORE file (`spore_<n>.nc`) takes its inputs from `spore_0.nc` in the same folder.
    """
    cfg = loaded_config if loaded_config is not None else load_config(config)
    (run,) = group_runs([Path(path)])
    if scenario:
        run.scenario = scenario
    return process_run(run, output_dir, cfg, checks)


def process(paths, output_dir: str | Path = "tidy-results", config: str | Path | dict | None = None,
            checks: bool = True, skip_unprocessable: bool = False) -> list[dict]:
    """Process result files: `paths` can mix .nc files, folders (all .nc inside) and glob patterns.

    Per-SPORE files (`spore_<n>.nc`, Calliope's `save_per_spore_path`) of one folder are processed together into
    one output folder named after that folder, with a `spores` column.
    With `skip_unprocessable`, a file that is not a result file or has no solution (`ResultFileError`) is skipped
    instead of stopping the run: its summary is `{"source": ..., "skipped": <reason>}`, and nothing is written for it.
    """
    cfg = load_config(config)
    if isinstance(paths, (str, Path)):
        paths = [paths]
    summaries = []
    for run in group_runs(find_result_files(paths)):
        try:
            summaries.append(process_run(run, output_dir, cfg, checks))
        except ResultFileError as err:
            if not skip_unprocessable:
                raise
            LOGGER.info(f"skipped {err}")
            summaries.append({"source": str(run.files[0]) if len(run.files) == 1 else [str(f) for f in run.files],
                              "skipped": str(err)})
    return summaries
