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
from tidy_calliope.reader import ResultFileError, find_result_files, open_results

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


def process_file(path: str | Path, output_dir: str | Path, config: str | Path | dict | None = None,
                 scenario: str | None = None, checks: bool = True, *, loaded_config: dict | None = None) -> dict:
    """Process one result file into `output_dir/<scenario>/`. Returns the run summary (also written as JSON).

    `config`: YAML path or dict with the settings to change; `loaded_config`: an already merged configuration.
    """
    cfg = loaded_config if loaded_config is not None else load_config(config)
    t0 = time.time()
    model = open_results(path)
    if scenario:
        model.scenario = scenario
    folder = Path(output_dir) / model.scenario
    folder.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"{model.scenario}: {path} -> {folder} ({model.time.describe()})")

    proc, tables = compute_tables(model, cfg)
    hourly_tables = {"flow_out", "flow_in", "storage_level", "net_import", "shadow_price_system_balance"}
    value_column = {"shadow_price_system_balance": "value", "shadow_price_constraints": "value"}
    written = {}
    for name, series in tables.items():
        hourly = name in hourly_tables
        fmt = cfg["hourly_format"] if hourly else cfg["format"]
        wide = hourly and cfg["hourly_layout"] == "wide"
        p = write_table(series, name, folder, model.scenario, cfg["scenario_dim_name"], fmt, wide=wide,
                        value_name=value_column.get(name))
        written[name] = {"file": p.name, "rows": int(len(series))}

    summary = {
        "scenario": model.scenario,
        "source": str(Path(path).resolve()),
        "created": datetime.now().isoformat(timespec="seconds"),
        "tidy_calliope_version": __version__,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "time": model.time.describe(),
        "tables": written,
        "runtime_seconds": None,
        "config": cfg,
    }
    if checks:
        summary["checks"] = run_checks(proc, tables)
        failed = [k for k, v in summary["checks"].items() if not v["ok"]]
        summary["checks_failed"] = failed
        (LOGGER.warning if failed else LOGGER.info)(
            f"{model.scenario}: checks {'FAILED: ' + ', '.join(failed) if failed else 'all passed'}")
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(folder / "metadata.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    model.close()
    return summary


def process(paths, output_dir: str | Path = "tidy-results", config: str | Path | dict | None = None,
            checks: bool = True, skip_unprocessable: bool = False) -> list[dict]:
    """Process result files: `paths` can mix .nc files, folders (all .nc inside) and glob patterns.

    With `skip_unprocessable`, a file that is not a result file or has no solution (`ResultFileError`) is skipped
    instead of stopping the run: its summary is `{"source": ..., "skipped": <reason>}`, and nothing is written for it.
    """
    cfg = load_config(config)
    if isinstance(paths, (str, Path)):
        paths = [paths]
    summaries = []
    for f in find_result_files(paths):
        try:
            summaries.append(process_file(f, output_dir, checks=checks, loaded_config=cfg))
        except ResultFileError as err:
            if not skip_unprocessable:
                raise
            LOGGER.info(f"skipped {err}")
            summaries.append({"source": str(f), "skipped": str(err)})
    return summaries
