"""Consistency checks of the processed tables against the model's own data.

Each check returns {"ok": bool, ...details}. They catch errors in the time rebuilding (weights, months, hours),
in the relabelling (lost or double-counted values) and in the storage rebuilding of clustered runs.
"""

import pandas as pd
import xarray as xr

from tidy_calliope.outputs import Processor

REL_TOL = 1e-6


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


def _row_sums(table: pd.Series, drop: str) -> pd.Series:
    levels = [n for n in table.index.names if n != drop]
    return table.groupby(level=levels).sum()


def run_checks(proc: Processor, tables: dict) -> dict:
    checks = {}
    time = proc.time
    info = time.describe()

    checks["timestep_weights"] = {"ok": time.check_weights() < 1e-9, "max_difference": time.check_weights()}
    hours = info["real_hours_represented"]
    checks["hours_in_year"] = {"ok": hours in (8760.0, 8784.0), "hours": hours}

    # monthly and hourly tables add up to the annual ones
    for annual_name in [n for n in tables if n.endswith("_sum")]:
        annual = tables[annual_name]
        for other_name, level in [(annual_name + "_1M", "month"), (annual_name[: -len("_sum")], "timesteps")]:
            if other_name not in tables:
                continue
            summed = _row_sums(tables[other_name], level).reindex(annual.index).fillna(0.0)
            diff = (summed - annual).abs()
            worst = float((diff / annual.abs().clip(lower=1e-9)).max()) if len(annual) else 0.0
            checks[f"{other_name}_adds_up_to_{annual_name}"] = {
                "ok": worst < 1e-6 or float(diff.max()) < 1e-9, "max_relative_difference": worst}

    # demand techs: annual inflow equals the demand input (model units, before any relabelling)
    inputs, results = proc.model.inputs, proc.model.results
    if "sink_use_equals" in inputs and "flow_in" in results:
        w = xr.DataArray(time.model_weights.values, coords={"timesteps": inputs["timesteps"].values}, dims="timesteps")
        demand = inputs["sink_use_equals"].fillna(0)
        demand_total = float((demand * w).sum())
        techs = [t for t in demand["techs"].values if float(demand.sel(techs=t).sum()) > 0]
        flow_total = float((results["flow_in"].sel(techs=techs).fillna(0) * w).sum())
        checks["demand_met"] = {"ok": _rel(flow_total, demand_total) < 1e-4, "flow_in_of_demand_techs": flow_total,
                                "sink_use_equals": demand_total, "relative_difference": _rel(flow_total, demand_total)}

    # costs: the sum of all techs' costs is the objective (cost classes weighted as in the model)
    if "cost" in results and "min_cost_optimisation" in results:
        weights = inputs["objective_cost_weights"].to_series() if "objective_cost_weights" in inputs else pd.Series(1.0, index=results["costs"].values)
        total = float(sum(float(results["cost"].sel(costs=c).sum()) * float(w) for c, w in weights.items() if c in results["costs"].values))
        objective = float(results["min_cost_optimisation"])
        checks["costs_sum_to_objective"] = {"ok": _rel(total, objective) < 1e-6, "weighted_cost_sum": total, "objective": objective}

    # every reported cost: (reported techs + transmission links) == model total, apart from dropped techs
    for cost_class, spec in proc.cfg["cost_classes"].items():
        name = spec["names"].get("cost")
        if name not in tables or "cost" not in results or cost_class not in results["costs"].values:
            continue
        scale = float(spec.get("scale", proc.cfg["units"]["cost"]["scale"]))
        reported = float(tables[name].sum())
        reported += float(tables.get(spec["names"].get("transmission_cost"), pd.Series(dtype=float)).sum())
        dropped = [t for t, new in proc.naming.tech.items() if new is None and t not in proc.naming.transmission_techs]
        model_total = float(results["cost"].sel(costs=cost_class).sum()) * scale
        dropped_total = float(results["cost"].sel(costs=cost_class, techs=dropped).sum()) * scale if dropped else 0.0
        intra_region_links = proc.naming.links.index[proc.naming.links["from_region"] == proc.naming.links["to_region"]]
        intra_total = float(results["cost"].sel(costs=cost_class, techs=list(intra_region_links)).sum()) * scale if len(intra_region_links) else 0.0
        expected = model_total - dropped_total - intra_total
        checks[f"{name}_complete"] = {"ok": _rel(reported, expected) < 1e-6, "reported": reported, "expected": expected,
                                      "dropped_techs_total": dropped_total, "intra_region_links_total": intra_total}

    # transmission: net imports are antisymmetric (A from B = - B from A)
    if "net_import_sum" in tables:
        s = tables["net_import_sum"]
        checks["net_import_antisymmetric"] = {"ok": abs(float(s.sum())) < 1e-6 * max(float(s.abs().sum()), 1.0),
                                              "sum_over_all_pairs": float(s.sum())}

    # storage rebuilt over the real year stays within [0, storage_cap]
    bounds = getattr(proc, "storage_bounds", None)
    if bounds is not None:
        tol = 1e-5 * max(bounds["largest_capacity"], 1.0)
        checks["storage_level_within_bounds"] = {
            "ok": bounds["min_level"] >= -tol and bounds["max_above_capacity"] <= tol, **bounds, "tolerance": tol}
    return checks
