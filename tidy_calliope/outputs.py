"""The output tables. Each function returns {table name: Series}, Series indexed by the table's dimensions.

Conventions:
- flows/storage per timestep are energy (unit by carrier, e.g. twh); `*_max` tables are power (e.g. tw);
- `*_sum` / `*_1M` sums are over the real year / month: for clustered runs each model timestep counts as many times
  as the real days it represents (`TimeInfo`);
- Calliope's `cost_operation_variable` already includes the timestep weights, so it is summed without weights.
"""

import numpy as np
import pandas as pd
import xarray as xr

from tidy_calliope.naming import Naming
from tidy_calliope.reader import ModelResults, to_long


def _long(model: ModelResults, var: str, cfg: dict, techs=None) -> pd.Series:
    if var not in model.results:
        return pd.Series(dtype=float)
    return to_long(model.results[var], cfg["zero_threshold"], cfg["read_chunk_size"], techs=techs)


def _wide(series: pd.Series) -> pd.DataFrame:
    """Long (..., timesteps) -> wide: rows = other levels, columns = model timesteps."""
    if series.empty:
        return pd.DataFrame()
    return series.unstack("timesteps")


def _stack_months(df: pd.DataFrame) -> pd.Series:
    return df.rename_axis(columns="month").stack().dropna()


def _stack_time(df: pd.DataFrame) -> pd.Series:
    return df.stack().dropna()


class Processor:
    def __init__(self, model: ModelResults, cfg: dict):
        self.model = model
        self.cfg = cfg
        self.time = model.time
        self.naming = Naming(cfg, model.inputs)
        self.non_transmission = [t for t, new in self.naming.tech.items() if new is not None]
        self._flow_wide: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ flows
    def flow_wide(self, direction: str) -> pd.DataFrame:
        """flow_out / flow_in of non-transmission techs, relabelled, as wide table (rows x model timesteps)."""
        if direction not in self._flow_wide:
            long = _long(self.model, f"flow_{direction}", self.cfg, techs=self.non_transmission)
            self._flow_wide[direction] = _wide(self.naming.relabel(long))
        return self._flow_wide[direction]

    def flows(self, annual: bool, monthly: bool, hourly: bool) -> dict:
        out = {}
        for direction in ("out", "in"):
            wide = self.flow_wide(direction)
            if wide.empty:
                continue
            name = f"flow_{direction}"
            power_index = self._power_index(wide.index)
            if annual:
                out[f"{name}_sum"] = self.time.annual_sum(wide).dropna()
                out[f"{name}_max"] = self.time.annual_max(wide).set_axis(power_index).dropna()
            if monthly:
                out[f"{name}_sum_1M"] = _stack_months(self.time.monthly_sum(wide))
                out[f"{name}_max_1M"] = _stack_months(self.time.monthly_max(wide).set_axis(power_index))
            if hourly:
                out[name] = _stack_time(self.time.to_real_time(wide))
        return out

    def _power_index(self, index: pd.MultiIndex) -> pd.MultiIndex:
        """Same index with energy unit labels replaced by power labels."""
        df = index.to_frame(index=False)
        df["unit"] = df["unit"].map(self.naming.power_unit)
        return pd.MultiIndex.from_frame(df)

    # ------------------------------------------------------------------ capacities and storage
    def capacities(self) -> dict:
        out = {}
        caps = _long(self.model, "flow_cap", self.cfg, techs=self.non_transmission)
        caps = self.naming.relabel(caps, power=True)
        if not caps.empty:
            keep = [not self.naming.excluded_from_capacities(t) for t in caps.index.get_level_values("techs")]
            caps = caps[keep]
            caps = self._add_flow_direction(caps)
            out["nameplate_capacity"] = caps
        storage_cap = self._storage_with_carrier(_long(self.model, "storage_cap", self.cfg, techs=self.non_transmission))
        if not storage_cap.empty:
            out["storage_capacity"] = self.naming.relabel(storage_cap)
        return out

    def _add_flow_direction(self, caps: pd.Series) -> pd.Series:
        """Add a `flow` level: in, out or both, from the carrier_in / carrier_out of the (renamed) tech and carrier."""
        inputs = self.model.inputs
        flags = {}
        for direction in ("in", "out"):
            da = inputs[f"carrier_{direction}"].fillna(0).astype(bool).any("nodes")
            s = da.to_series()
            s = s[s]
            for tech, carrier in s.index:
                key = (self.naming.tech.get(tech), self.naming.carrier.get(str(carrier)))
                flags.setdefault(key, set()).add(direction)
        df = caps.rename("value").reset_index()
        df["flow"] = [
            "both" if len(flags.get((t, c), ())) == 2 else next(iter(flags.get((t, c), {"n.a."})))
            for t, c in zip(df["techs"], df["carriers"])
        ]
        return df.set_index([c for c in df.columns if c != "value"])["value"]

    def _storage_with_carrier(self, series: pd.Series) -> pd.Series:
        """Add the stored carrier (the storage tech's output carrier) to a (nodes, techs[, timesteps]) Series."""
        if series.empty:
            return series
        carrier_out = self.model.inputs["carrier_out"].fillna(0).astype(bool).any("nodes").to_series()
        carrier_of = carrier_out[carrier_out].reset_index().groupby("techs")["carriers"].first()
        df = series.rename("value").reset_index()
        df["carriers"] = df["techs"].map(carrier_of)
        return df.set_index([c for c in df.columns if c != "value"])["value"]

    def storage_level(self) -> dict:
        intra = _long(self.model, "storage", self.cfg, techs=self.non_transmission)
        if intra.empty:
            return {}
        intra_wide = intra.unstack("timesteps")
        inter_wide = None
        if self.time.clustered and "storage_inter_cluster" in self.model.results:
            inter = to_long(self.model.results["storage_inter_cluster"], 0.0, self.cfg["read_chunk_size"])
            inter_wide = inter.unstack("datesteps").reindex(intra_wide.index)
            inter_wide.columns = pd.DatetimeIndex(inter_wide.columns)
        loss = self.model.inputs["storage_loss"].to_series() if "storage_loss" in self.model.inputs else pd.Series(dtype=float)
        tech_loss = intra_wide.index.get_level_values("techs").map(loss).fillna(0.0).values
        level = self.time.storage_to_real_time(intra_wide, inter_wide, tech_loss)
        # for the checks: the rebuilt level must stay within [0, storage_cap] (model units)
        cap = _long(self.model, "storage_cap", {**self.cfg, "zero_threshold": 0.0}).reindex(level.index).fillna(0.0)
        self.storage_bounds = {
            "min_level": float(level.min().min()),
            "max_above_capacity": float(level.max(axis=1).sub(cap).max()),
            "largest_capacity": float(cap.max()),
        }
        long = self._storage_with_carrier(_stack_time(level))
        return {"storage_level": self.naming.relabel(long)}

    # ------------------------------------------------------------------ costs
    def costs(self) -> dict:
        out = {}
        results = self.model.results
        for cost_class, spec in self.cfg["cost_classes"].items():
            if "costs" not in results.coords or cost_class not in results["costs"].values:
                continue
            scale = float(spec.get("scale", self.cfg["units"]["cost"]["scale"]))
            names = spec["names"]
            parts = {}
            for var in ("cost", "cost_investment_annualised", "cost_operation_fixed", "cost_operation_variable"):
                if var not in results:
                    continue
                da = results[var].sel(costs=cost_class).sel(techs=self.non_transmission)
                if "timesteps" in da.dims:  # already weighted by Calliope
                    da = da.sum("timesteps", min_count=1)
                parts[var] = self.naming.relabel(to_long(da, self.cfg["zero_threshold"]), unit=spec["unit"], scale=scale)
            for var, series in parts.items():
                if var in names and not series.empty:
                    out[names[var]] = series
            investment = [parts[v] for v in ("cost_investment_annualised", "cost_operation_fixed")
                          if v in parts and not parts[v].empty]
            if "total_investment" in names and investment:
                total = pd.concat(investment)
                out[names["total_investment"]] = total.groupby(level=list(total.index.names)).sum()
        return out

    def input_costs(self) -> dict:
        out = {}
        inputs = self.model.inputs
        per_unit = float(self.cfg["units"]["input_cost_per_unit_scale"])
        energy = self.cfg["units"]["energy"]["default"]
        power = self.naming.power_unit(energy)
        for cost_class, spec in self.cfg["cost_classes"].items():
            for var, name in spec.get("inputs", {}).items():
                if var not in inputs or "costs" not in inputs[var].dims or cost_class not in inputs[var]["costs"].values:
                    continue
                da = inputs[var].sel(costs=cost_class)
                if "techs" in da.dims:  # model-wide parameters (no techs, e.g. cost_interest_rate) are kept as is
                    da = da.sel(techs=[t for t in da.indexes["techs"] if t in set(self.non_transmission)])
                if var in ("cost_om_annual_investment_fraction", "cost_interest_rate"):
                    unit, scale = ("fraction_of_total_investment" if "fraction" in var else "fraction"), 1.0
                elif var == "cost_om_annual":
                    unit, scale = f"{spec['unit']}_per_{power}_per_year", per_unit
                elif var.endswith("_cap"):
                    unit, scale = f"{spec['unit']}_per_{power}", per_unit
                else:
                    unit, scale = f"{spec['unit']}_per_{energy}", per_unit
                if not da.dims:  # model-wide parameter: one value, reported with its unit only
                    value = float(da.values)
                    if not np.isnan(value):
                        out[name] = pd.Series([value * scale], index=pd.Index([unit], name="unit"))
                    continue
                # tech-level parameters (no `nodes` dimension) are reported without a region
                relabelled = self.naming.relabel(to_long(da, 0.0), how="mean", unit=unit, scale=scale)
                if not relabelled.empty:
                    out[name] = relabelled
        return out

    # ------------------------------------------------------------------ transmission
    def transmission(self, annual: bool, monthly: bool, hourly: bool) -> dict:
        links = self.naming.links
        if links.empty:
            return {}
        out = {}
        link_techs = list(links.index)
        # capacity: the same flow_cap is defined at both ends; reported in both directions
        cap = _long(self.model, "flow_cap", self.cfg, techs=link_techs).rename("value").reset_index()
        cap = cap.merge(links, left_on="techs", right_index=True)
        cap = cap[cap["nodes"] == cap["from_node"]]
        cap_unit = {c: self.naming.energy_unit(self.naming.carrier[c]) for c in cap["carriers"].unique()}
        cap["unit"] = cap["carriers"].map(lambda c: self.naming.power_unit(cap_unit[c][0]))
        cap["value"] *= cap["carriers"].map(lambda c: cap_unit[c][1])
        cap["carriers"] = cap["carriers"].map(self.naming.carrier)
        both = pd.concat([
            cap.rename(columns={"from_region": "exporting_region", "to_region": "importing_region"}),
            cap.rename(columns={"to_region": "exporting_region", "from_region": "importing_region"}),
        ])
        both = both[both["exporting_region"] != both["importing_region"]]
        out["net_transfer_capacity"] = both.groupby(
            ["type", "exporting_region", "importing_region", "carriers", "unit"])["value"].sum().rename_axis(
            index={"type": "techs"})

        # net import into `importing_region` from `exporting_region` per timestep
        net = self._net_import_wide(link_techs)
        if not net.empty:
            power_index = self._power_index(net.index)
            if annual:
                out["net_import_sum"] = self.time.annual_sum(net).dropna()
                out["net_import_max"] = self.time.annual_max(net).set_axis(power_index).dropna()
            if monthly:
                out["net_import_sum_1M"] = _stack_months(self.time.monthly_sum(net))
                out["net_import_max_1M"] = _stack_months(self.time.monthly_max(net).set_axis(power_index))
            if hourly:
                out["net_import"] = _stack_time(self.time.to_real_time(net))

        # costs per link (defined at both ends), per cost class
        for cost_class, spec in self.cfg["cost_classes"].items():
            name = spec["names"].get("transmission_cost")
            if name is None or "cost" not in self.model.results or cost_class not in self.model.results["costs"].values:
                continue
            cost = to_long(self.model.results["cost"].sel(costs=cost_class).sel(techs=link_techs), self.cfg["zero_threshold"])
            if cost.empty:
                continue
            df = cost.rename("value").reset_index().merge(links, left_on="techs", right_index=True)
            df = df[df["from_region"] != df["to_region"]]
            df["value"] *= float(spec.get("scale", self.cfg["units"]["cost"]["scale"]))
            df["unit"] = spec["unit"]
            out[name] = df.groupby(["type", "from_region", "to_region", "unit"])["value"].sum().rename_axis(
                index={"type": "techs", "from_region": "exporting_region", "to_region": "importing_region"})
        return out

    def _net_import_wide(self, link_techs: list[str]) -> pd.DataFrame:
        links = self.naming.links
        frames = []
        for direction in ("out", "in"):
            s = _long(self.model, f"flow_{direction}", self.cfg, techs=link_techs)
            if s.empty:
                continue
            df = s.rename("value").reset_index().merge(links, left_on="techs", right_index=True)
            # transfer A -> B (A = from, B = to) is sent at A (flow_in) and received at B (flow_out)
            at_from = df["nodes"] == df["from_node"]
            sending_from = at_from if direction == "in" else ~at_from
            df["sender"] = np.where(sending_from, df["from_region"], df["to_region"])
            df["receiver"] = np.where(sending_from, df["to_region"], df["from_region"])
            df["side"] = direction
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
        df = df[df["sender"] != df["receiver"]]
        # average of sent and received (averages the losses), per link and direction
        transfer = df.pivot_table(index=["techs", "sender", "receiver", "carriers", "timesteps"], columns="side",
                                  values="value", aggfunc="sum", fill_value=0.0)
        transfer = transfer.reindex(columns=["in", "out"], fill_value=0.0).mean(axis=1)
        transfer = transfer.groupby(level=["sender", "receiver", "carriers", "timesteps"]).sum()
        reverse = transfer.copy()
        reverse.index = reverse.index.set_names(["receiver", "sender", "carriers", "timesteps"])
        reverse = reverse.reorder_levels(["sender", "receiver", "carriers", "timesteps"])
        net = transfer.sub(reverse, fill_value=0.0)  # into receiver from sender, minus the opposite direction
        net = net.rename_axis(index={"sender": "exporting_region", "receiver": "importing_region"}).rename("value")
        df = net.reset_index()
        units = {c: self.naming.energy_unit(self.naming.carrier[c]) for c in df["carriers"].unique()}
        df["unit"] = df["carriers"].map(lambda c: units[c][0])
        df["value"] *= df["carriers"].map(lambda c: units[c][1])
        df["carriers"] = df["carriers"].map(self.naming.carrier)
        long = df.groupby(["exporting_region", "importing_region", "carriers", "unit", "timesteps"])["value"].sum()
        return long.unstack("timesteps")

    # ------------------------------------------------------------------ derived
    def primary_energy_supply(self, flow_out_sum: pd.Series, flow_in_sum: pd.Series, net_import_sum: pd.Series | None) -> dict:
        spec = self.cfg["primary_energy_supply"]

        def _by_category(series: pd.Series, mapping: dict) -> pd.Series:
            if series is None or series.empty:
                return pd.Series(dtype=float)
            df = series.rename("value").reset_index()
            df["category"] = df["techs"].map(mapping)
            df = df[df["category"].notna()]
            return df.groupby(["locs", "category", "unit"])["value"].sum()

        supply = _by_category(flow_out_sum, spec["producers"])
        exports = _by_category(flow_in_sum, spec["exports"])
        total = supply.sub(exports, fill_value=0.0)
        if net_import_sum is not None and not net_import_sum.empty:
            df = net_import_sum.rename("value").reset_index()
            df["category"] = df["carriers"].map(spec["net_import_carriers"])
            df = df[df["category"].notna()]
            imports = df.rename(columns={"importing_region": "locs"}).groupby(["locs", "category", "unit"])["value"].sum()
            total = total.add(imports, fill_value=0.0)
        total = total.rename_axis(index={"category": "carriers"})
        return {"primary_energy_supply": total.sort_index()}

    def names(self) -> dict:
        names = self.naming.tech_names(self.model.inputs)
        return {"names": names} if not names.empty else {}

    # ------------------------------------------------------------------ shadow prices
    def objective_unit(self) -> tuple[str, float]:
        """(unit label, scale) of the objective: the cost class with a non-zero objective weight, if only one."""
        inputs = self.model.inputs
        if "objective_cost_weights" in inputs:
            weights = inputs["objective_cost_weights"].to_series()
            classes = [str(c) for c, w in weights.items() if w != 0]
        else:
            classes = [str(c) for c in self.model.results["costs"].values] if "costs" in self.model.results.coords else []
        if len(classes) == 1 and classes[0] in self.cfg["cost_classes"]:
            spec = self.cfg["cost_classes"][classes[0]]
            return spec["unit"], float(spec.get("scale", self.cfg["units"]["cost"]["scale"]))
        return "objective_unit", 1.0

    def shadow_prices(self) -> dict:
        """Energy balance prices per node, carrier and timestep, and the other tracked shadow prices.

        Calliope weights each model timestep in the objective by `timestep_weights` (the real days a typical day
        stands for), so its dual of the balance is weight x price: it is divided by the weights. For clustered runs
        the prices of the typical days are then repeated for every real day they stand for (`shadow_prices.full_year`).
        Prices are per unit of energy of the carrier (model carrier names: carriers are not merged, prices cannot be
        summed) and use the nodes of the model (regions are not aggregated).
        """
        results = self.model.results
        out = {}
        if "shadow_price_system_balance" in results:
            long = to_long(results["shadow_price_system_balance"], 0.0)  # zero prices are kept
            if not long.empty:
                wide = long.unstack("timesteps")
                wide = wide.div(self.time.model_weights.reindex(wide.columns).values, axis=1)
                if self.cfg["shadow_prices"]["full_year"]:
                    wide = self.time.to_real_time(wide)
                cost_unit, cost_scale = self.objective_unit()
                df = wide.stack().rename("value").reset_index()
                units = {c: self.naming.energy_unit(self.naming.carrier.get(str(c), str(c))) for c in df["carriers"].unique()}
                df["unit"] = df["carriers"].map(lambda c: f"{cost_unit}_per_{units[c][0]}")
                df["value"] *= df["carriers"].map(lambda c: cost_scale / units[c][1])
                out["shadow_price_system_balance"] = df.set_index(["nodes", "carriers", "unit", "timesteps"])["value"]

        others = []
        for var in sorted(v for v in results.data_vars if v.startswith("shadow_price_") and v != "shadow_price_system_balance"):
            da = results[var]
            if "timesteps" in da.dims:  # per-timestep constraint: price = dual / weight, as for the balance
                da = da / xr.DataArray(self.time.model_weights.values, coords={"timesteps": da["timesteps"].values},
                                       dims="timesteps")
            if not da.dims:  # one value for the whole constraint
                value = float(da.values)
                if np.isnan(value):
                    continue
                s, index = pd.Series([value]), [""]
            else:
                s = to_long(da, 0.0)
                if s.empty:
                    continue
                index = [", ".join(f"{d}={v}" for d, v in zip(s.index.names, i if isinstance(i, tuple) else (i,)))
                         for i in s.index]
            others.append(pd.DataFrame({"constraint": var.removeprefix("shadow_price_"), "index": index,
                                        "value": s.values}))
        if others:
            out["shadow_price_constraints"] = pd.concat(others).set_index(["constraint", "index"])["value"]
        return out
