"""From model labels (nodes, techs, carriers) to reported labels (locs, techs, carriers, unit)."""

import re

import numpy as np
import pandas as pd
import xarray as xr

INDEX = ["techs", "locs", "carriers", "unit"]


def _apply_regex_renames(name: str, rules: dict) -> str:
    for pattern, replacement in rules.items():
        new = re.sub(pattern, replacement, name)
        if new != name:
            return new
    return name


class Naming:
    def __init__(self, cfg: dict, inputs: xr.Dataset):
        self.cfg = cfg
        base_tech = inputs["base_tech"].to_series()
        self.transmission_techs = set(base_tech.index[base_tech == "transmission"])
        self.base_tech = base_tech

        # regions
        nodes = [str(n) for n in inputs["nodes"].values]
        regions = cfg["regions"]
        if regions == "subcountries":
            self.region = {n: n for n in nodes}
        elif regions == "countries":
            self.region = {n: n.split("_")[0] for n in nodes}
        elif isinstance(regions, dict):
            missing = sorted(set(nodes) - set(regions))
            if missing:
                raise KeyError(f"`regions` mapping misses nodes {missing}")
            self.region = {n: regions[n] for n in nodes}
        else:
            raise ValueError(f"`regions` must be subcountries, countries or a mapping, not {regions!r}")

        # techs (non-transmission): dropped ones map to None
        drop = [re.compile(p) for p in cfg["techs"]["drop"]]
        self.tech = {}
        for t in base_tech.index:
            if t in self.transmission_techs or any(p.search(t) for p in drop):
                self.tech[t] = None
            else:
                self.tech[t] = _apply_regex_renames(t, cfg["techs"]["rename"])
        self._no_capacity = [re.compile(p) for p in cfg["techs"]["exclude_from_capacities"]]

        self.carrier = {str(c): _apply_regex_renames(str(c), cfg["carriers"]["rename"]) for c in inputs["carriers"].values}

        # transmission links
        if self.transmission_techs:
            links = pd.DataFrame({
                "from_node": inputs["link_from"].to_series(),
                "to_node": inputs["link_to"].to_series(),
            }).loc[sorted(self.transmission_techs)]
            prefix = links["from_node"] + "_to_" + links["to_node"] + "_"
            link_type = [t[len(p):] if t.startswith(p) else t for t, p in zip(links.index, prefix)]
            links["type"] = [_apply_regex_renames(t, cfg["transmission"]["rename"]) for t in link_type]
            links["from_region"] = links["from_node"].map(self.region)
            links["to_region"] = links["to_node"].map(self.region)
            self.links = links
        else:
            self.links = pd.DataFrame(columns=["from_node", "to_node", "type", "from_region", "to_region"])

        energy = cfg["units"]["energy"]
        self._energy_rules = [(re.compile(p), v) for p, v in energy["by_carrier"].items()]

    # ------------------------------------------------------------------ units
    def energy_unit(self, carrier: str) -> tuple[str, float]:
        """(unit label, scale) of an energy quantity of a (renamed) carrier."""
        energy = self.cfg["units"]["energy"]
        for pattern, rule in self._energy_rules:
            if pattern.search(carrier):
                if isinstance(rule, dict):
                    return rule["unit"], float(rule.get("scale", energy["scale"]))
                return rule, float(energy["scale"])
        return energy["default"], float(energy["scale"])

    def power_unit(self, energy_unit: str) -> str:
        power = self.cfg["units"]["power"]
        return power.get(energy_unit, energy_unit + power["suffix"])

    def excluded_from_capacities(self, tech: str) -> bool:
        return any(p.search(tech) for p in self._no_capacity)

    # ------------------------------------------------------------------ relabelling of long Series
    def relabel(self, series: pd.Series, how: str = "sum", power: bool = False, unit: str | None = None,
                scale: float | None = None) -> pd.Series:
        """Map a long Series indexed by (nodes, techs[, carriers], *other) to (techs, locs[, carriers], unit, *other).

        Values are scaled to the output unit: by carrier (energy, or power if `power`), or with the given
        `unit`/`scale` (e.g. costs). Labels that map together are combined with `how` (sum or mean).
        Transmission and dropped techs are removed.
        """
        if series.empty:
            return series
        df = series.rename("value").reset_index()
        if "techs" in df:
            df["techs"] = df["techs"].map(self.tech)
            df = df[df["techs"].notna()]
        if "nodes" in df:
            df["locs"] = df.pop("nodes").map(self.region)
        if "carriers" in df:
            df["carriers"] = df["carriers"].map(self.carrier)
            units = {c: self.energy_unit(c) for c in df["carriers"].unique()}
            df["unit"] = df["carriers"].map({c: u for c, (u, _) in units.items()})
            if power:
                df["unit"] = df["unit"].map(self.power_unit)
            if scale is None:
                df["value"] *= df["carriers"].map({c: s for c, (_, s) in units.items()})
        if unit is not None:
            df["unit"] = unit
        if scale is not None:
            df["value"] *= scale
        levels = [c for c in INDEX if c in df] + [c for c in df.columns if c not in INDEX + ["value"]]
        grouped = df.groupby(levels, sort=True, observed=True)["value"]
        return grouped.sum() if how == "sum" else grouped.mean()

    def tech_names(self, inputs: xr.Dataset) -> pd.Series:
        if "name" not in inputs:
            return pd.Series(dtype=str)
        names = inputs["name"].to_series().dropna()
        names.index = names.index.map(self.tech)
        names = names[names.index.notna()]
        return names.groupby(level=0).first().rename_axis("techs").rename("names")
