"""Time handling: resolution, clustering (typical days) and the mapping back to the real year.

Everything is read from the model inputs:
- `timesteps`, `timestep_resolution` (hours) and `timestep_weights`;
- for clustered runs (Calliope `config.init.time_cluster`): `datesteps` (the real days), `lookup_datestep_cluster`
  (real day -> cluster) and `timestep_cluster` (model timestep -> cluster).

Aggregations work on "wide" tables (rows: e.g. tech/region/carrier, columns: model timesteps) so they never need the
full real-year table, which is only built for hourly outputs.
"""

import numpy as np
import pandas as pd
import xarray as xr


class TimeInfo:
    def __init__(self, inputs: xr.Dataset):
        self.timesteps = pd.DatetimeIndex(inputs["timesteps"].values, name="timesteps")
        self.resolution = pd.Series(inputs["timestep_resolution"].values.astype(float), index=self.timesteps)
        if "timestep_weights" in inputs:
            self.model_weights = pd.Series(inputs["timestep_weights"].values.astype(float), index=self.timesteps)
        else:
            self.model_weights = pd.Series(1.0, index=self.timesteps)
        self.clustered = all(v in inputs for v in ("lookup_datestep_cluster", "timestep_cluster", "datesteps"))

        if self.clustered:
            self._init_clustered(inputs)
        else:
            self._init_full()
        # Weight of each model timestep in the year: number of real days it stands for (clustered) or 1.
        self.annual_weights = self.month_weights.sum(axis=1)

    # ------------------------------------------------------------------ setup
    def _init_clustered(self, inputs: xr.Dataset):
        days = pd.DatetimeIndex(inputs["datesteps"].values)
        day_cluster = pd.Series(inputs["lookup_datestep_cluster"].values, index=days).astype(int)
        ts_cluster = pd.Series(inputs["timestep_cluster"].values, index=self.timesteps).astype(int)
        # hours since the start of its (representative) day, at the start of each model timestep
        offset = self.resolution.groupby(ts_cluster).cumsum() - self.resolution
        self.cluster_of_day = day_cluster
        self.cluster_of_timestep = ts_cluster
        self.offset_hours = offset

        rows = []
        for cluster, ts in ts_cluster.groupby(ts_cluster).groups.items():
            ts = pd.DatetimeIndex(ts)
            real_days = day_cluster.index[day_cluster.values == cluster]
            for day in real_days:
                rows.append(pd.DataFrame({
                    "real": day + pd.to_timedelta(offset.loc[ts].values, unit="h"),
                    "rep": ts,
                    "day": day,
                }))
        self.expansion = pd.concat(rows).sort_values("real").reset_index(drop=True)

        months = pd.PeriodIndex(day_cluster.index, freq="M")
        days_per_cluster_month = pd.crosstab(day_cluster.values, months.astype(str))
        self.month_weights = days_per_cluster_month.reindex(ts_cluster.values).set_axis(self.timesteps).fillna(0.0)
        self.month_weights.columns.name = "month"

    def _init_full(self):
        """Not clustered: model timesteps are real (possibly resampled) timesteps; a timestep straddling two months
        is split between them in proportion to its hours."""
        self.expansion = pd.DataFrame({"real": self.timesteps, "rep": self.timesteps, "day": self.timesteps.normalize()})
        self.offset_hours = pd.Series(0.0, index=self.timesteps)
        weights = []
        for ts, res in self.resolution.items():
            n_hours = max(int(round(res)), 1)
            hours = pd.date_range(ts, periods=n_hours, freq="h")
            share = pd.Series(hours.to_period("M").astype(str)).value_counts() / n_hours
            weights.append(share.rename(ts))
        self.month_weights = pd.DataFrame(weights).fillna(0.0)
        self.month_weights.index = self.timesteps
        self.month_weights = self.month_weights.reindex(sorted(self.month_weights.columns), axis=1)
        self.month_weights.columns.name = "month"

    # ------------------------------------------------------------------ info
    def describe(self) -> dict:
        info = {
            "clustered": self.clustered,
            "model_timesteps": len(self.timesteps),
            "resolution_hours": sorted(set(self.resolution.round(6).tolist())),
            "real_hours_represented": float((self.resolution * self.annual_weights).sum()),
            "months": list(self.month_weights.columns),
        }
        if self.clustered:
            info["typical_days"] = int(self.cluster_of_day.nunique())
            info["real_days"] = len(self.cluster_of_day)
        return info

    def check_weights(self) -> float:
        """Largest difference between the model's `timestep_weights` and the weights rebuilt from the mapping."""
        return float((self.annual_weights - self.model_weights).abs().max())

    # ------------------------------------------------------------------ aggregations of wide tables
    def annual_sum(self, wide: pd.DataFrame) -> pd.Series:
        """Sum over the real year of quantities per timestep (energy). Rows that are all-NaN stay NaN."""
        out = wide.fillna(0.0) @ self.annual_weights.reindex(wide.columns).values
        return out.where(wide.notna().any(axis=1))

    def monthly_sum(self, wide: pd.DataFrame) -> pd.DataFrame:
        mw = self.month_weights.reindex(wide.columns)
        out = wide.fillna(0.0).values @ mw.values
        out = pd.DataFrame(out, index=wide.index, columns=mw.columns)
        present = wide.notna().values.astype(float) @ (mw.values > 0)
        return out.where(present > 0)

    def per_hour(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Quantity per timestep -> quantity per hour (energy -> power)."""
        return wide.div(self.resolution.reindex(wide.columns).values, axis=1)

    def annual_max(self, wide: pd.DataFrame) -> pd.Series:
        return self.per_hour(wide).max(axis=1)

    def monthly_max(self, wide: pd.DataFrame) -> pd.DataFrame:
        power = self.per_hour(wide)
        mw = self.month_weights.reindex(wide.columns)
        return pd.DataFrame({m: power.loc[:, (mw[m] > 0).values].max(axis=1) for m in mw.columns}).rename_axis(columns="month")

    def to_real_time(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Model timesteps -> real timesteps (the full year for clustered runs). Columns: real timesteps."""
        out = wide.reindex(columns=self.expansion["rep"].values)
        out.columns = pd.DatetimeIndex(self.expansion["real"].values, name="timesteps")
        return out

    def storage_to_real_time(self, intra: pd.DataFrame, inter: pd.DataFrame | None, loss: np.ndarray) -> pd.DataFrame:
        """Storage level over the real year.

        Not clustered: `intra` already is the level. Clustered (Calliope `storage_inter_cluster` math): the level is
        the level at the start of the real day (`storage_inter_cluster`, decayed by the hourly `loss`) plus the
        change within the representative day (`storage`), at the end of each timestep.
        """
        if not self.clustered or inter is None:
            return self.to_real_time(intra)
        level = self.to_real_time(intra).fillna(0.0)
        days = pd.DatetimeIndex(self.expansion["day"].values)
        start = inter.reindex(index=intra.index, columns=days).fillna(0.0).values
        hours = (self.offset_hours.reindex(self.expansion["rep"].values).values
                 + self.resolution.reindex(self.expansion["rep"].values).values)
        decay = (1.0 - np.asarray(loss, dtype=float))[:, None] ** hours[None, :]
        return level + start * decay
