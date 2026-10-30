"""Small synthetic Calliope 0.7 result files (same layout as `calliope.Model.to_netcdf`).

Two nodes (A_1, B_1), one carrier (electricity), a PV supply, a demand, a battery and one transmission link from A_1
to B_1 (10 % losses). Built so that the library's checks must pass: demand inflow = `sink_use_equals`, costs sum to
the objective, the storage level stays within [0, storage_cap].
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

NODES = ["A_1", "B_1"]
LINK = "A_1_to_B_1_ac_ohl_transmission"
TECHS = ["pv", "demand_elec", "battery", LINK, "syn_diesel_converter"]
CARRIERS = ["electricity"]
COSTS = ["monetary"]

DEMAND = {"A_1": 1.0, "B_1": 2.0}  # per hour, times the daily shape below
EXPORT = 0.5  # sent from A_1 to B_1 per hour


def price(node: str, timesteps: pd.DatetimeIndex) -> np.ndarray:
    """Electricity price per model timestep (model units): the solver's dual divided by the timestep weights."""
    day = timesteps.day_of_year.values - 1  # differs between the typical days of the clustered file
    return (0.002 + 0.001 * timesteps.hour.values / 24 + 0.0005 * day) * (1 if node == "A_1" else 2)


def _daily_shape(day_type: int) -> np.ndarray:
    hours = np.arange(24)
    return 1.0 + 0.5 * np.sin(2 * np.pi * (hours + 6 * day_type) / 24)


def make_model(timesteps: pd.DatetimeIndex, resolution: np.ndarray, weights: np.ndarray,
               profile: np.ndarray, clustered: dict | None = None) -> xr.DataTree:
    nt = len(timesteps)
    shape = (len(NODES), len(TECHS), len(CARRIERS), nt)
    flow_out = np.full(shape, np.nan)
    flow_in = np.full(shape, np.nan)
    sink = np.full((len(TECHS), nt, len(NODES)), np.nan)
    t = {name: i for i, name in enumerate(TECHS)}
    for n, node in enumerate(NODES):
        demand = DEMAND[node] * profile * resolution
        sink[t["demand_elec"], :, n] = demand
        flow_in[n, t["demand_elec"], 0] = demand
        flow_out[n, t["battery"], 0] = 0.0
        flow_in[n, t["battery"], 0] = 0.0
    sent = EXPORT * resolution
    flow_in[0, t[LINK], 0] = sent  # sent at A_1
    flow_out[1, t[LINK], 0] = 0.9 * sent  # received at B_1
    flow_out[0, t[LINK], 0] = 0.0
    flow_in[1, t[LINK], 0] = 0.0
    flow_out[0, t["pv"], 0] = flow_in[0, t["demand_elec"], 0] + sent
    flow_out[1, t["pv"], 0] = flow_in[1, t["demand_elec"], 0] - 0.9 * sent

    coords = {"nodes": NODES, "techs": TECHS, "carriers": CARRIERS, "costs": COSTS, "timesteps": timesteps}
    flow_cap = np.full(shape[:3], np.nan)
    flow_cap[:, t["pv"], 0] = [10.0, 10.0]
    flow_cap[:, t["battery"], 0] = [2.0, 2.0]
    flow_cap[:, t[LINK], 0] = [1.0, 1.0]
    storage_cap = np.full(shape[:2], np.nan)
    storage_cap[:, t["battery"]] = [20.0, 20.0]

    cost_annualised = np.full((2, len(TECHS), 1), np.nan)
    cost_annualised[:, t["pv"], 0] = [3.0, 4.0]
    cost_annualised[:, t["battery"], 0] = [1.0, 1.0]
    cost_annualised[:, t[LINK], 0] = [0.5, 0.5]
    cost_annualised[0, t["syn_diesel_converter"], 0] = 0.25  # dropped tech: reported as such by the checks
    cost_fixed = np.where(np.isnan(cost_annualised), np.nan, 0.1)
    cost_variable = np.full((2, len(TECHS), 1, nt), np.nan)
    cost_variable[:, t["pv"], 0, :] = 0.01 * weights  # Calliope's variable cost already includes the weights
    total = np.nan_to_num(cost_annualised) + np.nan_to_num(cost_fixed) + np.nansum(np.nan_to_num(cost_variable), axis=-1)
    cost = np.where(np.isnan(cost_annualised), np.nan, total)

    storage = np.full((2, len(TECHS), nt), np.nan)
    within_day = 2.0 * np.sin(np.pi * ((np.arange(nt) % 24) + 1) / 24) ** 2 if clustered else np.full(nt, 5.0)
    storage[:, t["battery"], :] = within_day

    results = xr.Dataset(
        {
            "flow_out": (("nodes", "techs", "carriers", "timesteps"), flow_out),
            "flow_in": (("nodes", "techs", "carriers", "timesteps"), flow_in),
            "flow_cap": (("nodes", "techs", "carriers"), flow_cap),
            "storage_cap": (("nodes", "techs"), storage_cap),
            "storage": (("nodes", "techs", "timesteps"), storage),
            "cost": (("nodes", "techs", "costs"), cost),
            "cost_investment_annualised": (("nodes", "techs", "costs"), cost_annualised),
            "cost_operation_fixed": (("nodes", "techs", "costs"), cost_fixed),
            "cost_operation_variable": (("nodes", "techs", "costs", "timesteps"), cost_variable),
            "min_cost_optimisation": ((), float(np.nansum(cost))),
            # Calliope's dual of the balance is weight x price
            "shadow_price_system_balance": (("nodes", "carriers", "timesteps"),
                                            np.stack([price(n, timesteps) * weights for n in NODES])[:, None, :]),
            "shadow_price_group_co2_cap": (("costs",), np.array([-0.5])),
        },
        coords={k: v for k, v in coords.items()},
    )

    is_in = np.zeros(shape[:3], dtype=np.int8)
    is_out = np.zeros(shape[:3], dtype=np.int8)
    is_out[:, t["pv"], 0] = 1
    is_in[:, t["demand_elec"], 0] = 1
    is_in[:, t["battery"], 0] = is_out[:, t["battery"], 0] = 1
    is_in[:, t[LINK], 0] = is_out[:, t[LINK], 0] = 1
    inputs = xr.Dataset(
        {
            "timestep_resolution": (("timesteps",), resolution.astype(float)),
            "timestep_weights": (("timesteps",), weights.astype(float)),
            "base_tech": (("techs",), np.array(["supply", "demand", "storage", "transmission", "conversion"], dtype=object)),
            "name": (("techs",), np.array(["PV", "Electricity demand", "Battery", "AC line", "Converter"], dtype=object)),
            "link_from": (("techs",), np.array([np.nan, np.nan, np.nan, "A_1", np.nan], dtype=object)),
            "link_to": (("techs",), np.array([np.nan, np.nan, np.nan, "B_1", np.nan], dtype=object)),
            "carrier_in": (("nodes", "techs", "carriers"), is_in),
            "carrier_out": (("nodes", "techs", "carriers"), is_out),
            "storage_loss": (("techs",), np.array([np.nan, np.nan, 0.0, np.nan, np.nan])),
            "sink_use_equals": (("techs", "timesteps", "nodes"), sink),
            "objective_cost_weights": (("costs",), np.array([1.0])),
            "cost_flow_cap": (("techs", "costs"), np.array([[0.5], [np.nan], [0.2], [np.nan], [np.nan]])),
        },
        coords={k: v for k, v in coords.items()},
    )
    if clustered:
        inputs = inputs.assign_coords(datesteps=clustered["days"])
        inputs["lookup_datestep_cluster"] = (("datesteps",), clustered["day_cluster"].astype(float))
        inputs["timestep_cluster"] = (("timesteps",), clustered["timestep_cluster"].astype(float))
        n_days = len(clustered["days"])
        inter = np.full((2, len(TECHS), n_days), np.nan)
        inter[:, t["battery"], :] = 10.0
        results = results.assign_coords(datesteps=clustered["days"])
        results["storage_inter_cluster"] = (("nodes", "techs", "datesteps"), inter)
    return xr.DataTree.from_dict({"inputs": inputs, "results": results})


@pytest.fixture(scope="session")
def clustered_file(tmp_path_factory) -> str:
    """Two typical days (24 h each) standing for 183 and 182 real days of 2018."""
    days = pd.date_range("2018-01-01", "2018-12-31", freq="D")
    day_cluster = np.arange(len(days)) % 2
    timesteps = pd.date_range("2018-01-01", periods=48, freq="h")  # relabelled typical days
    timestep_cluster = np.repeat([0, 1], 24)
    weights = np.where(timestep_cluster == 0, (day_cluster == 0).sum(), (day_cluster == 1).sum())
    profile = np.concatenate([_daily_shape(0), _daily_shape(1)])
    tree = make_model(timesteps, np.ones(48), weights, profile,
                      clustered={"days": days, "day_cluster": day_cluster, "timestep_cluster": timestep_cluster})
    path = tmp_path_factory.mktemp("results") / "clustered-2d.nc"
    tree.to_netcdf(path)
    return str(path)


@pytest.fixture(scope="session")
def unsolved_file(tmp_path_factory) -> str:
    """As Calliope saves a model that was not solved to optimality: inputs, but an empty `results` group."""
    timesteps = pd.date_range("2018-01-01", periods=12, freq="730h")
    tree = make_model(timesteps, np.full(12, 730.0), np.ones(12), np.ones(12))
    tree = xr.DataTree.from_dict({"inputs": tree["inputs"].to_dataset(), "results": xr.Dataset()})
    path = tmp_path_factory.mktemp("results") / "unsolved.nc"
    tree.to_netcdf(path)
    return str(path)


def _spores_model(cost_factor: float = 1.2) -> tuple[xr.Dataset, xr.Dataset]:
    """SPORES results as Calliope 0.7 stores them: a `spores` dimension (0 = cost-optimal). SPORE 1 has the same
    flows, a different PV capacity and costs x `cost_factor`; `spores_slack` is 0.5."""
    timesteps = pd.date_range("2018-01-01", periods=12, freq="730h")
    tree = make_model(timesteps, np.full(12, 730.0), np.ones(12), np.linspace(1.0, 2.0, 12))
    inputs, base = tree["inputs"].to_dataset(), tree["results"].to_dataset()
    inputs["spores_slack"] = ((), 0.5)
    spore = base.copy(deep=True)
    for var in ["cost", "cost_investment_annualised", "cost_operation_fixed", "cost_operation_variable",
                "min_cost_optimisation"]:
        spore[var] = spore[var] * cost_factor
    spore["flow_cap"] = spore["flow_cap"].where(spore["techs"] != "pv", spore["flow_cap"] * 2)
    results = xr.concat([base, spore], dim=pd.Index([0, 1], name="spores"), data_vars="all")
    return inputs, results


@pytest.fixture(scope="session")
def spores_file(tmp_path_factory) -> str:
    inputs, results = _spores_model()
    path = tmp_path_factory.mktemp("results") / "spores-run.nc"
    xr.DataTree.from_dict({"inputs": inputs, "results": results}).to_netcdf(path)
    return str(path)


@pytest.fixture(scope="session")
def per_spore_dir(tmp_path_factory) -> str:
    """Calliope's `save_per_spore_path` layout: spore_0.nc with inputs (with a `spores` dimension), spore_1.nc without."""
    inputs, results = _spores_model()
    folder = tmp_path_factory.mktemp("results") / "max_pv"
    folder.mkdir()
    inputs0 = inputs.copy()
    inputs0["objective_cost_weights"] = inputs0["objective_cost_weights"].expand_dims(spores=[0])
    xr.DataTree.from_dict({"results": results.sel(spores=[0]), "inputs": inputs0}).to_netcdf(folder / "spore_0.nc")
    xr.DataTree.from_dict({"results": results.sel(spores=[1])}).to_netcdf(folder / "spore_1.nc")
    return str(folder)


@pytest.fixture(scope="session")
def spores_over_slack_file(tmp_path_factory) -> str:
    inputs, results = _spores_model(cost_factor=1.6)  # above (1 + 0.5) x the cost-optimal cost
    path = tmp_path_factory.mktemp("results") / "spores-over-slack.nc"
    xr.DataTree.from_dict({"inputs": inputs, "results": results}).to_netcdf(path)
    return str(path)


@pytest.fixture(scope="session")
def resampled_file(tmp_path_factory) -> str:
    """12 timesteps of 730 h (as `res_730h`), not clustered."""
    timesteps = pd.date_range("2018-01-01", periods=12, freq="730h")
    profile = np.linspace(1.0, 2.0, 12)
    tree = make_model(timesteps, np.full(12, 730.0), np.ones(12), profile)
    path = tmp_path_factory.mktemp("results") / "resampled-730h.nc"
    tree.to_netcdf(path)
    return str(path)
