import json

import numpy as np
import pandas as pd
import pytest

from tidy_calliope import load_config, process
from tidy_calliope.cli import main
from tidy_calliope.reader import open_results

from conftest import DEMAND, EXPORT, LINK, _daily_shape

HOURLY = {"outputs": {"hourly": True}}


def _read(folder, name):
    return pd.read_csv(folder / f"{name}.csv")


@pytest.fixture(scope="module")
def clustered_out(clustered_file, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    (summary,) = process(clustered_file, out, HOURLY)
    return out / summary["scenario"], summary


@pytest.fixture(scope="module")
def resampled_out(resampled_file, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    (summary,) = process(resampled_file, out, HOURLY)
    return out / summary["scenario"], summary


@pytest.mark.parametrize("which", ["clustered_out", "resampled_out"])
def test_all_checks_pass(which, request):
    folder, summary = request.getfixturevalue(which)
    assert summary["checks_failed"] == [], summary["checks"]
    assert json.loads((folder / "metadata.json").read_text())["checks_failed"] == []


def test_time_info(clustered_file, resampled_file):
    clustered = open_results(clustered_file).time
    assert clustered.clustered and clustered.describe()["typical_days"] == 2
    assert len(clustered.expansion) == 8760
    resampled = open_results(resampled_file).time
    assert not resampled.clustered and resampled.describe()["real_hours_represented"] == 8760


def test_clustered_annual_demand_is_weighted(clustered_out):
    folder, _ = clustered_out
    demand = _read(folder, "flow_in_sum").set_index(["techs", "locs"])["flow_in_sum"]
    days = pd.date_range("2018-01-01", "2018-12-31", freq="D")
    n0, n1 = (np.arange(len(days)) % 2 == 0).sum(), (np.arange(len(days)) % 2 == 1).sum()
    for node, scale in DEMAND.items():
        expected = 0.1 * scale * (n0 * _daily_shape(0).sum() + n1 * _daily_shape(1).sum())
        assert demand[("demand_elec", node)] == pytest.approx(expected)


def test_clustered_hourly_is_rebuilt(clustered_out):
    folder, _ = clustered_out
    flow = _read(folder, "flow_in")
    flow["timesteps"] = pd.to_datetime(flow["timesteps"])
    a = flow[(flow.techs == "demand_elec") & (flow.locs == "A_1")].set_index("timesteps")["flow_in"]
    assert len(a) == 8760
    # 1 Jan is typical day 0, 2 Jan typical day 1, 3 Jan typical day 0 again
    assert np.allclose(a["2018-01-01"].values, 0.1 * _daily_shape(0))
    assert np.allclose(a["2018-01-02"].values, 0.1 * _daily_shape(1))
    assert np.allclose(a["2018-01-03"].values, 0.1 * _daily_shape(0))


def test_clustered_storage_level(clustered_out):
    folder, _ = clustered_out
    level = _read(folder, "storage_level")
    a = level[(level.techs == "battery") & (level.locs == "A_1")]["storage_level"].values
    # start-of-day level 10 (storage_inter_cluster) + change within the typical day, x 0.1
    assert len(a) == 8760 and a.min() >= 1.0 and a.max() <= 1.2 + 1e-12


def test_monthly_split_of_long_timesteps(resampled_out):
    folder, _ = resampled_out
    monthly = _read(folder, "flow_in_sum_1M")
    a = monthly[(monthly.techs == "demand_elec") & (monthly.locs == "A_1")].set_index("month")["flow_in_sum_1M"]
    annual = _read(folder, "flow_in_sum").set_index(["techs", "locs"]).loc[("demand_elec", "A_1"), "flow_in_sum"]
    assert len(a) == 12 and a.sum() == pytest.approx(annual)


def test_transmission(clustered_out):
    folder, _ = clustered_out
    net = _read(folder, "net_import_sum").set_index(["exporting_region", "importing_region"])["net_import_sum"]
    expected = 0.1 * EXPORT * 0.95 * 8760  # average of sent and received (5 % losses on average)
    assert net[("A_1", "B_1")] == pytest.approx(expected)
    assert net[("B_1", "A_1")] == pytest.approx(-expected)
    caps = _read(folder, "net_transfer_capacity")
    assert set(caps["techs"]) == {"ac_transmission"} and len(caps) == 2


def test_costs_and_dropped_techs(clustered_out):
    folder, summary = clustered_out
    total = _read(folder, "total_system_cost")
    assert "syn_diesel_converter" not in set(total["techs"]) and LINK not in set(total["techs"])
    assert summary["checks"]["total_system_cost_complete"]["dropped_techs_total"] == pytest.approx(0.25 + 0.1)
    variable = _read(folder, "variable_operation_cost")["variable_operation_cost"].sum()
    assert variable == pytest.approx(2 * 0.01 * 365 * 24)  # weights already in Calliope's variable cost


def test_regions_countries(clustered_file, tmp_path):
    (summary,) = process(clustered_file, tmp_path, {"regions": "countries"})
    caps = pd.read_csv(tmp_path / summary["scenario"] / "nameplate_capacity.csv")
    assert set(caps["locs"]) == {"A", "B"}


def test_config_rejects_typos():
    with pytest.raises(KeyError):
        load_config({"output": {"hourly": True}})
    with pytest.raises(KeyError):
        load_config({"outputs": {"hourlyy": True}})


def test_cli(resampled_file, tmp_path):
    assert main([resampled_file, "-o", str(tmp_path), "-q", "--strict"]) == 0
    assert (tmp_path / "resampled-730h" / "nameplate_capacity.csv").exists()
