import json

import numpy as np
import pandas as pd
import pytest

from tidy_calliope import ResultFileError, load_config, process
from tidy_calliope.cli import main
from tidy_calliope.reader import open_results

from conftest import DEMAND, EXPORT, LINK, _daily_shape, price

HOURLY = {"outputs": {"hourly": True, "shadow_prices": True}}


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


def test_unsolved_file_raises(unsolved_file, tmp_path):
    with pytest.raises(ResultFileError, match="not solved to optimality"):
        process(unsolved_file, tmp_path)
    assert not (tmp_path / "unsolved").exists()


def test_cli_skips_unsolved_file(unsolved_file, resampled_file, tmp_path, capsys):
    """An unsolved file is reported in one line and skipped; the other files are still processed."""
    assert main([unsolved_file, resampled_file, "-o", str(tmp_path), "-q"]) == 1
    err = capsys.readouterr().err
    assert "SKIPPED" in err and "unsolved.nc" in err and "Traceback" not in err
    assert "1 of 2 files processed, 1 skipped" in err
    assert (tmp_path / "resampled-730h" / "nameplate_capacity.csv").exists()
    assert not (tmp_path / "unsolved").exists()


def test_shadow_prices_clustered_full_year(clustered_out):
    """Prices are divided by the timestep weights and repeated for every real day of each typical day."""
    folder, summary = clustered_out
    sp = _read(folder, "shadow_price_system_balance")
    assert list(sp.columns) == ["scenario", "nodes", "carriers", "unit", "timesteps", "value"]
    assert set(sp["unit"]) == {"billion_2015eur_per_twh"}
    a = sp[sp.nodes == "A_1"].set_index("timesteps")["value"]
    assert len(a) == 8760
    # real day d stands for typical day d % 2, i.e. model timesteps on 2018-01-01 (day 0) or 2018-01-02 (day 1)
    real = pd.to_datetime(a.index)
    typical = pd.Timestamp("2018-01-01") + pd.to_timedelta((real.day_of_year - 1) % 2, unit="D") + pd.to_timedelta(real.hour, unit="h")
    expected = price("A_1", pd.DatetimeIndex(typical)) / 0.1  # weights removed; cost scale 1, energy scale 0.1
    assert np.allclose(a.values, expected)
    assert a.iloc[0] != a.iloc[24]  # 1 and 2 January come from different typical days
    assert summary["checks"]["shadow_price_timesteps"]["ok"]


def test_shadow_prices_per_model_timestep(clustered_file, tmp_path):
    (summary,) = process(clustered_file, tmp_path, {"outputs": {"shadow_prices": True},
                                                    "shadow_prices": {"full_year": False}})
    sp = pd.read_csv(tmp_path / summary["scenario"] / "shadow_price_system_balance.csv")
    assert len(sp[sp.nodes == "B_1"]) == 48  # 2 typical days x 24 h
    assert summary["checks_failed"] == []


def test_shadow_prices_resampled_and_constraints(resampled_out):
    folder, _ = resampled_out
    sp = _read(folder, "shadow_price_system_balance")
    b = sp[sp.nodes == "B_1"]["value"].values
    hours = pd.date_range("2018-01-01", periods=12, freq="730h")
    assert len(b) == 12 and np.allclose(b, price("B_1", hours) / 0.1)  # weights 1: no correction
    other = _read(folder, "shadow_price_constraints")
    assert other.loc[0, ["constraint", "index", "value"]].tolist() == ["group_co2_cap", "costs=monetary", -0.5]


def test_shadow_prices_off_by_default(resampled_file, tmp_path):
    (summary,) = process(resampled_file, tmp_path)
    assert "shadow_price_system_balance" not in summary["tables"]


def _cost_by_spore(folder):
    return _read(folder, "total_system_cost").groupby("spores")["total_system_cost"].sum()


def test_spores_combined_file(spores_file, tmp_path):
    (summary,) = process(spores_file, tmp_path, HOURLY)
    folder = tmp_path / "spores-run"
    assert summary["spores"] == [0, 1] and summary["checks_failed"] == []
    assert set(summary["checks"]) >= {"spores=0", "spores=1", "spores_cost_within_slack"}
    cost = _cost_by_spore(folder)
    assert cost[1] == pytest.approx(1.2 * cost[0])
    caps = _read(folder, "nameplate_capacity")
    pv = caps[caps.techs == "pv"].groupby("spores")["nameplate_capacity"].sum()
    assert pv[1] == pytest.approx(2 * pv[0])
    flow = _read(folder, "flow_out")
    assert list(flow.columns[:3]) == ["scenario", "spores", "techs"] and set(flow["spores"]) == {0, 1}


def test_spores_per_spore_files_match_combined(spores_file, per_spore_dir, tmp_path):
    process(spores_file, tmp_path / "combined")
    (summary,) = process(per_spore_dir, tmp_path / "split")  # a folder of spore_<n>.nc files: one run
    assert summary["scenario"] == "max_pv" and summary["spores"] == [0, 1] and summary["checks_failed"] == []
    pd.testing.assert_series_equal(_cost_by_spore(tmp_path / "combined" / "spores-run"),
                                   _cost_by_spore(tmp_path / "split" / "max_pv"))


def test_spores_single_file_borrows_inputs(per_spore_dir, tmp_path):
    (summary,) = process(f"{per_spore_dir}/spore_1.nc", tmp_path)  # no inputs: taken from spore_0.nc
    assert summary["spores"] == [1]
    assert set(_read(tmp_path / "max_pv", "total_system_cost")["spores"]) == {1}


def test_spores_cost_above_slack_fails(spores_over_slack_file, tmp_path):
    (summary,) = process(spores_over_slack_file, tmp_path)
    assert summary["checks_failed"] == ["spores_cost_within_slack"]
    assert list(summary["checks"]["spores_cost_within_slack"]["above_limit"]) == [1]


def test_subset_period_hours(spores_file):
    """A model of part of a year: the check compares with the modelled period, not with 8760 h."""
    from tidy_calliope.checks import run_checks
    from tidy_calliope.outputs import Processor
    from tidy_calliope.reader import open_results, spore_views

    model = spore_views(open_results(spores_file))[0]
    check = run_checks(Processor(model, load_config()), {})["hours_in_period"]
    assert check["ok"] and check["full_year"]


def test_cli_missing_path(tmp_path, capsys):
    assert main([str(tmp_path / "nothing-here.nc"), "-o", str(tmp_path), "-q"]) == 2
    assert capsys.readouterr().err.startswith("ERROR: ")
