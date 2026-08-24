"""Network diagnostics, the reporting cost model, and the foreign-network importers.

Nothing here is allowed to influence selection, and two tests say so directly: a
:class:`~rbfenetmap.core.cost.CostModel` never reaches a planner, and ``--cost-units``
cannot move an edge. The rest checks the numbers, and checks that the one stochastic
function in the package cannot be called without a seed.
"""

from __future__ import annotations

import json
import warnings

import pytest

from rbfenetmap.cli.main import main
from rbfenetmap.core.cost import COST_UNITS, CostModel, network_cost_summary
from rbfenetmap.core.diagnostics import (
    count_cycles,
    degree_summary,
    diameter,
    edge_budget_advice,
    failure_robustness,
    network_cost,
    network_efficiency,
    summarize,
)
from rbfenetmap.core.models import EdgeKind, Network
from rbfenetmap.core.options import NetworkOptions
from rbfenetmap.io.loaders import load_fepplus_network, load_orion_network
from rbfenetmap.io.networkio import dump_network
from rbfenetmap.plugins.planners import create_planner

from .conftest import make_transformation


@pytest.fixture
def triangle(benzamides):
    """Three ligands joined in a cycle, at costs 1, 2, and 3."""
    ligands = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl")}
    edges = (
        make_transformation("bza_H", "bza_F", cost=1.0),
        make_transformation("bza_F", "bza_Cl", cost=2.0),
        make_transformation("bza_H", "bza_Cl", cost=3.0),
    )
    return Network(ligands=ligands, edges=edges, planner="dummy", options=NetworkOptions())


@pytest.fixture
def path_of_three(benzamides):
    """Three ligands in a line, so the diameter is 2 and nothing lies on a cycle."""
    ligands = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl")}
    edges = (make_transformation("bza_H", "bza_F", cost=1.0), make_transformation("bza_F", "bza_Cl", cost=2.0))
    return Network(ligands=ligands, edges=edges, planner="dummy", options=NetworkOptions())


@pytest.fixture
def square_ligands(benzamides):
    """Four ligands and a full set of feasible pairs, for the "cost never selects" check."""
    return {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl", "bza_Me")}


@pytest.fixture
def square_candidates():
    """Six candidate edges over the four square ligands, at distinct costs."""
    costs = {
        ("bza_H", "bza_F"): 1.0,
        ("bza_F", "bza_Cl"): 2.0,
        ("bza_H", "bza_Cl"): 3.0,
        ("bza_H", "bza_Me"): 4.0,
        ("bza_Cl", "bza_Me"): 8.0,
        ("bza_F", "bza_Me"): 9.0,
    }
    return [make_transformation(a, b, cost=cost) for (a, b), cost in costs.items()]


@pytest.fixture
def split(benzamides):
    """Four ligands in two disjoint pairs -- a disconnected network."""
    ligands = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl", "bza_Me")}
    edges = (make_transformation("bza_H", "bza_F", cost=1.0), make_transformation("bza_Cl", "bza_Me", cost=1.0))
    return Network(ligands=ligands, edges=edges, planner="dummy", options=NetworkOptions())


class TestBasicMetrics:
    def test_cost_is_the_sum_and_efficiency_the_mean(self, triangle):
        assert network_cost(triangle) == pytest.approx(6.0)
        assert network_efficiency(triangle) == pytest.approx(2.0)

    def test_an_empty_network_has_no_efficiency_rather_than_a_division_error(self, benzamides):
        empty = Network(ligands=benzamides, edges=(), planner="dummy")
        assert network_efficiency(empty) == 0.0

    def test_counts_short_cycles(self, triangle, path_of_three):
        assert count_cycles(triangle) == 1
        assert count_cycles(path_of_three) == 0

    def test_a_cycle_shorter_than_three_is_refused(self, triangle):
        with pytest.raises(ValueError, match="at least 3"):
            count_cycles(triangle, max_length=2)

    def test_degree_summary_reports_the_spread_and_the_isolated(self, path_of_three, split, benzamides):
        summary = degree_summary(path_of_three)
        assert (summary.minimum, summary.maximum) == (1, 2)
        assert summary.mean == pytest.approx(4 / 3)
        assert summary.isolated == ()

        lonely = Network(
            ligands={n: benzamides[n] for n in ("bza_H", "bza_F", "bza_Cl")},
            edges=(make_transformation("bza_H", "bza_F", cost=1.0),),
            planner="dummy",
        )
        assert degree_summary(lonely).isolated == ("bza_Cl",)

    def test_diameter_is_none_when_disconnected(self, path_of_three, split):
        assert diameter(path_of_three) == 2
        assert diameter(split) is None


class TestFailureRobustness:
    def test_the_seed_is_mandatory(self, triangle):
        """A defaulted seed is a defaulted seed until someone leaves it off."""
        with pytest.raises(TypeError):
            failure_robustness(triangle)  # type: ignore[call-arg]

    def test_a_fixed_seed_is_reproducible(self, triangle):
        first = failure_robustness(triangle, seed=7, n_repeats=50)
        second = failure_robustness(triangle, seed=7, n_repeats=50)
        assert first == second

    def test_no_failures_means_always_connected(self, triangle):
        result = failure_robustness(triangle, failure_rate=0.0, n_repeats=10, seed=0)
        assert result.connected_fraction == 1.0
        assert result.mean_ligands_retained == pytest.approx(3.0)

    def test_every_edge_failing_shatters_the_network(self, triangle):
        result = failure_robustness(triangle, failure_rate=1.0, n_repeats=10, seed=0)
        assert result.connected_fraction == 0.0
        assert result.mean_ligands_retained == pytest.approx(1.0)

    def test_a_triangle_survives_more_than_a_path(self, triangle, path_of_three):
        """The point of buying a cycle, stated as a number."""
        cyclic = failure_robustness(triangle, failure_rate=0.3, n_repeats=400, seed=3)
        linear = failure_robustness(path_of_three, failure_rate=0.3, n_repeats=400, seed=3)
        assert cyclic.connected_fraction > linear.connected_fraction

    @pytest.mark.parametrize("rate,repeats", [(-0.1, 10), (1.1, 10), (0.1, 0)])
    def test_out_of_range_arguments_are_refused(self, triangle, rate, repeats):
        with pytest.raises(ValueError):
            failure_robustness(triangle, failure_rate=rate, n_repeats=repeats, seed=0)


class TestEdgeBudgetAdvice:
    def test_reports_the_shortfall_against_n_log_n(self):
        advice = edge_budget_advice(40, 40)
        assert advice.recommended == 148  # ceil(40 * ln 40)
        assert advice.shortfall == 108
        assert "below the n*ln(n) precision floor" in advice.message

    def test_a_dense_network_has_no_shortfall(self):
        advice = edge_budget_advice(5, 40)
        assert advice.shortfall == 0
        assert "meets the" in advice.message

    def test_a_single_ligand_has_no_budget(self):
        assert edge_budget_advice(1, 0).recommended == 0

    def test_it_is_advice_and_never_a_warning(self, triangle):
        """With edges_per_ligand=2 this would otherwise fire on every run ever planned."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            summarize(triangle, seed=0, n_repeats=5)


class TestSummarize:
    def test_gathers_every_metric(self, triangle):
        report = summarize(triangle, seed=0, n_repeats=5)
        assert report["n_ligands"] == 3
        assert report["n_edges"] == 3
        assert report["diameter"] == 1
        assert report["n_cycles"] == 1
        assert report["budget"].recommended == 4  # ceil(3 * ln 3)

    def test_is_reproducible_across_calls(self, triangle):
        assert summarize(triangle, seed=1) == summarize(triangle, seed=1)


class TestCostModel:
    def test_the_published_defaults(self):
        model = CostModel()
        assert model.rbfe_gpu_hours == pytest.approx(3.97)
        assert model.cbfe_gpu_hours == pytest.approx(12.704)

    def test_a_counterpoised_edge_costs_the_multiple(self, benzamides):
        network = Network(
            ligands={n: benzamides[n] for n in ("bza_H", "bza_F")},
            edges=(make_transformation("bza_H", "bza_F", cost=1.0, kind=EdgeKind.CBFE),),
            planner="dummy",
        )
        model = CostModel(rbfe_gpu_hours=1.0, cbfe_multiplier=3.0, price_per_gpu_hour=2.0)
        assert model.network_gpu_hours(network) == pytest.approx(3.0)
        assert model.network_price(network) == pytest.approx(6.0)

    def test_a_cheaper_counterpoised_edge_is_refused(self):
        with pytest.raises(ValueError, match="at least 1.0"):
            CostModel(cbfe_multiplier=0.5)

    @pytest.mark.parametrize("kwargs", [{"rbfe_gpu_hours": -1.0}, {"price_per_gpu_hour": -0.1}])
    def test_negative_costs_are_refused(self, kwargs):
        with pytest.raises(ValueError):
            CostModel(**kwargs)

    def test_the_summary_reports_both_scales(self, triangle):
        totals = network_cost_summary(triangle)
        assert totals["score"] == pytest.approx(6.0)
        assert totals["gpu_hours"] == pytest.approx(3 * 3.97)
        assert set(COST_UNITS) == {"score", "gpu_hours"}

    def test_the_cost_model_never_reaches_selection(self, square_ligands, square_candidates):
        """Phase 1's cost work is reporting only; Phase 3 is where cost may steer edges."""
        planned = create_planner("mst").plan(square_ligands, square_candidates, NetworkOptions())
        assert network_cost_summary(planned)["gpu_hours"] > 0
        again = create_planner("mst").plan(square_ligands, square_candidates, NetworkOptions())
        assert {e.unordered_key for e in planned.edges} == {e.unordered_key for e in again.edges}


class TestForeignNetworkImport:
    def test_reads_a_fepplus_edge_list(self, tmp_path):
        path = tmp_path / "map.edge"
        path.write_text("# planned map\nlig_1 >>> lig_2\n\nabc123:lig_2 >>> def456:lig_3\n")
        assert load_fepplus_network(path) == ("lig_1~lig_2", "lig_2~lig_3")

    def test_reads_an_orion_edge_list(self, tmp_path):
        path = tmp_path / "map.txt"
        path.write_text("# NES\nlig_1 >> lig_2\nlig_2 >> lig_3\n")
        assert load_orion_network(path) == ("lig_1~lig_2", "lig_2~lig_3")

    def test_a_repeated_edge_is_collapsed_not_duplicated(self, tmp_path):
        path = tmp_path / "map.txt"
        path.write_text("a >> b\nb >> a\n")
        assert load_orion_network(path) == ("a~b",)

    def test_names_are_sanitized_the_way_the_ligand_loader_sanitizes_them(self, tmp_path):
        """Otherwise an imported edge names a vertex no ligand loader would ever produce."""
        path = tmp_path / "map.txt"
        path.write_text("lig one >> lig/two\n")
        assert load_orion_network(path) == ("lig_one~lig_two",)

    def test_the_orion_parser_refuses_a_fepplus_file(self, tmp_path):
        """Splitting on the literal would have read 'a >>> b' as an edge to '> b'."""
        path = tmp_path / "map.edge"
        path.write_text("a >>> b\n")
        with pytest.raises(ValueError, match="Orion edge line"):
            load_orion_network(path)

    @pytest.mark.parametrize("line", ["a >> b >> c", "a", " >> b", "a >> a"])
    def test_an_unparsable_line_raises_rather_than_being_skipped(self, tmp_path, line):
        path = tmp_path / "map.txt"
        path.write_text(line + "\n")
        with pytest.raises(ValueError):
            load_orion_network(path)

    def test_a_file_with_no_edges_raises(self, tmp_path):
        path = tmp_path / "map.txt"
        path.write_text("# nothing here\n")
        with pytest.raises(ValueError, match="contains no"):
            load_orion_network(path)

    def test_the_specs_drive_the_explicit_planner(self, tmp_path, square_ligands, square_candidates):
        path = tmp_path / "map.txt"
        path.write_text("bza_H >> bza_F\nbza_F >> bza_Cl\nbza_H >> bza_Me\n")
        options = NetworkOptions(pair_strategy="explicit", explicit_pairs=load_orion_network(path))
        network = create_planner("explicit").plan(square_ligands, square_candidates, options)
        assert len(network.edges) == 3


class TestDiagnoseCommand:
    @pytest.fixture
    def planned(self, tmp_path, triangle):
        return dump_network(triangle, tmp_path / "n.json")

    def test_table_output(self, planned, capsys):
        assert main(["diagnose", "--network", str(planned)]) == 0
        out = capsys.readouterr().out
        assert "diameter" in out and "robustness" in out
        assert "n*ln(n)" in out

    def test_json_output_is_parsable(self, planned, capsys):
        assert main(["diagnose", "--network", str(planned), "--format", "json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["n_ligands"] == 3
        assert payload["robustness"]["seed"] == 0

    def test_the_same_seed_gives_the_same_report(self, planned, capsys):
        main(["diagnose", "--network", str(planned), "--format", "json", "--seed", "5"])
        first = capsys.readouterr().out
        main(["diagnose", "--network", str(planned), "--format", "json", "--seed", "5"])
        assert capsys.readouterr().out == first

    def test_cost_units_changes_the_wording_not_the_network(self, planned, capsys):
        main(["diagnose", "--network", str(planned), "--cost-units", "gpu_hours"])
        assert "GPU-hours" in capsys.readouterr().out
        main(["diagnose", "--network", str(planned), "--cost-units", "score"])
        assert "scorer units" in capsys.readouterr().out
