"""LMI ingestion, pruning, and the replan that refills the gaps.

The networks here are hand-built with ``make_transformation`` so that which edge the
planner picks next is checkable by adding up small numbers, and the LMI values are made up
for the same reason: nothing in this module computes an LMI, it only decides what to do
with the ones the analysis reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rbfenetmap.core.replanning import (
    cycle_closure_errors,
    load_edge_lmi,
    lmi_threshold,
    replan_after_diagnostics,
    select_high_lmi_edges,
)
from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.models import Ligand, Network
from rbfenetmap.core.options import NetworkOptions
from rbfenetmap.plugins.planners import create_planner

from .conftest import make_transformation


@pytest.fixture
def five(benzamides) -> dict[str, Ligand]:
    """Five co-posed ligands; only their names matter here."""
    return dict(benzamides)


@pytest.fixture
def candidates():
    """Every pair among the five, cheapest first in a predictable order."""
    names = ["bza_H", "bza_F", "bza_Cl", "bza_Me", "bza_CF3"]
    pool = []
    cost = 1.0
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pool.append(make_transformation(a, b, cost=cost))
            cost += 1.0
    return pool


@pytest.fixture
def network(five, candidates) -> Network:
    """A planned network over the full pool, so a replan has somewhere to go."""
    return create_planner("mst").plan(five, candidates, NetworkOptions())


class TestLoadEdgeLMI:
    def test_string_keys_become_unordered_pairs(self):
        assert load_edge_lmi({"b~a": 0.5}) == {("a", "b"): 0.5}

    def test_tuple_keys_are_accepted_in_memory(self):
        assert load_edge_lmi({("b", "a"): 0.5}) == {("a", "b"): 0.5}

    def test_a_json_file_is_read(self, tmp_path):
        path = tmp_path / "lmi.json"
        path.write_text(json.dumps({"a~b": 1.5}))
        assert load_edge_lmi(path) == {("a", "b"): 1.5}

    def test_an_edges_wrapper_is_unwrapped(self, tmp_path):
        path = tmp_path / "lmi.json"
        path.write_text(json.dumps({"protein": "bace1", "edges": {"a~b": 1.5}}))
        assert load_edge_lmi(path) == {("a", "b"): 1.5}

    def test_the_two_orientations_may_not_disagree(self):
        with pytest.raises(ValueError, match="appears twice with different LMI"):
            load_edge_lmi({"a~b": 1.0, "b~a": 2.0})

    def test_agreeing_orientations_are_fine(self):
        assert load_edge_lmi({"a~b": 1.0, "b~a": 1.0}) == {("a", "b"): 1.0}

    def test_a_non_numeric_value_is_refused(self):
        with pytest.raises(ValueError, match="is not a number"):
            load_edge_lmi({"a~b": "high"})

    def test_a_malformed_key_is_refused(self):
        with pytest.raises(ValueError, match="Malformed edge key"):
            load_edge_lmi({"a-b": 1.0})


class TestThreshold:
    def test_the_cut_isolates_an_outlier_even_on_a_small_network(self):
        """Nearest-rank would land on 5.0 here and prune nothing at all."""
        values = [0.1, 0.2, 0.3, 0.4, 5.0]
        cut = lmi_threshold(values, quantile=0.9)
        assert [v for v in values if v > cut] == [5.0]

    def test_all_equal_values_lose_nothing(self):
        values = [0.3, 0.3, 0.3, 0.3]
        assert not [v for v in values if v > lmi_threshold(values, quantile=0.5)]

    def test_the_top_quantile_cuts_at_the_largest_value(self):
        assert lmi_threshold([1.0, 2.0, 3.0], quantile=1.0) == 3.0

    def test_an_empty_set_is_refused(self):
        with pytest.raises(ValueError, match="empty set"):
            lmi_threshold([])

    def test_an_out_of_range_quantile_is_refused(self):
        with pytest.raises(ValueError, match="must lie in"):
            lmi_threshold([1.0], quantile=1.5)


class TestSelectHighLMIEdges:
    def _lmi(self, network, **overrides):
        """A flat LMI over the selected edges, with named edges pushed up."""
        values = {edge.unordered_key: 0.1 for edge in network.edges}
        for key, value in overrides.items():
            values[tuple(sorted(key.split("~")))] = value
        return values

    def test_only_edges_above_the_cut_are_returned(self, network):
        worst = network.edges[0].key
        lmi = self._lmi(network, **{worst: 9.0})
        assert select_high_lmi_edges(network, lmi) == (network.edges[0].unordered_key,)

    def test_they_come_back_worst_first(self, network):
        first, second = network.edges[0].key, network.edges[1].key
        lmi = self._lmi(network, **{first: 5.0, second: 9.0})
        selected = select_high_lmi_edges(network, lmi, threshold=1.0)
        assert selected == (network.edges[1].unordered_key, network.edges[0].unordered_key)

    def test_max_pruned_caps_the_damage(self, network):
        first, second = network.edges[0].key, network.edges[1].key
        lmi = self._lmi(network, **{first: 5.0, second: 9.0})
        assert len(select_high_lmi_edges(network, lmi, threshold=1.0, max_pruned=1)) == 1

    def test_a_missing_value_is_a_failure_by_default(self, network):
        lmi = self._lmi(network)
        del lmi[network.edges[0].unordered_key]
        with pytest.raises(ValueError, match="have no LMI value"):
            select_high_lmi_edges(network, lmi)

    def test_a_missing_value_can_be_waived_deliberately(self, network):
        lmi = self._lmi(network, **{network.edges[1].key: 9.0})
        del lmi[network.edges[0].unordered_key]
        assert select_high_lmi_edges(network, lmi, threshold=1.0, require_complete=False)

    def test_an_analysis_that_names_nothing_recognisable_says_so(self, network):
        with pytest.raises(ValueError, match="None of the selected edges"):
            select_high_lmi_edges(network, {("x", "y"): 9.0})

    def test_a_forced_edge_is_never_pruned(self, five, candidates):
        forced = "bza_H~bza_CF3"
        options = NetworkOptions(forced_edges=(forced,))
        network = create_planner("mst").plan(five, candidates, options)
        lmi = {edge.unordered_key: 0.1 for edge in network.edges}
        lmi[tuple(sorted(forced.split("~")))] = 99.0
        assert select_high_lmi_edges(network, lmi, threshold=1.0) == ()


class TestReplanAfterDiagnostics:
    def _flat(self, network, worst):
        values = {edge.unordered_key: 0.1 for edge in network.edges}
        values[tuple(sorted(worst.split("~")))] = 9.0
        return values

    def test_the_pruned_edge_is_gone_and_banned(self, network):
        worst = network.edges[0]
        replanned, pruned = replan_after_diagnostics(network, self._flat(network, worst.key))
        assert pruned == (worst.unordered_key,)
        assert worst.unordered_key not in {edge.unordered_key for edge in replanned.edges}
        assert f"{worst.unordered_key[0]}~{worst.unordered_key[1]}" in replanned.options.banned_edges

    def test_only_the_gap_changes(self, network):
        """The default holds the survivors in place; a running campaign depends on it."""
        worst = network.edges[0]
        replanned, _ = replan_after_diagnostics(network, self._flat(network, worst.key))
        before = {edge.unordered_key for edge in network.edges}
        after = {edge.unordered_key for edge in replanned.edges}
        assert before - after == {worst.unordered_key}
        assert len(after - before) <= 1

    def test_reselection_is_available_and_is_not_the_default(self, network):
        worst = network.edges[0]
        held, _ = replan_after_diagnostics(network, self._flat(network, worst.key))
        free, _ = replan_after_diagnostics(network, self._flat(network, worst.key), keep_existing=False)
        assert not held.options.forced_edges == free.options.forced_edges

    def test_the_replanned_network_still_spans(self, network):
        replanned, _ = replan_after_diagnostics(network, self._flat(network, network.edges[0].key))
        replanned.validate(require_connected=True)

    def test_nothing_above_the_cut_returns_the_network_unchanged(self, network):
        flat = {edge.unordered_key: 1.0 for edge in network.edges}
        replanned, pruned = replan_after_diagnostics(network, flat)
        assert pruned == ()
        assert replanned is network

    def test_a_network_with_no_candidate_pool_says_what_to_do(self, five):
        edges = (
            make_transformation("bza_H", "bza_F"),
            make_transformation("bza_F", "bza_Cl"),
            make_transformation("bza_Cl", "bza_Me"),
            make_transformation("bza_Me", "bza_CF3"),
        )
        bare = Network(ligands=five, edges=edges, options=NetworkOptions())
        lmi = {edge.unordered_key: 0.1 for edge in edges}
        lmi[edges[0].unordered_key] = 9.0
        with pytest.raises(NetworkPlanError, match="no candidate pool"):
            replan_after_diagnostics(bare, lmi, threshold=1.0)


class TestCycleClosureErrors:
    def test_a_triangle_sums_around_its_loop(self, benzamides):
        ligands = {n: benzamides[n] for n in ("bza_H", "bza_F", "bza_Cl")}
        edges = (
            make_transformation("bza_H", "bza_F"),
            make_transformation("bza_F", "bza_Cl"),
            make_transformation("bza_H", "bza_Cl"),
        )
        network = Network(ligands=ligands, edges=edges, options=NetworkOptions())
        # 1.0 + 1.0 going round, then -1.5 coming back: a 0.5 kcal/mol hysteresis.
        closures = cycle_closure_errors(network, {"bza_H~bza_F": 1.0, "bza_F~bza_Cl": 1.0, "bza_H~bza_Cl": 1.5})
        assert len(closures) == 1
        # The basis cycle may be traversed either way round, so it is the magnitude of the
        # hysteresis that is the physical quantity, not its sign.
        assert pytest.approx(abs(next(iter(closures.values()))), abs=1e-9) == 0.5

    def test_a_cycle_with_a_missing_edge_is_dropped_not_partially_summed(self, benzamides):
        ligands = {n: benzamides[n] for n in ("bza_H", "bza_F", "bza_Cl")}
        edges = (
            make_transformation("bza_H", "bza_F"),
            make_transformation("bza_F", "bza_Cl"),
            make_transformation("bza_H", "bza_Cl"),
        )
        network = Network(ligands=ligands, edges=edges, options=NetworkOptions())
        assert cycle_closure_errors(network, {"bza_H~bza_F": 1.0, "bza_F~bza_Cl": 1.0}) == {}

    def test_a_tree_has_no_cycles_to_close(self, benzamides):
        ligands = {n: benzamides[n] for n in ("bza_H", "bza_F", "bza_Cl")}
        edges = (make_transformation("bza_H", "bza_F"), make_transformation("bza_F", "bza_Cl"))
        network = Network(ligands=ligands, edges=edges, options=NetworkOptions())
        assert cycle_closure_errors(network, {"bza_H~bza_F": 1.0, "bza_F~bza_Cl": 1.0}) == {}


class TestReplanCommand:
    """The CLI leg of the loop: a planned network in, a replanned one out."""

    @pytest.mark.integration
    def test_it_prunes_replans_and_reports(self, tmp_path, capsys):
        from rbfenetmap.cli.main import main
        from rbfenetmap.io.networkio import load_network

        sdf = str(Path(__file__).resolve().parent / "data" / "golden_benzamides.sdf")
        planned = tmp_path / "network.json"
        assert main(["plan", "--ligands", sdf, "--out", str(planned)]) == 0
        network = load_network(planned)

        worst = network.edges[0]
        values = {edge.key: 0.1 for edge in network.edges}
        values[worst.key] = 9.0
        lmi = tmp_path / "lmi.json"
        lmi.write_text(json.dumps(values))

        out = tmp_path / "replanned.json"
        capsys.readouterr()
        assert main(["replan", "--network", str(planned), "--lmi", str(lmi), "--out", str(out)]) == 0
        # Printed as the unordered pair the ban is keyed by, not the edge direction.
        assert f"{worst.unordered_key[0]}~{worst.unordered_key[1]}" in capsys.readouterr().out

        replanned = load_network(out)
        assert worst.unordered_key not in {edge.unordered_key for edge in replanned.edges}
        replanned.validate(require_connected=True)

    @pytest.mark.integration
    def test_an_incomplete_analysis_fails_rather_than_exempting_the_gaps(self, tmp_path):
        from rbfenetmap.cli.main import main
        from rbfenetmap.io.networkio import load_network

        sdf = str(Path(__file__).resolve().parent / "data" / "golden_benzamides.sdf")
        planned = tmp_path / "network.json"
        assert main(["plan", "--ligands", sdf, "--out", str(planned)]) == 0
        network = load_network(planned)

        lmi = tmp_path / "lmi.json"
        lmi.write_text(json.dumps({edge.key: 0.1 for edge in network.edges[:-1]}))
        assert main(["replan", "--network", str(planned), "--lmi", str(lmi), "--out", str(tmp_path / "r.json")]) == 1
