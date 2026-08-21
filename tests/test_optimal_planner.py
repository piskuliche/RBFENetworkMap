"""The optimal-design planner.

Costs here are read as predicted standard deviations in kcal/mol, which is what the
``variance`` scorer returns -- but as everywhere else in the planner tests, no chemistry is
involved: a transformation carries only its endpoints, feasibility, and cost.

The load-bearing test is :meth:`TestHeuristicQuality.test_within_the_published_bound`. It
enumerates every edge subset of the chosen size on a five-ligand complete graph and checks
the heuristic lands within 1.10x of the true optimum -- the bound Xu's Appendix H is
published at. Anything that quietly degrades the selection shows up there and nowhere else,
because a worse design is still a perfectly valid network.
"""

from __future__ import annotations

import itertools
import math

import pytest

from rbfenetmap.core.design import a_optimal_criterion, criterion_value, fisher_information
from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.models import Ligand
from rbfenetmap.core.options import NetworkOptions
from rbfenetmap.plugins.planners import create_planner
from rbfenetmap.plugins.planners.optimal_planner import minimum_edge_count

from .conftest import make_transformation

#: Five ligands, and a reproducible spread of predicted standard deviations across the
#: complete graph. Written out rather than generated so the brute-force comparison below is
#: a fixed, reviewable problem instead of a different one on every run.
SIGMAS: dict[tuple[str, str], float] = {
    ("bza_CF3", "bza_Cl"): 2.31,
    ("bza_CF3", "bza_F"): 1.04,
    ("bza_CF3", "bza_H"): 2.05,
    ("bza_CF3", "bza_Me"): 0.62,
    ("bza_Cl", "bza_F"): 1.77,
    ("bza_Cl", "bza_H"): 0.88,
    ("bza_Cl", "bza_Me"): 2.44,
    ("bza_F", "bza_H"): 0.51,
    ("bza_F", "bza_Me"): 1.93,
    ("bza_H", "bza_Me"): 1.36,
}


@pytest.fixture
def five_ligands(benzamides) -> dict[str, Ligand]:
    """Five real ligands, so :class:`Network` validation is satisfied."""
    return {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl", "bza_Me", "bza_CF3")}


@pytest.fixture
def complete_candidates():
    """Every pair of the five, at the costs in :data:`SIGMAS`."""
    return [make_transformation(a, b, cost=sigma) for (a, b), sigma in SIGMAS.items()]


def _design(criterion: str, **overrides) -> NetworkOptions:
    """Options for a design run, with the best-effort targets silenced.

    ``edges_per_ligand`` and ``min_cycle_coverage`` are not what this planner optimises, and
    leaving them at their defaults would fill ``unmet_constraints`` with noise unrelated to
    the assertion under test.
    """
    return NetworkOptions(design=criterion, edges_per_ligand=1, min_cycle_coverage=0.0, **overrides)


def _criterion_of(network, criterion: str) -> float:
    """Evaluate *criterion* on a planned network, from its own edges and costs."""
    nodes = sorted(network.ligands)
    edges = sorted(edge.unordered_key for edge in network.edges)
    return criterion_value(fisher_information(nodes, edges, [SIGMAS[pair] for pair in edges]), criterion)


def _brute_force(nodes, criterion: str, k: int) -> float:
    """The best achievable criterion over every ``k``-subset of the complete graph."""
    pairs = sorted(SIGMAS)
    best = math.inf
    for subset in itertools.combinations(pairs, k):
        value = criterion_value(fisher_information(nodes, list(subset), [SIGMAS[p] for p in subset]), criterion)
        best = min(best, value)
    return best


class TestCriterionIsRequired:
    def test_design_none_is_refused_rather_than_guessed(self, five_ligands, complete_candidates):
        """The two criteria answer different questions; neither is a safe default."""
        with pytest.raises(NetworkPlanError, match="needs a design criterion"):
            create_planner("optimal").plan(five_ligands, complete_candidates, NetworkOptions())

    def test_a_planner_that_cannot_optimise_refuses_the_flag(self):
        """``--design`` under ``mst`` would be a knob that silently did nothing."""
        with pytest.raises(NetworkPlanError, match="does not optimise a design criterion"):
            create_planner("mst").check_design_support(NetworkOptions(design="a_optimal"))

    def test_the_optimal_planner_accepts_it(self):
        create_planner("optimal").check_design_support(NetworkOptions(design="d_optimal"))

    def test_design_none_is_accepted_by_every_planner(self):
        for name in ("mst", "star", "explicit", "complete", "optimal"):
            create_planner(name).check_design_support(NetworkOptions())

    def test_the_optimal_planner_refuses_cbfe_placement(self):
        """It has no notion of bridging, so the flag would be ignored."""
        with pytest.raises(NetworkPlanError, match="cannot place CBFE edges"):
            create_planner("optimal").check_cbfe_support(NetworkOptions(cbfe_mode="bridge"))


class TestSelection:
    @pytest.mark.parametrize("criterion", ["a_optimal", "d_optimal"])
    def test_the_network_spans_and_honours_the_budget(self, five_ligands, complete_candidates, criterion):
        network = create_planner("optimal").plan(five_ligands, complete_candidates, _design(criterion, n_edges=7))
        assert len(network.edges) == 7
        assert network.planner == "optimal"

    @pytest.mark.parametrize("criterion", ["a_optimal", "d_optimal"])
    def test_the_result_has_no_bridges(self, five_ligands, complete_candidates, criterion):
        """Not forced, but what the criterion arrives at on its own.

        A bridge is an edge no cycle checks. The planner does not buy a bridge cover -- that
        costs more than it buys, see the module docstring -- so this asserts the criterion
        reaches a bridgeless design by itself when the budget allows one.
        """
        import networkx as nx

        network = create_planner("optimal").plan(five_ligands, complete_candidates, _design(criterion, n_edges=7))
        graph = nx.Graph()
        graph.add_nodes_from(network.ligands)
        graph.add_edges_from(edge.unordered_key for edge in network.edges)
        assert not list(nx.bridges(graph))

    def test_an_unset_budget_uses_pitmans_floor(self, five_ligands, complete_candidates):
        """A design planner that ignored ``n ln n`` would optimise inside a known-bad budget."""
        network = create_planner("optimal").plan(five_ligands, complete_candidates, _design("a_optimal"))
        assert len(network.edges) == minimum_edge_count(5) == 8

    def test_a_forced_edge_is_selected_even_when_it_hurts(self, five_ligands, complete_candidates):
        network = create_planner("optimal").plan(
            five_ligands, complete_candidates, _design("a_optimal", n_edges=5, forced_edges=("bza_Cl~bza_Me",))
        )
        assert ("bza_Cl", "bza_Me") in {edge.unordered_key for edge in network.edges}

    def test_a_banned_edge_is_excluded(self, five_ligands, complete_candidates):
        network = create_planner("optimal").plan(
            five_ligands, complete_candidates, _design("a_optimal", n_edges=7, banned_edges=("bza_F~bza_H",))
        )
        assert ("bza_F", "bza_H") not in {edge.unordered_key for edge in network.edges}

    def test_an_infeasible_forced_edge_raises_with_the_reason(self, five_ligands):
        candidates = [make_transformation(a, b, cost=s) for (a, b), s in SIGMAS.items()]
        candidates.append(make_transformation("bza_F", "bza_H", feasible=False))
        with pytest.raises(NetworkPlanError, match="Forced edge"):
            create_planner("optimal").plan(
                five_ligands,
                [c for c in candidates if c.unordered_key != ("bza_F", "bza_H") or not c.feasible],
                _design("a_optimal", forced_edges=("bza_F~bza_H",)),
            )

    def test_a_disconnected_pool_raises(self, five_ligands):
        candidates = [
            make_transformation("bza_H", "bza_F", cost=1.0),
            make_transformation("bza_Cl", "bza_Me", cost=1.0),
        ]
        with pytest.raises(NetworkPlanError, match="disconnected"):
            create_planner("optimal").plan(five_ligands, candidates, _design("a_optimal"))

    def test_a_degree_shortfall_is_recorded_not_forced(self, five_ligands, complete_candidates):
        """Spending budget on a degree target would work against the stated objective."""
        network = create_planner("optimal").plan(
            five_ligands, complete_candidates, NetworkOptions(design="a_optimal", n_edges=4, edges_per_ligand=3)
        )
        assert any("edges_per_ligand" in c for c in network.unmet_constraints)

    def test_more_edges_than_the_pool_holds_is_recorded(self, five_ligands):
        candidates = [make_transformation(a, b, cost=s) for (a, b), s in list(SIGMAS.items())[:6]]
        network = create_planner("optimal").plan(five_ligands, candidates, _design("a_optimal", n_edges=9))
        assert any("only 6 feasible candidate" in c for c in network.unmet_constraints)


class TestHeuristicQuality:
    """The claim the whole planner rests on, checked against exhaustive enumeration."""

    @pytest.mark.parametrize("criterion", ["a_optimal", "d_optimal"])
    @pytest.mark.parametrize("k", [5, 6, 7, 8])
    def test_within_the_published_bound(self, five_ligands, complete_candidates, criterion, k):
        """Appendix H is published at 1.10 +/- 0.03x of the optimum. Hold it to 1.10x."""
        nodes = sorted(five_ligands)
        network = create_planner("optimal").plan(five_ligands, complete_candidates, _design(criterion, n_edges=k))
        achieved = _criterion_of(network, criterion)
        best = _brute_force(nodes, criterion, k)

        if criterion == "a_optimal":
            assert achieved <= 1.10 * best
        else:
            # ln det is a logarithm, so "within 1.10x" is a ratio of the determinants
            # themselves; comparing the logs directly would be a much tighter and
            # differently-shaped claim, and negative values make it meaningless besides.
            assert achieved - best <= math.log(1.10)

    @pytest.mark.parametrize("criterion", ["a_optimal", "d_optimal"])
    def test_the_bound_holds_across_randomised_costs(self, five_ligands, criterion):
        """One fixed instance can be lucky; twenty random ones cannot all be.

        Seeded, so a failure is reproducible and is a real regression rather than an
        unlucky draw that vanishes on re-run.
        """
        import numpy as np

        nodes = sorted(five_ligands)
        rng = np.random.default_rng(20260821)
        for _ in range(20):
            sigmas = {pair: float(rng.uniform(0.3, 3.0)) for pair in sorted(SIGMAS)}
            candidates = [make_transformation(a, b, cost=s) for (a, b), s in sigmas.items()]
            network = create_planner("optimal").plan(five_ligands, candidates, _design(criterion, n_edges=7))
            chosen = sorted(edge.unordered_key for edge in network.edges)
            achieved = criterion_value(fisher_information(nodes, chosen, [sigmas[p] for p in chosen]), criterion)
            best = min(
                criterion_value(fisher_information(nodes, list(s), [sigmas[p] for p in s]), criterion)
                for s in itertools.combinations(sorted(sigmas), 7)
            )
            ratio = achieved / best if criterion == "a_optimal" else math.exp(achieved - best)
            assert ratio <= 1.10

    def test_a_optimal_beats_the_cheapest_subset_of_the_same_size(self, five_ligands, complete_candidates):
        """Otherwise the planner is an expensive way to sort by cost."""
        nodes = sorted(five_ligands)
        network = create_planner("optimal").plan(five_ligands, complete_candidates, _design("a_optimal", n_edges=6))
        cheapest = sorted(SIGMAS, key=lambda pair: SIGMAS[pair])[:6]
        by_cost = a_optimal_criterion(fisher_information(nodes, cheapest, [SIGMAS[p] for p in cheapest]))
        assert _criterion_of(network, "a_optimal") < by_cost

    def test_d_optimal_is_at_least_as_cyclic_as_a_optimal(self, five_ligands, complete_candidates):
        """Pitman's argument for it: the pseudo-determinant counts spanning trees."""
        import networkx as nx

        def cycle_rank(criterion):
            network = create_planner("optimal").plan(five_ligands, complete_candidates, _design(criterion, n_edges=7))
            graph = nx.Graph()
            graph.add_nodes_from(network.ligands)
            graph.add_edges_from(edge.unordered_key for edge in network.edges)
            return len(nx.cycle_basis(graph))

        assert cycle_rank("d_optimal") >= cycle_rank("a_optimal")

    def test_refinement_never_worsens_the_criterion(self, five_ligands, complete_candidates):
        """Fedorov exchange only ever applies a swap that improves the criterion."""
        plain = create_planner("optimal").plan(five_ligands, complete_candidates, _design("a_optimal", n_edges=6))
        refined = create_planner("optimal").plan(
            five_ligands, complete_candidates, _design("a_optimal", n_edges=6, design_refine=True)
        )
        assert _criterion_of(refined, "a_optimal") <= _criterion_of(plain, "a_optimal") + 1e-12


class TestOptionsValidation:
    """Out-of-range design knobs are refused at construction, as everything else here is."""

    def test_an_unknown_criterion_is_refused(self):
        with pytest.raises(ValueError, match="design must be one of"):
            NetworkOptions(design="e_optimal")

    def test_a_pool_smaller_than_the_ligand_set_is_refused(self):
        with pytest.raises(ValueError, match="design_candidate_factor"):
            NetworkOptions(design_candidate_factor=0.5)

    def test_a_non_positive_budget_is_refused(self):
        with pytest.raises(ValueError, match="design_total_ns"):
            NetworkOptions(design_total_ns=0.0)

    def test_a_one_window_schedule_is_refused(self):
        with pytest.raises(ValueError, match="two end states"):
            NetworkOptions(design_lambda_min=1)

    def test_an_inverted_window_range_is_refused(self):
        with pytest.raises(ValueError, match="below design_lambda_min"):
            NetworkOptions(design_lambda_min=20, design_lambda_max=10)

    def test_the_defaults_are_the_no_op(self):
        """Everything new here defaults to what v0.4.0 did, which was nothing."""
        options = NetworkOptions()
        assert options.design == "none"
        assert options.design_total_ns is None
        assert options.design_refine is False


class TestMinimumEdgeCount:
    @pytest.mark.parametrize("n,expected", [(0, 0), (1, 0), (2, 1), (5, 8), (10, 23), (40, 148)])
    def test_pitmans_floor(self, n, expected):
        """40 ligands need 148 edges, against the ~40 ``edges_per_ligand=2`` buys."""
        assert minimum_edge_count(n) == expected

    def test_it_never_falls_below_spanning(self):
        for n in range(2, 12):
            assert minimum_edge_count(n) >= n - 1
