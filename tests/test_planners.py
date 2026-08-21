"""Planner tests, with hand-authored costs so every result is checkable by eye.

No chemistry here at all: the planner reads only endpoints, feasibility, and cost, so
these tests build transformations directly. That makes the expected minimum spanning tree
something a reader can verify by adding up four numbers.
"""

from __future__ import annotations

import networkx as nx
import pytest

from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.models import EdgeKind, Ligand
from rbfenetmap.core.options import NetworkOptions
from rbfenetmap.plugins.planners import create_planner

from .conftest import make_transformation


@pytest.fixture
def square_ligands(benzamides) -> dict[str, Ligand]:
    """Four ligands, reusing real molecules so :class:`Network` validation is satisfied."""
    return {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl", "bza_Me")}


@pytest.fixture
def five_ligands(benzamides) -> dict[str, Ligand]:
    """Five ligands for cycle-coverage tests."""
    return {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl", "bza_Me", "bza_CF3")}


@pytest.fixture
def square_candidates():
    """A 4-cycle with a known cheapest spanning tree.

    Costs::

        H --1-- F
        |       |
        4       2
        |       |
        Me --8-- Cl        plus the diagonals H-Cl (3) and F-Me (9)

    The minimum spanning tree is {H-F (1), F-Cl (2), H-Cl (3)}... but H-Cl closes a
    cycle, so Kruskal takes H-F (1), F-Cl (2), then H-Me (4): total 7.
    """
    costs = {
        ("bza_H", "bza_F"): 1.0,
        ("bza_F", "bza_Cl"): 2.0,
        ("bza_H", "bza_Cl"): 3.0,
        ("bza_H", "bza_Me"): 4.0,
        ("bza_Cl", "bza_Me"): 8.0,
        ("bza_F", "bza_Me"): 9.0,
    }
    return [make_transformation(a, b, cost=cost) for (a, b), cost in costs.items()]


class TestMSTPlanner:
    def test_selects_the_minimum_spanning_tree_then_adds_redundancy(self, square_ligands, square_candidates):
        planner = create_planner("mst")
        network = planner.plan(
            square_ligands, square_candidates, NetworkOptions(edges_per_ligand=1, min_cycle_coverage=0.0)
        )
        selected = {e.unordered_key for e in network.edges}
        assert selected == {("bza_F", "bza_H"), ("bza_Cl", "bza_F"), ("bza_H", "bza_Me")}
        assert sum(e.score.total for e in network.edges) == pytest.approx(7.0)

    def test_redundancy_raises_the_minimum_degree(self, square_ligands, square_candidates):
        planner = create_planner("mst")
        network = planner.plan(square_ligands, square_candidates, NetworkOptions(edges_per_ligand=2))
        degrees = dict(network.to_networkx().degree())
        assert min(degrees.values()) >= 2

    def test_redundancy_is_additive_and_keeps_the_tree(self, square_ligands, square_candidates):
        planner = create_planner("mst")
        sparse = planner.plan(
            square_ligands, square_candidates, NetworkOptions(edges_per_ligand=1, min_cycle_coverage=0.0)
        )
        dense = planner.plan(square_ligands, square_candidates, NetworkOptions(edges_per_ligand=3))
        assert {e.unordered_key for e in sparse.edges} <= {e.unordered_key for e in dense.edges}

    def test_forced_edge_is_selected_even_when_expensive(self, square_ligands, square_candidates):
        planner = create_planner("mst")
        network = planner.plan(
            square_ligands,
            square_candidates,
            NetworkOptions(forced_edges=("bza_F~bza_Me",), edges_per_ligand=1, min_cycle_coverage=0.0),
        )
        assert ("bza_F", "bza_Me") in {e.unordered_key for e in network.edges}

    def test_banned_edge_is_excluded(self, square_ligands, square_candidates):
        planner = create_planner("mst")
        network = planner.plan(square_ligands, square_candidates, NetworkOptions(banned_edges=("bza_H~bza_F",)))
        assert ("bza_F", "bza_H") not in {e.unordered_key for e in network.edges}

    def test_infeasible_forced_edge_raises_with_the_reason(self, square_ligands, square_candidates):
        candidates = [*square_candidates, make_transformation("bza_H", "bza_Me", feasible=False)]
        candidates = [c for c in candidates if c.unordered_key != ("bza_H", "bza_Me") or not c.feasible]
        planner = create_planner("mst")
        with pytest.raises(NetworkPlanError, match="softcore_too_large"):
            planner.plan(
                square_ligands, candidates, NetworkOptions(forced_edges=("bza_H~bza_Me",), require_connected=False)
            )

    def test_disconnected_pool_names_the_bridging_rejections(self, square_ligands):
        candidates = [
            make_transformation("bza_H", "bza_F", cost=1.0),
            make_transformation("bza_Cl", "bza_Me", cost=1.0),
            make_transformation("bza_F", "bza_Cl", feasible=False),
        ]
        planner = create_planner("mst")
        with pytest.raises(NetworkPlanError) as excinfo:
            planner.plan(square_ligands, candidates, NetworkOptions())
        message = str(excinfo.value)
        assert "disconnected" in message
        assert "bza_F~bza_Cl" in message, "the message must name the bridge that was rejected"
        assert "softcore_too_large" in message, "and why it was rejected"

    def test_allow_disconnected_records_an_unmet_constraint(self, square_ligands):
        candidates = [
            make_transformation("bza_H", "bza_F", cost=1.0),
            make_transformation("bza_Cl", "bza_Me", cost=1.0),
        ]
        planner = create_planner("mst")
        network = planner.plan(square_ligands, candidates, NetworkOptions(require_connected=False, edges_per_ligand=1))
        assert any("disconnected" in c for c in network.unmet_constraints)

    def test_unmet_degree_target_warns_rather_than_raising(self, square_ligands):
        candidates = [
            make_transformation("bza_H", "bza_F", cost=1.0),
            make_transformation("bza_F", "bza_Cl", cost=1.0),
            make_transformation("bza_Cl", "bza_Me", cost=1.0),
        ]
        planner = create_planner("mst")
        with pytest.warns(UserWarning, match="edges_per_ligand"):
            network = planner.plan(square_ligands, candidates, NetworkOptions(edges_per_ligand=3))
        assert any("edges_per_ligand" in c for c in network.unmet_constraints)

    def test_connectivity_then_cycles_prefers_short_cycle_coverage(self, five_ligands):
        candidates = [
            make_transformation("bza_H", "bza_F", cost=1.0),
            make_transformation("bza_F", "bza_Cl", cost=1.0),
            make_transformation("bza_Cl", "bza_Me", cost=1.0),
            make_transformation("bza_Me", "bza_CF3", cost=1.0),
            make_transformation("bza_H", "bza_CF3", cost=1.1),
            make_transformation("bza_H", "bza_Cl", cost=2.0),
            make_transformation("bza_Cl", "bza_CF3", cost=2.0),
        ]
        planner = create_planner("mst")
        network = planner.plan(
            five_ligands,
            candidates,
            NetworkOptions(
                edges_per_ligand=2,
                min_cycle_coverage=0.6,
                n_edges=5,
                selection_objective="connectivity_then_cycles",
                max_cycle_size=3,
            ),
        )
        selected = {e.unordered_key for e in network.edges}
        assert ("bza_CF3", "bza_H") not in selected, "the short-cycle objective should skip the cheap 5-cycle"
        assert len(selected & {("bza_Cl", "bza_H"), ("bza_CF3", "bza_Cl")}) == 1
        assert any("edges_per_ligand" in c for c in network.unmet_constraints)


class TestCBFEPlacement:
    """Where the planner is and is not allowed to spend a counterpoised edge.

    The disconnected pool here is the same one ``test_disconnected_pool_names_the_bridging_rejections``
    proves is a hard failure with CBFE off, so these tests read as the delta the feature buys.
    """

    @staticmethod
    def _split_pool():
        """Two components, {H, F} and {Cl, Me}, with the only bridge rejected."""
        return [
            make_transformation("bza_H", "bza_F", cost=1.0),
            make_transformation("bza_Cl", "bza_Me", cost=1.0),
            make_transformation("bza_F", "bza_Cl", feasible=False),
        ]

    def test_bridge_joins_components_that_would_otherwise_raise(self, square_ligands):
        network = create_planner("mst").plan(
            square_ligands, self._split_pool(), NetworkOptions(cbfe_mode="bridge", edges_per_ligand=1)
        )
        assert len(network.cbfe_edges) == 1, "two components need exactly one bridge"
        assert len(network.rbfe_edges) == 2
        assert not any("disconnected" in c for c in network.unmet_constraints)

    def test_bridge_adds_nothing_to_an_already_connected_pool(self, square_ligands, square_candidates):
        network = create_planner("mst").plan(square_ligands, square_candidates, NetworkOptions(cbfe_mode="bridge"))
        assert network.cbfe_edges == ()

    def test_bridge_alone_does_not_close_cycles(self, square_ligands):
        with pytest.warns(UserWarning, match="min_cycle_coverage"):
            network = create_planner("mst").plan(
                square_ligands, self._split_pool(), NetworkOptions(cbfe_mode="bridge", edges_per_ligand=1)
            )
        assert any("min_cycle_coverage" in c for c in network.unmet_constraints)
        assert len(network.cbfe_edges) == 1

    def test_cycles_spends_further_edges_to_meet_the_coverage_target(self, square_ligands):
        network = create_planner("mst").plan(
            square_ligands, self._split_pool(), NetworkOptions(cbfe_mode="cycles", edges_per_ligand=1)
        )
        assert len(network.cbfe_edges) > 1, "beyond the single bridge, to close a cycle"
        assert not any("min_cycle_coverage" in c for c in network.unmet_constraints)

    def test_cycles_does_not_spend_cbfe_on_degree_targets(self, square_ligands):
        """The injection ordering, pinned.

        Degree raising runs against a pool with the CBFE edges filtered out, so an
        unreachable ``edges_per_ligand`` must still be reported rather than quietly bought
        with counterpoised calculations.
        """
        candidates = [
            make_transformation("bza_H", "bza_F", cost=1.0),
            make_transformation("bza_F", "bza_Cl", cost=1.0),
            make_transformation("bza_Cl", "bza_Me", cost=1.0),
            make_transformation("bza_H", "bza_Me", cost=1.0),
        ]
        options = NetworkOptions(cbfe_mode="cycles", edges_per_ligand=3)
        with pytest.warns(UserWarning, match="edges_per_ligand"):
            network = create_planner("mst").plan(square_ligands, candidates, options)
        assert any("edges_per_ligand" in c for c in network.unmet_constraints)
        assert every_ligand_has_degree_two(network)

    def test_an_rbfe_edge_is_preferred_over_a_cheaper_cbfe_edge_for_the_same_coverage(self, square_ligands):
        """Cost already carries the CBFE penalty; the ranking is about provenance."""
        candidates = [
            make_transformation("bza_H", "bza_F", cost=1.0),
            make_transformation("bza_F", "bza_Cl", cost=1.0),
            make_transformation("bza_Cl", "bza_Me", cost=1.0),
            make_transformation("bza_H", "bza_Me", cost=20.0),
        ]
        # A CBFE edge would cost ~9, well under the 20 of the closing RBFE edge.
        network = create_planner("mst").plan(
            square_ligands, candidates, NetworkOptions(cbfe_mode="cycles", edges_per_ligand=1)
        )
        assert network.cbfe_edges == (), "the expensive relative edge should still win"
        assert ("bza_H", "bza_Me") in {e.unordered_key for e in network.edges}

    def test_a_forced_pair_only_cbfe_can_supply_is_honoured_and_reported(self, square_ligands):
        candidates = [
            make_transformation("bza_H", "bza_F", cost=1.0),
            make_transformation("bza_F", "bza_Cl", cost=1.0),
            make_transformation("bza_Cl", "bza_Me", cost=1.0),
            make_transformation("bza_H", "bza_Me", feasible=False),
        ]
        options = NetworkOptions(cbfe_mode="bridge", forced_edges=("bza_H~bza_Me",), edges_per_ligand=1)
        network = create_planner("mst").plan(square_ligands, candidates, options)

        forced = next(e for e in network.edges if e.unordered_key == ("bza_H", "bza_Me"))
        assert forced.kind is EdgeKind.CBFE
        assert any("supplied as a CBFE edge" in c for c in network.unmet_constraints), (
            "a substitution the user did not ask for must be reported, not silent"
        )

    def test_a_forced_pair_is_honoured_even_inside_one_component(self, square_ligands, square_candidates):
        """Forced outranks the mode ladder: bridge mode still supplies a non-bridging pair."""
        candidates = [c for c in square_candidates if c.unordered_key != ("bza_Cl", "bza_Me")]
        candidates.append(make_transformation("bza_Cl", "bza_Me", feasible=False))
        options = NetworkOptions(cbfe_mode="bridge", forced_edges=("bza_Cl~bza_Me",))
        network = create_planner("mst").plan(square_ligands, candidates, options)
        assert ("bza_Cl", "bza_Me") in {e.unordered_key for e in network.cbfe_edges}

    def test_banned_pairs_are_never_used_as_bridges(self, square_ligands):
        """With every crossing pair banned, the connectivity failure must still be a failure."""
        options = NetworkOptions(
            cbfe_mode="bridge", banned_edges=("bza_H~bza_Cl", "bza_H~bza_Me", "bza_F~bza_Cl", "bza_F~bza_Me")
        )
        with pytest.raises(NetworkPlanError) as excinfo:
            create_planner("mst").plan(square_ligands, self._split_pool(), options)
        message = str(excinfo.value)
        assert "banned_edges" in message
        assert "Loosen the soft-core budget" not in message, "the soft-core budget is irrelevant to a CBFE bridge"

    def test_off_mode_advertises_the_bridge_option(self, square_ligands):
        with pytest.raises(NetworkPlanError, match="cbfe_mode='bridge'"):
            create_planner("mst").plan(square_ligands, self._split_pool(), NetworkOptions())

    @pytest.mark.parametrize("planner", ["star", "complete", "explicit"])
    @pytest.mark.parametrize("mode", ["bridge", "cycles"])
    def test_planners_that_cannot_place_cbfe_edges_say_so(self, square_ligands, planner, mode):
        options = NetworkOptions(cbfe_mode=mode, explicit_pairs=("bza_H~bza_F",), hub="bza_H")
        with pytest.raises(NetworkPlanError, match="cannot place CBFE edges"):
            create_planner(planner).plan(square_ligands, [], options)

    @pytest.mark.parametrize("planner", ["star", "complete"])
    def test_all_mode_needs_no_planner_support(self, square_ligands, square_candidates, planner):
        """``all`` is a pool substitution made upstream, so any planner can honour it."""
        create_planner(planner).plan(square_ligands, square_candidates, NetworkOptions(cbfe_mode="all", hub="bza_H"))


def every_ligand_has_degree_two(network) -> bool:
    """Whether no ligand exceeded the degree the RBFE pool alone could support."""
    graph = network.to_networkx()
    return all(graph.degree(node) <= 2 for node in graph.nodes)


class TestKnobPrecedence:
    def test_forced_and_banned_overlap_rejected_at_construction(self):
        with pytest.raises(ValueError, match="both forced_edges and banned_edges"):
            NetworkOptions(forced_edges=("a~b",), banned_edges=("b~a",))

    def test_n_edges_below_spanning_minimum_is_a_hard_error(self):
        with pytest.raises(ValueError, match="cannot connect 12 ligands"):
            NetworkOptions(n_edges=8).check_edge_budget(12)

    def test_n_edges_allowed_when_connectivity_is_not_required(self):
        NetworkOptions(n_edges=3, require_connected=False).check_edge_budget(12)

    def test_edge_specs_normalize_to_unordered_pairs(self):
        options = NetworkOptions(banned_edges=("b~a",))
        assert options.banned_pairs == frozenset({("a", "b")})

    def test_star_strategy_requires_a_hub(self):
        with pytest.raises(ValueError, match="requires a hub"):
            NetworkOptions(pair_strategy="star")

    def test_selection_objective_is_validated(self):
        with pytest.raises(ValueError, match="selection_objective"):
            NetworkOptions(selection_objective="nope")  # type: ignore[arg-type]

    def test_max_cycle_size_must_allow_a_cycle(self):
        with pytest.raises(ValueError, match="max_cycle_size"):
            NetworkOptions(max_cycle_size=2)

    def test_adaptive_batch_sizes_are_validated(self):
        with pytest.raises(ValueError, match="adaptive_initial_neighbors"):
            NetworkOptions(adaptive_initial_neighbors=0)
        with pytest.raises(ValueError, match="adaptive_batch_size"):
            NetworkOptions(adaptive_batch_size=0)


class TestSimplePlanners:
    def test_complete_selects_every_feasible_edge(self, square_ligands, square_candidates):
        network = create_planner("complete").plan(square_ligands, square_candidates, NetworkOptions())
        assert len(network.edges) == len(square_candidates)

    def test_complete_honours_the_edge_cap(self, square_ligands, square_candidates):
        network = create_planner("complete").plan(square_ligands, square_candidates, NetworkOptions(n_edges=4))
        assert len(network.edges) == 4
        assert any("truncated" in c for c in network.unmet_constraints)

    def test_star_connects_everything_to_the_hub(self, square_ligands, square_candidates):
        network = create_planner("star").plan(
            square_ligands, square_candidates, NetworkOptions(pair_strategy="star", hub="bza_H")
        )
        assert all("bza_H" in e.unordered_key for e in network.edges)
        assert len(network.edges) == 3

    def test_explicit_selects_exactly_what_it_is_given(self, square_ligands, square_candidates):
        network = create_planner("explicit").plan(
            square_ligands,
            square_candidates,
            NetworkOptions(
                pair_strategy="explicit",
                explicit_pairs=("bza_H~bza_F", "bza_F~bza_Cl", "bza_H~bza_Me"),
                require_connected=True,
            ),
        )
        assert len(network.edges) == 3

    def test_explicit_refuses_to_silently_omit_an_infeasible_edge(self, square_ligands):
        candidates = [make_transformation("bza_H", "bza_F", feasible=False)]
        with pytest.raises(NetworkPlanError, match="not feasible"):
            create_planner("explicit").plan(
                square_ligands,
                candidates,
                NetworkOptions(pair_strategy="explicit", explicit_pairs=("bza_H~bza_F",), require_connected=False),
            )


def _graph_of(network) -> nx.Graph:
    """The selected edges as a plain graph, for shape assertions."""
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(network.ligands)
    graph.add_edges_from(edge.unordered_key for edge in network.edges)
    return graph


@pytest.fixture
def path_ligands(benzamides) -> dict[str, Ligand]:
    """Six ligands. Real molecules only so that Network validation is satisfied."""
    names = ("bza_H", "bza_F", "bza_Cl", "bza_Me", "bza_CF3")
    ligands = {name: benzamides[name] for name in names}
    ligands["bza_extra"] = benzamides["bza_H"]
    return ligands


@pytest.fixture
def path_order() -> tuple[str, ...]:
    """The order the cheap edges lay the six ligands out in."""
    return ("bza_H", "bza_F", "bza_Cl", "bza_Me", "bza_CF3", "bza_extra")


@pytest.fixture
def path_candidates(path_order):
    """A cheap path plus expensive shortcuts, so the spanning tree is the path itself.

    The MST is therefore a line of five edges with diameter 5, which is the shape a
    diameter bound exists to fix.
    """
    candidates = []
    for i, a in enumerate(path_order):
        for b in path_order[i + 1 :]:
            gap = path_order.index(b) - i
            candidates.append(make_transformation(a, b, cost=1.0 if gap == 1 else 10.0 + gap))
    return candidates


class TestDiameterBound:
    """``max_diameter`` as an additive third redundancy pass."""

    def _plan(self, ligands, candidates, **kwargs):
        options = NetworkOptions(edges_per_ligand=1, min_cycle_coverage=0.0, **kwargs)
        return create_planner("mst").plan(ligands, candidates, options)

    def test_unset_leaves_the_long_path_alone(self, path_ligands, path_candidates):
        network = self._plan(path_ligands, path_candidates)
        assert nx.diameter(_graph_of(network)) == 5
        assert len(network.edges) == 5

    def test_a_bound_buys_exactly_the_shortcuts_it_needs(self, path_ligands, path_candidates):
        network = self._plan(path_ligands, path_candidates, max_diameter=3)
        assert nx.diameter(_graph_of(network)) <= 3
        assert len(network.edges) == 6  # the five path edges plus one shortcut
        assert network.unmet_constraints == ()

    def test_an_impossible_bound_warns_and_is_recorded(self, path_ligands, path_order):
        """Only the path edges exist, so nothing can shorten it. Best-effort, never a raise."""
        only_the_path = [make_transformation(a, b, cost=1.0) for a, b in zip(path_order, path_order[1:])]
        with pytest.warns(UserWarning, match="max_diameter=2 unmet"):
            network = self._plan(path_ligands, only_the_path, max_diameter=2)
        assert any("max_diameter=2 unmet" in c for c in network.unmet_constraints)
        assert len(network.edges) == 5

    def test_a_disconnected_network_reports_that_instead_of_a_diameter(self, path_ligands, path_order):
        """Diameter is undefined across components; saying "unmet" would name the wrong knob."""
        halves = [make_transformation(a, b, cost=1.0) for a, b in (path_order[:2], path_order[2:4], path_order[4:6])]
        options = NetworkOptions(edges_per_ligand=1, min_cycle_coverage=0.0, max_diameter=2, require_connected=False)
        network = create_planner("mst").plan(path_ligands, halves, options)
        assert any("diameter is undefined" in c for c in network.unmet_constraints)

    def test_the_edge_budget_still_caps_the_pass(self, path_ligands, path_candidates):
        network = self._plan(path_ligands, path_candidates, max_diameter=2, n_edges=5)
        assert len(network.edges) == 5


class TestCycleCoverageMode:
    """``node`` counts ligands on a cycle; ``edge`` counts edges, i.e. 2-edge-connectivity."""

    @pytest.fixture
    def two_triangles(self, path_ligands, path_order):
        """Two cheap triangles joined by one cheap-ish edge, everything else dear.

        The node rule is satisfied by this shape -- every ligand is on a cycle -- while the
        joining edge is a bridge that nothing checks. That gap is the whole reason the edge
        mode exists.
        """
        left, right = path_order[:3], path_order[3:]
        cheap = {
            tuple(sorted((a, b))): 1.0 for group in (left, right) for i, a in enumerate(group) for b in group[i + 1 :]
        }
        cheap[tuple(sorted((left[2], right[0])))] = 2.0
        candidates = []
        for i, a in enumerate(path_order):
            for b in path_order[i + 1 :]:
                key = tuple(sorted((a, b)))
                candidates.append(make_transformation(a, b, cost=cheap.get(key, 20.0)))
        return candidates

    def test_node_mode_is_satisfied_while_a_bridge_remains(self, path_ligands, two_triangles):
        network = create_planner("mst").plan(path_ligands, two_triangles, NetworkOptions())
        graph = _graph_of(network)
        assert list(nx.bridges(graph))  # the joining edge is checked by nothing
        assert network.unmet_constraints == ()

    def test_edge_mode_leaves_no_bridges_at_full_coverage(self, path_ligands, two_triangles):
        network = create_planner("mst").plan(path_ligands, two_triangles, NetworkOptions(cycle_coverage_mode="edge"))
        assert list(nx.bridges(_graph_of(network))) == []
        assert network.unmet_constraints == ()

    def test_edge_mode_is_the_stricter_target(self, path_ligands, two_triangles):
        node = create_planner("mst").plan(path_ligands, two_triangles, NetworkOptions())
        edge = create_planner("mst").plan(path_ligands, two_triangles, NetworkOptions(cycle_coverage_mode="edge"))
        assert len(edge.edges) > len(node.edges)

    def test_the_default_is_the_node_rule(self):
        assert NetworkOptions().cycle_coverage_mode == "node"

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="cycle_coverage_mode"):
            NetworkOptions(cycle_coverage_mode="ligand")  # type: ignore[arg-type]


class TestRedundantMSTPlanner:
    """``n_redundancy`` overlaid trees, per Konnektor."""

    def test_one_tree_reproduces_the_plain_mst(self, square_ligands, square_candidates):
        options = NetworkOptions(n_redundancy=1, edges_per_ligand=1, min_cycle_coverage=0.0)
        plain = create_planner("mst").plan(square_ligands, square_candidates, options)
        overlaid = create_planner("redundant-mst").plan(square_ligands, square_candidates, options)
        assert {e.unordered_key for e in overlaid.edges} == {e.unordered_key for e in plain.edges}

    def test_two_trees_add_a_second_independent_spanning_structure(self, square_ligands, square_candidates):
        options = NetworkOptions(n_redundancy=2, edges_per_ligand=1, min_cycle_coverage=0.0)
        plain = create_planner("mst").plan(square_ligands, square_candidates, options)
        overlaid = create_planner("redundant-mst").plan(square_ligands, square_candidates, options)
        first = {e.unordered_key for e in plain.edges}
        both = {e.unordered_key for e in overlaid.edges}
        assert first < both
        # Removing the first tree entirely still leaves every ligand attached.
        remainder: nx.Graph = nx.Graph()
        remainder.add_nodes_from(square_ligands)
        remainder.add_edges_from(both - first)
        assert min(dict(remainder.degree()).values()) >= 1

    def test_asking_for_more_trees_than_the_pool_holds_is_not_an_error(self, square_ligands, square_candidates):
        options = NetworkOptions(n_redundancy=9, edges_per_ligand=1, min_cycle_coverage=0.0)
        network = create_planner("redundant-mst").plan(square_ligands, square_candidates, options)
        assert len(network.edges) == len(square_candidates)

    def test_it_supports_cbfe(self):
        assert create_planner("redundant-mst").supports_cbfe is True

    def test_zero_trees_is_refused(self):
        with pytest.raises(ValueError, match="n_redundancy"):
            NetworkOptions(n_redundancy=0)


class TestHubSelection:
    """Hub choice is, by OpenEye's own account, the dominant factor in a star map."""

    @pytest.fixture
    def lopsided(self, square_ligands):
        """bza_H reaches everything at cost 5; bza_Cl reaches two ligands far more cheaply."""
        costs = {("bza_H", "bza_F"): 5.0, ("bza_H", "bza_Cl"): 5.0, ("bza_H", "bza_Me"): 5.0, ("bza_Cl", "bza_F"): 0.1}
        return [make_transformation(a, b, cost=cost) for (a, b), cost in costs.items()]

    def test_the_default_prefers_the_ligand_with_the_most_partners(self, square_ligands, lopsided):
        network = create_planner("star").plan(square_ligands, lopsided, NetworkOptions())
        assert {e.unordered_key for e in network.edges} == {
            ("bza_F", "bza_H"),
            ("bza_Cl", "bza_H"),
            ("bza_H", "bza_Me"),
        }

    def test_min_total_cost_prefers_the_cheapest_reachable_hub(self, square_ligands, lopsided):
        options = NetworkOptions(hub_selection="min_total_cost", require_connected=False)
        network = create_planner("star").plan(square_ligands, lopsided, options)
        assert {e.unordered_key for e in network.edges} == {("bza_Cl", "bza_F"), ("bza_Cl", "bza_H")}

    def test_an_explicit_hub_still_wins_over_either_rule(self, square_ligands, lopsided):
        options = NetworkOptions(hub="bza_H", hub_selection="min_total_cost")
        network = create_planner("star").plan(square_ligands, lopsided, options)
        assert all("bza_H" in edge.unordered_key for edge in network.edges)

    def test_an_unknown_rule_is_refused(self):
        with pytest.raises(ValueError, match="hub_selection"):
            NetworkOptions(hub_selection="most_central")  # type: ignore[arg-type]
