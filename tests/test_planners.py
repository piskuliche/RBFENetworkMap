"""Planner tests, with hand-authored costs so every result is checkable by eye.

No chemistry here at all: the planner reads only endpoints, feasibility, and cost, so
these tests build transformations directly. That makes the expected minimum spanning tree
something a reader can verify by adding up four numbers.
"""

from __future__ import annotations

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
