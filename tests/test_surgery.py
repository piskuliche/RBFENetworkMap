"""Network surgery: appending, deleting, merging, concatenating, cyclizing.

Most of these need no chemistry -- surgery reads endpoints, feasibility, and cost, exactly
as the planner does -- so they build transformations with ``make_transformation`` and the
expected answer is checkable by eye. The two operations that must map new pairs
(:func:`append_ligand`, :func:`concatenate_networks`) use the co-posed ``benzamides``
fixture, because anything that reaches the geometry check needs ligands in a shared frame.
"""

from __future__ import annotations

import pytest

from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.models import EdgeKind, Ligand, Network
from rbfenetmap.core.options import NetworkOptions
from rbfenetmap.core.surgery import (
    append_ligand,
    concatenate_networks,
    cyclize_around_component,
    delete_edge,
    merge_networks,
)
from rbfenetmap.core.pipeline import build_network

from .conftest import make_ligand, make_transformation


@pytest.fixture
def four(benzamides) -> dict[str, Ligand]:
    """Four co-posed ligands."""
    return {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl", "bza_Me")}


@pytest.fixture
def path_network(four) -> Network:
    """H - F - Cl - Me: every edge is a bridge."""
    edges = (
        make_transformation("bza_H", "bza_F", cost=1.0),
        make_transformation("bza_F", "bza_Cl", cost=2.0),
        make_transformation("bza_Cl", "bza_Me", cost=3.0),
    )
    return Network(ligands=four, edges=edges, candidates=edges, planner="dummy", options=NetworkOptions())


@pytest.fixture
def cycle_network(four) -> Network:
    """The same path, closed into a 4-cycle: no edge is a bridge."""
    edges = (
        make_transformation("bza_H", "bza_F", cost=1.0),
        make_transformation("bza_F", "bza_Cl", cost=2.0),
        make_transformation("bza_Cl", "bza_Me", cost=3.0),
        make_transformation("bza_H", "bza_Me", cost=4.0),
    )
    return Network(ligands=four, edges=edges, candidates=edges, planner="dummy", options=NetworkOptions())


class TestDeleteEdge:
    def test_refuses_a_bridge_and_names_both_sides(self, path_network):
        with pytest.raises(ValueError) as excinfo:
            delete_edge(path_network, "bza_F~bza_Cl")
        message = str(excinfo.value)
        assert "bza_Cl~bza_F is a bridge" in message
        assert "bza_H" in message and "bza_Me" in message

    def test_a_bridge_can_be_deleted_deliberately_and_is_recorded(self, path_network):
        result = delete_edge(path_network, "bza_F~bza_Cl", must_stay_connected=False)
        assert len(result.edges) == 2
        assert any("disconnected" in note for note in result.unmet_constraints)

    def test_a_cycle_edge_deletes_and_leaves_the_rest_identical(self, cycle_network):
        result = delete_edge(cycle_network, ("bza_H", "bza_Me"))
        assert len(result.edges) == 3
        # Identity, not equality: nothing is re-scored, so the survivors are the same objects.
        assert all(edge is original for edge, original in zip(result.edges, cycle_network.edges))
        result.validate()

    def test_the_input_network_is_untouched(self, cycle_network):
        delete_edge(cycle_network, "bza_H~bza_Me")
        assert len(cycle_network.edges) == 4

    def test_direction_is_irrelevant(self, cycle_network):
        assert len(delete_edge(cycle_network, "bza_Me~bza_H").edges) == 3

    def test_an_unknown_edge_lists_the_ones_present(self, cycle_network):
        with pytest.raises(ValueError, match="is not in the network"):
            delete_edge(cycle_network, "bza_H~bza_CF3")

    def test_a_self_loop_specification_is_refused(self, cycle_network):
        with pytest.raises(ValueError, match="same ligand twice"):
            delete_edge(cycle_network, ("bza_H", "bza_H"))


class TestAppendLigand:
    @pytest.mark.integration
    def test_the_result_still_validates_and_still_spans(self, four, benzamides):
        network = build_network(four)
        result = append_ligand(network, benzamides["bza_CF3"])
        result.validate(require_connected=True)
        assert set(result.ligands) == set(four) | {"bza_CF3"}
        assert len(result.edges) == len(network.edges) + 2

    @pytest.mark.integration
    def test_existing_edges_are_carried_over_untouched(self, four, benzamides):
        network = build_network(four)
        result = append_ligand(network, benzamides["bza_CF3"])
        assert result.edges[: len(network.edges)] == network.edges

    @pytest.mark.integration
    def test_new_candidates_join_the_audit_trail(self, four, benzamides):
        network = build_network(four)
        result = append_ligand(network, benzamides["bza_CF3"])
        added = {c.unordered_key for c in result.candidates} - {c.unordered_key for c in network.candidates}
        assert added == {tuple(sorted(("bza_CF3", name))) for name in four}

    @pytest.mark.integration
    def test_a_shortfall_is_best_effort_and_recorded(self, four, benzamides):
        network = build_network(four)
        result = append_ligand(network, benzamides["bza_CF3"], n_edges=len(four) + 2)
        assert any("unmet for appended ligand bza_CF3" in note for note in result.unmet_constraints)

    @pytest.mark.integration
    def test_a_duplicate_name_is_refused(self, four, benzamides):
        network = build_network(four)
        with pytest.raises(ValueError, match="already in the network"):
            append_ligand(network, benzamides["bza_H"])

    @pytest.mark.integration
    def test_an_unmappable_ligand_raises_with_the_reasons(self, four):
        network = build_network(four)
        stranger = make_ligand("CCCCCCCCCC", "decane")
        with pytest.raises(NetworkPlanError) as excinfo:
            append_ligand(network, stranger)
        assert "No feasible edge connects 'decane'" in str(excinfo.value)

    @pytest.mark.integration
    def test_cbfe_attaches_an_unmappable_ligand_when_the_mode_allows(self, four):
        options = NetworkOptions(cbfe_mode="bridge")
        network = build_network(four, network_options=options)
        result = append_ligand(network, make_ligand("CCCCCCCCCC", "decane"))
        assert [edge.kind for edge in result.edges[len(network.edges) :]] == [EdgeKind.CBFE]
        assert any("attached as a CBFE edge" in note for note in result.unmet_constraints)
        result.validate()


class TestMergeNetworks:
    def test_the_union_is_taken_over_a_shared_ligand(self, benzamides):
        left = Network(
            ligands={n: benzamides[n] for n in ("bza_H", "bza_F", "bza_Cl")},
            edges=(make_transformation("bza_H", "bza_F"), make_transformation("bza_F", "bza_Cl")),
            planner="mst",
            options=NetworkOptions(),
        )
        right = Network(
            ligands={n: benzamides[n] for n in ("bza_Cl", "bza_Me", "bza_CF3")},
            edges=(make_transformation("bza_Cl", "bza_Me"), make_transformation("bza_Me", "bza_CF3")),
            planner="mst",
            options=NetworkOptions(),
        )
        merged = merge_networks(left, right)
        assert set(merged.ligands) == set(left.ligands) | set(right.ligands)
        assert len(merged.edges) == 4
        merged.validate(require_connected=True)

    def test_a_pair_selected_by_both_keeps_the_cheaper_edge(self, benzamides):
        ligands = {n: benzamides[n] for n in ("bza_H", "bza_F", "bza_Cl")}
        left = Network(
            ligands=ligands,
            edges=(make_transformation("bza_H", "bza_F", cost=5.0), make_transformation("bza_F", "bza_Cl")),
            options=NetworkOptions(),
        )
        right = Network(
            ligands=ligands, edges=(make_transformation("bza_H", "bza_F", cost=1.0),), options=NetworkOptions()
        )
        merged = merge_networks(left, right)
        assert len(merged.edges) == 2
        assert {e.score.total for e in merged.edges} == {1.0, 1.0}

    def test_disjoint_networks_are_pointed_at_concatenate(self, benzamides):
        left = Network(ligands={n: benzamides[n] for n in ("bza_H", "bza_F")}, options=NetworkOptions())
        right = Network(ligands={n: benzamides[n] for n in ("bza_Cl", "bza_Me")}, options=NetworkOptions())
        with pytest.raises(ValueError, match="concatenate_networks"):
            merge_networks(left, right)

    def test_a_shared_name_that_is_a_different_molecule_is_refused(self, benzamides):
        left = Network(ligands={n: benzamides[n] for n in ("bza_H", "bza_F")}, options=NetworkOptions())
        impostor = make_ligand("CCCCCCCCCC", "bza_F")
        right = Network(ligands={"bza_F": impostor, "bza_Cl": benzamides["bza_Cl"]}, options=NetworkOptions())
        with pytest.raises(ValueError, match="not the same molecule"):
            merge_networks(left, right)


class TestConcatenateNetworks:
    @pytest.mark.integration
    def test_two_disjoint_networks_are_joined_and_span(self, benzamides):
        left = build_network({n: benzamides[n] for n in ("bza_H", "bza_F")})
        right = build_network({n: benzamides[n] for n in ("bza_Cl", "bza_Me")})
        joined = concatenate_networks(left, right)
        joined.validate(require_connected=True)
        assert len(joined.edges) == len(left.edges) + len(right.edges) + 2

    @pytest.mark.integration
    def test_the_second_bridge_avoids_the_first_bridge_endpoints(self, benzamides):
        left = build_network({n: benzamides[n] for n in ("bza_H", "bza_F")})
        right = build_network({n: benzamides[n] for n in ("bza_Cl", "bza_Me")})
        joined = concatenate_networks(left, right)
        bridges = [e.unordered_key for e in joined.edges[len(left.edges) + len(right.edges) :]]
        assert len({name for pair in bridges for name in pair}) == 4

    @pytest.mark.integration
    def test_overlapping_networks_are_pointed_at_merge(self, benzamides):
        left = build_network({n: benzamides[n] for n in ("bza_H", "bza_F")})
        right = build_network({n: benzamides[n] for n in ("bza_F", "bza_Cl")})
        with pytest.raises(ValueError, match="merge_networks"):
            concatenate_networks(left, right)


class TestCyclizeAroundComponent:
    @pytest.mark.integration
    def test_a_single_bridge_join_is_put_on_a_cycle(self, benzamides):
        left = build_network({n: benzamides[n] for n in ("bza_H", "bza_F")})
        right = build_network({n: benzamides[n] for n in ("bza_Cl", "bza_Me")})
        joined = concatenate_networks(left, right, n_bridges=1)
        cyclized = cyclize_around_component(joined)
        assert len(cyclized.edges) > len(joined.edges)
        assert not any("lie on no cycle" in note for note in cyclized.unmet_constraints)
        cyclized.validate(require_connected=True)

    def test_a_ligand_that_cannot_be_cycled_is_reported_not_raised(self, path_network):
        result = cyclize_around_component(path_network)
        assert any("lie on no cycle" in note for note in result.unmet_constraints)
        assert len(result.edges) == len(path_network.edges)

    def test_an_unknown_ligand_is_refused(self, cycle_network):
        with pytest.raises(ValueError, match="not in the network"):
            cyclize_around_component(cycle_network, ["bza_CF3"])

    def test_it_never_spends_a_cbfe_edge_where_an_rbfe_one_would_do(self, four):
        """The planner's rule, restated here: kind outranks cycle length and cost."""
        edges = (
            make_transformation("bza_H", "bza_F", cost=1.0),
            make_transformation("bza_F", "bza_Cl", cost=1.0),
            make_transformation("bza_Cl", "bza_Me", cost=1.0),
        )
        candidates = (
            *edges,
            make_transformation("bza_H", "bza_Me", cost=9.0),
            make_transformation("bza_H", "bza_Cl", cost=0.1, kind=EdgeKind.CBFE),
        )
        network = Network(ligands=four, edges=edges, candidates=candidates, options=NetworkOptions())
        result = cyclize_around_component(network)
        added = [e for e in result.edges if e not in edges]
        assert [e.kind for e in added] == [EdgeKind.RBFE]
