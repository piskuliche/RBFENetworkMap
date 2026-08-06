"""Tests for the soft-core connectivity repair.

Split into two layers. The graph-level tests use hand-built :class:`networkx.Graph`
objects with no chemistry at all, so the closure rules and the Steiner solver can be
checked against sets computed by hand. The chemistry-level tests then confirm the
documented behaviour on real molecules, including the cases the algorithm is expected to
*reject*.
"""

from __future__ import annotations

import networkx as nx
import pytest

from rbfenetmap.core.models import AtomMapping, RejectionReason
from rbfenetmap.core.molgraph import connected_components_of, node_weighted_steiner
from rbfenetmap.core.options import SoftcorePolicy
from rbfenetmap.core.softcore import (
    RepairContext,
    detect_fragments,
    joint_closure,
    precheck_mapping,
    repair_softcore_connectivity,
)

from .conftest import make_coposed, make_ligand


def _chain(n: int) -> nx.Graph:
    """A path graph ``0-1-...-(n-1)`` with no ring bonds."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n - 1):
        graph.add_edge(i, i + 1, in_ring=False)
    return graph


def _context(graph_1, graph_2, *, rings_1=(), rings_2=(), forward=None, policy=None) -> RepairContext:
    """Build a bare :class:`RepairContext` for graph-only tests."""
    forward = forward or {}
    return RepairContext(
        graph_1=graph_1,
        graph_2=graph_2,
        rings_1=tuple(frozenset(r) for r in rings_1),
        rings_2=tuple(frozenset(r) for r in rings_2),
        hydrogen_parent_1={},
        hydrogen_parent_2={},
        forward=dict(forward),
        reverse={v: k for k, v in forward.items()},
        heavy_1=frozenset(graph_1.nodes),
        heavy_2=frozenset(graph_2.nodes),
        policy=policy or SoftcorePolicy(),
        n_atoms_1=graph_1.number_of_nodes(),
        n_atoms_2=graph_2.number_of_nodes(),
    )


class TestFragmentDetection:
    def test_empty_softcore_has_no_regions(self):
        assert detect_fragments(_chain(5), set()) == []

    def test_contiguous_softcore_is_one_region(self):
        assert detect_fragments(_chain(5), {1, 2, 3}) == [{1, 2, 3}]

    def test_split_softcore_is_two_regions(self):
        assert detect_fragments(_chain(5), {0, 4}) == [{0}, {4}]

    def test_regions_are_ordered_largest_first(self):
        assert detect_fragments(_chain(6), {0, 3, 4}) == [{3, 4}, {0}]


class TestJointClosure:
    def test_whole_ring_rule_absorbs_the_ring(self):
        ring = nx.cycle_graph(6)
        nx.set_edge_attributes(ring, True, "in_ring")
        context = _context(ring, _chain(1), rings_1=[range(6)])
        closed_1, _ = joint_closure({0}, set(), context)
        assert closed_1 == set(range(6))

    def test_ring_policy_none_leaves_the_ring_alone(self):
        ring = nx.cycle_graph(6)
        context = _context(ring, _chain(1), rings_1=[range(6)], policy=SoftcorePolicy(ring_policy="none"))
        closed_1, _ = joint_closure({0}, set(), context)
        assert closed_1 == {0}

    def test_fused_rings_cascade_without_a_special_case(self):
        # Two rings sharing atoms 0 and 1. Touching one must pull in both.
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 5), (5, 6), (6, 7), (7, 0)])
        context = _context(graph, _chain(1), rings_1=[[0, 1, 2, 3, 4], [0, 1, 5, 6, 7]])
        closed_1, _ = joint_closure({2}, set(), context)
        assert closed_1 == set(range(8))

    def test_partner_rule_propagates_across_the_sides(self):
        context = _context(_chain(3), _chain(3), forward={0: 0, 1: 1, 2: 2})
        closed_1, closed_2 = joint_closure({1}, set(), context)
        assert closed_1 == {1}
        assert closed_2 == {1}

    def test_hydrogen_rule_is_one_way(self):
        # A soft-core hydrogen on a common-core parent must NOT drag the parent in;
        # that is the R-H -> R-CH3 case, the most common transformation there is.
        graph = nx.Graph()
        graph.add_edges_from([(0, 1)], in_ring=False)
        context = RepairContext(
            graph_1=graph,
            graph_2=_chain(1),
            rings_1=(),
            rings_2=(),
            hydrogen_parent_1={1: 0},
            hydrogen_parent_2={},
            forward={},
            reverse={},
            heavy_1=frozenset({0}),
            heavy_2=frozenset({0}),
            policy=SoftcorePolicy(),
            n_atoms_1=2,
            n_atoms_2=1,
        )
        closed_1, _ = joint_closure({1}, set(), context)
        assert closed_1 == {1}, "a soft-core hydrogen must not demote its common-core parent"

        # The reverse direction does propagate.
        closed_1, _ = joint_closure({0}, set(), context)
        assert closed_1 == {0, 1}


class TestNodeWeightedSteiner:
    def test_fewer_than_two_terminals_is_a_no_op(self):
        nodes, approximate = node_weighted_steiner(_chain(5), [{0}], lambda n: 1.0)
        assert nodes == set() and not approximate

    def test_two_terminals_take_the_shortest_path(self):
        nodes, approximate = node_weighted_steiner(_chain(5), [{0}, {4}], lambda n: 1.0)
        assert nodes == {1, 2, 3}
        assert not approximate, "two terminals must be solved exactly"

    def test_cost_steers_the_route(self):
        # Two routes between 0 and 4: via 1-2-3 (expensive) or via 5-6 (cheap).
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 4)], in_ring=False)
        expensive = {1: 100.0, 2: 100.0, 3: 100.0}
        nodes, _ = node_weighted_steiner(graph, [{0}, {4}], lambda n: expensive.get(n, 1.0))
        assert nodes == {5, 6}

    def test_three_terminals_find_the_shared_hub(self):
        # A star: three arms meeting at the hub. Three or more terminals is NP-hard, so
        # the greedy merge is flagged approximate even when -- as here -- it is optimal.
        graph = nx.Graph()
        graph.add_edges_from([(0, 9), (1, 9), (2, 9)], in_ring=False)
        nodes, approximate = node_weighted_steiner(graph, [{0}, {1}, {2}], lambda n: 1.0)
        assert nodes == {9}
        assert approximate

    def test_many_terminals_on_a_wide_graph_terminate_quickly(self):
        # Guards the bug this solver was rewritten for: the previous exhaustive subset
        # search enumerated C(n, k) candidate sets and hung on real ligand-sized inputs.
        import time

        graph = _chain(40)
        terminals = [{0}, {12}, {24}, {36}]
        start = time.monotonic()
        nodes, _ = node_weighted_steiner(graph, terminals, lambda n: 1.0)
        assert time.monotonic() - start < 5.0
        # The chain forces every intermediate node between the outermost terminals.
        assert nodes == set(range(1, 36)) - {12, 24}

    def test_result_is_deterministic(self):
        graph = nx.Graph()
        graph.add_edges_from([(0, 9), (1, 9), (2, 9), (0, 8), (1, 8), (2, 8)], in_ring=False)
        results = [node_weighted_steiner(graph, [{0}, {1}, {2}], lambda n: 1.0)[0] for _ in range(5)]
        assert all(r == results[0] for r in results)

    def test_disconnected_terminals_raise(self):
        graph = nx.Graph()
        graph.add_nodes_from([0, 1])
        with pytest.raises(ValueError, match="no path exists"):
            node_weighted_steiner(graph, [{0}, {1}], lambda n: 1.0)


class TestRepairOnMolecules:
    """Behaviour on real chemistry, including the documented rejections."""

    @staticmethod
    def _mcs_mapping(source, target):
        """Map with the default MCS mapper."""
        from rbfenetmap.core.options import MappingOptions
        from rbfenetmap.plugins.mappers import create_mapper

        return create_mapper("mcss-e2").map_pair(source, target, MappingOptions())

    def test_single_substituent_needs_no_repair(self):
        ligands = make_coposed({"a": "c1ccccc1C(=O)N", "b": "Cc1ccccc1C(=O)N"}, "c1ccccc1C(=O)N")
        mapping = self._mcs_mapping(ligands["a"], ligands["b"])
        _, repair = repair_softcore_connectivity(ligands["a"], ligands["b"], mapping)
        assert repair.rejection is None
        assert repair.n_fragments_before == (1, 1)
        assert not repair.applied

    def test_three_regions_merge_into_one(self):
        # CF3 -> ethyl: three fluorines and the ethyl fragments each start disconnected.
        ligands = make_coposed({"a": "FC(F)(F)c1ccccc1C(=O)N", "b": "CCc1ccccc1C(=O)N"}, "c1ccccc1C(=O)N")
        mapping = self._mcs_mapping(ligands["a"], ligands["b"])
        repaired, repair = repair_softcore_connectivity(ligands["a"], ligands["b"], mapping)
        assert repair.rejection is None
        assert max(repair.n_fragments_before) > 1, "this pair should start fragmented"
        assert repair.n_fragments_after[0] <= 1 and repair.n_fragments_after[1] <= 1
        assert repair.applied
        # The repaired mapping is itself valid, which AtomMapping enforces on construction.
        assert set(repaired.sc1) | set(repaired.cc1) == set(range(repaired.n_atoms_1))

    def test_para_disubstitution_on_a_bare_ring_consumes_the_molecule(self):
        # Benzene -> p-xylene. The two vanishing hydrogens sit across the ring, so the
        # bridge recruits ring carbons, the whole-ring rule absorbs the rest, and nothing
        # is left as common core. Rejecting is the correct answer, not a failure.
        a = make_ligand("c1ccccc1", "benzene")
        b = make_ligand("Cc1ccc(C)cc1", "pxylene")
        mapping = self._mcs_mapping(a, b)
        _, repair = repair_softcore_connectivity(a, b, mapping)
        assert repair.rejection is RejectionReason.NO_COMMON_CORE

    def test_budget_rejects_an_oversized_softcore(self):
        ligands = make_coposed({"a": "c1ccccc1C(=O)N", "b": "c1ccccc1-c1ccccc1C(=O)N"}, "c1ccccc1C(=O)N")
        mapping = self._mcs_mapping(ligands["a"], ligands["b"])
        _, repair = repair_softcore_connectivity(
            ligands["a"], ligands["b"], mapping, SoftcorePolicy(max_softcore_atoms=2)
        )
        assert repair.rejection in (RejectionReason.SOFTCORE_TOO_LARGE, RejectionReason.SOFTCORE_FRACTION)

    def test_rejection_returns_the_original_mapping_unchanged(self):
        a = make_ligand("c1ccccc1", "benzene")
        b = make_ligand("Cc1ccc(C)cc1", "pxylene")
        mapping = self._mcs_mapping(a, b)
        returned, repair = repair_softcore_connectivity(a, b, mapping)
        assert repair.rejection is not None
        assert returned == mapping, "a rejected edge must not be silently mutated"

    def test_repair_records_a_trace(self):
        ligands = make_coposed({"a": "FC(F)(F)c1ccccc1C(=O)N", "b": "CCc1ccccc1C(=O)N"}, "c1ccccc1C(=O)N")
        mapping = self._mcs_mapping(ligands["a"], ligands["b"])
        _, repair = repair_softcore_connectivity(ligands["a"], ligands["b"], mapping)
        assert repair.trace
        assert any("bridged" in line for line in repair.trace)


class TestPrecheck:
    def test_charge_change_rejected_under_reject_policy(self):
        ligands = make_coposed({"a": "c1ccccc1C(=O)O", "b": "c1ccccc1C(=O)[O-]"}, "c1ccccc1C=O")
        mapping = AtomMapping.from_core_pairs(
            {i: i for i in range(6)}, n_atoms_1=ligands["a"].n_atoms, n_atoms_2=ligands["b"].n_atoms
        )
        reason = precheck_mapping(ligands["a"], ligands["b"], mapping, SoftcorePolicy(charge_change_policy="reject"))
        assert reason is RejectionReason.NET_CHARGE_CHANGE

    def test_charge_change_allowed_under_penalize_policy(self):
        ligands = make_coposed({"a": "c1ccccc1C(=O)O", "b": "c1ccccc1C(=O)[O-]"}, "c1ccccc1C=O")
        mapping = AtomMapping.from_core_pairs(
            {i: i for i in range(9)}, n_atoms_1=ligands["a"].n_atoms, n_atoms_2=ligands["b"].n_atoms
        )
        reason = precheck_mapping(ligands["a"], ligands["b"], mapping, SoftcorePolicy(charge_change_policy="penalize"))
        assert reason is not RejectionReason.NET_CHARGE_CHANGE

    def test_empty_core_rejected(self, benzene):
        mapping = AtomMapping.from_core_pairs({}, n_atoms_1=benzene.n_atoms, n_atoms_2=benzene.n_atoms)
        assert precheck_mapping(benzene, benzene, mapping, SoftcorePolicy()) is RejectionReason.NO_COMMON_CORE


def test_connected_components_of_is_deterministic():
    graph = _chain(6)
    for _ in range(5):
        assert connected_components_of(graph, {0, 1, 4}) == [{0, 1}, {4}]
