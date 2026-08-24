"""Counterpoised (CBFE) edge construction, costing, and bridge ranking.

The units here are the pieces the planner composes: what a CBFE edge *is*, what it costs,
which pairs are eligible, and how a bridge is chosen. Planner-level behaviour -- when those
pieces actually get used -- lives in ``test_planners.py``.
"""

from __future__ import annotations

import networkx as nx
import pytest

from rbfenetmap.core.cbfe import (
    BRIDGE_CENTRALITY_WEIGHT,
    bridge_rank_key,
    build_cbfe_pool,
    cbfe_cost,
    component_centrality,
    make_cbfe_transformation,
    select_bridges,
    select_cbfe_bridges,
)
from rbfenetmap.core.models import EdgeKind
from rbfenetmap.core.options import NetworkOptions


class TestCost:
    def test_cost_is_base_plus_weight_times_both_heavy_counts(self, benzamides):
        options = NetworkOptions(cbfe_base_cost=8.0, cbfe_atom_weight=0.05)
        source, target = benzamides["bza_H"], benzamides["bza_F"]
        expected = 8.0 + 0.05 * (source.n_heavy + target.n_heavy)
        assert cbfe_cost(source, target, options) == pytest.approx(expected)

    def test_cost_is_symmetric(self, benzamides):
        options = NetworkOptions()
        a, b = benzamides["bza_H"], benzamides["bza_CF3"]
        assert cbfe_cost(a, b, options) == pytest.approx(cbfe_cost(b, a, options))

    def test_a_zero_atom_weight_makes_every_edge_cost_the_base(self, benzamides):
        options = NetworkOptions(cbfe_base_cost=5.0, cbfe_atom_weight=0.0)
        costs = {
            cbfe_cost(benzamides[a], benzamides[b], options) for a, b in (("bza_H", "bza_F"), ("bza_H", "bza_CF3"))
        }
        assert costs == {5.0}


class TestTransformation:
    def test_mapping_is_entirely_softcore(self, benzamides):
        source, target = benzamides["bza_H"], benzamides["bza_CF3"]
        edge = make_cbfe_transformation(source, target, NetworkOptions())

        assert edge.kind is EdgeKind.CBFE
        assert edge.mapping.cc1 == () and edge.mapping.cc2 == ()
        assert edge.mapping.sc1 == tuple(range(source.n_atoms))
        assert edge.mapping.sc2 == tuple(range(target.n_atoms))
        assert edge.mapping.n_common_core == 0

    def test_is_always_feasible_with_a_finite_cost(self, benzamides):
        edge = make_cbfe_transformation(benzamides["bza_H"], benzamides["bza_CF3"], NetworkOptions())
        assert edge.feasible
        assert edge.score.rejections == ()
        assert edge.score.total == pytest.approx(sum(edge.score.contributions.values()))

    def test_reversing_preserves_the_kind(self, benzamides):
        edge = make_cbfe_transformation(benzamides["bza_H"], benzamides["bza_F"], NetworkOptions())
        flipped = edge.reversed()
        assert flipped.kind is EdgeKind.CBFE
        assert (flipped.source, flipped.target) == (edge.target, edge.source)


class TestPool:
    def test_spans_every_unordered_pair(self, benzamides):
        pool = build_cbfe_pool(benzamides, NetworkOptions())
        n = len(benzamides)
        assert len(pool) == n * (n - 1) // 2
        assert all(pair[0] < pair[1] for pair in pool)

    def test_banned_pairs_are_excluded(self, benzamides):
        options = NetworkOptions(banned_edges=("bza_F~bza_H",))
        assert ("bza_F", "bza_H") not in build_cbfe_pool(benzamides, options)

    def test_excluded_pairs_are_omitted(self, benzamides):
        """The planner passes the pairs that already have a feasible RBFE edge.

        Without this a pair could enter the network twice, which ``Network.validate``
        rejects far downstream of the mistake.
        """
        pool = build_cbfe_pool(benzamides, NetworkOptions(), exclude=[("bza_F", "bza_H")])
        assert ("bza_F", "bza_H") not in pool

    def test_ignores_the_fingerprint_prefilter(self, benzamides):
        """Dissimilar pairs are exactly the ones that need a bridge, so they must survive."""
        options = NetworkOptions(prefilter="fingerprint", prefilter_k=1, prefilter_min_tanimoto=0.99)
        n = len(benzamides)
        assert len(build_cbfe_pool(benzamides, options)) == n * (n - 1) // 2


class TestBridgeRanking:
    def test_a_singleton_component_scores_maximum_centrality(self):
        """Its only entry point must not rank as the worst one available."""
        graph = nx.Graph()
        graph.add_edge("a", "b")
        graph.add_node("lonely")
        centrality = component_centrality(graph, [{"a", "b"}, {"lonely"}])
        assert centrality["lonely"] == 1.0

    def test_centrality_is_relative_to_the_ligands_own_component(self):
        graph = nx.Graph()
        graph.add_edges_from([("hub", "x"), ("hub", "y"), ("hub", "z")])
        centrality = component_centrality(graph, [{"hub", "x", "y", "z"}])
        assert centrality["hub"] == 1.0
        assert centrality["x"] == pytest.approx(1 / 3)

    def test_more_central_endpoint_wins_at_equal_similarity(self):
        similarity = {("a", "c"): 0.5, ("b", "c"): 0.5}
        centrality = {"a": 1.0, "b": 0.0, "c": 1.0}
        cost = {("a", "c"): 9.0, ("b", "c"): 9.0}
        keys = {p: bridge_rank_key(p, similarity=similarity, centrality=centrality, cost=cost) for p in cost}
        assert keys[("a", "c")] < keys[("b", "c")]

    def test_more_similar_pair_wins_at_equal_centrality(self):
        similarity = {("a", "c"): 0.9, ("b", "c"): 0.1}
        centrality = {"a": 0.5, "b": 0.5, "c": 0.5}
        cost = {("a", "c"): 9.0, ("b", "c"): 9.0}
        keys = {p: bridge_rank_key(p, similarity=similarity, centrality=centrality, cost=cost) for p in cost}
        assert keys[("a", "c")] < keys[("b", "c")]

    def test_centrality_weight_is_stated_in_units_of_similarity(self):
        """A full centrality swing is worth exactly ``BRIDGE_CENTRALITY_WEIGHT`` Tanimoto."""
        centrality = {"a": 1.0, "b": 0.0, "c": 0.0}
        similarity = {("a", "c"): 0.0, ("b", "c"): BRIDGE_CENTRALITY_WEIGHT * 0.5}
        cost = {("a", "c"): 9.0, ("b", "c"): 9.0}
        keys = {p: bridge_rank_key(p, similarity=similarity, centrality=centrality, cost=cost) for p in cost}
        assert keys[("a", "c")][0] == pytest.approx(keys[("b", "c")][0])

    def test_cheaper_edge_breaks_a_merit_tie(self):
        similarity = {("a", "c"): 0.5, ("b", "c"): 0.5}
        centrality = {"a": 0.5, "b": 0.5, "c": 0.5}
        cost = {("a", "c"): 12.0, ("b", "c"): 9.0}
        keys = {p: bridge_rank_key(p, similarity=similarity, centrality=centrality, cost=cost) for p in cost}
        assert keys[("b", "c")] < keys[("a", "c")]


class TestBridgeSelection:
    def test_joins_every_component_with_exactly_one_bridge_short_of_the_count(self, benzamides):
        graph = nx.Graph()
        graph.add_nodes_from(benzamides)
        graph.add_edge("bza_H", "bza_F")
        graph.add_edge("bza_Cl", "bza_Me")
        # Three components: {H,F}, {Cl,Me}, {CF3}
        pool = build_cbfe_pool(benzamides, NetworkOptions(), exclude=graph.edges)

        bridges = select_cbfe_bridges(graph, benzamides, pool)

        assert len(bridges) == 2
        graph.add_edges_from(bridges)
        assert nx.is_connected(graph)

    def test_an_already_connected_graph_needs_no_bridges(self, benzamides):
        graph = nx.Graph()
        graph.add_nodes_from(benzamides)
        names = list(benzamides)
        graph.add_edges_from(zip(names, names[1:]))
        assert select_cbfe_bridges(graph, benzamides, build_cbfe_pool(benzamides, NetworkOptions())) == []

    def test_is_deterministic(self, benzamides):
        graph = nx.Graph()
        graph.add_nodes_from(benzamides)
        graph.add_edge("bza_H", "bza_F")
        pool = build_cbfe_pool(benzamides, NetworkOptions(), exclude=graph.edges)
        runs = {tuple(select_cbfe_bridges(graph, benzamides, pool)) for _ in range(3)}
        assert len(runs) == 1

    def test_an_empty_pool_yields_no_bridges(self, benzamides):
        graph = nx.Graph()
        graph.add_nodes_from(benzamides)
        assert select_cbfe_bridges(graph, benzamides, {}) == []


class TestPartitionBridges:
    """:func:`select_bridges` with a partition that is *not* the connected components.

    This is the seam clustered planning uses. The connectivity case is covered above
    through :func:`select_cbfe_bridges`; what matters here is that an arbitrary partition
    behaves the same way, and that ``n_per_pair`` buys the extra crossings that put a
    bridge on a cycle.
    """

    @staticmethod
    def _two_groups(benzamides):
        """A partition of the benzamides into two groups, and a pool spanning every pair."""
        names = sorted(benzamides)
        partition = {name: int(index >= 2) for index, name in enumerate(names)}
        return partition, build_cbfe_pool(benzamides, NetworkOptions())

    def test_one_bridge_per_joined_group_pair_by_default(self, benzamides):
        partition, pool = self._two_groups(benzamides)
        assert len(select_bridges(partition, benzamides, pool)) == 1

    def test_n_per_pair_takes_that_many_crossings(self, benzamides):
        partition, pool = self._two_groups(benzamides)
        bridges = select_bridges(partition, benzamides, pool, n_per_pair=3)
        assert len(bridges) == 3
        assert len(set(bridges)) == 3
        assert all(partition[a] != partition[b] for a, b in bridges)

    def test_the_extra_crossings_extend_the_ranked_prefix(self, benzamides):
        """The second bridge is the second-best one, not an arbitrary spare."""
        partition, pool = self._two_groups(benzamides)
        assert (
            select_bridges(partition, benzamides, pool)[0]
            == select_bridges(partition, benzamides, pool, n_per_pair=2)[0]
        )

    def test_three_groups_are_joined_by_a_spanning_selection(self, benzamides):
        names = sorted(benzamides)
        partition = {name: index % 3 for index, name in enumerate(names)}
        pool = build_cbfe_pool(benzamides, NetworkOptions())
        bridges = select_bridges(partition, benzamides, pool, n_per_pair=2)
        assert len(bridges) == 4, "two group pairs joined, two crossings each"
        graph: nx.Graph = nx.Graph()
        graph.add_nodes_from(benzamides)
        graph.add_edges_from(bridges)
        assert nx.number_connected_components(graph) == len(benzamides) - len(bridges)

    def test_a_single_group_needs_no_bridges(self, benzamides):
        partition = dict.fromkeys(benzamides, 0)
        assert select_bridges(partition, benzamides, build_cbfe_pool(benzamides, NetworkOptions())) == []

    def test_ranking_without_a_graph_falls_back_to_similarity(self, benzamides):
        """Centrality has no answer without a graph, so it must simply drop out."""
        partition, pool = self._two_groups(benzamides)
        assert select_bridges(partition, benzamides, pool, graph=None) != []

    def test_is_deterministic(self, benzamides):
        partition, pool = self._two_groups(benzamides)
        runs = {tuple(select_bridges(partition, benzamides, pool, n_per_pair=2)) for _ in range(3)}
        assert len(runs) == 1


class TestOptionValidation:
    @pytest.mark.parametrize("mode", ["off", "bridge", "cycles", "all"])
    def test_every_documented_mode_is_accepted(self, mode):
        assert NetworkOptions(cbfe_mode=mode).cbfe_mode == mode

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="cbfe_mode"):
            NetworkOptions(cbfe_mode="sometimes")

    @pytest.mark.parametrize("field", ["cbfe_base_cost", "cbfe_atom_weight"])
    def test_negative_costs_are_refused(self, field):
        with pytest.raises(ValueError, match=field):
            NetworkOptions(**{field: -1.0})

    def test_the_mode_ladder_is_strict(self):
        """``cycles`` includes everything ``bridge`` does; ``all`` needs no placement help."""
        assert not NetworkOptions(cbfe_mode="off").cbfe_bridges_components
        assert NetworkOptions(cbfe_mode="bridge").cbfe_bridges_components
        assert not NetworkOptions(cbfe_mode="bridge").cbfe_closes_cycles
        assert NetworkOptions(cbfe_mode="cycles").cbfe_bridges_components
        assert NetworkOptions(cbfe_mode="cycles").cbfe_closes_cycles
