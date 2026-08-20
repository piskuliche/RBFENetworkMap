"""Core-based clustering: the partition, and the guarantee that ``report`` changes nothing.

The second half is the load-bearing part. ``core_clusters`` is a ladder whose bottom two
rungs must leave selection exactly as it was, so the tests that compare ``off`` against
``report`` are what make "keeps the current approach" a checked property rather than an
intention.
"""

from __future__ import annotations

import networkx as nx
import pytest

from rbfenetmap.core.clustering import cluster_core_size, core_clusters, describe_clusters
from rbfenetmap.core.mcs import mcs_query_many
from rbfenetmap.core.models import Ligand
from rbfenetmap.core.options import ClusteringPolicy, MappingOptions, NetworkOptions, SoftcorePolicy
from rbfenetmap.core.pairs import generate_candidate_pairs
from rbfenetmap.core.pipeline import build_network, evaluate_pairs
from rbfenetmap.plugins.mappers import create_mapper
from rbfenetmap.plugins.scorers import create_scorer


def feasible_graph(ligands: dict[str, Ligand]) -> nx.Graph:
    """Build the pairwise-feasible RBFE graph the partition is seeded from."""
    options = NetworkOptions()
    pairs, _ = generate_candidate_pairs(ligands, options)
    candidates = evaluate_pairs(
        ligands, pairs, create_mapper("mcss-e2"), create_scorer("linear"), MappingOptions(), options
    )
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(ligands)
    for candidate in candidates:
        if candidate.feasible:
            graph.add_edge(*candidate.unordered_key, weight=candidate.score.total)
    return graph


def partition(ligands: dict[str, Ligand], **policy: object) -> tuple[frozenset[str], ...]:
    """Cluster *ligands* under a policy given as keyword overrides."""
    return core_clusters(ligands, feasible_graph(ligands), ClusteringPolicy(**policy), MappingOptions())


class TestNWayMCS:
    """The primitive the partition is built on."""

    def test_fewer_than_two_molecules_has_no_common_substructure(self, benzamides):
        """A group of one has nothing to intersect, so the query is None rather than itself."""
        mols = [ligand.mol for ligand in benzamides.values()]
        assert mcs_query_many(mols[:1], MappingOptions()) is None
        assert mcs_query_many([], MappingOptions()) is None

    def test_core_shrinks_as_the_group_grows(self, three_scaffolds):
        """Adding a molecule can only shrink the shared core, never grow it."""
        options = MappingOptions()
        within = cluster_core_size([f"bza_{s}" for s in ("H", "F", "Cl", "Me")], three_scaffolds, options)
        across = cluster_core_size(list(three_scaffolds), three_scaffolds, options)
        assert within > across, "the whole set cannot share more core than one scaffold does"

    @pytest.mark.parametrize("size", [2, 3, 4, 12])
    def test_core_never_exceeds_the_smallest_member(self, three_scaffolds, size):
        """A shared substructure has to embed in every member, so it cannot be larger.

        The regression guard for counting MCS *query* atoms instead of matched ones:
        ``CompareAny`` emits generic atoms whose atomic number is 0, and counting those as
        heavy reported a 10-atom core shared by a 9-heavy-atom ligand.
        """
        names = sorted(three_scaffolds)[:size]
        core = cluster_core_size(names, three_scaffolds, MappingOptions())
        smallest = min(three_scaffolds[name].n_heavy for name in names)
        assert core <= smallest, f"core {core} exceeds smallest member {smallest}"

    def test_singleton_shares_all_of_itself(self, benzamides):
        """A lone ligand's core is its own heavy-atom count, not zero."""
        name, ligand = next(iter(benzamides.items()))
        assert cluster_core_size([name], benzamides, MappingOptions()) == ligand.n_heavy


class TestPartition:
    """Properties the partition must hold regardless of thresholds."""

    @pytest.mark.parametrize("min_core_atoms", [1, 4, 6, 8, 12, 20])
    def test_every_ligand_lands_in_exactly_one_cluster(self, three_scaffolds, min_core_atoms):
        """A partition, not a grouping: singletons included, nothing dropped or duplicated."""
        clusters = partition(three_scaffolds, min_core_atoms=min_core_atoms)
        members = [name for cluster in clusters for name in cluster]
        assert sorted(members) == sorted(three_scaffolds)
        assert len(members) == len(set(members))

    def test_clusters_are_connected_in_the_feasible_graph(self, three_scaffolds):
        """A cluster whose members cannot be mapped to one another has no network to build."""
        graph = feasible_graph(three_scaffolds)
        for cluster in core_clusters(three_scaffolds, graph, ClusteringPolicy(min_core_atoms=6), MappingOptions()):
            assert nx.is_connected(graph.subgraph(cluster)), f"{sorted(cluster)} is not internally mappable"

    def test_raising_the_threshold_never_merges_more(self, three_scaffolds):
        """Monotonicity: a tighter core requirement can only split, never join."""
        counts = [len(partition(three_scaffolds, min_core_atoms=n)) for n in (2, 4, 6, 8, 10, 12)]
        assert counts == sorted(counts), f"cluster count fell as the threshold rose: {counts}"

    def test_separates_scaffolds(self, three_scaffolds):
        """The headline case: three scaffolds come back as three clusters."""
        clusters = partition(three_scaffolds, min_core_atoms=6)
        assert len(clusters) == 3
        for cluster in clusters:
            prefixes = {name.split("_")[0] for name in cluster}
            assert len(prefixes) == 1, f"{sorted(cluster)} mixes scaffolds {prefixes}"

    def test_homogeneous_series_is_one_cluster(self, benzamides):
        """A congeneric series has nothing to split, and must not be split anyway."""
        assert len(partition(benzamides, min_core_atoms=6)) == 1

    def test_threshold_above_every_ligand_shatters(self, three_scaffolds):
        """Documented failure mode: no group can share more core than its smallest member has.

        The bound is per-cluster, not global -- a threshold above the *smallest* ligand in
        the set can still merge a cluster whose own members are all larger. Only a threshold
        above the largest ligand is guaranteed to leave singletons.
        """
        largest = max(ligand.n_heavy for ligand in three_scaffolds.values())
        clusters = partition(three_scaffolds, min_core_atoms=largest + 1)
        assert all(len(cluster) == 1 for cluster in clusters)

    def test_max_cluster_size_caps_the_merge(self, benzamides):
        """A cap splits a series that would otherwise fuse into one cluster."""
        clusters = partition(benzamides, min_core_atoms=4, max_cluster_size=2, min_cluster_size=2)
        assert clusters, "expected a partition"
        assert max(len(cluster) for cluster in clusters) <= 2

    def test_partition_is_reproducible(self, three_scaffolds):
        """Ties break deterministically, so two runs agree."""
        first = partition(three_scaffolds, min_core_atoms=6)
        second = partition(three_scaffolds, min_core_atoms=6)
        assert first == second

    def test_describe_flags_clusters_too_small_to_cycle(self, three_scaffolds):
        """A cluster below min_cluster_size is reported, not rejected."""
        clusters = partition(three_scaffolds, min_core_atoms=6)
        lines = describe_clusters(clusters, three_scaffolds, MappingOptions(), ClusteringPolicy(min_cluster_size=99))
        assert len(lines) == len(clusters)
        assert all("too small to carry a cycle" in line for line in lines)


class TestReportChangesNothing:
    """The backward-compatibility guarantee, stated as a test."""

    @staticmethod
    def _edges(network):
        """The selected edge set, with costs, in a comparable form."""
        return sorted((edge.source, edge.target, round(edge.score.total, 9)) for edge in network.edges)

    @pytest.mark.parametrize("max_softcore_atoms", [6, 12, 20])
    @pytest.mark.parametrize("min_core_atoms", [4, 8])
    def test_report_selects_exactly_what_off_selects(self, benzamides, max_softcore_atoms, min_core_atoms):
        """``report`` is information only; the network must be identical to ``off``."""
        softcore = SoftcorePolicy(max_softcore_atoms=max_softcore_atoms)
        off = build_network(benzamides, network_options=NetworkOptions(softcore=softcore, cbfe_mode="bridge"))
        report = build_network(
            benzamides,
            network_options=NetworkOptions(
                softcore=softcore,
                cbfe_mode="bridge",
                core_clusters="report",
                clustering=ClusteringPolicy(min_core_atoms=min_core_atoms),
            ),
        )
        assert self._edges(off) == self._edges(report)

    def test_off_records_no_clusters(self, benzamides):
        """The default costs nothing and says nothing about clusters."""
        network = build_network(benzamides, network_options=NetworkOptions())
        assert network.clusters == ()

    def test_report_records_the_partition(self, three_scaffolds):
        """``report`` is how you see the clustering without changing the network."""
        network = build_network(
            three_scaffolds,
            network_options=NetworkOptions(
                cbfe_mode="bridge", core_clusters="report", clustering=ClusteringPolicy(min_core_atoms=6)
            ),
        )
        assert len(network.clusters) == 3
        assert sorted(name for cluster in network.clusters for name in cluster) == sorted(three_scaffolds)


class TestOptionConflicts:
    """Contradictory knobs are refused at construction, not several minutes into a run."""

    def test_clustering_needs_mappings(self):
        """``cbfe_mode='all'`` skips mapping, so there are no cores to cluster on."""
        with pytest.raises(ValueError, match="no common"):
            NetworkOptions(core_clusters="report", cbfe_mode="all")

    def test_plan_needs_something_to_bridge_with(self):
        """``plan`` joining clusters by CBFE cannot work with CBFE switched off."""
        with pytest.raises(ValueError, match="cbfe_mode='off'"):
            NetworkOptions(core_clusters="plan")

    def test_plan_with_prefer_rbfe_needs_no_cbfe(self):
        """The escape hatch: relative edges across the boundary need no counterpoised pool."""
        options = NetworkOptions(core_clusters="plan", clustering=ClusteringPolicy(inter_cluster="prefer_rbfe"))
        assert options.clusters_drive_selection

    def test_unknown_mode_is_rejected(self):
        """A typo must not fall through to the default and look like it worked."""
        with pytest.raises(ValueError, match="core_clusters must be one of"):
            NetworkOptions(core_clusters="cluster")

    @pytest.mark.parametrize(
        "policy,match",
        [
            ({"min_core_atoms": 0}, "min_core_atoms"),
            ({"min_core_fraction": 1.5}, "min_core_fraction"),
            ({"min_cluster_size": 0}, "min_cluster_size"),
            ({"max_cluster_size": 1}, "max_cluster_size"),
            ({"max_cluster_size": 2, "min_cluster_size": 5}, "below min_cluster_size"),
            ({"inter_cluster": "sometimes"}, "inter_cluster"),
        ],
    )
    def test_policy_rejects_nonsense(self, policy, match):
        """Every threshold is range-checked where it is declared."""
        with pytest.raises(ValueError, match=match):
            ClusteringPolicy(**policy)


class TestSerialization:
    """Clusters survive the JSON round trip, and old files still load."""

    def test_round_trip(self, three_scaffolds, tmp_path):
        """A planned partition comes back identical."""
        from rbfenetmap.io.networkio import dump_network, load_network

        network = build_network(
            three_scaffolds,
            network_options=NetworkOptions(
                cbfe_mode="bridge", core_clusters="report", clustering=ClusteringPolicy(min_core_atoms=6)
            ),
        )
        path = dump_network(network, tmp_path / "network.json")
        restored = load_network(path)
        assert sorted(sorted(c) for c in restored.clusters) == sorted(sorted(c) for c in network.clusters)
        assert restored.options.core_clusters == "report"
        assert restored.options.clustering.min_core_atoms == 6

    def test_json_without_clusters_still_loads(self, benzamides, tmp_path):
        """Networks written before the field existed must not become unreadable."""
        import json

        from rbfenetmap.io.networkio import dump_network, load_network

        path = dump_network(build_network(benzamides), tmp_path / "network.json")
        data = json.loads(path.read_text())
        del data["clusters"]
        del data["options"]["core_clusters"]
        del data["options"]["clustering"]
        path.write_text(json.dumps(data))

        restored = load_network(path)
        assert restored.clusters == ()
        assert restored.options.core_clusters == "off"


class TestClusterDrivenSelection:
    """``plan`` mode: cycles inside clusters, single bridges between them."""

    @staticmethod
    def _split(network):
        """Return ``(membership, intra, cross)`` for the selected edges."""
        membership = {name: index for index, cluster in enumerate(network.clusters) for name in cluster}
        intra = [e for e in network.edges if membership[e.source] == membership[e.target]]
        cross = [e for e in network.edges if membership[e.source] != membership[e.target]]
        return membership, intra, cross

    def test_scaffolds_become_cycled_clusters_joined_by_bridges(self, three_scaffolds):
        """The headline result: three internally cycled clusters, two counterpoised bridges."""
        network = build_network(
            three_scaffolds,
            network_options=NetworkOptions(
                core_clusters="plan",
                cbfe_mode="bridge",
                min_cycle_coverage=1.0,
                clustering=ClusteringPolicy(min_core_atoms=6),
            ),
        )
        _, intra, cross = self._split(network)
        assert len(network.clusters) == 3
        assert len(cross) == len(network.clusters) - 1, "expected a spanning set of bridges, no more"
        assert all(edge.kind.value == "cbfe" for edge in cross)
        assert all(edge.kind.value == "rbfe" for edge in intra)

    def test_clusters_are_internally_cycled(self, three_scaffolds):
        """Each cluster large enough to hold a cycle has one."""
        network = build_network(
            three_scaffolds,
            network_options=NetworkOptions(
                core_clusters="plan",
                cbfe_mode="bridge",
                min_cycle_coverage=1.0,
                clustering=ClusteringPolicy(min_core_atoms=6),
            ),
        )
        _, intra, _ = self._split(network)
        graph: nx.Graph = nx.Graph()
        graph.add_nodes_from(network.ligands)
        graph.add_edges_from((edge.source, edge.target) for edge in intra)
        for cluster in network.clusters:
            if len(cluster) < 3:
                continue
            subgraph = graph.subgraph(cluster)
            assert subgraph.number_of_edges() >= len(cluster), f"{sorted(cluster)} is a tree, not cycled"

    def test_no_cycle_crosses_a_cluster_boundary(self, three_scaffolds):
        """Bridges form a spanning forest over the clusters, so the quotient graph is a tree."""
        network = build_network(
            three_scaffolds,
            network_options=NetworkOptions(
                core_clusters="plan", cbfe_mode="bridge", clustering=ClusteringPolicy(min_core_atoms=6)
            ),
        )
        membership, _, cross = self._split(network)
        quotient: nx.Graph = nx.Graph()
        quotient.add_nodes_from(range(len(network.clusters)))
        for edge in cross:
            assert not quotient.has_edge(membership[edge.source], membership[edge.target]), "duplicate bridge"
            quotient.add_edge(membership[edge.source], membership[edge.target])
        assert quotient.number_of_edges() == len(network.clusters) - 1

    def test_a_connected_pool_can_still_be_clustered(self, benzamides):
        """The capability ``cbfe_mode='bridge'`` alone cannot provide.

        On a single-scaffold series every pair maps, so there are no disconnected components
        for bridging to find. Only a deliberate partition can split it.
        """
        options = dict(cbfe_mode="bridge", edges_per_ligand=1, min_cycle_coverage=0.0)
        unclustered = build_network(benzamides, network_options=NetworkOptions(**options))
        assert unclustered.clusters == ()

        clustered = build_network(
            benzamides,
            network_options=NetworkOptions(
                core_clusters="plan", clustering=ClusteringPolicy(min_core_atoms=10, min_cluster_size=2), **options
            ),
        )
        assert len(clustered.clusters) > 1, "a connected pool was not split"
        _, _, cross = self._split(clustered)
        assert all(edge.kind.value == "cbfe" for edge in cross)

    def test_cross_cluster_bridge_is_priced_as_cbfe(self, benzamides):
        """Regression: a bridged pair that also had a feasible RBFE mapping.

        Such a pair sits in both pools -- set aside for crossing a boundary, then adopted as
        a bridge. Materializing it from the relative pool would hand back an RBFE
        transformation for an edge the planner selected, priced, and reported as CBFE.
        """
        network = build_network(
            benzamides,
            network_options=NetworkOptions(
                core_clusters="plan",
                cbfe_mode="bridge",
                edges_per_ligand=1,
                clustering=ClusteringPolicy(min_core_atoms=10, min_cluster_size=2),
            ),
        )
        _, _, cross = self._split(network)
        assert cross, "expected at least one bridge"
        for edge in cross:
            assert edge.kind.value == "cbfe"
            assert edge.score.total >= network.options.cbfe_base_cost
            assert edge.mapping.n_common_core == 0, "a counterpoised edge has no common core"

    def test_prefer_rbfe_uses_relative_edges_and_stays_spanning(self, benzamides):
        """``prefer_rbfe`` swaps the bridge kind without spending more of them."""
        policy = dict(min_core_atoms=10, min_cluster_size=2)
        network = build_network(
            benzamides,
            network_options=NetworkOptions(
                core_clusters="plan",
                cbfe_mode="bridge",
                edges_per_ligand=1,
                clustering=ClusteringPolicy(inter_cluster="prefer_rbfe", **policy),
            ),
        )
        _, _, cross = self._split(network)
        assert len(cross) == len(network.clusters) - 1, "prefer_rbfe must not restore a quadratic set"
        assert any(edge.kind.value == "rbfe" for edge in cross)

    def test_homogeneous_series_plans_as_one_cluster(self, benzamides):
        """One cluster means no bridges: ``plan`` degrades to the ordinary network."""
        network = build_network(
            benzamides,
            network_options=NetworkOptions(
                core_clusters="plan", cbfe_mode="bridge", clustering=ClusteringPolicy(min_core_atoms=6)
            ),
        )
        assert len(network.clusters) == 1
        assert all(edge.kind.value == "rbfe" for edge in network.edges)

    def test_small_clusters_are_reported_not_rejected(self, benzamides):
        """A cluster too small to cycle still forms, and says so."""
        network = build_network(
            benzamides,
            network_options=NetworkOptions(
                core_clusters="plan",
                cbfe_mode="bridge",
                edges_per_ligand=1,
                clustering=ClusteringPolicy(min_core_atoms=10, min_cluster_size=3),
            ),
        )
        assert any("cannot carry a cycle" in message for message in network.unmet_constraints)

    def test_other_planners_refuse_plan_but_allow_report(self, benzamides):
        """``report`` needs no planner support; ``plan`` does, and says which to use."""
        from rbfenetmap.core.exceptions import NetworkPlanError

        # cbfe_mode stays 'off' and the bridges would be relative, so the CBFE support check
        # passes and the clustering one is what fires. Testing them separately keeps this
        # about clustering rather than about which guard happens to run first.
        with pytest.raises(NetworkPlanError, match="cannot let clusters drive selection"):
            build_network(
                benzamides,
                planner="complete",
                network_options=NetworkOptions(
                    core_clusters="plan", clustering=ClusteringPolicy(min_core_atoms=6, inter_cluster="prefer_rbfe")
                ),
            )
        network = build_network(benzamides, planner="complete", network_options=NetworkOptions(core_clusters="report"))
        assert len(network.clusters) == 1


class TestClusterVisualisation:
    """Clusters reach the SVG and the HTML report, and never by colour alone."""

    @staticmethod
    def _clustered(ligands):
        """A network whose benzamides fall into more than one cluster."""
        return build_network(
            ligands,
            network_options=NetworkOptions(
                core_clusters="plan",
                cbfe_mode="bridge",
                edges_per_ligand=1,
                clustering=ClusteringPolicy(min_core_atoms=10, min_cluster_size=2),
            ),
        )

    def test_svg_fills_nodes_by_cluster(self, benzamides):
        """One fill class per cluster, and only the ones actually used are defined."""
        import re

        from rbfenetmap.viz.network_svg import render_network_svg

        network = self._clustered(benzamides)
        svg = render_network_svg(network)
        applied = set(re.findall(r'class="node (node-cluster-\d+)"', svg))
        defined = set(re.findall(r"\.(node-cluster-\d+)\{", svg))
        assert applied, "no node was filled by cluster"
        assert applied == defined, "the style block and the nodes disagree"
        assert len(defined) == len(network.clusters)

    def test_svg_names_the_cluster_in_the_tooltip(self, benzamides):
        """Colour is never the only signal -- the same rule the CBFE edges follow."""
        import re

        from rbfenetmap.viz.network_svg import render_network_svg

        svg = render_network_svg(self._clustered(benzamides))
        assert len(re.findall(r"cluster \d+\)", svg)) == len(self._clustered(benzamides).ligands)

    def test_unclustered_svg_is_untouched(self, benzamides):
        """A network with no partition renders exactly as it did before the feature."""
        from rbfenetmap.viz.network_svg import render_network_svg

        svg = render_network_svg(build_network(benzamides, network_options=NetworkOptions(edges_per_ligand=1)))
        assert "node-cluster" not in svg

    def test_report_lists_the_clusters(self, benzamides):
        """The SVG palette repeats past eight clusters, so the list is what resolves them."""
        from rbfenetmap.viz.gallery import render_report

        network = self._clustered(benzamides)
        html = render_report(network)
        assert "Core clusters" in html
        assert html.count("<strong>Cluster ") == len(network.clusters)
        for name in network.ligands:
            assert name in html
