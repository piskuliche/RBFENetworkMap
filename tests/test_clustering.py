"""Clustered planning: the partition, and what the planner does with it.

Two levels, kept apart deliberately. The clusterers are pure functions of the molecules and
are checked against chemistry a reader can verify by eye -- two charge classes really are
two charge classes. The planner tests then check the only claims that matter about the knob:
the crossings are few, they are on a cycle, and turning it off changes nothing.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from rbfenetmap.core.clustering import (
    CLUSTER_METHODS,
    assign_clusters,
    cluster_by_charge,
    cluster_by_fingerprint,
    cluster_by_scaffold,
    cluster_edge_budget,
    cluster_sizes,
)
from rbfenetmap.core.options import NetworkOptions
from rbfenetmap.plugins.planners import create_planner

from .conftest import make_ligand, make_transformation

#: Three benzamides and their three carboxylate analogues: two clean charge classes over
#: molecules similar enough that nothing else in the pipeline would separate them.
MIXED_CHARGE_SMILES = {
    "amide_H": "c1ccccc1C(=O)N",
    "amide_F": "Fc1ccccc1C(=O)N",
    "amide_Cl": "Clc1ccccc1C(=O)N",
    "acid_H": "c1ccccc1C(=O)[O-]",
    "acid_F": "Fc1ccccc1C(=O)[O-]",
    "acid_Cl": "Clc1ccccc1C(=O)[O-]",
}

NEUTRAL = {"amide_H", "amide_F", "amide_Cl"}
ANIONIC = {"acid_H", "acid_F", "acid_Cl"}


@pytest.fixture(scope="module")
def mixed_charges():
    """Ligands spanning two charge classes.

    Independently embedded rather than co-posed, which is safe here only because nothing
    under test reaches geometry: clustering reads charges, scaffolds, and fingerprints, and
    the planner tests below drive it with hand-authored transformations that carry no
    chemistry at all.
    """
    return {name: make_ligand(smiles, name) for name, smiles in MIXED_CHARGE_SMILES.items()}


def groups_of(partition):
    """Return a partition as a set of frozen membership sets, ignoring the numbering."""
    members: dict[int, set[str]] = {}
    for name, cluster in partition.items():
        members.setdefault(cluster, set()).add(name)
    return {frozenset(group) for group in members.values()}


class TestChargeClusterer:
    def test_yields_exactly_the_charge_classes(self, mixed_charges):
        assert groups_of(cluster_by_charge(mixed_charges)) == {frozenset(NEUTRAL), frozenset(ANIONIC)}

    def test_a_single_charge_class_is_one_cluster(self, benzamides):
        assert set(cluster_by_charge(benzamides).values()) == {0}

    def test_indices_are_contiguous_from_zero(self, mixed_charges):
        assert sorted(set(cluster_by_charge(mixed_charges).values())) == [0, 1]

    def test_numbering_is_stable_under_input_order(self, mixed_charges):
        reversed_input = dict(reversed(list(mixed_charges.items())))
        assert cluster_by_charge(reversed_input) == cluster_by_charge(mixed_charges)


class TestScaffoldClusterer:
    def test_a_shared_framework_is_one_cluster(self, benzamides):
        """The co-posed benzamides differ only in substituent, so Murcko sees one series."""
        assert set(cluster_by_scaffold(benzamides).values()) == {0}

    def test_a_different_ring_system_separates(self):
        ligands = {
            name: make_ligand(smiles, name)
            for name, smiles in {
                "benzamide": "c1ccccc1C(=O)N",
                "toluamide": "Cc1ccccc1C(=O)N",
                "pyridine": "c1ccncc1C(=O)N",
            }.items()
        }
        partition = cluster_by_scaffold(ligands)
        assert partition["benzamide"] == partition["toluamide"]
        assert partition["pyridine"] != partition["benzamide"]

    def test_acyclic_ligands_share_the_empty_scaffold(self):
        """Grouped, not scattered: "has no ring system" is a real shared property."""
        ligands = {name: make_ligand(smiles, name) for name, smiles in {"ea": "CCN", "pa": "CCCN"}.items()}
        assert set(cluster_by_scaffold(ligands).values()) == {0}


class TestFingerprintClusterer:
    def test_n_clusters_is_honoured(self, benzamides):
        assert len(groups_of(cluster_by_fingerprint(benzamides, n_clusters=3))) == 3

    def test_a_cutoff_of_zero_makes_every_ligand_its_own_cluster(self, benzamides):
        partition = cluster_by_fingerprint(benzamides, cutoff=0.0)
        assert len(groups_of(partition)) == len(benzamides)

    def test_a_cutoff_of_one_puts_everything_together(self, benzamides):
        assert set(cluster_by_fingerprint(benzamides, cutoff=1.0).values()) == {0}

    def test_distinct_chemotypes_separate(self):
        """Substructure, not charge: the fingerprint groups on what the molecules are made of."""
        ligands = {
            name: make_ligand(smiles, name)
            for name, smiles in {
                "ring_H": "c1ccccc1C(=O)N",
                "ring_F": "Fc1ccccc1C(=O)N",
                "chain_a": "CCCCCCN",
                "chain_b": "CCCCCCCN",
            }.items()
        }
        assert groups_of(cluster_by_fingerprint(ligands, n_clusters=2)) == {
            frozenset({"ring_H", "ring_F"}),
            frozenset({"chain_a", "chain_b"}),
        }

    def test_is_deterministic(self, benzamides):
        runs = {tuple(sorted(cluster_by_fingerprint(benzamides, n_clusters=3).items())) for _ in range(3)}
        assert len(runs) == 1

    def test_n_clusters_above_the_ligand_count_is_clamped(self, benzamides):
        assert len(groups_of(cluster_by_fingerprint(benzamides, n_clusters=99))) == len(benzamides)

    def test_a_single_ligand_needs_no_linkage(self, benzamides):
        one = {"bza_H": benzamides["bza_H"]}
        assert cluster_by_fingerprint(one) == {"bza_H": 0}

    @pytest.mark.parametrize("kwargs", [{"n_clusters": 0}, {"cutoff": -0.1}, {"cutoff": 1.5}])
    def test_out_of_range_settings_are_refused(self, benzamides, kwargs):
        with pytest.raises(ValueError):
            cluster_by_fingerprint(benzamides, **kwargs)


class TestAssignClusters:
    def test_none_is_the_single_cluster(self, mixed_charges):
        assert set(assign_clusters(mixed_charges, "none").values()) == {0}

    def test_dispatch_matches_the_clusterer(self, mixed_charges):
        assert assign_clusters(mixed_charges, "charge") == cluster_by_charge(mixed_charges)

    def test_an_unknown_method_is_refused(self, mixed_charges):
        with pytest.raises(ValueError, match="Unknown cluster_by"):
            assign_clusters(mixed_charges, "murcko")

    def test_options_meant_for_the_fingerprint_clusterer_are_refused_elsewhere(self, mixed_charges):
        with pytest.raises(ValueError, match="takes no options"):
            assign_clusters(mixed_charges, "charge", n_clusters=2)

    def test_every_declared_method_runs(self, mixed_charges):
        for method in CLUSTER_METHODS:
            assert set(assign_clusters(mixed_charges, method)) == set(mixed_charges)


class TestEdgeBudget:
    def test_a_partition_beats_the_unclustered_floor(self, mixed_charges):
        budget = cluster_edge_budget(cluster_by_charge(mixed_charges))
        assert budget["clustered_floor"] < budget["unclustered_floor"]
        assert budget["saving"] > 0.0

    def test_one_cluster_saves_nothing(self, benzamides):
        budget = cluster_edge_budget(assign_clusters(benzamides, "none"))
        assert budget["clustered_floor"] == pytest.approx(budget["unclustered_floor"])
        assert budget["saving"] == pytest.approx(0.0)

    def test_the_floors_are_n_log_n(self):
        partition = {f"l{i}": i // 5 for i in range(10)}
        budget = cluster_edge_budget(partition)
        assert budget["unclustered_floor"] == pytest.approx(10 * math.log(10))
        assert budget["clustered_floor"] == pytest.approx(2 * 5 * math.log(5))

    def test_sizes_are_reported_largest_first(self):
        assert cluster_sizes({"a": 0, "b": 1, "c": 1, "d": 1}) == [3, 1]

    def test_a_single_ligand_has_no_floor_to_save(self):
        assert cluster_edge_budget({"only": 0})["saving"] == 0.0


def complete_candidates(names, *, intra, cost_within=1.0, cost_across=0.5):
    """Every pair, priced so that crossing *intra* is *cheaper* than staying inside it.

    Deliberately the wrong way round. Clustering is a selection-level objective, not a
    price: if the crossings were also the dearest edges, a clustered network would be
    indistinguishable from one the cost model produced on its own, and the test would prove
    nothing. Priced like this, the unclustered planner reaches for crossings and the
    clustered one still refuses.
    """
    return [
        make_transformation(a, b, cost=cost_within if ((a in intra) == (b in intra)) else cost_across)
        for index, a in enumerate(sorted(names))
        for b in sorted(names)[index + 1 :]
    ]


def crossings_of(network, partition):
    """The selected edges whose endpoints lie in different clusters."""
    return [edge.unordered_key for edge in network.edges if partition[edge.source] != partition[edge.target]]


def edges_on_a_cycle(network):
    """Selected edges belonging to a biconnected component of two or more edges."""
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(network.ligands)
    graph.add_edges_from(edge.unordered_key for edge in network.edges)
    on_cycle: set[frozenset[str]] = set()
    for component in nx.biconnected_component_edges(graph):
        edges = list(component)
        if len(edges) >= 2:
            on_cycle |= {frozenset(edge) for edge in edges}
    return on_cycle


class TestClusteredPlanning:
    """The knob on the planner, driven with hand-authored costs.

    Charge clustering is used throughout because it is the one clusterer with no threshold
    in it: the expected partition is a fact about the molecules rather than about a cut.
    """

    @staticmethod
    def _plan(ligands, **overrides):
        candidates = complete_candidates(ligands, intra=NEUTRAL)
        return create_planner("mst").plan(ligands, candidates, NetworkOptions(**overrides))

    def test_off_by_default(self, mixed_charges):
        assert NetworkOptions().cluster_by == "none"

    def test_clustering_reduces_the_crossings_to_the_bridge_count(self, mixed_charges):
        partition = cluster_by_charge(mixed_charges)
        unclustered = crossings_of(self._plan(mixed_charges), partition)
        clustered = crossings_of(self._plan(mixed_charges, cluster_by="charge"), partition)
        assert len(clustered) == 2, "two clusters, joined once, at cluster_bridges=2"
        assert len(unclustered) > len(clustered), "the unclustered planner prefers the cheap crossings"

    def test_two_bridges_put_every_crossing_on_a_cycle(self, mixed_charges):
        """The reason ``cluster_bridges`` defaults to 2 rather than to 1.

        A crossing is the least similar edge in the network and therefore the least
        trustworthy; two of them between the same pair of clusters close a loop through
        both, which is the every-edge-in-a-cycle invariant applied where it buys most.
        """
        partition = cluster_by_charge(mixed_charges)
        network = self._plan(mixed_charges, cluster_by="charge", cluster_bridges=2)
        on_cycle = edges_on_a_cycle(network)
        crossings = crossings_of(network, partition)
        assert crossings, "the clusters must actually be joined"
        assert all(frozenset(pair) in on_cycle for pair in crossings)

    def test_one_bridge_leaves_the_crossing_unchecked(self, mixed_charges):
        """The contrast that gives the previous test its meaning."""
        partition = cluster_by_charge(mixed_charges)
        network = self._plan(mixed_charges, cluster_by="charge", cluster_bridges=1)
        crossings = crossings_of(network, partition)
        assert len(crossings) == 1
        assert frozenset(crossings[0]) not in edges_on_a_cycle(network)

    def test_the_network_still_spans_every_ligand(self, mixed_charges):
        network = self._plan(mixed_charges, cluster_by="charge")
        graph = network.to_networkx()
        assert nx.is_connected(graph)

    def test_a_single_cluster_changes_nothing(self, mixed_charges):
        """``cluster_by`` may only ever remove crossings, and there are none to remove."""
        plain = self._plan(mixed_charges)
        one_cluster = self._plan(mixed_charges, cluster_by="scaffold")
        assert {e.key for e in one_cluster.edges} == {e.key for e in plain.edges}

    def test_a_forced_crossing_survives_pruning(self, mixed_charges):
        """A forced edge outranks the partition; it is intent, not a heuristic."""
        network = self._plan(mixed_charges, cluster_by="charge", forced_edges=("acid_Cl~amide_H",))
        assert ("acid_Cl", "amide_H") in {edge.unordered_key for edge in network.edges}

    def test_a_crossing_is_restored_rather_than_disconnecting_the_network(self, mixed_charges):
        """Pruning is an optimisation, and one that changed the answer would be a bug.

        Here the anions reach each other *only* through the amides, so honouring the
        partition literally would split the network. The planner restores the crossing and
        says so, rather than returning a network the user did not ask for.
        """
        candidates = [
            make_transformation("amide_H", "amide_F", cost=1.0),
            make_transformation("amide_F", "amide_Cl", cost=1.0),
            make_transformation("amide_H", "acid_H", cost=5.0),
            make_transformation("amide_F", "acid_F", cost=5.0),
            make_transformation("amide_Cl", "acid_Cl", cost=5.0),
        ]
        with pytest.warns(UserWarning):
            network = create_planner("mst").plan(
                mixed_charges, candidates, NetworkOptions(cluster_by="charge", cluster_bridges=1)
            )
        assert nx.is_connected(network.to_networkx())
        assert any("restored" in constraint for constraint in network.unmet_constraints)

    def test_clustering_composes_with_cbfe_bridging(self, mixed_charges):
        """The reason this is a knob and not a separate planner.

        With no candidate at all between the classes, clustering has nothing to prune and
        CBFE still supplies the bridge -- the two features stack because they are stages of
        one planner rather than two planners the user must choose between.
        """
        candidates = [
            make_transformation(a, b, cost=1.0)
            for group in (sorted(NEUTRAL), sorted(ANIONIC))
            for index, a in enumerate(group)
            for b in group[index + 1 :]
        ]
        network = create_planner("mst").plan(
            mixed_charges, candidates, NetworkOptions(cluster_by="charge", cbfe_mode="bridge")
        )
        assert len(network.cbfe_edges) == 1
        assert nx.is_connected(network.to_networkx())


class TestOptionValidation:
    @pytest.mark.parametrize("method", CLUSTER_METHODS)
    def test_every_documented_method_is_accepted(self, method):
        assert NetworkOptions(cluster_by=method).cluster_by == method

    def test_an_unknown_method_is_refused(self):
        with pytest.raises(ValueError, match="cluster_by must be one of"):
            NetworkOptions(cluster_by="murcko")

    def test_zero_bridges_cannot_join_anything(self):
        with pytest.raises(ValueError, match="cluster_bridges must be at least 1"):
            NetworkOptions(cluster_bridges=0)

    def test_the_defaults_are_the_unclustered_behaviour(self):
        options = NetworkOptions()
        assert (options.cluster_by, options.cluster_bridges) == ("none", 2)


class TestSerialization:
    def test_the_knobs_round_trip(self, benzamides, tmp_path):
        from rbfenetmap.io.networkio import dump_network, load_network

        candidates = complete_candidates(benzamides, intra=set(list(benzamides)[:2]))
        network = create_planner("mst").plan(
            benzamides, candidates, NetworkOptions(cluster_by="fingerprint", cluster_bridges=3)
        )
        loaded = load_network(dump_network(network, tmp_path / "n.json"))
        assert (loaded.options.cluster_by, loaded.options.cluster_bridges) == ("fingerprint", 3)

    def test_a_file_predating_the_knobs_loads_unclustered(self, benzamides, tmp_path):
        """An absent key must mean v0.4's behaviour, not a crash."""
        import json

        from rbfenetmap.io.networkio import dump_network, load_network

        candidates = complete_candidates(benzamides, intra=set(list(benzamides)[:2]))
        network = create_planner("mst").plan(benzamides, candidates, NetworkOptions())
        path = dump_network(network, tmp_path / "n.json")
        payload = json.loads(path.read_text())
        del payload["options"]["cluster_by"]
        del payload["options"]["cluster_bridges"]
        path.write_text(json.dumps(payload))
        assert load_network(path).options.cluster_by == "none"


class TestCLI:
    """The flags reach the planner, end to end."""

    @pytest.mark.integration
    def test_plan_accepts_the_clustering_flags(self, tmp_path):
        from pathlib import Path

        from rbfenetmap.cli.main import main
        from rbfenetmap.io.networkio import load_network

        # tests/data/, not examples/data/: the latter is gitignored and regenerated on
        # demand, so it is absent from a fresh clone and from CI.
        sdf = Path(__file__).resolve().parent / "data" / "golden_benzamides.sdf"
        out = tmp_path / "n.json"
        argv = ["plan", "--ligands", str(sdf), "--out", str(out), "--cluster-by", "scaffold", "--cluster-bridges", "2"]
        assert main(argv) == 0
        options = load_network(out).options
        assert (options.cluster_by, options.cluster_bridges) == ("scaffold", 2)
