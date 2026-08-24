"""``--consistency graph``: one core per ligand rather than one per edge.

The series used here is the tracked golden set, and it contains the case the whole feature
exists for. Mapped pairwise, ``bza_Me~bza_Et`` gets an 18-atom core -- the two ligands are
more like each other than either is like the rest of the series -- while every other edge on
those two ligands gets 15. Graph consistency is exactly the rule that says a ligand cannot
hold both.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rbfenetmap.core.consistency import apply_graph_consistency, graph_consistent_cores
from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.models import EdgeKind, Network
from rbfenetmap.core.options import NetworkOptions, SoftcorePolicy
from rbfenetmap.core.pipeline import build_network
from rbfenetmap.io.loaders import load_ligands
from rbfenetmap.io.networkio import dump_network, load_network, network_to_dict

from .conftest import make_transformation

GOLDEN_SDF = Path(__file__).resolve().parent / "data" / "golden_benzamides.sdf"


@pytest.fixture(scope="module")
def series():
    """The tracked nine-ligand series, indexed by name."""
    return {ligand.name: ligand for ligand in load_ligands([str(GOLDEN_SDF)])}


@pytest.fixture(scope="module")
def pairwise(series) -> Network:
    """The default network: every edge carries its own core."""
    return build_network(series)


@pytest.fixture(scope="module")
def consistent(series) -> Network:
    """The same network with each ligand reduced to one core."""
    return build_network(series, network_options=NetworkOptions(consistency="graph"))


def _softcore_by_ligand(network: Network) -> dict[str, set[frozenset[int]]]:
    """The distinct **soft-cores** each ligand holds across its RBFE edges.

    Deliberately separate from :func:`_core_by_ligand` and asserted on directly rather than
    inferred from it. The soft-core is what the feature is asked for by name -- an Amber
    ``scmask`` is a soft-core, not a core -- and "the complement is a partition, so the two
    are equivalent" is a true statement about ``AtomMapping`` that a bug in this module
    could not violate but a bug in ``_restrict`` very much could, by handing back a mapping
    whose halves no longer correspond.
    """
    held: dict[str, set[frozenset[int]]] = {}
    for edge in network.edges:
        if edge.kind is not EdgeKind.RBFE:
            continue
        held.setdefault(edge.source, set()).add(frozenset(edge.mapping.sc1))
        held.setdefault(edge.target, set()).add(frozenset(edge.mapping.sc2))
    return held


def _core_by_ligand(network: Network) -> dict[str, set[frozenset[int]]]:
    """The distinct cores each ligand holds across its RBFE edges."""
    held: dict[str, set[frozenset[int]]] = {}
    for edge in network.edges:
        if edge.kind is not EdgeKind.RBFE:
            continue
        held.setdefault(edge.source, set()).add(frozenset(edge.mapping.cc1))
        held.setdefault(edge.target, set()).add(frozenset(edge.mapping.cc2))
    return held


class TestOption:
    def test_an_unknown_value_is_refused(self):
        with pytest.raises(ValueError, match="consistency must be"):
            NetworkOptions(consistency="graphy")

    def test_it_round_trips_through_the_network_file(self, tmp_path, series):
        network = build_network({n: series[n] for n in ("bza_H", "bza_F", "bza_Cl")})
        options = network_to_dict(network)["options"]
        assert options["consistency"] == "pairwise"
        path = dump_network(network, tmp_path / "n.json")
        assert load_network(path).options.consistency == "pairwise"

    def test_a_file_written_before_the_key_existed_loads_as_pairwise(self, tmp_path, series):
        network = build_network({n: series[n] for n in ("bza_H", "bza_F", "bza_Cl")})
        path = dump_network(network, tmp_path / "n.json")
        payload = json.loads(path.read_text())
        del payload["options"]["consistency"]
        path.write_text(json.dumps(payload))
        assert load_network(path).options.consistency == "pairwise"


class TestGraphConsistency:
    @pytest.mark.integration
    def test_pairwise_lets_one_ligand_hold_several_cores(self, pairwise):
        """The premise. If this stops being true the feature has nothing to fix."""
        assert any(len(cores) > 1 for cores in _core_by_ligand(pairwise).values())

    @pytest.mark.integration
    def test_graph_leaves_every_ligand_holding_exactly_one(self, consistent):
        assert all(len(cores) == 1 for cores in _core_by_ligand(consistent).values())

    @pytest.mark.integration
    def test_the_outlying_core_is_the_one_that_shrinks(self, pairwise, consistent):
        before = {edge.unordered_key: edge.mapping.n_common_core for edge in pairwise.edges}
        after = {edge.unordered_key: edge.mapping.n_common_core for edge in consistent.edges}
        assert before[("bza_Et", "bza_Me")] == 18
        assert after[("bza_Et", "bza_Me")] == 15
        assert {pair for pair in before if before[pair] != after[pair]} == {("bza_Et", "bza_Me")}

    @pytest.mark.integration
    def test_a_reduced_core_is_re_costed_upward(self, pairwise, consistent):
        """A smaller core means more soft-core, and the cost must say so."""
        before = {edge.unordered_key: edge.score.total for edge in pairwise.edges}
        after = {edge.unordered_key: edge.score.total for edge in consistent.edges}
        assert after[("bza_Et", "bza_Me")] > before[("bza_Et", "bza_Me")]

    @pytest.mark.integration
    def test_selection_is_unchanged(self, pairwise, consistent):
        """It refines mappings; it does not re-select. Same edges, same directions."""
        assert [e.key for e in consistent.edges] == [e.key for e in pairwise.edges]

    @pytest.mark.integration
    def test_the_result_still_validates(self, consistent):
        consistent.validate(require_connected=True)

    @pytest.mark.integration
    def test_the_surviving_cores_are_subsets_of_the_pairwise_ones(self, pairwise):
        cores = graph_consistent_cores(pairwise)
        for edge in pairwise.edges:
            assert cores[edge.source] <= frozenset(edge.mapping.cc1)
            assert cores[edge.target] <= frozenset(edge.mapping.cc2)

    @pytest.mark.integration
    def test_an_edge_left_infeasible_raises_rather_than_reverting(self, pairwise):
        """A network handed back with an infeasible selected edge cannot be run."""
        strict = SoftcorePolicy(min_core_atoms=16)
        with pytest.raises(NetworkPlanError, match="leaves these selected edge"):
            apply_graph_consistency(pairwise, policy=strict)

    def test_a_cbfe_edge_does_not_erase_its_endpoints_cores(self, benzamides):
        """A counterpoised edge has no core by construction; reading it as one would be
        an artefact of the bridge rather than a statement about the scaffold."""
        ligands = {n: benzamides[n] for n in ("bza_H", "bza_F", "bza_Cl")}
        edges = (
            make_transformation("bza_H", "bza_F", n_atoms=6),
            make_transformation("bza_F", "bza_Cl", n_atoms=6, kind=EdgeKind.CBFE),
        )
        network = Network(ligands=ligands, edges=edges, options=NetworkOptions())
        cores = graph_consistent_cores(network)
        assert cores["bza_F"] == frozenset(range(6))
        assert "bza_Cl" not in cores


class TestSoftcoreUniformity:
    """The property the feature is actually asked for, asserted on ``sc1``/``sc2``.

    Not marked ``integration``. The core-side equivalent is, which is why it took a user
    asking "could you add a knob for this?" to notice the feature already existed -- the
    assertion that would have shown it was deselected in the run everyone watches.
    """

    def test_pairwise_lets_a_ligand_hold_several_softcores(self, pairwise):
        """The premise, so the assertion below cannot pass vacuously."""
        offenders = {name for name, held in _softcore_by_ligand(pairwise).items() if len(held) > 1}
        assert offenders == {"bza_Et", "bza_Me"}

    def test_graph_leaves_every_ligand_holding_exactly_one(self, consistent):
        assert all(len(held) == 1 for held in _softcore_by_ligand(consistent).values())

    def test_the_softcore_is_the_exact_complement_of_the_core(self, consistent):
        """What makes one core per ligand *mean* one soft-core per ligand."""
        for edge in consistent.edges:
            for cc, sc, total in (
                (edge.mapping.cc1, edge.mapping.sc1, edge.mapping.n_atoms_1),
                (edge.mapping.cc2, edge.mapping.sc2, edge.mapping.n_atoms_2),
            ):
                assert set(cc).isdisjoint(sc)
                assert set(cc) | set(sc) == set(range(total))


@pytest.fixture(scope="module")
def by_component(series) -> Network:
    """The same network with the rule scoped to each connected component."""
    return build_network(series, network_options=NetworkOptions(consistency="component"))


class TestScope:
    """``component`` asks for the rule within a component; ``graph`` across the network."""

    def test_component_also_leaves_one_softcore_per_ligand(self, by_component):
        assert all(len(held) == 1 for held in _softcore_by_ligand(by_component).values())

    def test_on_a_connected_series_the_two_scopes_agree(self, by_component, consistent):
        """One component means one group, so there is nothing for the scope to change."""
        assert _softcore_by_ligand(by_component) == _softcore_by_ligand(consistent)

    def test_a_boundary_edge_is_exempt(self, series):
        """The tradeoff scoping makes, pinned rather than left to be discovered.

        Two groups joined by one edge: the ligands either side of it are not being asked to
        share anything, so that edge keeps its pairwise core. Feeding it into both groups'
        intersections is not the alternative -- the constraint would propagate across the
        join and collapse the scope back to ``graph``.
        """
        from rbfenetmap.core.consistency import consistency_groups

        network = build_network(series, network_options=NetworkOptions(consistency="component"))
        groups = consistency_groups(network, "component")
        assert len(set(groups.values())) == 1  # this series is one component
        assert consistency_groups(network, "graph") is None

    def test_pairwise_is_the_only_scope_that_does_nothing(self, series, pairwise):
        untouched = build_network(series, network_options=NetworkOptions(consistency="pairwise"))
        assert _softcore_by_ligand(untouched) == _softcore_by_ligand(pairwise)


class TestHydrogensAreIntersectedByCount:
    """Why the intersection is heavy-atom-first rather than index-wise.

    Two edges out of one ligand routinely core *different* hydrogens of a symmetric group;
    which one got paired is an artefact of the embedding. Intersecting raw indices drops
    both and keeps the parent, and a demoted hydrogen on a common-core parent is its own
    soft-core region -- so the repair pays to bridge regions the intersection invented, and
    the cascade eats the core. Measured before the fix: every edge of a three-scaffold set
    failed with the heavy core untouched at nine atoms.
    """

    def test_a_cored_hydrogen_keeps_its_parent(self, consistent, series):
        from rbfenetmap.core.molgraph import hydrogen_parents

        for edge in consistent.edges:
            if edge.kind is not EdgeKind.RBFE:
                continue
            parents = hydrogen_parents(series[edge.source].mol)
            core = set(edge.mapping.cc1)
            orphans = {h for h in core if h in parents and parents[h] not in core}
            assert not orphans, f"{edge.key}: cored hydrogen(s) {sorted(orphans)} without their parent"

    def test_the_documented_core_sizes_are_unchanged(self, consistent):
        """The heavy-first rewrite must not move the series this feature is documented on."""
        after = {edge.unordered_key: edge.mapping.n_common_core for edge in consistent.edges}
        assert after[("bza_Et", "bza_Me")] == 15
