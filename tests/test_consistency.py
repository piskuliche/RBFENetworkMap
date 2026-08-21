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
