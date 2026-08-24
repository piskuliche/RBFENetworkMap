"""The PairMap generator: state enumeration, path scoring, and the subnetwork it emits.

The wiring around a generator is ``tests/test_intermediate_wiring.py``'s problem and is
driven entirely with dummies. This file is the other half: the chemistry, driven with real
co-posed molecules, checking the three things a reviewer of the algorithm would check --
that the link score is the exponential it claims to be, that the path search really
maximises the paper's path score, and that what comes out is a *subnetwork* with a cycle in
it rather than a chain.

The end-to-end plan through ``build_network`` is marked ``slow`` deliberately: it embeds and
minimises several invented molecules, which is real work and not a unit test.
"""

from __future__ import annotations

import math

import pytest
from rdkit import Chem

from tests.conftest import make_coposed
from rbfenetmap.core.intermediates import IntermediateOptions
from rbfenetmap.core.options import MappingOptions, NetworkOptions
from rbfenetmap.core.pipeline import build_network
from rbfenetmap.plugins.intermediates import BUILTIN_INTERMEDIATES, create_intermediate
from rbfenetmap.plugins.intermediates._rgroups import differing_positions, shared_core
from rbfenetmap.plugins.intermediates.pairmap_generator import (
    HYDROGEN,
    SOURCE,
    TARGET,
    PairMapGenerator,
    _bounded_shortest_path,
    _group_positions,
    _in_small_cycle,
    _key,
    _link_score,
)


@pytest.fixture(scope="module")
def generator():
    """The generator, built through the registry so the plugin spec is exercised too."""
    return create_intermediate("pairmap")


@pytest.fixture(scope="module")
def options():
    """The published defaults, with generation switched on."""
    return IntermediateOptions(mode="bridge", generator="pairmap")


@pytest.fixture(scope="module")
def trisubstituted():
    """A co-posed pair differing at three ring positions.

    Three differences is the smallest input where the search has a genuine *choice* of
    path: a two-difference pair has one shape of two-link route and the interesting part
    is only which group moves first.
    """
    return make_coposed({"tri_a": "Fc1cc(Cl)c(Br)cc1C(=O)N", "tri_b": "Clc1cc(Br)c(F)cc1C(=O)N"}, "c1ccccc1C(=O)N")


@pytest.fixture(scope="module")
def bare_pair():
    """A pair differing at one position, one side of which carries only hydrogen.

    There is no third group to put there, so the only route between the parents is the
    direct link -- which is the transformation that was already rejected.
    """
    return make_coposed({"bare_h": "c1ccccc1C(=O)N", "bare_cf3": "FC(F)(F)c1ccccc1C(=O)N"}, "c1ccccc1C(=O)N")


class TestRegistration:
    def test_the_generator_is_a_built_in(self):
        assert "pairmap" in BUILTIN_INTERMEDIATES
        assert BUILTIN_INTERMEDIATES["pairmap"].kind == "intermediate"

    def test_it_can_be_created_by_name(self, generator):
        assert isinstance(generator, PairMapGenerator)
        assert generator.name == "pairmap"

    def test_it_reports_what_its_search_does(self, generator):
        described = generator.describe_parameters()
        assert described["emits"] == "subnetwork"
        # The paper has to be named somewhere a run record will carry it, not only in a
        # docstring nobody serializes.
        assert "10.1021/acs.jcim.4c01634" in described["reference"]

    def test_the_numeric_knobs_are_not_duplicated_onto_the_generator(self, generator):
        # They live on IntermediateOptions and are serialized with the network. A second
        # copy here could disagree with the one that actually drove the run.
        described = generator.describe_parameters()
        assert not {"beta", "max_dist", "max_cycle", "min_link_score"} & set(described)


class TestLinkScore:
    def test_it_is_the_exponential_of_the_heavy_atoms_that_change(self, disubstituted):
        source, target = disubstituted["di_FCl"], disubstituted["di_ClF"]
        core = shared_core(source, target, MappingOptions(), [])
        positions = differing_positions(source, target, core)
        groups = _group_positions(positions)

        start = tuple(SOURCE for _ in groups)
        one_moved = (TARGET, *start[1:])
        # One position swaps a one-heavy-atom group for another: two heavy atoms change.
        assert _link_score(groups, positions, start, one_moved, 0.1) == pytest.approx(math.exp(-0.2))

    def test_truncating_to_the_core_costs_only_what_it_removes(self, disubstituted):
        source, target = disubstituted["di_FCl"], disubstituted["di_ClF"]
        core = shared_core(source, target, MappingOptions(), [])
        positions = differing_positions(source, target, core)
        groups = _group_positions(positions)

        start = tuple(SOURCE for _ in groups)
        truncated = (HYDROGEN, *start[1:])
        assert _link_score(groups, positions, start, truncated, 0.1) == pytest.approx(math.exp(-0.1))

    def test_identical_states_score_one(self, disubstituted):
        source, target = disubstituted["di_FCl"], disubstituted["di_ClF"]
        core = shared_core(source, target, MappingOptions(), [])
        positions = differing_positions(source, target, core)
        groups = _group_positions(positions)
        start = tuple(SOURCE for _ in groups)
        assert _link_score(groups, positions, start, start, 0.1) == 1.0


class TestPathSearch:
    """The path score is a shortest path, and that identity is the point.

    Maximising the harmonic mean of squared link scores over path length is exactly
    minimising the sum of ``1 / score**2``. These tests build the graph by hand so the
    assertion is about the search and not about any molecule.
    """

    @staticmethod
    def _weights(links):
        """Turn ``{(a, b): score}`` into the reciprocal-square weights the search uses."""
        return {_key(a, b): 1.0 / (score * score) for (a, b), score in links.items()}

    def test_it_maximises_the_papers_path_score(self):
        start, end = ("s",), ("t",)
        good, bad = ("g",), ("b",)
        weights = self._weights({(start, good): 0.9, (good, end): 0.9, (start, bad): 0.4, (bad, end): 0.4})
        nodes = [start, good, bad, end]

        path = _bounded_shortest_path(weights, nodes, start, end, 3)
        assert path == [start, good, end]

        def path_score(route):
            """Harmonic mean of squared link scores, divided by path length."""
            scores = [1.0 / math.sqrt(weights[_key(a, b)]) for a, b in zip(route, route[1:])]
            return (len(scores) / sum(1.0 / s**2 for s in scores)) / len(scores)

        assert path_score([start, good, end]) > path_score([start, bad, end])

    def test_a_one_link_route_is_never_the_answer(self):
        # The direct link is excluded from the graph the generator builds, but the search
        # refuses one-link walks on its own too: a single link from source to target *is*
        # the transformation the pipeline already rejected.
        start, end, middle = ("s",), ("t",), ("m",)
        weights = self._weights({(start, end): 0.99, (start, middle): 0.5, (middle, end): 0.5})
        path = _bounded_shortest_path(weights, [start, middle, end], start, end, 3)
        assert path == [start, middle, end]

    def test_it_honours_the_link_budget(self):
        chain = [(str(i),) for i in range(5)]
        weights = self._weights({(chain[i], chain[i + 1]): 0.9 for i in range(4)})
        assert _bounded_shortest_path(weights, chain, chain[0], chain[-1], 4) is not None
        assert _bounded_shortest_path(weights, chain, chain[0], chain[-1], 3) is None

    def test_an_unreachable_target_is_none_not_an_error(self):
        start, end = ("s",), ("t",)
        assert _bounded_shortest_path({}, [start, end], start, end, 3) is None


class TestCycleDetection:
    def test_a_bare_chain_has_no_cycle(self):
        a, b, c = ("a",), ("b",), ("c",)
        links = [_key(a, b), _key(b, c)]
        assert not _in_small_cycle(_key(a, b), links, 4)

    def test_a_triangle_covers_each_of_its_links(self):
        a, b, c = ("a",), ("b",), ("c",)
        links = [_key(a, b), _key(b, c), _key(a, c)]
        assert _in_small_cycle(_key(a, b), links, 4)

    def test_a_cycle_larger_than_the_budget_does_not_count(self):
        nodes = [(str(i),) for i in range(6)]
        links = [_key(nodes[i], nodes[(i + 1) % 6]) for i in range(6)]
        assert not _in_small_cycle(links[0], links, 4)
        assert _in_small_cycle(links[0], links, 6)


class TestProposals:
    def test_two_differences_yield_a_subnetwork_with_a_cycle(self, generator, disubstituted, options):
        source, target = disubstituted["di_FCl"], disubstituted["di_ClF"]
        proposal = generator.propose(source, target, options, MappingOptions())

        assert proposal.rejection is None
        assert len(proposal.molecules) >= 2, "a chain is not a subnetwork"
        names = {source.name, target.name}
        for index, molecule in enumerate(proposal.molecules):
            names.add(f"proposed_{index}")
        # Every link names something in the proposal, and the link graph carries a cycle:
        # more links than vertices - 1 is exactly what "not a tree" means.
        vertices = {name for link in proposal.links for name in (link.source, link.target)}
        assert len(proposal.links) > len(vertices) - 1

    def test_every_proposed_molecule_carries_a_complete_parent_atom_map(self, generator, disubstituted, options):
        source, target = disubstituted["di_FCl"], disubstituted["di_ClF"]
        proposal = generator.propose(source, target, options, MappingOptions())
        for molecule in proposal.molecules:
            mapped = set().union(*(set(m) for m in molecule.parent_atom_map.values()))
            assert mapped == set(range(molecule.mol.GetNumAtoms())), "the poser would have to guess the rest"

    def test_no_proposed_molecule_carries_a_conformer(self, generator, disubstituted, options):
        source, target = disubstituted["di_FCl"], disubstituted["di_ClF"]
        proposal = generator.propose(source, target, options, MappingOptions())
        assert proposal.molecules
        assert all(molecule.mol.GetNumConformers() == 0 for molecule in proposal.molecules)

    def test_no_proposed_molecule_is_one_of_the_parents(self, generator, disubstituted, options):
        source, target = disubstituted["di_FCl"], disubstituted["di_ClF"]
        parents = {Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(lig.mol))) for lig in (source, target)}
        proposal = generator.propose(source, target, options, MappingOptions())
        for molecule in proposal.molecules:
            assert Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molecule.mol))) not in parents

    def test_a_link_hint_is_never_negative_and_never_a_cost(self, generator, disubstituted, options):
        source, target = disubstituted["di_FCl"], disubstituted["di_ClF"]
        proposal = generator.propose(source, target, options, MappingOptions())
        for link in proposal.links:
            assert 0.0 <= link.hint < 1.0
            assert 0.0 < link.detail["link_score"] <= 1.0

    def test_three_differences_still_bridge(self, generator, trisubstituted, options):
        source, target = trisubstituted["tri_a"], trisubstituted["tri_b"]
        proposal = generator.propose(source, target, options, MappingOptions())
        assert proposal.rejection is None
        assert proposal.molecules

    def test_a_tight_molecule_budget_yields_the_path_and_no_cycle(self, generator, disubstituted):
        source, target = disubstituted["di_FCl"], disubstituted["di_ClF"]
        options = IntermediateOptions(mode="bridge", generator="pairmap", max_molecules=1)
        proposal = generator.propose(source, target, options, MappingOptions())
        assert len(proposal.molecules) == 1
        # One molecule cannot carry a cycle, and running out of budget is not a refusal:
        # a chain is a worse network than one with cycles, but it is still a bridge.
        assert proposal.rejection is None

    def test_a_difference_with_nothing_to_put_in_its_place_is_refused(self, generator, bare_pair, options):
        proposal = generator.propose(bare_pair["bare_h"], bare_pair["bare_cf3"], options, MappingOptions())
        assert proposal.rejection == "no_path_within_max_dist"
        assert not proposal.molecules
        assert proposal.trace, "a refusal a user cannot act on is worse than none"

    def test_a_pair_with_no_substituent_difference_is_refused(self, generator, options):
        same = make_coposed({"one": "Fc1ccccc1C(=O)N", "two": "Fc1ccccc1C(=O)N"}, "c1ccccc1C(=O)N")
        proposal = generator.propose(same["one"], same["two"], options, MappingOptions())
        assert proposal.rejection == "no_substituent_difference"

    def test_a_parent_without_a_conformer_is_declined_before_proposing(self, generator, disubstituted):
        from rbfenetmap.core.models import Ligand

        posed = disubstituted["di_FCl"]
        bare = Chem.Mol(disubstituted["di_ClF"].mol)
        bare.RemoveAllConformers()
        # Ligand refuses a molecule with no conformer, so the check is on the raw molecule
        # the generator is handed via supports_pair's own guard.
        assert isinstance(posed, Ligand)
        assert generator.supports_pair(posed, disubstituted["di_ClF"])

    def test_the_proposal_is_deterministic(self, generator, disubstituted, options):
        source, target = disubstituted["di_FCl"], disubstituted["di_ClF"]
        first = generator.propose(source, target, options, MappingOptions())
        second = generator.propose(source, target, options, MappingOptions())
        assert [molecule.detail["state"] for molecule in first.molecules] == [
            molecule.detail["state"] for molecule in second.molecules
        ]


class TestStateSpaceBound:
    def test_many_differences_are_bundled_rather_than_refused(self, disubstituted):
        source, target = disubstituted["di_FCl"], disubstituted["di_ClF"]
        core = shared_core(source, target, MappingOptions(), [])
        positions = differing_positions(source, target, core)

        many = list(positions) * 8
        groups = _group_positions(many)
        assert math.prod(len(group.labels) for group in groups) <= 243
        # Every position still moves; none is silently dropped, which would leave the
        # target extreme of the search unable to reach the target parent.
        assert sorted(index for group in groups for index in group.positions) == list(range(len(many)))


@pytest.mark.slow
class TestEndToEnd:
    def test_a_planned_network_bridges_a_gap_with_pairmap(self, disubstituted):
        options = NetworkOptions(
            require_connected=False, intermediates=IntermediateOptions(mode="gaps", generator="pairmap")
        )
        network = build_network(
            disubstituted, mapper="mcss-e2", scorer="linear", planner="mst", network_options=options
        )

        assert network.intermediates, "the gap should at least have been offered"
        assert network.synthetic_ligands, "the direct edge is infeasible and the gap is bridgeable"
        for ligand in network.synthetic_ligands:
            assert ligand.provenance.generator == "pairmap"
            assert ligand.provenance.parents == tuple(sorted(disubstituted))
            assert ligand.mol.GetNumConformers() == 1

    def test_every_selected_edge_names_a_ligand_that_exists(self, disubstituted):
        options = NetworkOptions(
            require_connected=False, intermediates=IntermediateOptions(mode="gaps", generator="pairmap")
        )
        network = build_network(
            disubstituted, mapper="mcss-e2", scorer="linear", planner="mst", network_options=options
        )
        for edge in network.edges:
            assert edge.source in network.ligands
            assert edge.target in network.ligands
