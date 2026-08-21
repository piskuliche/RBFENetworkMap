"""Intermediate ligands: the model, the serialization, the plugin seam, and the poser.

Nothing here reaches the pipeline, because nothing in this phase wires generation into
it. Each layer is exercised against the one below it and against the ``Dummy`` plugins,
which is what makes a failure attributable: a bad pose fails a posing test, not a
planning one.

``examples/data/benzamides.sdf`` is deliberately never touched. It is gitignored and
regenerated on demand, so a test that read it would pass locally and fail in CI on a
missing file. Everything below is built in memory with ``make_coposed``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from tests.conftest import DummyIntermediateGenerator, make_coposed, make_ligand, scramble_frame
from rbfenetmap.core.exceptions import PluginError
from rbfenetmap.core.intermediates import (
    INTERMEDIATE_NAME_PREFIX,
    IntermediateOptions,
    IntermediateProposal,
    ProposedLink,
    ProposedMolecule,
    intermediate_name,
    reserve_intermediate_names,
    synthesize_ligand,
)
from rbfenetmap.core.meta.intermediates import AbstractIntermediateGenerator
from rbfenetmap.core.models import (
    _LIGAND_NAME_RE,
    EDGE_SEPARATOR,
    IntermediateRecord,
    Ligand,
    LigandProvenance,
    Network,
    RejectionReason,
    edge_key,
    parse_edge_key,
)
from rbfenetmap.core.options import MappingOptions, NetworkOptions, SoftcorePolicy
from rbfenetmap.core.pipeline import build_network
from rbfenetmap.core.posing import POSE_RMSD_FACTOR, PoseDonor, PoseRejection, pose_intermediate
from rbfenetmap.io.networkio import SCHEMA_VERSION, dump_network, load_network, network_to_dict
from rbfenetmap.plugins.intermediates import (
    BUILTIN_INTERMEDIATES,
    available_intermediates,
    create_intermediate,
    list_active_intermediates,
    require_intermediates,
)

PROVENANCE = LigandProvenance(
    kind="intermediate", generator="dummy", parents=("lig_a", "lig_b"), pose_method="parent_atom_map", pose_rmsd=0.12
)


def coords(mol: Chem.Mol) -> np.ndarray:
    """The molecule's single conformer as an ``(n, 3)`` array."""
    return np.asarray(mol.GetConformer().GetPositions(), dtype=float)


@pytest.fixture
def swapped(disubstituted):
    """The two hybrids ``FragmentSwapGenerator`` proposes for the disubstituted pair."""
    source, target = disubstituted["di_FCl"], disubstituted["di_ClF"]
    generator = create_intermediate("fragment-swap")
    return generator.propose(source, target, IntermediateOptions(), MappingOptions())


# --- Step 1: the model -----------------------------------------------------------


class TestLigandProvenance:
    def test_rejects_an_empty_kind(self):
        with pytest.raises(ValueError, match="kind must be"):
            LigandProvenance(kind="", generator="g", parents=("a",), pose_method="m", pose_rmsd=0.0)

    def test_rejects_an_empty_generator(self):
        with pytest.raises(ValueError, match="generator must be"):
            LigandProvenance(kind="intermediate", generator="", parents=("a",), pose_method="m", pose_rmsd=0.0)

    def test_rejects_a_parentless_provenance(self):
        with pytest.raises(ValueError, match="at least one ligand"):
            LigandProvenance(kind="intermediate", generator="g", parents=(), pose_method="m", pose_rmsd=0.0)

    def test_rejects_a_negative_rmsd(self):
        with pytest.raises(ValueError, match="non-negative"):
            LigandProvenance(kind="intermediate", generator="g", parents=("a",), pose_method="m", pose_rmsd=-1.0)


class TestLigandField:
    def test_from_mol_leaves_provenance_unset(self, benzene):
        assert benzene.provenance is None
        assert benzene.synthetic is False

    def test_every_coposed_ligand_is_real(self, benzamides):
        assert all(ligand.provenance is None and not ligand.synthetic for ligand in benzamides.values())

    def test_positional_construction_still_works(self, benzene):
        """The field is appended last and defaulted, so the old five-argument form is intact."""
        ligand = Ligand("copy", benzene.mol, 0, None, {})
        assert ligand.provenance is None and not ligand.synthetic

    def test_synthesized_sets_provenance_and_charge(self, benzene):
        ligand = Ligand.synthesized(benzene.mol, "int_a_b_abc123", PROVENANCE)
        assert ligand.synthetic
        assert ligand.provenance is PROVENANCE
        assert ligand.charge == 0

    def test_from_mol_metadata_named_provenance_is_metadata(self, benzene):
        """``from_mol`` collects ``**metadata``, which is precisely why ``synthesized`` exists."""
        ligand = Ligand.from_mol(benzene.mol, "annotated", provenance="ChEMBL")
        assert ligand.metadata["provenance"] == "ChEMBL"
        assert ligand.provenance is None and not ligand.synthetic

    def test_replace_preserves_provenance(self, benzene):
        """``core.align`` moves ligands with ``dataclasses.replace``; a field survives that."""
        import dataclasses

        ligand = Ligand.synthesized(benzene.mol, "int_a_b_abc123", PROVENANCE)
        assert dataclasses.replace(ligand, name="int_a_b_abc124").synthetic


class TestNetworkField:
    def test_intermediates_default_to_empty(self, benzamides):
        assert Network(ligands=benzamides).intermediates == ()

    def test_synthetic_ligands_are_derived(self, benzamides, benzene):
        invented = Ligand.synthesized(benzene.mol, "int_a_b_abc123", PROVENANCE)
        network = Network(ligands={**benzamides, invented.name: invented})
        assert [ligand.name for ligand in network.synthetic_ligands] == [invented.name]

    def test_an_all_real_network_has_no_synthetic_ligands(self, benzamides):
        assert Network(ligands=benzamides).synthetic_ligands == ()


class TestNaming:
    def test_is_a_legal_ligand_name(self, benzene):
        name = intermediate_name(("bza_H", "bza_F"), benzene.mol)
        assert _LIGAND_NAME_RE.match(name)
        assert EDGE_SEPARATOR not in name
        assert name.startswith(INTERMEDIATE_NAME_PREFIX)

    def test_survives_the_edge_key_round_trip(self, benzene):
        name = intermediate_name(("bza_H", "bza_F"), benzene.mol)
        assert parse_edge_key(edge_key(name, "bza_H")) == (name, "bza_H")

    def test_parents_are_sorted(self, benzene):
        assert intermediate_name(("b", "a"), benzene.mol) == intermediate_name(("a", "b"), benzene.mol)

    def test_the_same_molecule_from_the_same_gap_gets_one_name(self, benzene):
        """Content-addressed, so a gap reached from both ends collapses to a single ligand."""
        again = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(benzene.mol)))
        assert intermediate_name(("a", "b"), benzene.mol) == intermediate_name(("a", "b"), again)

    def test_a_different_molecule_gets_a_different_name(self, benzene, benzamides):
        assert intermediate_name(("a", "b"), benzene.mol) != intermediate_name(("a", "b"), benzamides["bza_H"].mol)

    def test_hydrogen_treatment_does_not_change_the_name(self, benzene):
        assert intermediate_name(("a", "b"), benzene.mol) == intermediate_name(("a", "b"), Chem.RemoveHs(benzene.mol))

    def test_needs_a_parent(self, benzene):
        with pytest.raises(ValueError, match="at least one parent"):
            intermediate_name((), benzene.mol)


class TestReservedPrefix:
    def test_not_reserved_when_the_feature_is_off(self):
        reserve_intermediate_names(["int_3", "lig_a"], enabled=False)

    def test_reserved_when_the_feature_is_on(self):
        with pytest.raises(ValueError, match="reserved prefix"):
            reserve_intermediate_names(["int_3", "lig_a"], enabled=True)

    def test_ordinary_names_pass_either_way(self):
        reserve_intermediate_names(["lig_a", "integral"], enabled=True)

    def test_build_network_still_accepts_an_int_name(self, benzamides, dummy_mapper, dummy_scorer, dummy_planner):
        """Nothing in this phase enables generation, so the prefix is never reserved yet."""
        renamed = {}
        for index, ligand in enumerate(sorted(benzamides.values(), key=lambda lig: lig.name)[:3]):
            name = f"int_{index}"
            renamed[name] = Ligand(name, ligand.mol, ligand.charge)
        network = build_network(
            renamed,
            mapper=dummy_mapper,
            scorer=dummy_scorer,
            planner=dummy_planner,
            network_options=NetworkOptions(require_connected=False),
        )
        assert set(network.ligands) == set(renamed)


# --- Step 2: serialization -------------------------------------------------------


class TestSerialization:
    def test_schema_version_is_unchanged(self):
        """Absent-means-default is a complete compatibility story; a bump would not be."""
        assert SCHEMA_VERSION == 1

    def test_an_all_real_network_gains_no_keys(self, benzamides, dummy_mapper, dummy_scorer, dummy_planner):
        network = build_network(
            benzamides,
            mapper=dummy_mapper,
            scorer=dummy_scorer,
            planner=dummy_planner,
            network_options=NetworkOptions(require_connected=False),
        )
        data = network_to_dict(network)
        assert "intermediates" not in data
        assert all("provenance" not in ligand for ligand in data["ligands"])

    def test_a_pre_phase_4_file_loads_unchanged(self, tmp_path, benzamides, dummy_mapper, dummy_scorer, dummy_planner):
        """The test that proves no schema bump was needed."""
        network = build_network(
            benzamides,
            mapper=dummy_mapper,
            scorer=dummy_scorer,
            planner=dummy_planner,
            network_options=NetworkOptions(require_connected=False),
        )
        path = Path(tmp_path) / "legacy.json"
        data = network_to_dict(network)
        # A file written before this feature existed carries neither key anywhere.
        assert "intermediates" not in data
        path.write_text(json.dumps(data, indent=2))
        loaded = load_network(path)
        assert loaded.intermediates == ()
        assert all(not ligand.synthetic for ligand in loaded.ligands.values())
        assert network_to_dict(loaded) == data

    def test_provenance_round_trips(self, tmp_path, benzamides, benzene):
        invented = Ligand.synthesized(benzene.mol, "int_bza_F_bza_H_abc123", PROVENANCE)
        network = Network(ligands={**benzamides, invented.name: invented}, options=NetworkOptions())
        path = dump_network(network, Path(tmp_path) / "network.json")
        loaded = load_network(path)
        restored = loaded.ligands[invented.name]
        assert restored.synthetic
        assert restored.provenance == PROVENANCE
        assert all(not loaded.ligands[name].synthetic for name in benzamides)

    def test_intermediate_records_round_trip(self, tmp_path, benzamides):
        records = (
            IntermediateRecord("bza_H", "bza_F", "fragment-swap", True, ("int_bza_F_bza_H_abc123",), None, ("built",)),
            IntermediateRecord("bza_H", "bza_Cl", "fragment-swap", False, (), "single_substituent_difference", ()),
        )
        network = Network(ligands=benzamides, options=NetworkOptions(), intermediates=records)
        loaded = load_network(dump_network(network, Path(tmp_path) / "network.json"))
        assert loaded.intermediates == records

    def test_a_rejected_attempt_is_retained(self, tmp_path, benzamides):
        """A network where generation ran and found nothing must not look like one where it never ran."""
        network = Network(
            ligands=benzamides,
            options=NetworkOptions(),
            intermediates=(IntermediateRecord("bza_H", "bza_F", "fragment-swap", rejection="no_common_core"),),
        )
        loaded = load_network(dump_network(network, Path(tmp_path) / "network.json"))
        assert loaded.intermediates[0].rejection == "no_common_core"
        assert not loaded.intermediates[0].accepted


# --- Step 3: the plugin seam -----------------------------------------------------


class TestRegistry:
    def test_the_builtin_is_registered(self):
        assert "fragment-swap" in BUILTIN_INTERMEDIATES
        assert BUILTIN_INTERMEDIATES["fragment-swap"].kind == "intermediate"

    def test_it_is_available_and_creatable(self):
        assert "fragment-swap" in available_intermediates()
        assert isinstance(create_intermediate("fragment-swap"), AbstractIntermediateGenerator)

    def test_active_listing(self):
        assert list_active_intermediates() == ["fragment-swap"]

    def test_unknown_name_names_the_known_ones(self):
        with pytest.raises(PluginError, match="fragment-swap"):
            create_intermediate("no-such-generator")

    def test_require_accepts_the_builtin(self):
        require_intermediates(("fragment-swap",))

    def test_the_cli_lists_the_kind(self, capsys):
        from rbfenetmap.cli.main import main

        assert main(["plugins", "--kind", "intermediate"]) == 0
        assert "fragment-swap" in capsys.readouterr().out


class TestProposalTypes:
    def test_a_proposed_molecule_carries_no_conformer(self, benzene):
        """Posing is centralised, so coordinates handed in are discarded rather than trusted."""
        proposed = ProposedMolecule(mol=benzene.mol, parents=("b", "a"))
        assert proposed.mol.GetNumConformers() == 0
        assert benzene.mol.GetNumConformers() == 1, "the caller's molecule must not be mutated"

    def test_parents_are_sorted(self, benzene):
        assert ProposedMolecule(mol=benzene.mol, parents=("b", "a")).parents == ("a", "b")

    def test_a_map_for_a_non_parent_is_refused(self, benzene):
        with pytest.raises(ValueError, match="non-parent"):
            ProposedMolecule(mol=benzene.mol, parents=("a",), parent_atom_map={"c": {0: 0}})

    def test_a_parentless_molecule_is_refused(self, benzene):
        with pytest.raises(ValueError, match="at least one parent"):
            ProposedMolecule(mol=benzene.mol, parents=())

    def test_a_link_hint_is_not_a_score(self):
        """Advisory only: nothing reads it as a cost, and it has no ``total`` to be mistaken for one."""
        link = ProposedLink(source="a", target="int_a_b_abc123", hint=0.25)
        assert not hasattr(link, "total")
        assert ProposedLink(source="a", target="b").hint is None

    def test_a_rejection_is_a_plain_string(self):
        proposal = IntermediateProposal(source="a", target="b", generator="dummy", rejection="no_common_core")
        assert isinstance(proposal.rejection, str)
        assert not isinstance(proposal.rejection, RejectionReason)
        assert not proposal.proposed

    def test_a_gap_needs_two_ligands(self):
        with pytest.raises(ValueError, match="two distinct ligands"):
            IntermediateProposal(source="a", target="a", generator="dummy")


class TestDummyGenerator:
    def test_returns_the_supplied_proposal_verbatim(self, benzamides):
        proposal = IntermediateProposal(source="bza_H", target="bza_F", generator="dummy", trace=("hand-authored",))
        generator = DummyIntermediateGenerator(proposal)
        returned = generator.propose(benzamides["bza_H"], benzamides["bza_F"], IntermediateOptions(), MappingOptions())
        assert returned is proposal

    def test_declines_by_default(self, benzamides, dummy_intermediate_generator):
        returned = dummy_intermediate_generator.propose(
            benzamides["bza_H"], benzamides["bza_F"], IntermediateOptions(), MappingOptions()
        )
        assert returned.rejection == "dummy_declined"
        assert returned.molecules == ()

    def test_supports_every_pair_by_default(self, benzamides, dummy_intermediate_generator):
        assert dummy_intermediate_generator.supports_pair(benzamides["bza_H"], benzamides["bza_F"])
        assert dummy_intermediate_generator.describe_parameters() == {}


class TestFragmentSwapGenerator:
    def test_one_difference_leaves_nothing_to_invent(self, benzamides):
        """Every hybrid of a single-substituent pair *is* one of the parents."""
        generator = create_intermediate("fragment-swap")
        proposal = generator.propose(benzamides["bza_F"], benzamides["bza_Cl"], IntermediateOptions(), MappingOptions())
        assert proposal.rejection == "single_substituent_difference"
        assert proposal.molecules == ()

    def test_two_differences_yield_one_hybrid_each(self, swapped):
        assert swapped.rejection is None
        assert len(swapped.molecules) == 2
        assert len(swapped.links) == 4

    def test_each_hybrid_is_a_real_new_molecule(self, swapped, disubstituted):
        parents = {Chem.MolToSmiles(Chem.RemoveHs(lig.mol)) for lig in disubstituted.values()}
        hybrids = {Chem.MolToSmiles(Chem.RemoveHs(m.mol)) for m in swapped.molecules}
        assert len(hybrids) == 2
        assert not hybrids & parents

    def test_the_atom_map_is_complete(self, swapped):
        """Every atom of the hybrid is accounted for, which is what spares the poser a search."""
        for molecule in swapped.molecules:
            mapped = {idx for per_parent in molecule.parent_atom_map.values() for idx in per_parent}
            assert mapped == set(range(molecule.mol.GetNumAtoms()))

    def test_both_parents_contribute(self, swapped):
        for molecule in swapped.molecules:
            assert all(molecule.parent_atom_map[parent] for parent in molecule.parents)

    def test_the_links_name_the_predicted_ligand(self, swapped, disubstituted):
        names = {intermediate_name(m.parents, m.mol) for m in swapped.molecules}
        endpoints = {link.source for link in swapped.links} | {link.target for link in swapped.links}
        assert names <= endpoints
        assert set(disubstituted) <= endpoints

    def test_max_molecules_caps_the_output(self, disubstituted):
        generator = create_intermediate("fragment-swap")
        proposal = generator.propose(
            disubstituted["di_FCl"], disubstituted["di_ClF"], IntermediateOptions(max_molecules=1), MappingOptions()
        )
        assert len(proposal.molecules) == 1

    def test_it_describes_its_parameters(self):
        assert create_intermediate("fragment-swap").describe_parameters() == {"swaps_per_molecule": 1}


class TestIntermediateOptions:
    @pytest.mark.parametrize(
        "kwargs", [{"max_molecules": 0}, {"max_pose_attempts": 0}, {"pose_rmsd_factor": 0.0}, {"pose_rmsd_factor": 2.0}]
    )
    def test_rejects_nonsense(self, kwargs):
        with pytest.raises(ValueError):
            IntermediateOptions(**kwargs)

    def test_defaults_to_the_named_factor(self):
        assert IntermediateOptions().pose_rmsd_factor == POSE_RMSD_FACTOR


# --- Step 4: posing --------------------------------------------------------------


class TestPosing:
    def test_the_raw_coordinates_lie_in_the_parents_frame(self, swapped, disubstituted):
        """Compared **directly**, atom by atom, not as an RMSD after a fit.

        An RMSD-after-superposition assertion passes for a molecule posed in an entirely
        wrong frame -- the fit removes exactly the error being tested for. Only the raw
        deviation from the donor coordinate says the pose is where the parents are.
        """
        for molecule in swapped.molecules:
            ligand, result = synthesize_ligand(molecule, disubstituted, generator="fragment-swap")
            assert ligand is not None, result.rejection
            posed = coords(ligand.mol)
            for parent, atom_map in molecule.parent_atom_map.items():
                donor = coords(disubstituted[parent].mol)
                for here, there in atom_map.items():
                    assert np.linalg.norm(posed[here] - donor[there]) < 0.5

    def test_the_pose_is_recorded_on_the_ligand(self, swapped, disubstituted):
        ligand, result = synthesize_ligand(swapped.molecules[0], disubstituted, generator="fragment-swap")
        assert ligand.synthetic
        assert ligand.provenance.generator == "fragment-swap"
        assert ligand.provenance.parents == ("di_ClF", "di_FCl")
        assert ligand.provenance.pose_method == "parent_atom_map"
        assert ligand.provenance.pose_rmsd == pytest.approx(result.rmsd)

    def test_the_posed_ligand_is_a_legal_ligand(self, swapped, disubstituted):
        """``Ligand.__post_init__`` forbids implicit hydrogens, which is why AddHs runs first."""
        ligand, _ = synthesize_ligand(swapped.molecules[0], disubstituted, generator="fragment-swap")
        assert ligand.mol.GetNumConformers() == 1
        assert not any(atom.GetNumImplicitHs() for atom in ligand.mol.GetAtoms())

    def test_it_is_reproducible(self, swapped, disubstituted):
        first, _ = synthesize_ligand(swapped.molecules[0], disubstituted, generator="fragment-swap")
        second, _ = synthesize_ligand(swapped.molecules[0], disubstituted, generator="fragment-swap")
        assert first.name == second.name
        assert np.allclose(coords(first.mol), coords(second.mol), atol=1e-6)

    def test_a_scrambled_parent_is_rejected_by_the_gate(self, swapped, disubstituted):
        """The load-bearing negative: a parent in a bad frame must not yield an accepted pose."""
        broken = {**disubstituted, "di_FCl": scramble_frame(disubstituted["di_FCl"], seed=7)}
        ligand, result = synthesize_ligand(swapped.molecules[0], broken, generator="fragment-swap")
        assert ligand is None
        assert result.rejection in {PoseRejection.POSE_RMSD_EXCEEDED.value, PoseRejection.EMBED_FAILED.value}

    def test_the_gate_is_half_the_feasibility_threshold(self, swapped, disubstituted):
        _, generous = synthesize_ligand(swapped.molecules[0], disubstituted, generator="fragment-swap")
        ligand, strict = synthesize_ligand(
            swapped.molecules[0],
            disubstituted,
            generator="fragment-swap",
            softcore=SoftcorePolicy(core_rmsd_threshold=2.0),
            options=IntermediateOptions(pose_rmsd_factor=generous.rmsd / 4.0),
        )
        assert ligand is None
        assert strict.rejection == PoseRejection.POSE_RMSD_EXCEEDED.value
        assert strict.rmsd == pytest.approx(generous.rmsd, rel=1e-3)


class TestPoseFailuresAreData:
    def test_an_embed_failure_yields_a_rejection_not_an_exception(self, monkeypatch, swapped, disubstituted):
        monkeypatch.setattr(AllChem, "EmbedMolecule", lambda *args, **kwargs: -1)
        ligand, result = synthesize_ligand(swapped.molecules[0], disubstituted, generator="fragment-swap")
        assert ligand is None
        assert result.rejection == PoseRejection.EMBED_FAILED.value
        assert result.mol is None

    def test_an_embed_that_raises_is_also_data(self, monkeypatch, swapped, disubstituted):
        def explode(*args, **kwargs):
            raise RuntimeError("Invariant Violation")

        monkeypatch.setattr(AllChem, "EmbedMolecule", explode)
        _, result = synthesize_ligand(swapped.molecules[0], disubstituted, generator="fragment-swap")
        assert result.rejection == PoseRejection.EMBED_FAILED.value

    def test_a_charge_change_is_refused(self, benzene):
        """One hard edge becoming two harder ones is not an improvement."""
        charged = make_ligand("CC(=O)[O-]", "acetate")
        result = pose_intermediate(Chem.RemoveHs(benzene.mol), [PoseDonor("acetate", charged.mol, {})])
        assert result.rejection == PoseRejection.CHARGE_MISMATCH.value

    def test_undefined_stereo_is_refused(self, benzene):
        ambiguous = Chem.MolFromSmiles("CC(N)C(=O)O")
        result = pose_intermediate(ambiguous, [PoseDonor("benzene", benzene.mol, {})])
        assert result.rejection == PoseRejection.STEREO_UNDEFINED.value

    def test_an_unsanitizable_molecule_is_refused(self, benzene):
        broken = Chem.MolFromSmiles("c1cc1", sanitize=False)
        result = pose_intermediate(broken, [PoseDonor("benzene", benzene.mol, {})])
        assert result.rejection == PoseRejection.INVALID_MOLECULE.value

    def test_no_correspondence_is_refused(self, benzene):
        result = pose_intermediate(Chem.RemoveHs(benzene.mol), [PoseDonor("benzene", benzene.mol, {})])
        assert result.rejection == PoseRejection.NO_DONOR_ATOMS.value

    def test_a_forcefield_failure_is_refused(self, monkeypatch, swapped, disubstituted):
        monkeypatch.setattr(AllChem, "MMFFGetMoleculeProperties", lambda *args, **kwargs: None)
        monkeypatch.setattr(AllChem, "UFFGetMoleculeForceField", lambda *args, **kwargs: None)
        ligand, result = synthesize_ligand(swapped.molecules[0], disubstituted, generator="fragment-swap")
        assert ligand is None
        assert result.rejection == PoseRejection.FORCEFIELD_FAILED.value

    def test_nothing_raises_from_synthesize(self, swapped, disubstituted):
        for molecule in swapped.molecules:
            ligand, result = synthesize_ligand(molecule, disubstituted, generator="fragment-swap")
            assert (ligand is None) == (result.rejection is not None)


class TestMCSFallback:
    def test_a_missing_atom_map_is_recovered_and_reported(self, benzamides):
        """Weaker, and named as such so the provenance shows it."""
        parent = benzamides["bza_Cl"]
        bare = Chem.RemoveHs(Chem.Mol(parent.mol))
        bare.RemoveAllConformers()
        result = pose_intermediate(bare, [PoseDonor(parent.name, parent.mol, None)])
        assert result.posed, result.rejection
        assert result.method == "mcs_fallback"
        assert result.rmsd < 1.0

    def test_the_recovered_pose_is_in_the_parents_frame(self, benzamides):
        parent = benzamides["bza_Cl"]
        bare = Chem.RemoveHs(Chem.Mol(parent.mol))
        bare.RemoveAllConformers()
        result = pose_intermediate(bare, [PoseDonor(parent.name, parent.mol, None)])
        centroid = coords(result.mol).mean(axis=0)
        assert np.linalg.norm(centroid - coords(parent.mol).mean(axis=0)) < 1.0


def test_a_coposed_series_is_the_only_input(disubstituted):
    """Guards the fixture itself: an independently embedded pair would make every test above vacuous."""
    reference = make_coposed({"di_FCl": "Fc1ccc(Cl)cc1C(=O)N"}, "c1ccccc1C(=O)N")["di_FCl"]
    assert np.allclose(coords(reference.mol), coords(disubstituted["di_FCl"].mol), atol=1e-6)
