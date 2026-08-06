"""Invariant tests for the core data model.

Every malformed mapping must be rejected at construction with a message naming the
offending indices, because a mapping that escapes validation fails much later in a form
that looks like a chemistry problem rather than a bookkeeping one.
"""

from __future__ import annotations

import math

import pytest

from rbfenetmap.core.models import (
    AtomMapping,
    EdgeScore,
    Ligand,
    Network,
    RejectionReason,
    Transformation,
    edge_key,
    parse_edge_key,
)

from .conftest import make_transformation


def _valid_kwargs(**overrides):
    """Return a valid three-atom mapping, with *overrides* applied."""
    kwargs = {"cc1": (0, 1), "cc2": (0, 1), "sc1": (2,), "sc2": (2,), "n_atoms_1": 3, "n_atoms_2": 3}
    kwargs.update(overrides)
    return kwargs


class TestAtomMappingInvariants:
    """Each case is a distinct way a mapper can produce a broken partition."""

    def test_valid_mapping_constructs(self):
        mapping = AtomMapping(**_valid_kwargs())
        assert mapping.n_common_core == 2
        assert mapping.forward == {0: 0, 1: 1}
        assert mapping.reverse == {0: 0, 1: 1}

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"cc1": (0, 0), "sc1": (1, 2), "cc2": (0, 1), "sc2": (2,)}, "duplicate atom indices"),
            ({"cc2": (0, 0), "sc2": (1, 2)}, "duplicate atom indices"),
            ({"cc1": (0, 9), "sc1": (1, 2)}, "outside range"),
            ({"sc1": (1, 2)}, "in both the common core and the soft-core"),
            ({"sc1": ()}, "in neither the common core nor the soft-core"),
            ({"cc1": (0,), "sc1": (1, 2)}, "Common core sizes disagree"),
        ],
    )
    def test_malformed_mapping_is_rejected(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            AtomMapping(**_valid_kwargs(**overrides))

    def test_non_injective_correspondence_is_rejected(self):
        # This is also Amber's linear-scaling constraint, caught here rather than in pmemd.
        with pytest.raises(ValueError, match="injective"):
            AtomMapping(cc1=(0, 1), cc2=(0, 0), sc1=(2,), sc2=(1, 2), n_atoms_1=3, n_atoms_2=3)

    def test_contract_round_trips(self):
        mapping = AtomMapping(**_valid_kwargs())
        rebuilt = AtomMapping.from_contract(mapping.to_contract(), n_atoms_1=3, n_atoms_2=3, method=mapping.method)
        assert rebuilt == mapping

    def test_contract_preserves_pairing_not_sort_order(self):
        # cc2 is paired positionally; re-sorting it independently would scramble the map.
        contract = {"cc1": [0, 1], "cc2": [1, 0], "sc1": [2], "sc2": [2]}
        mapping = AtomMapping.from_contract(contract, n_atoms_1=3, n_atoms_2=3)
        assert mapping.forward == {0: 1, 1: 0}

    def test_missing_contract_key_is_named(self):
        with pytest.raises(ValueError, match=r"missing key\(s\) \['sc2'\]"):
            AtomMapping.from_contract({"cc1": [], "cc2": [], "sc1": []}, n_atoms_1=0, n_atoms_2=0)

    def test_from_core_pairs_infers_softcore(self):
        mapping = AtomMapping.from_core_pairs({0: 1}, n_atoms_1=3, n_atoms_2=3)
        assert mapping.sc1 == (1, 2)
        assert mapping.sc2 == (0, 2)

    def test_swapped_preserves_correspondence(self):
        mapping = AtomMapping.from_core_pairs({0: 2, 1: 0}, n_atoms_1=3, n_atoms_2=3)
        swapped = mapping.swapped()
        assert swapped.forward == {2: 0, 0: 1}
        assert swapped.swapped() == mapping


class TestLigand:
    def test_rejects_implicit_hydrogens(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC")
        AllChem.EmbedMolecule(mol, randomSeed=1)
        with pytest.raises(ValueError, match="implicit hydrogens"):
            Ligand.from_mol(mol, "ethane")

    def test_rejects_name_containing_edge_separator(self, benzene):
        with pytest.raises(ValueError, match="Invalid ligand name"):
            Ligand(name="a~b", mol=benzene.mol, charge=0)

    def test_rejects_missing_conformer(self):
        from rdkit import Chem

        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        with pytest.raises(ValueError, match="conformers"):
            Ligand.from_mol(mol, "methane")

    def test_atom_names_are_unique(self, benzene):
        names = benzene.atom_names
        assert len(set(names)) == len(names)

    def test_heavy_indices_exclude_hydrogens(self, benzene):
        assert benzene.n_heavy == 6
        assert benzene.n_atoms == 12


class TestEdgeScore:
    def test_feasible_score_cannot_carry_rejections(self):
        with pytest.raises(ValueError, match="feasible but carries rejections"):
            EdgeScore(total=1.0, feasible=True, rejections=(RejectionReason.CORE_TOO_SMALL,))

    def test_infeasible_score_must_give_a_reason(self):
        with pytest.raises(ValueError, match="records no RejectionReason"):
            EdgeScore(total=math.inf, feasible=False)

    def test_feasible_score_must_be_finite(self):
        with pytest.raises(ValueError, match="not finite"):
            EdgeScore(total=math.inf, feasible=True)

    def test_rejected_helper_requires_a_reason(self):
        with pytest.raises(ValueError, match="at least one RejectionReason"):
            EdgeScore.rejected()


class TestEdgeKeys:
    def test_round_trip(self):
        assert parse_edge_key(edge_key("a", "b")) == ("a", "b")

    @pytest.mark.parametrize("bad", ["ab", "a~b~c", "~b", "a~"])
    def test_malformed_keys_rejected(self, bad):
        with pytest.raises(ValueError, match="Malformed edge key"):
            parse_edge_key(bad)


class TestTransformationAndNetwork:
    def test_self_loop_rejected(self):
        with pytest.raises(ValueError, match="self-loops are not edges"):
            make_transformation("a", "a")

    def test_unordered_key_is_direction_independent(self):
        forward = make_transformation("b", "a")
        assert forward.unordered_key == ("a", "b")
        assert forward.reversed().unordered_key == ("a", "b")

    def test_reversed_notes_the_orientation_flip_in_the_trace(self):
        from rbfenetmap.core.models import SoftcoreRepair

        edge = make_transformation("a", "b")
        edge = Transformation(
            source=edge.source,
            target=edge.target,
            mapping=edge.mapping,
            repair=SoftcoreRepair(trace=("iter 1 side 1: demoted 3",)),
            score=edge.score,
        )
        flipped = edge.reversed()
        assert "orientation flipped" in flipped.repair.trace[0]
        assert flipped.repair.trace[1:] == edge.repair.trace

    def test_validate_detects_duplicate_pair(self, benzamides):
        network = Network(
            ligands=benzamides, edges=(make_transformation("bza_H", "bza_F"), make_transformation("bza_F", "bza_H"))
        )
        with pytest.raises(ValueError, match="duplicates an already-selected pair"):
            network.validate(require_connected=False)

    def test_validate_detects_unknown_endpoint(self, benzamides):
        network = Network(ligands=benzamides, edges=(make_transformation("bza_H", "nope"),))
        with pytest.raises(ValueError, match="unknown ligand"):
            network.validate(require_connected=False)

    def test_validate_detects_disconnection(self, benzamides):
        network = Network(ligands=benzamides, edges=(make_transformation("bza_H", "bza_F"),))
        with pytest.raises(ValueError, match="disconnected"):
            network.validate(require_connected=True)
