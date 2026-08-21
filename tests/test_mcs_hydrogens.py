"""The MCS runs on heavy atoms; hydrogens are re-paired afterwards.

Suppressing hydrogens for the search is a large speedup that is only correct because the
mapper puts them back. These tests pin the putting-back, which is the part that fails
quietly: an unpaired hydrogen is soft-core, and a molecule's worth of them arrive as
scattered one-atom regions that drive the repair to demote the entire core.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from rbfenetmap.core.mcs import _suppress_hydrogens, mcs_query
from rbfenetmap.core.options import MappingOptions
from rbfenetmap.plugins.mappers import create_mapper
from rbfenetmap.plugins.mappers.mcss_mapper import _pair_hydrogens


class TestSuppression:
    """The search runs on the heavy-atom graph."""

    def test_removes_hydrogens(self, benzamides):
        """The suppressed copy has exactly the heavy atoms."""
        ligand = benzamides["bza_Me"]
        assert _suppress_hydrogens(ligand.mol).GetNumAtoms() == ligand.n_heavy

    def test_falls_back_when_removal_fails(self, monkeypatch, benzamides):
        """A molecule RDKit will not strip is searched as-is rather than raising."""

        def boom(_mol):
            raise ValueError("cannot remove")

        monkeypatch.setattr(Chem, "RemoveHs", boom)
        ligand = benzamides["bza_Me"]
        assert _suppress_hydrogens(ligand.mol) is ligand.mol

    def test_pattern_still_matches_the_full_molecule(self, benzamides):
        """No index from the suppressed copy escapes: the SMARTS matches the real mol."""
        source, target = benzamides["bza_H"], benzamides["bza_Me"]
        pattern = mcs_query(source.mol, target.mol, MappingOptions())
        assert pattern is not None
        assert source.mol.GetSubstructMatches(pattern)
        assert target.mol.GetSubstructMatches(pattern)

    def test_pattern_carries_no_hydrogens(self, benzamides):
        """The MCS is over heavy atoms, so a heavy atom can never pair with a hydrogen."""
        pattern = mcs_query(benzamides["bza_H"].mol, benzamides["bza_Me"].mol, MappingOptions())
        assert all(atom.GetAtomicNum() != 1 for atom in pattern.GetAtoms())


class TestHydrogenPairing:
    """Hydrogens follow their parents back into the core."""

    def test_hydrogens_are_paired_with_their_parents(self, benzamides):
        """A mapped heavy pair brings its hydrogens with it."""
        source, target = benzamides["bza_H"], benzamides["bza_F"]
        mapping = create_mapper("mcss-e2").map_pair(source, target, MappingOptions())
        heavy = set(source.heavy_indices)
        paired_h = [a for a in mapping.cc1 if a not in heavy]
        assert paired_h, "no hydrogen reached the common core"
        # every cored hydrogen's parent must be cored too, and map to its partner's parent
        parents_1 = {h: n.GetIdx() for h in paired_h for n in source.mol.GetAtomWithIdx(h).GetNeighbors()}
        for hydrogen in paired_h:
            assert parents_1[hydrogen] in mapping.cc1

    def test_pairing_is_deterministic(self, benzamides):
        """Interchangeable hydrogens are assigned in a reproducible order."""
        source, target = benzamides["bza_Me"], benzamides["bza_CF3"]
        options = MappingOptions()
        first = create_mapper("mcss-e2").map_pair(source, target, options)
        second = create_mapper("mcss-e2").map_pair(source, target, options)
        assert first.forward == second.forward

    def test_uneven_hydrogen_counts_leave_a_remainder(self, benzamides):
        """``R-CH3 -> R-CF3``: zip pairs what it can, the rest stays soft-core."""
        source, target = benzamides["bza_Me"], benzamides["bza_CF3"]
        core = {}
        # the methyl carbon of bza_Me against the CF3 carbon of bza_CF3
        methyl = next(
            a.GetIdx()
            for a in source.mol.GetAtoms()
            if a.GetSymbol() == "C" and sum(1 for n in a.GetNeighbors() if n.GetAtomicNum() == 1) == 3
        )
        fluoro = next(
            a.GetIdx()
            for a in target.mol.GetAtoms()
            if a.GetSymbol() == "C" and sum(1 for n in a.GetNeighbors() if n.GetSymbol() == "F") == 3
        )
        core[methyl] = fluoro
        extended = _pair_hydrogens(source, target, core)
        # the CF3 carbon has no hydrogens, so none of the methyl's three can pair
        assert extended == core

    def test_empty_core_extends_to_nothing(self, benzamides):
        """No heavy pairs means no hydrogen pairs, not an error."""
        assert _pair_hydrogens(benzamides["bza_H"], benzamides["bza_F"], {}) == {}


class TestEndToEnd:
    """The suppression must not change what the pipeline concludes."""

    @pytest.mark.parametrize("mapper", ["mcss", "mcss-e", "mcss-e2"])
    def test_congeneric_pair_still_maps_with_a_small_softcore(self, benzamides, mapper):
        """The everyday case: R-H -> R-CH3 keeps a large core and a one-atom soft-core.

        The regression this guards is total rather than subtle. Without hydrogen
        re-pairing every hydrogen is soft-core, so this assertion fails by a wide margin
        rather than by a rounding error.
        """
        source, target = benzamides["bza_H"], benzamides["bza_Me"]
        mapping = create_mapper(mapper).map_pair(source, target, MappingOptions())
        heavy = set(source.heavy_indices)
        assert len(heavy & set(mapping.cc1)) >= 7, "the benzamide scaffold should survive as core"
        assert mapping.n_softcore_1 <= 2, "R-H side should be at most the one hydrogen"
