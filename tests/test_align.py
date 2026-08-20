"""Tests for bringing a ligand set into a common frame.

The central fixture is ``scrambled_benzamides``: co-posed ligands whose conformers are
identical and whose frames are not. That separation is what makes the assertions sharp --
perfect alignment is achievable, so a residual RMSD above the noise floor is a defect in
the aligner and nothing else.

Two tests here are really regression guards for the *rest* of the package rather than for
this module. ``test_scrambled_ligands_cannot_be_planned`` pins the behaviour that motivated
the feature, and must fail loudly if the geometry gate ever stops firing. And
``test_atom_names_survive_alignment`` guards the molecule copy: Tripos atom names feed the
Amber mask builder, and losing them would corrupt exported topologies silently.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from rbfenetmap.core.align import align_ligands, choose_reference
from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.models import Ligand, RejectionReason
from rbfenetmap.core.options import AlignmentOptions
from rbfenetmap.core.pipeline import build_network

from .conftest import make_coposed, make_ligand, scramble_frame

SCAFFOLD = "c1ccccc1C(=O)N"


def _positions(ligand: Ligand) -> np.ndarray:
    """The ligand's conformer coordinates."""
    return np.asarray(ligand.mol.GetConformer().GetPositions(), dtype=float)


def _by_name(ligands) -> dict[str, Ligand]:
    """Index an aligned result's ligands by name."""
    return {ligand.name: ligand for ligand in ligands}


def _scaffold_rmsd(a: Ligand, b: Ligand) -> float:
    """In-place RMSD over the shared benzamide scaffold of two aligned ligands."""
    from rdkit import Chem

    pattern = Chem.MolFromSmiles(SCAFFOLD)
    match_a, match_b = a.mol.GetSubstructMatch(pattern), b.mol.GetSubstructMatch(pattern)
    assert match_a and match_b
    delta = _positions(a)[list(match_a)] - _positions(b)[list(match_b)]
    return float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))


class TestChooseReference:
    def test_picks_the_largest_ligand(self, benzamides):
        assert choose_reference(benzamides) == "bza_CF3"

    def test_is_independent_of_input_order(self, benzamides):
        reversed_order = dict(reversed(list(benzamides.items())))
        assert choose_reference(reversed_order) == choose_reference(benzamides)

    def test_honours_an_explicit_choice(self, benzamides):
        assert choose_reference(benzamides, "bza_H") == "bza_H"

    def test_unknown_reference_names_the_loaded_ligands(self, benzamides):
        with pytest.raises(ValueError, match="Unknown alignment reference 'nope'"):
            choose_reference(benzamides, "nope")
        with pytest.raises(ValueError, match="bza_CF3"):
            choose_reference(benzamides, "nope")

    def test_empty_set_raises(self):
        with pytest.raises(ValueError, match="empty ligand set"):
            choose_reference({})


class TestAlignLigands:
    def test_recovers_the_common_frame(self, scrambled_benzamides):
        aligned = _by_name(align_ligands(scrambled_benzamides).ligands)
        names = sorted(aligned)
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                assert _scaffold_rmsd(aligned[first], aligned[second]) < 0.2

    def test_scrambled_ligands_cannot_be_planned(self, scrambled_benzamides):
        # The reported bug, pinned. Every candidate is rejected for geometry, so the
        # planner has nothing to connect.
        with pytest.raises(NetworkPlanError, match=RejectionReason.CORE_GEOMETRY_MISMATCH.value):
            build_network(scrambled_benzamides, mapper="mcss-e2")

    @pytest.mark.integration
    def test_alignment_makes_the_network_planable(self, scrambled_benzamides):
        aligned = _by_name(align_ligands(scrambled_benzamides).ligands)
        network = build_network(aligned, mapper="mcss-e2")
        network.validate()
        assert network.edges
        assert not any(RejectionReason.CORE_GEOMETRY_MISMATCH in e.score.rejections for e in network.edges)

    def test_input_ligands_are_not_mutated(self, scrambled_benzamides):
        before = {name: _positions(ligand).copy() for name, ligand in scrambled_benzamides.items()}
        align_ligands(scrambled_benzamides)
        for name, coords in before.items():
            assert np.array_equal(_positions(scrambled_benzamides[name]), coords)

    def test_the_reference_is_not_moved(self, scrambled_benzamides):
        result = align_ligands(scrambled_benzamides)
        aligned = _by_name(result.ligands)[result.reference]
        assert np.array_equal(_positions(aligned), _positions(scrambled_benzamides[result.reference]))

    def test_atom_names_survive_alignment(self, scrambled_benzamides):
        result = align_ligands(scrambled_benzamides)
        for aligned in result.ligands:
            assert aligned.atom_names == scrambled_benzamides[aligned.name].atom_names

    def test_charge_and_source_survive_alignment(self, scrambled_benzamides):
        for aligned in align_ligands(scrambled_benzamides).ligands:
            original = scrambled_benzamides[aligned.name]
            assert aligned.charge == original.charge
            assert aligned.source == original.source

    def test_every_ligand_records_its_alignment(self, scrambled_benzamides):
        result = align_ligands(scrambled_benzamides)
        for aligned in result.ligands:
            record = dict(aligned.metadata)["alignment"]
            assert record["ok"] is True
            assert record["method"] in ("mcs", "reference")
        assert {r.name for r in result.records} == set(scrambled_benzamides)

    def test_existing_metadata_is_preserved(self, benzamides):
        tagged = Ligand.from_mol(benzamides["bza_H"].mol, "bza_H", series="acme")
        result = align_ligands([tagged, benzamides["bza_CF3"]])
        aligned = _by_name(result.ligands)["bza_H"]
        assert aligned.metadata["series"] == "acme"
        assert "alignment" in aligned.metadata

    def test_is_deterministic(self, scrambled_benzamides):
        first = align_ligands(scrambled_benzamides)
        second = align_ligands(scrambled_benzamides)
        assert first.reference == second.reference
        assert [r.reference for r in first.records] == [r.reference for r in second.records]
        for a, b in zip(first.ligands, second.ligands):
            assert np.array_equal(_positions(a), _positions(b))

    def test_result_order_matches_input_order(self, scrambled_benzamides):
        result = align_ligands(scrambled_benzamides)
        assert [ligand.name for ligand in result.ligands] == list(scrambled_benzamides)
        assert [record.name for record in result.records] == list(scrambled_benzamides)

    def test_accepts_a_sequence_as_well_as_a_mapping(self, scrambled_benzamides):
        as_list = align_ligands(list(scrambled_benzamides.values()))
        as_mapping = align_ligands(scrambled_benzamides)
        assert as_list.reference == as_mapping.reference

    def test_median_rmsd_excludes_the_reference(self, scrambled_benzamides):
        result = align_ligands(scrambled_benzamides)
        moved = [r.rmsd for r in result.records if r.reference is not None]
        assert result.median_rmsd == pytest.approx(float(np.median(moved)))

    def test_progressive_alignment_uses_the_nearest_aligned_parent(self):
        # C shares a long alkyl chain with B and almost nothing with the reference A, so a
        # star to A would be the wrong tree. Its record must name B.
        ligands = make_coposed(
            {
                "lig_a": "c1ccc(cc1)C(=O)Nc1ccccc1",
                "lig_b": "c1ccc(cc1)C(=O)NCCCCCC",
                "lig_c": "c1ccc(cc1)C(=O)NCCCCCCC",
            },
            "c1ccc(cc1)C(=O)N",
        )
        scrambled = {n: scramble_frame(lig, seed=i + 1) for i, (n, lig) in enumerate(sorted(ligands.items()))}
        result = align_ligands(scrambled, options=AlignmentOptions(reference="lig_a"))
        records = {r.name: r for r in result.records}
        assert records["lig_c"].reference == "lig_b"

    def test_an_unalignable_ligand_is_left_in_place_and_reported(self, benzamides, caplog):
        stranger = make_ligand("[He]", "helium")
        scrambled = {n: scramble_frame(lig, seed=i + 1) for i, (n, lig) in enumerate(sorted(benzamides.items()))}
        scrambled["helium"] = scramble_frame(stranger, seed=99)
        before = _positions(scrambled["helium"]).copy()

        with caplog.at_level(logging.WARNING):
            result = align_ligands(scrambled)

        record = next(r for r in result.records if r.name == "helium")
        assert record.ok is False
        assert record.method == "none"
        assert result.failures == (record,)
        assert np.array_equal(_positions(_by_name(result.ligands)["helium"]), before)
        assert "Could not align helium" in caplog.text

    def test_duplicate_names_are_refused(self, benzamides):
        duplicate = Ligand.from_mol(benzamides["bza_H"].mol, "bza_F")
        with pytest.raises(ValueError, match="names must be unique"):
            align_ligands([benzamides["bza_F"], duplicate])

    def test_a_single_ligand_is_its_own_reference(self, benzamides):
        result = align_ligands([benzamides["bza_H"]])
        assert result.reference == "bza_H"
        assert result.records[0].method == "reference"
        assert result.median_rmsd == 0.0


class TestO3AAlignment:
    def test_aligns_a_set_with_no_shared_substructure(self):
        ligands = {"naph": make_ligand("c1ccc2ccccc2c1", "naph"), "quin": make_ligand("c1ccc2ncccc2c1", "quin")}
        scrambled = {n: scramble_frame(lig, seed=i + 5) for i, (n, lig) in enumerate(sorted(ligands.items()))}
        result = align_ligands(scrambled, options=AlignmentOptions(method="o3a"))
        assert not result.failures
        assert all(r.method in ("o3a", "reference") for r in result.records)

    def test_falls_back_to_crippen_when_mmff_typing_fails(self, caplog):
        # Boron is outside MMFF94's coverage, so GetO3A raises where GetCrippenO3A copes.
        ligands = [make_ligand("OB(O)c1ccccc1", "boronic"), make_ligand("OB(O)c1ccc(C)cc1", "boronic_me")]
        with caplog.at_level(logging.WARNING):
            result = align_ligands(ligands, options=AlignmentOptions(method="o3a"))
        assert "falling back to the Crippen variant" in caplog.text
        assert not result.failures
