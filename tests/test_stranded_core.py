"""A core atom the soft-core strands is demoted, not grounds for rejection.

The invariant is that a soft-core region hangs off the common core by exactly one bond.
A mapper can break it without producing anything chemically awkward: keep a terminal
methyl in the core while the atom joining it to the rest of the core goes soft, and the
soft-core becomes a bridge, ``core -- soft-core -- core``. The repair used to see two
attachment bonds and reject. Absorbing the stranded fragment restores the invariant and
costs a handful of atoms.

Driven through :func:`build_candidate` on co-posed anilides rather than through a
hand-authored contract, because the point of the fix is what a *real* MCS hands over. The
contract test below covers the shape the mapper does not happen to produce.
"""

from __future__ import annotations

import pytest

from rbfenetmap.core.models import RejectionReason
from rbfenetmap.core.options import MappingOptions, NetworkOptions, SoftcorePolicy
from rbfenetmap.core.pipeline import build_candidate
from rbfenetmap.core.softcore import softcore_attachment_edges
from rbfenetmap.core.molgraph import mol_to_graph
from rbfenetmap.plugins.mappers.mcss_mapper import MCSSExtended2Mapper
from rbfenetmap.plugins.scorers.linear_scorer import LinearScorer

#: The pairs whose MCS strands a methyl. ``an_prop~an_piv`` is the awkward one: it is not
#: stranded when the mapper hands it over, only after the repair bridges its two regions,
#: so a fix applied once before the loop would miss it.
STRANDED_PAIRS = [("an_piv", "an_carb"), ("an_ibu", "an_carb"), ("an_prop", "an_carb"), ("an_prop", "an_piv")]

#: Pairs whose mapping was already fine. They pin that the new rule is inert on a good
#: edge -- a demotion pass that fired on these would quietly grow every soft-core.
CLEAN_PAIRS = [("an_ibu", "an_piv"), ("an_prop", "an_ibu")]


def candidate(ligands, a, b, policy=None):
    """Map, repair, and score one pair with the real MCS mapper."""
    options = NetworkOptions(softcore=policy) if policy else NetworkOptions()
    return build_candidate(ligands[a], ligands[b], MCSSExtended2Mapper(), LinearScorer(), MappingOptions(), options)


class TestStrandedCoreIsAbsorbed:
    @pytest.mark.parametrize("a,b", STRANDED_PAIRS)
    def test_the_pair_is_feasible(self, anilides, a, b):
        assert candidate(anilides, a, b).score.rejections == ()

    @pytest.mark.parametrize("a,b", STRANDED_PAIRS)
    def test_each_side_is_singly_attached(self, anilides, a, b):
        """The invariant itself, checked directly rather than via the absence of a reason."""
        edge = candidate(anilides, a, b)
        for ligand, softcore in ((anilides[a], edge.mapping.sc1), (anilides[b], edge.mapping.sc2)):
            if not softcore:
                continue
            assert len(softcore_attachment_edges(mol_to_graph(ligand.mol), softcore)) == 1

    def test_the_stranded_methyl_ends_up_soft_core(self, anilides):
        """Not merely feasible: the specific atom that was stranded moved.

        ``an_piv~an_carb`` is pivaloyl against a methyl carbamate. Every heavy atom of the
        t-butyl has to be soft-core once the quaternary carbon is, because the methyl the
        MCS kept is reachable only through it.
        """
        edge = candidate(anilides, "an_piv", "an_carb")
        mol = anilides["an_piv"].mol
        heavy = {i for i in edge.mapping.sc1 if mol.GetAtomWithIdx(i).GetAtomicNum() > 1}
        assert len(heavy) == 4, f"expected the whole t-butyl soft-core, got {sorted(heavy)}"

    def test_the_repair_says_what_it_absorbed(self, anilides):
        edge = candidate(anilides, "an_piv", "an_carb")
        assert any("stranded" in line for line in edge.repair.trace), edge.repair.trace


class TestGoodEdgesAreUntouched:
    @pytest.mark.parametrize("a,b", CLEAN_PAIRS)
    def test_still_feasible(self, anilides, a, b):
        assert candidate(anilides, a, b).score.rejections == ()

    @pytest.mark.parametrize("a,b", CLEAN_PAIRS)
    def test_the_soft_core_did_not_grow(self, anilides, a, b):
        """A pair with nothing stranded keeps the small soft-core it already had."""
        edge = candidate(anilides, a, b)
        mol_a, mol_b = anilides[a].mol, anilides[b].mol
        heavy_1 = {i for i in edge.mapping.sc1 if mol_a.GetAtomWithIdx(i).GetAtomicNum() > 1}
        heavy_2 = {i for i in edge.mapping.sc2 if mol_b.GetAtomWithIdx(i).GetAtomicNum() > 1}
        assert len(heavy_1) <= 1 and len(heavy_2) <= 4

    def test_no_absorption_is_traced(self, anilides):
        edge = candidate(anilides, "an_ibu", "an_piv")
        assert not any("stranded" in line for line in edge.repair.trace), edge.repair.trace


class TestBudgetStillGoverns:
    """Absorbing a stranded fragment must not become a way around the size limits."""

    def test_a_tight_budget_rejects_on_size_not_attachment(self, anilides):
        """The better diagnosis.

        With the soft-core capped below what the absorption needs, the honest answer is
        that the edge is too large -- not that it has two attachment bonds, which sends
        the reader after a knob that cannot help.
        """
        edge = candidate(anilides, "an_piv", "an_carb", policy=SoftcorePolicy(max_softcore_atoms=2))
        assert RejectionReason.SOFTCORE_TOO_LARGE in edge.score.rejections
        assert RejectionReason.SOFTCORE_MULTIPLE_ATTACHMENTS not in edge.score.rejections
