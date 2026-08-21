"""Scorer tests, driven entirely by hand-written descriptor dictionaries.

Not one of these tests imports RDKit or constructs a molecule. That is the payoff of
giving :class:`~rbfenetmap.core.meta.scorers.AbstractScorer` a ``Mapping[str, float]``
and nothing else: scoring behaviour is checkable in isolation from chemistry.
"""

from __future__ import annotations

import math

import pytest

from rbfenetmap.core.models import RejectionReason
from rbfenetmap.plugins.scorers import available_scorers, create_scorer
from rbfenetmap.plugins.scorers.linear_scorer import DEFAULT_SCORE_WEIGHTS, TERM_DEFINITIONS


def descriptors(**overrides) -> dict[str, float]:
    """A neutral descriptor set: a small, clean, single-atom perturbation."""
    base = {
        "n_softcore_max_heavy": 0.0,
        "softcore_asymmetry": 0.0,
        "heavy_atom_delta": 0.0,
        "charge_delta": 0.0,
        "ring_delta": 0.0,
        "n_ring_atoms_in_softcore": 0.0,
        "mcs_fraction": 1.0,
        "core_rmsd": 0.0,
        "rotatable_delta": 0.0,
        "n_demoted_atoms": 0.0,
        "logp_delta": 0.0,
    }
    base.update(overrides)
    return base


class TestLinearScorer:
    def test_perfect_edge_costs_nothing(self):
        score = create_scorer("linear").score_edge(descriptors(), rejections=[])
        assert score.feasible
        assert score.total == pytest.approx(0.0)

    def test_contributions_sum_to_the_total(self):
        score = create_scorer("linear").score_edge(
            descriptors(n_softcore_max_heavy=6, charge_delta=1, core_rmsd=0.4, mcs_fraction=0.7), rejections=[]
        )
        assert sum(score.contributions.values()) == pytest.approx(score.total)

    def test_cost_increases_with_softcore_size(self):
        scorer = create_scorer("linear")
        small = scorer.score_edge(descriptors(n_softcore_max_heavy=2), rejections=[]).total
        large = scorer.score_edge(descriptors(n_softcore_max_heavy=10), rejections=[]).total
        assert large > small

    def test_charge_change_dominates_a_small_softcore(self):
        scorer = create_scorer("linear")
        charged = scorer.score_edge(descriptors(charge_delta=1), rejections=[]).total
        bulky = scorer.score_edge(descriptors(n_softcore_max_heavy=8), rejections=[]).total
        assert charged > bulky, "a unit charge change should outweigh a typical soft-core"

    def test_mcs_deficit_is_inverted(self):
        scorer = create_scorer("linear")
        full = scorer.score_edge(descriptors(mcs_fraction=1.0), rejections=[]).total
        partial = scorer.score_edge(descriptors(mcs_fraction=0.5), rejections=[]).total
        assert partial > full

    def test_caps_bound_a_single_pathological_descriptor(self):
        scorer = create_scorer("linear")
        big = scorer.score_edge(descriptors(core_rmsd=1e6), rejections=[]).total
        capped = DEFAULT_SCORE_WEIGHTS["core_rmsd"] * TERM_DEFINITIONS["core_rmsd"][2]
        assert big == pytest.approx(capped)

    def test_unknown_weight_is_rejected(self):
        # A silently ignored typo would make a tuning run look effective while using the
        # defaults, which is the worst possible failure for this knob.
        with pytest.raises(ValueError, match="Unknown scoring term"):
            create_scorer("linear", weights={"softcore_atom": 2.0})

    def test_weights_are_applied(self):
        heavy = create_scorer("linear", weights={"softcore_atoms": 10.0})
        light = create_scorer("linear", weights={"softcore_atoms": 0.1})
        data = descriptors(n_softcore_max_heavy=8)
        assert heavy.score_edge(data, rejections=[]).total > light.score_edge(data, rejections=[]).total

    def test_rejections_propagate_without_being_invented(self):
        score = create_scorer("linear").score_edge(descriptors(), rejections=[RejectionReason.SOFTCORE_TOO_LARGE])
        assert not score.feasible
        assert score.rejections == (RejectionReason.SOFTCORE_TOO_LARGE,)
        assert math.isinf(score.total)


class TestLomapLikeScorer:
    def test_cost_is_non_negative_and_finite(self):
        score = create_scorer("lomaplike").score_edge(descriptors(n_softcore_max_heavy=5, ring_delta=1), rejections=[])
        assert score.feasible
        assert 0.0 <= score.total < math.inf

    def test_contributions_sum_to_the_total(self):
        score = create_scorer("lomaplike").score_edge(
            descriptors(n_softcore_max_heavy=4, charge_delta=1, core_rmsd=0.5), rejections=[]
        )
        assert sum(score.contributions.values()) == pytest.approx(score.total)

    def test_a_single_bad_factor_dominates(self):
        # Multiplicative composition: one hopeless factor should sink the edge regardless
        # of how good everything else is.
        scorer = create_scorer("lomaplike")
        clean = scorer.score_edge(descriptors(), rejections=[]).total
        charged = scorer.score_edge(descriptors(charge_delta=2), rejections=[]).total
        assert charged > clean * 10 or clean == pytest.approx(0.0) and charged > 1.0

    def test_hopeless_edge_stays_finite(self):
        score = create_scorer("lomaplike").score_edge(
            descriptors(n_softcore_max_heavy=500, charge_delta=5, ring_delta=5), rejections=[]
        )
        assert math.isfinite(score.total), "a bad-but-feasible edge must not look like a rejection"

    def test_unknown_parameter_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            create_scorer("lomaplike", parameters={"betta": 0.2})


class TestSoftcoreSizeScorer:
    def test_cost_is_the_softcore_size(self):
        score = create_scorer("softcore-size").score_edge(descriptors(n_softcore_max_heavy=7), rejections=[])
        assert score.total == pytest.approx(7.0)


class TestVarianceScorer:
    """NetBFE eq. 19, checked against numbers a reader can compute on paper.

    ``s = w0 + w1 sqrt(h) + w2 sqrt(H)`` with ``w = (1.0, 1.0, 0.5)``, ``h`` the larger
    soft-core heavy-atom count and ``H`` the larger total heavy-atom count. Every expected
    value below is written as that arithmetic rather than as a decimal, so a reader checks
    the formula rather than trusting a constant.
    """

    def test_a_zero_atom_edge_is_the_intercept(self):
        """The floor: an edge that transforms nothing still carries one run's noise."""
        score = create_scorer("variance").score_edge(
            descriptors(n_softcore_max_heavy=0.0, n_heavy_1=0.0, n_heavy_2=0.0), rejections=[]
        )
        assert score.total == pytest.approx(1.0)

    def test_it_matches_the_published_form(self):
        score = create_scorer("variance").score_edge(
            descriptors(n_softcore_max_heavy=4.0, n_heavy_1=16.0, n_heavy_2=9.0), rejections=[]
        )
        assert score.total == pytest.approx(1.0 + math.sqrt(4.0) + 0.5 * math.sqrt(16.0))

    def test_total_heavy_takes_the_larger_side(self):
        """``max(H_ij, H_ji)``: order of the two ligands must not change the prediction."""
        scorer = create_scorer("variance")
        forward = scorer.score_edge(descriptors(n_heavy_1=25.0, n_heavy_2=4.0), rejections=[])
        reverse = scorer.score_edge(descriptors(n_heavy_1=4.0, n_heavy_2=25.0), rejections=[])
        assert forward.total == pytest.approx(reverse.total)
        assert forward.total == pytest.approx(1.0 + 0.5 * math.sqrt(25.0))

    def test_the_contributions_name_the_three_terms(self):
        score = create_scorer("variance").score_edge(
            descriptors(n_softcore_max_heavy=9.0, n_heavy_1=36.0), rejections=[]
        )
        assert set(score.contributions) == {"intercept", "softcore_heavy", "total_heavy"}
        assert score.contributions["softcore_heavy"] == pytest.approx(3.0)
        assert score.contributions["total_heavy"] == pytest.approx(3.0)

    def test_it_grows_sublinearly_in_the_softcore(self):
        """Doubling the soft-core must not double the predicted noise."""
        scorer = create_scorer("variance")
        small = scorer.score_edge(descriptors(n_softcore_max_heavy=4.0), rejections=[]).total
        large = scorer.score_edge(descriptors(n_softcore_max_heavy=8.0), rejections=[]).total
        assert small < large < 2 * small

    def test_missing_descriptors_degrade_to_the_intercept(self):
        score = create_scorer("variance").score_edge({}, rejections=[])
        assert score.total == pytest.approx(1.0)

    def test_a_rejection_propagates_without_a_cost(self):
        score = create_scorer("variance").score_edge(
            descriptors(n_softcore_max_heavy=4.0), rejections=[RejectionReason.SOFTCORE_TOO_LARGE]
        )
        assert not score.feasible
        assert score.rejections == (RejectionReason.SOFTCORE_TOO_LARGE,)

    def test_the_prediction_is_always_positive(self):
        """The Fisher weight is ``1 / sigma ** 2``; a zero would be infinite information."""
        score = create_scorer("variance").score_edge({}, rejections=[])
        assert score.total > 0

    def test_weights_are_overridable(self):
        score = create_scorer("variance", weights={"intercept": 0.5}).score_edge(
            descriptors(n_softcore_max_heavy=4.0, n_heavy_1=0.0, n_heavy_2=0.0), rejections=[]
        )
        assert score.total == pytest.approx(0.5 + 2.0)

    def test_an_unknown_term_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown variance term"):
            create_scorer("variance", weights={"softcore_atoms": 2.0})

    def test_a_negative_weight_is_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            create_scorer("variance", weights={"softcore_heavy": -1.0})

    def test_describe_weights_reports_the_effective_values(self):
        assert create_scorer("variance").describe_weights()["total_heavy"] == pytest.approx(0.5)


def test_every_builtin_scorer_is_available_without_optional_deps():
    assert set(available_scorers()) == {"linear", "lomaplike", "softcore-size", "variance"}
