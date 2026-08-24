"""The HTML report draws its rejected candidates, not merely names them.

A reason string on its own rarely settles "why isn't X joined to Y". The mapping that
provoked the rejection is already on the candidate, so these tests pin that it reaches the
page -- and that the two ways it could go wrong at scale are handled: an uncapped report
inlining hundreds of SVGs, and a candidate with no correspondence to draw at all.
"""

from __future__ import annotations

import dataclasses

import pytest

from rbfenetmap.core.models import AtomMapping, EdgeScore, Network, RejectionReason, SoftcoreRepair, Transformation
from rbfenetmap.core.options import NetworkOptions, SoftcorePolicy
from rbfenetmap.core.pipeline import build_network
from rbfenetmap.viz.gallery import render_report


@pytest.fixture(scope="module")
def rejected_network(benzamides):
    """A real network carrying real rejections.

    Driven through the pipeline rather than assembled by hand: the depiction reads
    ``mapping`` and ``repair`` fields that a hand-built candidate would have to guess at,
    and guessing them is how a test like this passes while the report stays blank.
    """
    options = NetworkOptions(softcore=SoftcorePolicy(min_mcs_fraction=0.95), require_connected=False)
    network = build_network(benzamides, network_options=options)
    assert network.rejected, "fixture no longer rejects anything; the thresholds moved"
    return network


class TestRejectedAreDrawn:
    def test_each_rejected_candidate_gets_a_card(self, rejected_network):
        report = render_report(rejected_network)
        for candidate in rejected_network.rejected:
            assert f"id='reject-{candidate.key}'" in report

    def test_the_reason_is_shown_as_a_badge_on_the_card(self, rejected_network):
        report = render_report(rejected_network)
        reasons = {r.value for c in rejected_network.rejected for r in c.score.rejections}
        for reason in reasons:
            assert f"<span class='badge rej'>{reason}</span>" in report

    def test_drawing_adds_two_depictions_per_rejected_candidate(self, rejected_network):
        """The card carries structures, not just a heading.

        Counting SVGs against the same report with depictions off is what makes this able
        to fail: asserting only that the count is "large" would pass on the network
        diagram alone.
        """
        with_draw = render_report(rejected_network).count("<svg")
        without = render_report(rejected_network, reject_depictions=False).count("<svg")
        assert with_draw - without == 2 * len(rejected_network.rejected)

    def test_off_switch_removes_every_card(self, rejected_network):
        report = render_report(rejected_network, reject_depictions=False)
        assert "id='reject-" not in report
        # The table is the point of the section and must survive the depictions going away.
        assert "Rejected candidates" in report


class TestCap:
    def test_the_cap_limits_the_cards(self, rejected_network):
        report = render_report(rejected_network, max_reject_depictions=1)
        assert report.count("id='reject-") == 1

    def test_the_cap_says_how_many_it_dropped(self, rejected_network):
        omitted = len(rejected_network.rejected) - 1
        report = render_report(rejected_network, max_reject_depictions=1)
        assert f"<strong>{omitted}</strong> not shown" in report

    def test_an_uncapped_report_draws_all_of_them(self, rejected_network):
        report = render_report(rejected_network, max_reject_depictions=0)
        assert report.count("id='reject-") == len(rejected_network.rejected)
        assert "not shown" not in report

    def test_a_cap_that_is_not_reached_says_nothing(self, rejected_network):
        report = render_report(rejected_network, max_reject_depictions=len(rejected_network.rejected))
        assert "not shown" not in report


class TestNoCommonCore:
    """A pair the mapper could not relate is still drawn, and labelled.

    ``AtomMapping`` requires every atom to hold a role, so ``mapper_failed`` and
    ``no_common_core`` do not produce an empty mapping -- they produce a wholly soft-core
    one. Suppressing the depiction for those would be suppressing the most informative
    case; the report explains the all-warm picture instead.
    """

    @staticmethod
    def _no_core(network: Network) -> Network:
        """Add a candidate the mapper declined outright."""
        candidate = network.rejected[0]
        empty = AtomMapping.from_core_pairs(
            {}, n_atoms_1=candidate.mapping.n_atoms_1, n_atoms_2=candidate.mapping.n_atoms_2, method="mcss-e2"
        )
        declined = Transformation(
            source=candidate.source,
            target=candidate.target,
            mapping=empty,
            repair=SoftcoreRepair(rejection=RejectionReason.MAPPER_FAILED, trace=("mapper declined the pair",)),
            score=EdgeScore.rejected(RejectionReason.MAPPER_FAILED, scorer="linear"),
        )
        return dataclasses.replace(network, candidates=(*network.edges, declined))

    def test_the_partition_is_wholly_soft_core_not_empty(self, rejected_network):
        """The premise the rest of this class rests on."""
        mapping = self._no_core(rejected_network).rejected[0].mapping
        assert mapping.cc1 == () and mapping.cc2 == ()
        assert len(mapping.sc1) == mapping.n_atoms_1
        assert len(mapping.sc2) == mapping.n_atoms_2

    def test_it_is_still_drawn(self, rejected_network):
        network = self._no_core(rejected_network)
        drawn = render_report(network).count("<svg")
        suppressed = render_report(network, reject_depictions=False).count("<svg")
        assert drawn - suppressed == 2 * len(network.rejected)

    def test_the_all_soft_core_picture_is_explained(self, rejected_network):
        report = render_report(self._no_core(rejected_network))
        assert "No common core was found" in report

    def test_a_normal_rejection_gets_no_such_note(self, rejected_network):
        """Guards the note against firing on every card, which would make it noise."""
        assert all(c.mapping.cc1 for c in rejected_network.rejected)
        assert "No common core was found" not in render_report(rejected_network)

    def test_the_reason_reaches_the_table(self, rejected_network):
        report = render_report(self._no_core(rejected_network))
        assert RejectionReason.MAPPER_FAILED.value in report
