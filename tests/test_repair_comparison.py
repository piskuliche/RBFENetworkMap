"""A repaired edge can be toggled back to the soft-core the mapper proposed.

Two things are load-bearing and neither is obvious from the rendered page: that the
pre-repair partition is *reconstructed* rather than stored, and that the repaired view is
the one a reader sees without touching anything. Both are pinned here.
"""

from __future__ import annotations

import pytest

from rbfenetmap.core.options import MappingOptions, NetworkOptions
from rbfenetmap.core.pipeline import build_candidate, build_network
from rbfenetmap.plugins.mappers.mcss_mapper import MCSSExtended2Mapper
from rbfenetmap.plugins.scorers.linear_scorer import LinearScorer
from rbfenetmap.viz.gallery import pre_repair_partition, render_report


@pytest.fixture(scope="module")
def repaired_network(anilides):
    """A network whose edges the repair actually touched."""
    network = build_network(anilides, network_options=NetworkOptions(require_connected=False))
    assert any(e.repair.applied for e in network.edges), "fixture no longer repairs anything"
    return network


class TestPartitionReconstruction:
    """``sc_before = sc_after - demoted`` has to reproduce the mapper's own output."""

    @pytest.mark.parametrize("a,b", [("an_piv", "an_carb"), ("an_prop", "an_piv")])
    def test_it_matches_what_the_mapper_returned(self, anilides, a, b):
        mapper = MCSSExtended2Mapper()
        raw = mapper.map_pair(anilides[a], anilides[b], MappingOptions())
        edge = build_candidate(anilides[a], anilides[b], mapper, LinearScorer(), MappingOptions(), NetworkOptions())
        assert edge.repair.applied, "this pair is supposed to need repair"
        sc1, cc1, sc2, cc2 = pre_repair_partition(edge)
        assert set(sc1) == set(raw.sc1)
        assert set(sc2) == set(raw.sc2)

    def test_the_partition_stays_complete(self, anilides):
        """Every atom keeps a role, or the depiction would silently drop atoms."""
        edge = build_candidate(
            anilides["an_piv"],
            anilides["an_carb"],
            MCSSExtended2Mapper(),
            LinearScorer(),
            MappingOptions(),
            NetworkOptions(),
        )
        sc1, cc1, sc2, cc2 = pre_repair_partition(edge)
        assert len(set(sc1) | set(cc1)) == edge.mapping.n_atoms_1
        assert len(set(sc2) | set(cc2)) == edge.mapping.n_atoms_2
        assert not set(sc1) & set(cc1)

    def test_it_is_smaller_than_the_repaired_soft_core(self, anilides):
        """The repair only ever grows the soft-core, so 'before' must be a subset."""
        edge = build_candidate(
            anilides["an_piv"],
            anilides["an_carb"],
            MCSSExtended2Mapper(),
            LinearScorer(),
            MappingOptions(),
            NetworkOptions(),
        )
        sc1, _, sc2, _ = pre_repair_partition(edge)
        assert set(sc1) < set(edge.mapping.sc1)
        assert set(sc2) <= set(edge.mapping.sc2)


class TestRendering:
    def test_one_toggle_per_repaired_edge(self, repaired_network):
        report = render_report(repaired_network)
        assert report.count("class='repair-toggle'") == sum(1 for e in repaired_network.edges if e.repair.applied)

    def test_an_untouched_edge_gets_no_toggle(self, repaired_network):
        """A toggle on an unrepaired edge would promise a difference that is not there."""
        report = render_report(repaired_network)
        for edge in repaired_network.edges:
            if edge.repair.applied:
                continue
            assert f"id='before-{edge.key}'" not in report

    def test_both_pane_sets_are_present(self, repaired_network):
        report = render_report(repaired_network)
        n = sum(1 for e in repaired_network.edges if e.repair.applied)
        assert report.count("panes scroll after") == n
        assert report.count("panes scroll before") == n

    def test_the_repaired_view_is_the_default(self, repaired_network):
        """Unchecked must show 'after'.

        Asserted through the CSS rather than the markup order: the before panes are hidden
        until the checkbox is checked, which is the only thing that makes the default hold.
        """
        report = render_report(repaired_network)
        assert ".panes.before { display: none; }" in report
        assert ".repair-toggle:checked ~ .panes.before { display: flex; }" in report

    def test_no_checkbox_is_pre_checked(self, repaired_network):
        assert "checked>" not in render_report(repaired_network)
        assert "checked=" not in render_report(repaired_network)

    def test_the_report_stays_script_free(self, repaired_network):
        """The toggle is CSS. A script would break the self-contained guarantee."""
        assert "<script" not in render_report(repaired_network)


class TestSwitches:
    def test_it_can_be_turned_off(self, repaired_network):
        report = render_report(repaired_network, repair_comparison=False)
        # The stylesheet is static, so it still carries the .repair-toggle rules. What must
        # be gone is the markup: no control, and no second set of panes to reveal.
        assert "class='repair-toggle'" not in report
        assert "panes scroll before" not in report

    def test_off_leaves_the_ordinary_panes(self, repaired_network):
        report = render_report(repaired_network, repair_comparison=False)
        assert report.count("class='panes scroll'") == len(repaired_network.edges)

    def test_the_cap_limits_the_toggles(self, repaired_network):
        assert render_report(repaired_network, max_repair_comparisons=1).count("class='repair-toggle'") == 1

    def test_zero_means_no_limit(self, repaired_network):
        n = sum(1 for e in repaired_network.edges if e.repair.applied)
        assert render_report(repaired_network, max_repair_comparisons=0).count("class='repair-toggle'") == n


class TestTheCapSpeaksUp:
    """A control that appears on some cards and not others has to explain itself."""

    def test_it_says_how_many_can_be_toggled(self, repaired_network):
        n = sum(1 for e in repaired_network.edges if e.repair.applied)
        assert f"{n} of the {n} repaired edges" in render_report(repaired_network)

    def test_it_names_what_it_left_out(self, repaired_network):
        n = sum(1 for e in repaired_network.edges if e.repair.applied)
        report = render_report(repaired_network, max_repair_comparisons=1)
        assert f"<strong>{n - 1}</strong> are repaired but carry no toggle" in report

    def test_an_unreached_cap_says_nothing_about_omissions(self, repaired_network):
        assert "carry no toggle" not in render_report(repaired_network)

    def test_no_note_when_the_feature_is_off(self, repaired_network):
        assert "can be toggled" not in render_report(repaired_network, repair_comparison=False)
