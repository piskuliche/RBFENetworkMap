"""Resolving one edge, and saying what is known about it.

Two front-ends ask this question -- ``rbfenet inspect`` and the GUI's soft-core panel --
and the whole reason the logic is shared is that a question answered twice gets two
answers. These tests pin the resolution rule, which is subtler than it looks, and the
serialization, which had a bug that only a browser would have noticed.
"""

from __future__ import annotations

import json
import math

import pytest

from rbfenetmap.core.inspect import edge_facts, resolve_edge
from rbfenetmap.core.models import EdgeKind, Network, RejectionReason
from rbfenetmap.core.options import NetworkOptions
from rbfenetmap.core.pipeline import build_network

from .conftest import make_transformation


@pytest.fixture(scope="module")
def planned(benzamides):
    """A real planned network, so resolution meets real oriented edges."""
    return build_network(benzamides, network_options=NetworkOptions())


@pytest.fixture(scope="module")
def repaired_network(anilides):
    """A series whose edges actually need the soft-core repair.

    The benzamide fixture plans without demoting anything, so it cannot exercise the
    trace. This is the same series ``test_repair_comparison.py`` reaches for.
    """
    return build_network(anilides, network_options=NetworkOptions())


class TestResolution:
    def test_an_edge_resolves_by_its_own_key(self, planned):
        edge = planned.edges[0]
        assert resolve_edge(planned, edge.key, scope="edges") is edge

    def test_an_edge_resolves_by_its_reversed_key(self, planned):
        """The pass that matters most, and the one a naive lookup omits.

        Every planner orients its selected edges through ``orient_edge``, so a selected
        edge's key routinely differs from the key of the candidate it came from. Asking
        for a pair in the direction you happen to have must not miss.
        """
        edge = planned.edges[0]
        assert resolve_edge(planned, f"{edge.target}~{edge.source}", scope="edges") is edge

    def test_the_exact_direction_wins_over_the_reverse(self, benzamides):
        """``candidates`` may legitimately hold both directions of one pair.

        ``pair_strategy="all_pairs"`` enumerates permutations rather than combinations, so
        a caller who typed ``a~b`` has to get ``a~b`` and not its mirror.
        """
        network = build_network(benzamides, network_options=NetworkOptions(pair_strategy="all_pairs"))
        both = {c.key for c in network.candidates}
        pair = next((k for k in both if f"{k.split('~')[1]}~{k.split('~')[0]}" in both), None)
        if pair is None:  # pragma: no cover - depends on the strategy enumerating both ways
            pytest.skip("no pair enumerated in both directions")
        assert resolve_edge(network, pair, scope="candidates").key == pair

    def test_scope_selects_which_collection_is_searched(self, planned):
        """The scope is a parameter because one key can name two different edges."""
        edge = planned.edges[0]
        assert resolve_edge(planned, edge.key, scope="edges") is edge
        with pytest.raises(ValueError, match="candidate pool"):
            resolve_edge(Network(ligands=planned.ligands, edges=planned.edges), edge.key, scope="candidates")

    def test_a_selected_cbfe_edge_and_a_rejected_rbfe_candidate_can_share_a_key(self, benzamides):
        """The case that forced scope to be a parameter rather than a fixed order.

        Under ``cbfe_mode="bridge"`` a pair the mapper could not relate is materialized as
        a counterpoised edge, while the relative candidate that was refused stays in the
        pool with the same ``source~target`` key. Searching one fixed order can only ever
        answer one of the two questions.
        """
        rbfe = make_transformation("a", "b", feasible=False)
        cbfe = make_transformation("a", "b", kind=EdgeKind.CBFE)
        network = Network(ligands={}, edges=(cbfe,), candidates=(rbfe,))
        assert resolve_edge(network, "a~b", scope="edges").kind is EdgeKind.CBFE
        assert resolve_edge(network, "a~b", scope="candidates").kind is EdgeKind.RBFE
        # "any" is the CLI's sense of the question, and prefers what was selected.
        assert resolve_edge(network, "a~b", scope="any").kind is EdgeKind.CBFE

    def test_an_unknown_edge_says_what_is_available(self, planned):
        with pytest.raises(ValueError, match="is not in"):
            resolve_edge(planned, "nosuch~pair", scope="edges")

    def test_a_malformed_key_is_refused(self, planned):
        with pytest.raises(ValueError):
            resolve_edge(planned, "not-an-edge-key", scope="edges")

    def test_an_unknown_scope_is_refused(self, planned):
        with pytest.raises(ValueError, match="scope must be one of"):
            resolve_edge(planned, planned.edges[0].key, scope="everywhere")


class TestFacts:
    def test_the_facts_describe_the_edge(self, planned):
        edge = planned.edges[0]
        facts = edge_facts(planned, edge)
        assert facts["key"] == edge.key
        assert facts["kind"] == edge.kind.value
        assert facts["selected"] is True
        assert facts["feasible"] is True
        assert facts["n_common_core"] == edge.mapping.n_common_core
        assert facts["n_softcore_1"] == edge.mapping.n_softcore_1
        assert facts["cost"] == pytest.approx(edge.score.total)

    def test_selected_is_a_statement_about_the_pair(self, benzamides):
        """A pair bridged by a counterpoised edge is selected even when the object in hand
        is the relative candidate that was refused. That is what ``rbfenet inspect`` has
        always printed, and the panel must not disagree with it."""
        rbfe = make_transformation("a", "b", feasible=False)
        cbfe = make_transformation("a", "b", kind=EdgeKind.CBFE)
        network = Network(ligands={}, edges=(cbfe,), candidates=(rbfe,))
        assert edge_facts(network, rbfe)["selected"] is True

    def test_an_infeasible_cost_is_null_not_infinity(self, planned):
        """The bug that would have broken every hover on a rejected pair.

        ``EdgeScore.total`` is ``math.inf`` on every infeasible candidate, and
        ``json.dumps`` renders that as the bare token ``Infinity``, which ``JSON.parse``
        rejects. ``io/networkio.py`` already spells this fix; so does this.
        """
        rejected = make_transformation("a", "b", feasible=False)
        assert math.isinf(rejected.score.total)
        facts = edge_facts(Network(ligands={}, candidates=(rejected,)), rejected)
        assert facts["cost"] is None
        assert facts["feasible"] is False

    def test_every_fact_survives_strict_json(self, planned):
        """Serialized the way the GUI serializes it, refusing what a browser cannot read."""
        for edge in (*planned.edges, *planned.candidates):
            blob = json.dumps(edge_facts(planned, edge), allow_nan=False)
            assert "Infinity" not in blob and "NaN" not in blob

    def test_the_trace_is_returned_whole(self, repaired_network):
        """``Transformation.reversed`` prepends an orientation note as line zero.

        Drop or reflow it and the trace's side-1/side-2 language transposes against the
        pictures a caller draws beside it.
        """
        repaired = next((e for e in repaired_network.edges if e.repair.applied), None)
        assert repaired is not None, "fixture no longer repairs anything"
        assert edge_facts(repaired_network, repaired)["trace"] == list(repaired.repair.trace)
        assert repaired.repair.trace, "a repaired edge records why"

    def test_rejections_are_reported_by_name(self):
        rejected = make_transformation("a", "b", feasible=False)
        facts = edge_facts(Network(ligands={}, candidates=(rejected,)), rejected)
        assert facts["rejections"]
        assert all(isinstance(r, str) for r in facts["rejections"])
        assert RejectionReason(facts["rejections"][0])

    def test_a_counterpoised_edge_is_marked(self):
        cbfe = make_transformation("a", "b", kind=EdgeKind.CBFE)
        facts = edge_facts(Network(ligands={}, edges=(cbfe,)), cbfe)
        assert facts["counterpoised"] is True
        assert facts["n_common_core"] == 0

    def test_an_invented_endpoint_is_named(self, planned):
        """So a caller can badge it, rather than showing an invention as a real ligand."""
        edge = planned.edges[0]
        assert edge_facts(planned, edge)["synthetic"] == []


class TestInteractiveSvg:
    """The opt-in hooks a host needs to respond to an edge being pointed at."""

    def test_the_default_output_is_unchanged(self, planned):
        """Load-bearing. The report calls this renderer and its 21 checked-in outputs
        must stay byte-identical, so the new parameter has to be invisible when unset."""
        from rbfenetmap.viz.network_svg import render_network_svg

        assert "data-edge" not in render_network_svg(planned)
        assert "edge-hit" not in render_network_svg(planned)

    def test_interactive_marks_every_edge(self, planned):
        from rbfenetmap.viz.network_svg import render_network_svg

        svg = render_network_svg(planned, interactive=True)
        assert svg.count('class="edge-hit"') == len(planned.edges)
        assert svg.count("data-edge=") == len(planned.edges)

    def test_every_marker_names_a_real_transformation(self, planned):
        """The value has to round-trip, or the host resolves a key nothing answers to."""
        import re

        from rbfenetmap.viz.network_svg import render_network_svg

        keys = re.findall(r'data-edge="([^"]+)"', render_network_svg(planned, interactive=True))
        assert set(keys) == {edge.key for edge in planned.edges}
        for key in keys:
            assert resolve_edge(planned, key, scope="edges").key == key

    def test_the_key_is_the_transformations_own(self, planned):
        """Not the loop's endpoints. networkx hands back an undirected edge's endpoints in
        either order, while the key must name the edge as the planner oriented it."""
        import re

        from rbfenetmap.viz.network_svg import render_network_svg

        keys = set(re.findall(r'data-edge="([^"]+)"', render_network_svg(planned, interactive=True)))
        reversed_keys = {f"{e.target}~{e.source}" for e in planned.edges}
        assert not keys & reversed_keys or keys == {e.key for e in planned.edges}

    def test_the_hit_line_is_wide_and_invisible(self, planned):
        """A one-pixel diagonal target cannot be pointed at; stroke width here encodes
        cost, so the thinnest real edge is 1px."""
        from rbfenetmap.viz.network_svg import _HIT_WIDTH, render_network_svg

        svg = render_network_svg(planned, interactive=True)
        assert f'stroke-width="{_HIT_WIDTH}"' in svg
        assert 'stroke="transparent"' in svg
        assert _HIT_WIDTH > 8

    def test_it_stays_well_formed_xml(self, planned):
        from xml.etree import ElementTree

        from rbfenetmap.viz.network_svg import render_network_svg

        ElementTree.fromstring(render_network_svg(planned, interactive=True))

    def test_it_is_reachable_from_the_keyboard(self, planned):
        """This codebase clips its report checkbox rather than hiding it, precisely to keep
        it in the tab order. A pointer-only affordance would be out of character."""
        from rbfenetmap.viz.network_svg import render_network_svg

        svg = render_network_svg(planned, interactive=True)
        assert svg.count('tabindex="0"') == len(planned.edges)
        assert svg.count('role="button"') == len(planned.edges)
        assert svg.count("aria-label=") >= len(planned.edges)

    def test_no_script_is_emitted(self, planned):
        """viz/ output stays script-free even when it grows interaction hooks."""
        from rbfenetmap.viz.network_svg import render_network_svg

        assert "<script" not in render_network_svg(planned, interactive=True)
