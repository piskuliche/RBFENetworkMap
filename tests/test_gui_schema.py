"""The GUI form schema, and the guarantee that it cannot drift from the CLI.

The GUI's whole design is that it owns no option surface: the form is derived from the
CLI's argument groups and serialized back to flags the CLI itself parses. These tests are
what make that a guarantee rather than an intention.
"""

from __future__ import annotations

import argparse

import pytest

from rbfenetmap.cli._args import build_mapping_options, build_network_options, explicit_dests, resolve_compat
from rbfenetmap.cli.main import build_parser
from rbfenetmap.core.options import MappingOptions, NetworkOptions
from rbfenetmap.gui.schema import (
    EXPORT_KNOBS,
    KNOB_EXCLUSIONS,
    OUTPUT_DESTS,
    PIPELINE_KNOBS,
    PLANNER_KNOBS,
    WIDGETS,
    inactive_dests,
    knob_parser,
    plan_schema,
    to_argv,
)


def _plan_dests() -> set[str]:
    """Every destination ``rbfenet plan`` sets, from a real parse.

    Parsing rather than walking the subparser tree: ``vars()`` on the namespace is the
    public way to ask this, and it stays right if the parser is ever restructured.
    """
    args = build_parser().parse_args(["plan", "--ligands", "x.sdf"])
    return set(vars(args)) - {"command", "verbose"}


def _schema_dests() -> set[str]:
    """Every destination the form offers as a knob."""
    return {field["dest"] for group in plan_schema()["groups"] for field in group["fields"]}


def _options_from(values: dict) -> tuple[NetworkOptions, MappingOptions]:
    """Push form *values* through the CLI exactly as ``rbfenet plan`` would."""
    argv = ["plan", "--ligands", "x.sdf", *to_argv(values)]
    args = build_parser().parse_args(argv)
    # build_parser() a second time on purpose: explicit_dests suppresses the defaults of
    # whatever it is handed, so it must not be handed the parser whose result we use.
    resolve_compat(args, explicit_dests(build_parser(), argv))
    return build_network_options(args), build_mapping_options(args)


class TestDrift:
    def test_every_plan_flag_is_classified(self):
        """A flag added to the CLI must be offered by the GUI or explicitly excluded.

        This is the test the whole subpackage exists to make possible. Set equality in both
        directions, so it catches a new CLI flag that the form would silently not have, and
        equally an exclusion left behind after its flag was removed. When it fails, the fix
        is to classify the flag -- as a knob, as output plumbing, or in KNOB_EXCLUSIONS with
        a reason -- not to loosen the assertion.
        """
        assert _schema_dests() | set(KNOB_EXCLUSIONS) | OUTPUT_DESTS == _plan_dests()

    def test_knobs_and_exclusions_are_disjoint(self):
        """Nothing may be both offered and excluded."""
        assert not _schema_dests() & set(KNOB_EXCLUSIONS)

    def test_every_exclusion_states_a_reason(self):
        """An excluded flag without a reason is indistinguishable from an oversight."""
        for dest, reason in KNOB_EXCLUSIONS.items():
            assert reason.strip(), dest

    def test_relevance_tables_name_real_knobs(self):
        """The advisory tables must not accumulate destinations that no longer exist."""
        known = _schema_dests()
        assert PIPELINE_KNOBS <= known
        assert EXPORT_KNOBS <= known
        for planner, dests in PLANNER_KNOBS.items():
            assert dests <= known, (planner, sorted(dests - known))


class TestRoundTrip:
    def test_empty_form_reproduces_the_library_defaults(self):
        """The strongest assertion here: the GUI's idea of "unchanged" is the CLI's.

        An empty form emits no flags, so what comes out the far end must be exactly the
        dataclass defaults. If a default ever moves and the form does not follow, this is
        where it shows -- and because the form reads the parser rather than a copied table,
        it cannot fail for that reason without the parser having genuinely diverged.
        """
        network, mapping = _options_from({})
        assert network == NetworkOptions()
        assert mapping == MappingOptions()

    def test_an_empty_form_emits_nothing(self):
        """No flags, so the copied command line is the short one a person would write."""
        assert to_argv({}) == []

    def test_a_value_equal_to_its_default_emits_nothing(self):
        """Setting a knob to what it already was is not a change worth printing."""
        assert to_argv({"edges_per_ligand": 2, "cbfe": "off"}) == []

    @pytest.mark.parametrize(
        "values,check",
        [
            ({"edges_per_ligand": 3}, lambda n: n.edges_per_ligand == 3),
            ({"cbfe": "bridge"}, lambda n: n.cbfe_mode == "bridge"),
            ({"max_diameter": 4}, lambda n: n.max_diameter == 4),
            ({"allow_disconnected": True}, lambda n: n.require_connected is False),
            ({"banned_edge": ["a~b"]}, lambda n: n.banned_pairs == frozenset({("a", "b")})),
            ({"cluster_by": "charge"}, lambda n: n.cluster_by == "charge"),
            ({"n_redundancy": 3}, lambda n: n.n_redundancy == 3),
            ({"min_cycle_coverage": 0.5}, lambda n: n.min_cycle_coverage == 0.5),
            ({"max_softcore_atoms": 20}, lambda n: n.softcore.max_softcore_atoms == 20),
            ({"charge_change_policy": "reject"}, lambda n: n.softcore.charge_change_policy == "reject"),
            ({"intermediates": "bridge"}, lambda n: n.intermediates.mode == "bridge"),
        ],
    )
    def test_a_set_knob_reaches_the_options_object(self, values, check):
        """Each widget kind actually lands where the CLI would put it.

        Covers the store, store_true, append, choice and float paths, and both nested
        options objects -- a knob that serialized correctly but landed on the wrong
        dataclass would give a command line that does not reproduce its own network.
        """
        network, _ = _options_from(values)
        assert check(network)

    def test_repeatable_values_each_get_their_own_flag(self):
        """``append`` actions repeat the flag rather than joining the values."""
        assert to_argv({"banned_edge": ["a~b", "c~d"]}) == ["--banned-edge", "a~b", "--banned-edge", "c~d"]

    def test_compat_still_refuses_a_knob_it_pins(self):
        """The GUI inherits the CLI's contradiction check rather than reimplementing it.

        ``--compat`` pins every algorithmic knob, so naming one alongside it is a
        contradiction only the user can resolve. That rule lives in resolve_compat, and the
        form gets it for free precisely because it goes through the parser.
        """
        with pytest.raises(ValueError, match="pins every algorithmic knob"):
            _options_from({"compat": "v0.4", "max_diameter": 5})

    def test_compat_alone_is_accepted(self):
        """Pinning without contradicting it is the ordinary case."""
        network, _ = _options_from({"compat": "v0.4"})
        assert network.compat == "v0.4"


class TestFields:
    def test_every_widget_is_a_known_kind(self):
        """The page switches on widget exhaustively; an unknown kind would fall through."""
        for group in plan_schema()["groups"]:
            for field in group["fields"]:
                assert field["widget"] in WIDGETS, field

    def test_a_choice_field_offers_choices(self):
        """A select with nothing in it is a broken control, not an empty one."""
        for group in plan_schema()["groups"]:
            for field in group["fields"]:
                if field["widget"] == "choice":
                    assert field["choices"], field["dest"]

    def test_help_text_has_no_unexpanded_placeholders(self):
        """argparse expands %(default)s while formatting; the browser cannot.

        A tooltip reading "default: %(default)s" is worse than no tooltip.
        """
        for group in plan_schema()["groups"]:
            for field in group["fields"]:
                assert "%(default)s" not in field["help"], field["dest"]

    def test_groups_arrive_in_pipeline_order(self):
        """The form reads in the order the pipeline runs, because the parser declares it so."""
        titles = [group["title"] for group in plan_schema()["groups"]]
        assert titles == ["ligand input", "mapping", "scoring", "soft-core policy", "network selection"]

    def test_optional_marks_exactly_the_knobs_with_no_default(self):
        """A knob whose default is None has an "unset" state that is not one of its values."""
        for group in plan_schema()["groups"]:
            for field in group["fields"]:
                assert field["optional"] == (field["default"] is None), field["dest"]

    def test_plugin_fields_name_their_kind(self):
        """A plugin select has to know which registry to draw its options from."""
        kinds = {
            field["dest"]: field.get("plugin_kind")
            for group in plan_schema()["groups"]
            for field in group["fields"]
            if field["widget"] == "plugin"
        }
        assert kinds == {
            "mapper": "mapper",
            "scorer": "scorer",
            "planner": "planner",
            "intermediate_generator": "intermediate",
        }

    def test_the_knob_parser_carries_no_output_flags(self):
        """It is assembled from the five knob groups, so nothing that writes a file is on it."""
        dests = {action.dest for action in knob_parser()._actions}  # noqa: SLF001
        assert not dests & OUTPUT_DESTS


class TestPlugins:
    def test_every_kind_is_listed_with_availability(self):
        """The form must be able to grey out a plugin whose backend is missing."""
        plugins = plan_schema()["plugins"]
        assert set(plugins) == {"mapper", "scorer", "planner", "exporter", "intermediate"}
        for kind, entries in plugins.items():
            assert entries, kind
            for entry in entries:
                assert entry["available"] == (not entry["missing"]), entry

    def test_the_defaults_are_available_plugins(self):
        """Whatever the form opens on has to be selectable."""
        plugins = plan_schema()["plugins"]
        by_kind = {kind: {e["name"]: e for e in entries} for kind, entries in plugins.items()}
        for kind, name in (("mapper", "mcss-e2"), ("scorer", "linear"), ("planner", "mst")):
            assert by_kind[kind][name]["available"], (kind, name)

    def test_scorer_weight_tables_are_offered_for_the_configurable_scorers(self):
        """So the page can show a slider per term instead of a free-text K=V box."""
        weights = plan_schema()["scorer_weights"]
        assert set(weights) == {"linear", "lomaplike", "variance"}
        assert "softcore_atoms" in weights["linear"]
        for table in weights.values():
            assert table and all(isinstance(v, float) for v in table.values())


class TestRelevance:
    def test_a_star_network_ignores_the_cycle_and_cluster_knobs(self):
        """The single most useful thing the GUI adds over the CLI.

        star, explicit and complete accept these flags and then never read them. Only
        --design and --cbfe are refused out loud, by the planner's own support checks.
        """
        ignored = set(inactive_dests("star"))
        assert {"min_cycle_coverage", "max_diameter", "cluster_by", "cycle_coverage_mode"} <= ignored

    def test_the_mst_planner_ignores_the_design_knobs(self):
        """Design is an objective, and mst does not optimise one."""
        assert {"design", "design_candidate_factor", "design_refine"} <= set(inactive_dests("mst"))

    def test_soft_core_knobs_are_never_inactive(self):
        """Feasibility is decided before selection, so no planner can make it irrelevant."""
        for planner in PLANNER_KNOBS:
            assert not {"max_softcore_atoms", "core_rmsd_threshold"} & set(inactive_dests(planner))

    def test_an_unknown_planner_gets_no_claims_made_about_it(self):
        """A third-party planner reads knobs this table cannot know about.

        Returning everything would grey out the whole form for a plugin that may well use
        all of it, which is worse than saying nothing.
        """
        assert inactive_dests("some-third-party-planner") == ()


class TestSerialization:
    def test_an_unknown_destination_is_refused(self):
        """A silently dropped value would break the one promise the module makes.

        The command line shown beside a network has to be the command line that produces
        it, so a value the form cannot serialize is an error rather than an omission.
        """
        with pytest.raises(ValueError, match="Not knob destination"):
            to_argv({"no_such_knob": 1})

    def test_an_excluded_destination_is_refused_and_says_so(self):
        """Excluded flags are named in the message, so the caller learns why, not just that."""
        with pytest.raises(ValueError, match="mapper_opt"):
            to_argv({"mapper_opt": "k=v"})

    def test_the_emitted_flags_parse(self):
        """Whatever to_argv emits, the real parser must accept."""
        argv = to_argv({"planner": "redundant-mst", "n_redundancy": 3, "cbfe": "cycles", "design_refine": True})
        args = build_parser().parse_args(["plan", "--ligands", "x.sdf", *argv])
        assert args.planner == "redundant-mst"
        assert args.n_redundancy == 3
        assert args.cbfe == "cycles"
        assert args.design_refine is True


class TestScorerArgumentsMove:
    def test_the_scoring_group_is_assemblable_from_public_functions(self):
        """add_scorer_arguments moved out of cli.main so the form could reach it.

        Its four siblings were already public in cli._args; this asserts the fifth is too,
        which is what lets knob_parser build the whole plan surface without reaching into
        another module's underscore.
        """
        from rbfenetmap.cli import _args

        parser = argparse.ArgumentParser(add_help=False)
        _args.add_scorer_arguments(parser)
        assert {action.dest for action in parser._actions} == {"scorer", "weights", "weights_file"}  # noqa: SLF001

    def test_plan_score_and_map_still_share_the_scoring_flags(self):
        """The move must not have changed which subcommands carry the group."""
        parser = build_parser()
        for command in ("plan", "score", "map"):
            args = parser.parse_args([command, "--ligands", "x.sdf"])
            assert args.scorer == "linear"
            assert args.weights is None
