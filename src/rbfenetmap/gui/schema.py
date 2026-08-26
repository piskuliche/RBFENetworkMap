"""The GUI's form schema, derived from the CLI's own argument parser.

**This module defines no knob.** It reads the same argument groups ``rbfenet plan``
assembles, and turns each :mod:`argparse` action into a form field. The GUI then serializes
the filled form straight back to an argv list, which the CLI's own
:func:`~rbfenetmap.cli._args.build_network_options` and friends turn into options.

That round trip is the whole design. A GUI that kept its own list of knobs would be a
second option surface, and a second option surface drifts: a flag added to the CLI would
quietly not exist in the GUI, and a default moved in one place would silently disagree with
the other. Here a new flag appears in the form with no change to this file, and
``tests/test_gui_schema.py`` fails until any flag deliberately left out is classified.

It also means the GUI can always show the user the exact command that produced what they
are looking at, which is the point of the tool: explore in the browser, paste the
``rbfenet plan ...`` line into a job script.

Reaching into ``parser._action_groups`` is unavoidable -- argparse exposes no public
equivalent -- and has precedent in :func:`rbfenetmap.cli._args.explicit_dests`, which does
the same thing for the same reason.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from rbfenetmap.cli._args import (
    COMPAT_CLI_PINS,
    add_ligand_arguments,
    add_mapping_arguments,
    add_network_arguments,
    add_scorer_arguments,
    add_softcore_arguments,
)

__all__ = (
    "EXPORT_KNOBS",
    "KNOB_EXCLUSIONS",
    "OUTPUT_DESTS",
    "PIPELINE_KNOBS",
    "PLANNER_KNOBS",
    "WIDGETS",
    "inactive_dests",
    "knob_parser",
    "plan_schema",
    "to_argv",
)

#: Every widget kind a field may carry. Closed so the page can switch on it exhaustively,
#: and so a new argparse action type fails the schema test rather than rendering as a
#: text box that quietly mangles its value.
WIDGETS: tuple[str, ...] = (
    "text",
    "int",
    "float",
    "bool",
    "choice",
    "plugin",
    "repeatable",
    "path",
    "path_list",
    "tristate",
)

#: Destinations that appear in the five shared argument groups but are deliberately not
#: rendered as knobs, each for a stated reason. Classified rather than merely skipped:
#: the drift test in ``tests/test_gui_schema.py`` requires every parser action to be either
#: a field or a member of this set, so a flag added to the CLI cannot slip past unnoticed.
KNOB_EXCLUSIONS: Mapping[str, str] = {
    "ligands": "Input, not a knob. The session loads ligands and owns the path.",
    "name_property": "Input, not a knob. Applied at load time by the session.",
    "write_aligned": "Output plumbing. The GUI writes nothing to the user's tree on a run.",
    "weights_file": (
        "The page edits scoring weights inline and emits them as --weights, so the copied "
        "command shows the values that produced the network. A path to a JSON file the GUI "
        "cannot display would be a knob whose effect is invisible in the command line."
    ),
    "mapper_opt": "Dead flag: parsed and never read at any create_mapper call site. See issue #55.",
    "progress": "The GUI reports its own progress; the stderr renderer has nowhere to go.",
}

#: Destinations that exist on the ``plan`` subcommand but not in the five shared groups.
#: Output plumbing the GUI supplies for itself. Listed so the drift test can account for
#: every action on the real parser, not only the ones this module builds.
OUTPUT_DESTS: frozenset[str] = frozenset(
    {"out", "show_rejected", "cost_units", "export", "export_dir", "exporter_opt", "validate_exporter"}
)

#: Knobs that apply whatever planner is chosen, because they are consumed before or after
#: selection rather than by the planner: input preparation, mapping, the soft-core
#: feasibility policy, candidate generation (``core/pairs.py``), counterpoised pricing
#: (``core/cbfe.py``), the post-selection consistency pass (``core/consistency.py``),
#: intermediate generation, and the operational knobs.
PIPELINE_KNOBS: frozenset[str] = frozenset(
    {
        "align",
        "align_reference",
        "align_min_atoms",
        "mapper",
        "match_selection",
        "mcs_timeout",
        "distance_threshold",
        "scorer",
        "weights",
        "max_softcore_atoms",
        "max_softcore_fraction",
        "min_core_atoms",
        "min_mcs_fraction",
        "core_rmsd_threshold",
        "ring_policy",
        "charge_change_policy",
        "compat",
        "planner",
        "pair_strategy",
        "explicit_edge",
        "prefilter",
        "prefilter_k",
        "prefilter_min_tanimoto",
        "cbfe_base_cost",
        "cbfe_atom_weight",
        "consistency",
        "jobs",
        "intermediates",
        "intermediate_generator",
        "max_intermediates",
        "max_intermediate_gaps",
        "intermediates_per_gap",
        "intermediate_seed",
        "intermediate_pose_attempts",
        "intermediate_pose_rmsd_factor",
        "intermediate_min_link_score",
        "intermediate_max_dist",
        "intermediate_max_cycle",
        "intermediate_max_subgraph_dist",
        "intermediate_beta",
        # Honoured only under the mst planner: pipeline.py falls back to eager evaluation
        # for every other one, with a log line and no error. Listed here because it is
        # never *inactive* -- eager is what it degrades to -- and flagged to the user by
        # the page rather than by this table.
        "pair_evaluation",
        "adaptive_initial_neighbors",
        "adaptive_batch_size",
    }
)

#: Knobs that change nothing about the planned network. The A-optimal sample allocation is
#: read only by the Amber exporter, so moving these cannot move an edge -- worth saying out
#: loud in a tool whose whole display is the network.
EXPORT_KNOBS: frozenset[str] = frozenset({"design_total_ns", "design_lambda_min", "design_lambda_max"})

#: Which selection knobs each built-in planner actually reads, as CLI destinations.
#:
#: Transcribed from what each planner module touches on its ``options`` argument. Advisory:
#: it drives a "this planner ignores that" badge, and nothing depends on it being complete.
#: That matters because the alternative is worse -- ``star``, ``explicit`` and ``complete``
#: silently no-op some fourteen network flags today, and only ``--design`` and ``--cbfe``
#: are refused out loud.
PLANNER_KNOBS: Mapping[str, frozenset[str]] = {
    "mst": frozenset(
        {
            "banned_edge",
            "forced_edge",
            "hub",
            "n_edges",
            "edges_per_ligand",
            "min_cycle_coverage",
            "allow_disconnected",
            "edge_direction",
            "selection_objective",
            "cycle_coverage_mode",
            "max_cycle_size",
            "max_diameter",
            "cbfe",
            "cluster_by",
            "cluster_bridges",
        }
    ),
    "optimal": frozenset(
        {
            "banned_edge",
            "forced_edge",
            "n_edges",
            "edges_per_ligand",
            "allow_disconnected",
            "edge_direction",
            "cbfe",
            "design",
            "design_candidate_factor",
            "design_refine",
        }
    ),
    "star": frozenset({"banned_edge", "hub", "hub_selection", "n_edges", "allow_disconnected"}),
}
# redundant-mst is the mst planner plus one knob; explicit and complete share the simple
# planners' surface, minus the hub rule that only star consults.
PLANNER_KNOBS = {
    **PLANNER_KNOBS,
    "redundant-mst": PLANNER_KNOBS["mst"] | {"n_redundancy"},
    "explicit": frozenset({"banned_edge", "explicit_edge", "n_edges", "allow_disconnected"}),
    "complete": frozenset({"banned_edge", "n_edges", "allow_disconnected"}),
}

#: Plugin-name fields, and the plugin kind each one selects from.
_PLUGIN_FIELDS: Mapping[str, str] = {
    "mapper": "mapper",
    "scorer": "scorer",
    "planner": "planner",
    "intermediate_generator": "intermediate",
}


def knob_parser() -> argparse.ArgumentParser:
    """Return a throwaway parser carrying exactly the knobs ``rbfenet plan`` accepts.

    Assembled from the same five public group builders
    :func:`rbfenetmap.cli.main.build_parser` uses, so the flags, defaults, choices and help
    text are the CLI's, not a copy of them.

    Returns
    -------
    argparse.ArgumentParser
        Not fit for parsing a real command line: it has no subcommands and none of the
        output flags. It exists to be walked.
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_ligand_arguments(parser)
    add_mapping_arguments(parser)
    add_scorer_arguments(parser)
    add_softcore_arguments(parser)
    add_network_arguments(parser)
    return parser


def _widget_for(action: argparse.Action) -> str:
    """Classify one argparse action into a member of :data:`WIDGETS`.

    Raises
    ------
    ValueError
        If the action uses a construct this module has never seen. Loud on purpose: a new
        action type rendered as a text box would silently mangle whatever the user typed.
    """
    if isinstance(action, argparse.BooleanOptionalAction):
        return "tristate"
    if isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001 - argparse exposes no public equivalent
        return "bool"
    if isinstance(action, argparse._AppendAction):  # noqa: SLF001 - as above
        return "repeatable"
    if not isinstance(action, argparse._StoreAction):  # noqa: SLF001 - as above
        raise ValueError(f"No widget for {type(action).__name__} (dest {action.dest!r}).")
    if action.type is Path:
        return "path_list" if action.nargs in ("+", "*") else "path"
    if action.dest in _PLUGIN_FIELDS:
        return "plugin"
    if action.choices:
        return "choice"
    if action.type is int:
        return "int"
    if action.type is float:
        return "float"
    return "text"


def _field(action: argparse.Action) -> dict[str, Any]:
    """Describe one action as a form field."""
    widget = _widget_for(action)
    field: dict[str, Any] = {
        "dest": action.dest,
        "flag": max(action.option_strings, key=len),
        "widget": widget,
        "default": action.default,
        "choices": list(action.choices) if action.choices else None,
        "metavar": action.metavar,
        # The help strings carry %(default)s, which argparse only expands while formatting.
        # Substitute it here so the browser does not have to know argparse's dialect.
        "help": (action.help or "").replace("%(default)s", str(action.default)),
        # A knob whose default is None has a meaningful "unset" state that is not any of
        # its choices -- no --n-edges cap, no --compat pin, no --hub. The page renders that
        # as a blank option rather than inventing a sentinel value.
        "optional": action.default is None,
    }
    if widget == "plugin":
        field["plugin_kind"] = _PLUGIN_FIELDS[action.dest]
    return field


def _plugin_table() -> dict[str, list[dict[str, Any]]]:
    """List every built-in plugin with its availability.

    Reads the five ``BUILTIN_*`` tables directly. They are plain ``dict[str, PluginSpec]``
    and importing them imports no backend -- ``PluginSpec.missing_requirements`` probes with
    ``find_spec``, which is exactly what lets this report on kartograf without it installed.
    """
    from rbfenetmap.plugins.exporters import BUILTIN_EXPORTERS
    from rbfenetmap.plugins.intermediates import BUILTIN_INTERMEDIATES
    from rbfenetmap.plugins.mappers import BUILTIN_MAPPERS
    from rbfenetmap.plugins.planners import BUILTIN_PLANNERS
    from rbfenetmap.plugins.scorers import BUILTIN_SCORERS

    tables = {
        "mapper": BUILTIN_MAPPERS,
        "scorer": BUILTIN_SCORERS,
        "planner": BUILTIN_PLANNERS,
        "exporter": BUILTIN_EXPORTERS,
        "intermediate": BUILTIN_INTERMEDIATES,
    }
    return {
        kind: [
            {
                "name": name,
                "available": spec.available,
                "missing": list(spec.missing_requirements),
                "description": spec.description,
            }
            for name, spec in sorted(table.items())
        ]
        for kind, table in tables.items()
    }


def _scorer_weights() -> dict[str, dict[str, float]]:
    """Return each configurable scorer's default weights, keyed by scorer name.

    Lets the page offer a real slider per term instead of a free-text ``K=V`` box. Every
    one of these scorers raises on an unknown key, so a form built from these dicts can
    only produce keys the scorer accepts.
    """
    from rbfenetmap.plugins.scorers.linear_scorer import DEFAULT_SCORE_WEIGHTS
    from rbfenetmap.plugins.scorers.lomaplike_scorer import DEFAULT_LOMAP_PARAMETERS
    from rbfenetmap.plugins.scorers.variance_scorer import DEFAULT_VARIANCE_WEIGHTS

    return {
        "linear": dict(DEFAULT_SCORE_WEIGHTS),
        "lomaplike": dict(DEFAULT_LOMAP_PARAMETERS),
        "variance": dict(DEFAULT_VARIANCE_WEIGHTS),
    }


def _groups() -> list[dict[str, Any]]:
    """Walk the knob parser into form groups, dropping the ones left entirely empty."""
    groups = []
    for group in knob_parser()._action_groups:  # noqa: SLF001 - argparse exposes no public equivalent
        fields = [
            _field(action)
            for action in group._group_actions  # noqa: SLF001 - as above
            if action.dest not in KNOB_EXCLUSIONS and action.dest != argparse.SUPPRESS
        ]
        if fields:
            groups.append({"title": group.title, "fields": fields})
    return groups


def plan_schema() -> dict[str, Any]:
    """Return everything the GUI needs to render the knob form, as JSON-ready data.

    Returns
    -------
    dict
        ``groups``
            One entry per argparse argument group, in the order ``rbfenet plan`` declares
            them, each with a ``title`` and a list of fields.
        ``plugins``
            Every built-in plugin by kind, with availability and missing requirements.
        ``scorer_weights``
            Default weight tables for the three configurable scorers.
        ``planner_knobs``, ``pipeline_knobs``, ``export_knobs``
            The advisory relevance tables, so the page can grey out a knob the chosen
            planner will ignore.
        ``compat_pins``
            Destinations each ``--compat`` level pins, which it refuses to be combined with.
        ``exclusions``
            Destinations deliberately not offered, and why.
    """
    return {
        "groups": _groups(),
        "plugins": _plugin_table(),
        "scorer_weights": _scorer_weights(),
        "planner_knobs": {name: sorted(dests) for name, dests in PLANNER_KNOBS.items()},
        "pipeline_knobs": sorted(PIPELINE_KNOBS),
        "export_knobs": sorted(EXPORT_KNOBS),
        "compat_pins": {level: sorted(pins) for level, pins in COMPAT_CLI_PINS.items()},
        "exclusions": dict(KNOB_EXCLUSIONS),
    }


def inactive_dests(planner: str) -> tuple[str, ...]:
    """Return the knobs *planner* will silently ignore.

    Parameters
    ----------
    planner : str
        A planner plugin name.

    Returns
    -------
    tuple of str
        Sorted destinations. Empty for a planner not in :data:`PLANNER_KNOBS` -- an
        unknown, probably third-party, planner gets no claims made about it rather than
        having every knob declared inactive.

    Notes
    -----
    Advisory. The pipeline refuses only two of these out loud, ``--design`` and ``--cbfe``,
    through the planner's own ``check_design_support`` and ``check_cbfe_support``. The rest
    are accepted and then not read, which is exactly the failure a form can prevent and a
    command line cannot.
    """
    known = PLANNER_KNOBS.get(planner)
    if known is None:
        return ()
    always = known | PIPELINE_KNOBS | EXPORT_KNOBS
    every = {field["dest"] for group in _groups() for field in group["fields"]}
    return tuple(sorted(every - always))


def _field_index() -> dict[str, dict[str, Any]]:
    """Map destination to field description, for serialization."""
    return {field["dest"]: field for group in _groups() for field in group["fields"]}


def to_argv(values: Mapping[str, Any]) -> list[str]:
    """Serialize filled form *values* into ``rbfenet plan`` flags.

    Only what differs from the default is emitted, so the copied command line is the short
    one a person would have written rather than sixty flags most of which say nothing.

    Parameters
    ----------
    values : Mapping
        Destination to value. A destination absent from the mapping takes its default and
        emits nothing.

    Returns
    -------
    list of str
        The knob flags alone. The caller prepends ``plan`` and the input and output flags;
        those are the session's business, not the form's.

    Raises
    ------
    ValueError
        If *values* names a destination that is not a knob. Loud rather than ignored: a
        silently dropped value would produce a command line that does not reproduce the
        network shown beside it, which is the one promise this module exists to keep.
    """
    fields = _field_index()
    unknown = sorted(set(values) - set(fields))
    if unknown:
        raise ValueError(
            f"Not knob destination(s): {unknown}. Known: {sorted(fields)}. "
            f"Deliberately excluded: {sorted(KNOB_EXCLUSIONS)}."
        )

    argv: list[str] = []
    for dest, field in fields.items():
        if dest not in values:
            continue
        value = values[dest]
        if value is None or value == field["default"]:
            continue
        flag, widget = field["flag"], field["widget"]
        if widget == "bool":
            # A store_true default is False, so reaching here means the user turned it on.
            argv.append(flag)
        elif widget in ("repeatable", "path_list"):
            for item in value:
                argv += [flag, str(item)]
        elif widget == "tristate":
            argv.append(flag if value else f"--no-{dest.replace('_', '-')}")
        else:
            argv += [flag, str(value)]
    return argv
