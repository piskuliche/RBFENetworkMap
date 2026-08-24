"""``rbfenet`` command-line entry point.

Standard argparse shape: :func:`build_parser` assembles subcommands, :func:`dispatch`
routes to a thin handler, :func:`main` handles exit codes.

Exit codes: ``0`` success, ``1`` a package-level failure (unsatisfiable constraints, a
missing plugin, unreadable input), ``2`` argparse usage error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from rbfenetmap import __version__
from rbfenetmap.cli import commands
from rbfenetmap.cli._args import (
    add_cost_units_argument,
    add_ligand_arguments,
    add_mapping_arguments,
    add_network_arguments,
    add_softcore_arguments,
    explicit_dests,
    resolve_compat,
)
from rbfenetmap.core.exceptions import RBFENetworkMapError

__all__ = ("build_parser", "dispatch", "main")


def _add_scorer_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the scorer-selection flags."""
    group = parser.add_argument_group("scoring")
    group.add_argument("--scorer", default="linear", help="Scorer plugin (default: %(default)s).")
    group.add_argument("--weights", action="append", metavar="K=V", help="Override a scoring weight, repeatable.")
    group.add_argument("--weights-file", type=Path, metavar="PATH", help="JSON file of scoring weights.")


def build_parser() -> argparse.ArgumentParser:
    """Assemble the full argument parser."""
    parser = argparse.ArgumentParser(
        prog="rbfenet",
        description="Plan Relative Binding Free Energy perturbation networks from RDKit molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase log verbosity, repeatable.")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # plan ---------------------------------------------------------------------
    plan = subparsers.add_parser("plan", help="Map, score, and select a network.")
    add_ligand_arguments(plan)
    add_mapping_arguments(plan)
    _add_scorer_arguments(plan)
    add_softcore_arguments(plan)
    add_network_arguments(plan)
    plan.add_argument("--out", type=Path, default=Path("network.json"), help="Network JSON output path.")
    plan.add_argument("--show-rejected", action="store_true", help="List rejected candidates and why.")
    add_cost_units_argument(plan)
    plan.add_argument("--export", nargs="+", metavar="NAME", help="Also run these exporters.")
    plan.add_argument("--export-dir", type=Path, default=Path("."), help="Directory for exports.")
    plan.add_argument("--exporter-opt", action="append", metavar="K=V", help="Exporter option, repeatable.")
    plan.add_argument(
        "--validate-exporter",
        metavar="NAME",
        help="Check this exporter's format constraints before the expensive mapping stage.",
    )

    # score --------------------------------------------------------------------
    score = subparsers.add_parser("score", help="Score candidate edges without selecting a network.")
    add_ligand_arguments(score)
    add_mapping_arguments(score)
    _add_scorer_arguments(score)
    add_softcore_arguments(score)
    add_network_arguments(score)
    score.add_argument("--top", type=int, metavar="N", help="Show only the N best candidates.")
    score.add_argument("--explain", action="store_true", help="Add a column per cost contribution.")
    score.add_argument("--show-rejected", action="store_true", help="Include infeasible candidates.")
    score.add_argument("--format", choices=("table", "csv", "json"), default="table")

    # map ----------------------------------------------------------------------
    mapping = subparsers.add_parser("map", help="Compute mappings for specific pairs.")
    add_ligand_arguments(mapping)
    add_mapping_arguments(mapping)
    _add_scorer_arguments(mapping)
    add_softcore_arguments(mapping)
    add_network_arguments(mapping)
    mapping.add_argument("--pair", action="append", metavar="A~B", help="Pair to map, repeatable.")
    mapping.add_argument("--draw", type=Path, metavar="DIR", help="Write SVG depictions here.")

    # export -------------------------------------------------------------------
    export = subparsers.add_parser("export", help="Export an already-planned network.")
    export.add_argument("--network", type=Path, required=True, help="Network JSON from `rbfenet plan`.")
    export.add_argument("--exporter", nargs="+", required=True, metavar="NAME", help="Exporters to run.")
    export.add_argument("--dest", type=Path, default=Path("."), help="Output directory.")
    export.add_argument("--exporter-opt", action="append", metavar="K=V", help="Exporter option, repeatable.")

    # replan -------------------------------------------------------------------
    # No ligand, mapping, or soft-core flags: replanning maps nothing. It selects a new
    # network from the candidate pool the original run already scored and stored, which is
    # what makes it cheap enough to run after every analysis.
    replan = subparsers.add_parser("replan", help="Prune high-LMI edges from a planned network and replan the gaps.")
    replan.add_argument("--network", type=Path, required=True, help="Network JSON from `rbfenet plan`.")
    replan.add_argument(
        "--lmi",
        type=Path,
        required=True,
        help=(
            "JSON mapping 'lig_a~lig_b' to a Lagrange Multiplier Index, as produced from an "
            "edgembar network analysis. See the replanning guide for the format; this is the "
            "package's own small schema, not edgembar's on-disk output."
        ),
    )
    cut = replan.add_mutually_exclusive_group()
    cut.add_argument("--lmi-threshold", type=float, metavar="F", help="Prune edges with an LMI strictly above F.")
    cut.add_argument(
        "--lmi-quantile",
        type=float,
        default=0.9,
        metavar="Q",
        help="Prune edges above this quantile of the observed LMIs (default: %(default)s).",
    )
    replan.add_argument(
        "--max-pruned", type=int, metavar="N", help="Prune at most the N worst edges, whatever the cut selects."
    )
    replan.add_argument(
        "--allow-missing-lmi",
        action="store_true",
        help="Treat a selected edge with no LMI value as beyond reproach instead of failing.",
    )
    replan.add_argument(
        "--reselect",
        action="store_true",
        help=(
            "Re-select the whole network over the pruned pool instead of holding the surviving "
            "edges in place. Right before anything has been submitted; after that it can move "
            "edges that are already set up or running."
        ),
    )
    replan.add_argument("--planner", default="mst", help="Planner used for the replan (default: %(default)s).")
    replan.add_argument("--out", type=Path, default=Path("replanned.json"), help="Replanned network JSON path.")

    # report -------------------------------------------------------------------
    report = subparsers.add_parser("report", help="Render a self-contained HTML report.")
    report.add_argument("--network", type=Path, required=True, help="Network JSON from `rbfenet plan`.")
    report.add_argument("--out", type=Path, default=Path("network.html"), help="Output HTML path.")
    report.add_argument("--title", default="RBFE network", help="Report title.")
    report.add_argument("--show-indices", action="store_true", help="Label atoms with their indices.")

    # diagnose -----------------------------------------------------------------
    diagnose = subparsers.add_parser("diagnose", help="Report network-level metrics for a planned network.")
    diagnose.add_argument("--network", type=Path, required=True, help="Network JSON from `rbfenet plan`.")
    diagnose.add_argument(
        "--failure-rate",
        type=float,
        default=0.05,
        metavar="P",
        help="Per-edge failure probability used by the robustness estimate (default: %(default)s).",
    )
    diagnose.add_argument(
        "--repeats", type=int, default=100, metavar="N", help="Monte-Carlo trials (default: %(default)s)."
    )
    diagnose.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="Seed for the robustness estimate (default: %(default)s). The same seed always gives the same number.",
    )
    diagnose.add_argument(
        "--max-cycle-length",
        type=int,
        default=4,
        metavar="N",
        help="Longest cycle counted (default: %(default)s). Short cycles are the ones that localise an error.",
    )
    add_cost_units_argument(diagnose)
    diagnose.add_argument("--format", choices=("table", "json"), default="table")

    # plugins ------------------------------------------------------------------
    plugins = subparsers.add_parser("plugins", help="List plugins and their availability.")
    plugins.add_argument("--kind", choices=("mapper", "scorer", "planner", "exporter"))
    plugins.add_argument("--all", action="store_true", help="Include plugins whose backends are missing.")

    # inspect ------------------------------------------------------------------
    inspect = subparsers.add_parser("inspect", help="Show everything known about one edge.")
    inspect.add_argument("--network", type=Path, required=True, help="Network JSON from `rbfenet plan`.")
    inspect.add_argument("--edge", required=True, metavar="A~B", help="Edge to inspect.")
    inspect.add_argument("--show-repair-trace", action="store_true", help="Print the soft-core repair log.")
    inspect.add_argument("--show-descriptors", action="store_true", help="Print descriptors and cost terms.")
    inspect.add_argument("--show-masks", action="store_true", help="Print the Amber timask/scmask strings.")
    inspect.add_argument("--draw", type=Path, metavar="PATH", help="Write an HTML depiction of the edge.")

    return parser


_HANDLERS = {
    "plan": commands.cmd_plan,
    "score": commands.cmd_score,
    "map": commands.cmd_map,
    "export": commands.cmd_export,
    "replan": commands.cmd_replan,
    "report": commands.cmd_report,
    "plugins": commands.cmd_plugins,
    "inspect": commands.cmd_inspect,
    "diagnose": commands.cmd_diagnose,
}


def dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Route parsed arguments to the matching handler."""
    handler = _HANDLERS.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces the choice
        parser.error(f"Unknown command {args.command!r}")
    return handler(args)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``rbfenet`` console script.

    Package-level errors are reported as a single message rather than a traceback: an
    unsatisfiable constraint or a missing plugin is a user-facing condition, and its
    message already says what to do. ``-v`` restores the traceback for debugging.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Which flags the user actually typed, which the parsed values cannot tell us once a
    # release moves a default. build_parser() is called a second time deliberately:
    # explicit_dests suppresses the defaults of whatever it is handed, so it must not be
    # handed the parser above.
    try:
        resolve_compat(args, explicit_dests(build_parser(), argv))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    try:
        return dispatch(args, parser)
    except RBFENetworkMapError as exc:
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
