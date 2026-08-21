"""Shared argument groups and parsing helpers for the CLI.

Factored out so that ``plan``, ``score``, and ``map`` accept the same flags with the same
defaults. A knob that means one thing under ``plan`` and another under ``score`` is worse
than no knob.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from rbfenetmap.core.intermediates import IntermediateOptions
from rbfenetmap.core.options import (
    COMPAT_LEVELS,
    AlignmentOptions,
    CorePruningPolicy,
    MappingOptions,
    NetworkOptions,
    SoftcorePolicy,
)

__all__ = (
    "COMPAT_CLI_PINS",
    "add_compat_argument",
    "add_ligand_arguments",
    "add_mapping_arguments",
    "add_network_arguments",
    "add_softcore_arguments",
    "build_alignment_options",
    "build_mapping_options",
    "build_network_options",
    "explicit_dests",
    "parse_key_values",
    "resolve_compat",
)


#: What ``--compat LEVEL`` pins, as CLI destination names, per level.
#:
#: **These are literal transcriptions of what the named release did, not a view onto the
#: current defaults.** Deriving them from the parser would be shorter and would defeat the
#: mechanism entirely: when a later version moves a default, a derived table moves with it
#: and silently stops reproducing the version it names.
#:
#: The pinned surface is the *algorithmic* one -- the knobs whose meaning or default may
#: change between releases. Deliberately absent are the settings that describe **this run**
#: rather than **this behaviour**:
#:
#: - ligand-specific intent (``hub``, ``forced_edge``, ``banned_edge``, ``explicit_edge``):
#:   banning an edge is a statement about one ligand set, not about a version;
#: - input preparation (``ligands``, ``align`` and friends): that is which molecules go in,
#:   not how they are planned;
#: - operational knobs (``jobs``, ``progress``, ``out``, ``export``): they cannot change
#:   which network is produced.
#:
#: All three stay usable alongside a compat level, which is what makes it practical rather
#: than merely principled.
COMPAT_CLI_PINS: dict[str, dict[str, Any]] = {
    "v0.4": {
        # mapping
        "mapper": "mcss-e2",
        "match_selection": "fewest_fragments",
        "mcs_timeout": 60,
        "distance_threshold": 2.0,
        "mapper_opt": None,
        # scoring
        "scorer": "linear",
        "weights": None,
        "weights_file": None,
        # soft-core feasibility
        "max_softcore_atoms": 12,
        "max_softcore_fraction": 0.6,
        "min_core_atoms": 4,
        "min_mcs_fraction": 0.35,
        "core_rmsd_threshold": 2.0,
        "ring_policy": "ring_system",
        "charge_change_policy": "penalize",
        # selection
        "planner": "mst",
        "pair_strategy": "all_unordered_pairs",
        "n_edges": None,
        "edges_per_ligand": 2,
        "min_cycle_coverage": 1.0,
        "allow_disconnected": False,
        "edge_direction": "fewer_softcore_first",
        "prefilter": "none",
        "prefilter_k": 8,
        "prefilter_min_tanimoto": 0.4,
        "selection_objective": "uniform_redundancy",
        "max_cycle_size": None,
        "pair_evaluation": "eager",
        "adaptive_initial_neighbors": 3,
        "adaptive_batch_size": 32,
        "cbfe": "off",
        "cbfe_base_cost": 8.0,
        "cbfe_atom_weight": 0.05,
        # v0.4 could not invent a ligand, so every one of these pins to "do not". They are
        # pinned rather than omitted because they are algorithmic: leaving them out would
        # let `--compat v0.4 --intermediates bridge` quietly plan a network v0.4 could not
        # have produced, which is the exact surprise --compat exists to prevent.
        "intermediates": "off",
        "intermediate_generator": "pairmap",
        "max_intermediates": None,
        "max_intermediate_gaps": None,
        "intermediates_per_gap": 4,
        "intermediate_seed": 0xF00D,
        "intermediate_pose_attempts": 10,
        "intermediate_pose_rmsd_factor": 0.5,
        "intermediate_min_link_score": 0.2,
        "intermediate_max_dist": 3,
        "intermediate_max_cycle": 4,
        "intermediate_max_subgraph_dist": 4,
        "intermediate_beta": 0.1,
        "consistency": "pairwise",
    }
}

#: Reverse lookup from destination to the flag that sets it, for error messages. A user
#: who typed ``--edges-per-ligand`` should be told about ``--edges-per-ligand``, not about
#: an internal attribute name they have never seen.
_DEST_TO_FLAG: dict[str, str] = {
    "mapper_opt": "--mapper-opt",
    "mcs_timeout": "--mcs-timeout",
    "cbfe": "--cbfe",
    "allow_disconnected": "--allow-disconnected",
    "weights": "--weights",
    "weights_file": "--weights-file",
    "intermediates": "--intermediates",
}


def _flag_for(dest: str) -> str:
    """Return the user-facing flag that sets *dest*."""
    return _DEST_TO_FLAG.get(dest, "--" + dest.replace("_", "-"))


def add_compat_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--compat``, which pins every algorithmic knob to a released behaviour."""
    parser.add_argument(
        "--compat",
        choices=COMPAT_LEVELS,
        metavar="LEVEL",
        help=(
            "Reproduce a released version's behaviour exactly, pinning every algorithmic "
            f"knob to what that version used. Choices: {', '.join(COMPAT_LEVELS)}. Ligand "
            "intent (--hub, --banned-edge), input preparation (--align) and operational "
            "flags (--jobs, --progress) stay available; naming an algorithmic knob "
            "alongside this is a contradiction and is refused."
        ),
    )


def explicit_dests(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> frozenset[str]:
    """Return the destinations the user actually named on the command line.

    Comparing the parsed value against the current default cannot answer this, and the
    difference is the whole reason ``--compat`` exists: once a release moves a default, an
    unnamed flag and a deliberately-set one become indistinguishable that way, and every
    run would be reported as conflicting with the level it asked for.

    So this re-parses the same argv against a parser whose defaults are all suppressed.
    Only what the user typed survives, which is exactly the question being asked.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        **A throwaway.** Its defaults are suppressed in place, which leaves it unfit to
        parse anything else, so callers pass a freshly built one rather than the parser
        whose result they intend to use.
    argv : Sequence[str], optional
        The same argv the real parse saw. ``None`` means ``sys.argv[1:]``.
    """
    _suppress_defaults(parser)
    try:
        return frozenset(vars(parser.parse_args(list(argv) if argv is not None else None)))
    except SystemExit:  # pragma: no cover - argparse already reported the usage error
        return frozenset()


def _suppress_defaults(parser: argparse.ArgumentParser) -> None:
    """Set every default in *parser*, and in any subparser, to ``SUPPRESS``."""
    for action in parser._actions:  # noqa: SLF001 - argparse exposes no public equivalent
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            for sub in action.choices.values():
                _suppress_defaults(sub)
        action.default = argparse.SUPPRESS


def resolve_compat(args: argparse.Namespace, explicit: frozenset[str]) -> None:
    """Apply ``--compat`` to *args* in place, refusing any knob it contradicts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. Modified in place.
    explicit : frozenset[str]
        Destinations the user named, from :func:`explicit_dests`.

    Raises
    ------
    SystemExit
        Via ``argparse``-style exit, if the user named an algorithmic knob alongside
        ``--compat``.

    Notes
    -----
    Rejecting rather than silently letting one win follows the rule the package already
    applies to ``n_edges`` against ``require_connected``: both resolutions are defensible,
    so neither may be chosen on the user's behalf. ``--compat v0.4 --max-diameter 5`` is a
    request for v0.4's behaviour and for something v0.4 could not do, and only the user can
    say which they meant.
    """
    level = getattr(args, "compat", None)
    if level is None:
        return

    pins = COMPAT_CLI_PINS[level]
    conflicts = sorted(_flag_for(dest) for dest in pins if dest in explicit)
    if conflicts:
        raise ValueError(
            f"--compat {level} pins every algorithmic knob to what {level} used, so it cannot be "
            f"combined with {', '.join(conflicts)}. Drop --compat to set "
            f"{'them' if len(conflicts) > 1 else 'it'} explicitly, or drop "
            f"{'those flags' if len(conflicts) > 1 else 'that flag'} to reproduce {level}. "
            "Ligand intent (--hub, --forced-edge, --banned-edge), input preparation "
            "(--align) and operational flags (--jobs, --progress) are not pinned and may be "
            "combined with --compat freely."
        )

    for dest, value in pins.items():
        if hasattr(args, dest):
            setattr(args, dest, value)


def parse_key_values(items: Sequence[str] | None, *, numeric: bool = True) -> dict[str, Any]:
    """Parse repeated ``key=value`` arguments into a dictionary.

    Parameters
    ----------
    items : Sequence[str], optional
    numeric : bool, optional
        Convert values that parse as numbers, and ``true``/``false`` to booleans.

    Raises
    ------
    argparse.ArgumentTypeError
        If an item has no ``=``. Failing loudly matters here: a mistyped
        ``--weights softcore_atoms 2`` would otherwise be read as two separate items and
        silently ignored, and the run would quietly use the defaults.
    """
    parsed: dict[str, Any] = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise argparse.ArgumentTypeError(f"Expected key=value, got {item!r}.")
        if numeric:
            lowered = value.strip().lower()
            if lowered in ("true", "false"):
                parsed[key] = lowered == "true"
                continue
            try:
                parsed[key] = float(value)
                continue
            except ValueError:
                pass
        parsed[key] = value
    return parsed


def add_ligand_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ligand-input flags."""
    group = parser.add_argument_group("ligand input")
    group.add_argument(
        "--ligands",
        nargs="+",
        type=Path,
        required=True,
        metavar="PATH",
        help="SDF, mol2, .smi files, or directories containing them.",
    )
    group.add_argument(
        "--name-property",
        default="_Name",
        metavar="PROP",
        help="Molecule property to read ligand names from (default: %(default)s).",
    )
    # nargs="?" is unambiguous only because no subcommand takes a positional argument:
    # `--align` followed by another flag, or by the end of the line, can only mean the
    # const. Adding a positional to plan/score/map would break that quietly.
    group.add_argument(
        "--align",
        nargs="?",
        const="mcs",
        choices=("mcs", "o3a"),
        default=None,
        metavar="METHOD",
        help=(
            "Rigidly align the ligands into a common frame before mapping. Bare --align uses "
            "'mcs' (maximum common substructure plus Kabsch, applied outward along a similarity "
            "tree); 'o3a' uses Open3DAlign, for sets with no substructure large enough to fit on. "
            "Off by default: ligands prepared together are already co-posed, and aligning those "
            "would hide a real pose problem rather than solve one."
        ),
    )
    group.add_argument(
        "--align-reference",
        metavar="LIGAND",
        help="Ligand whose frame the others are brought into (default: the one with the most heavy atoms).",
    )
    group.add_argument(
        "--align-min-atoms",
        type=int,
        default=3,
        metavar="N",
        help="Refuse to align a ligand on fewer than N corresponding heavy atoms (default: %(default)s).",
    )
    group.add_argument(
        "--write-aligned",
        type=Path,
        metavar="DIR",
        help="Write the aligned structures here, one SDF per ligand, for inspection.",
    )


def add_mapping_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the mapper-selection flags."""
    group = parser.add_argument_group("mapping")
    group.add_argument("--mapper", default="mcss-e2", help="Mapper plugin (default: %(default)s).")
    group.add_argument("--mapper-opt", action="append", metavar="K=V", help="Mapper option, repeatable.")
    group.add_argument(
        "--match-selection",
        choices=("fewest_fragments", "best_rmsd", "first"),
        default="fewest_fragments",
        help=(
            "How to resolve a symmetric common substructure (default: %(default)s). "
            "'first' reproduces the historical arbitrary choice."
        ),
    )
    group.add_argument(
        "--mcs-timeout", type=int, default=60, metavar="SEC", help="MCS search timeout (default: %(default)s)."
    )
    group.add_argument(
        "--distance-threshold",
        type=float,
        default=2.0,
        metavar="A",
        help="Geometric cutoff for geometry-based mappers, in angstroms (default: %(default)s).",
    )


def add_softcore_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the soft-core feasibility flags."""
    group = parser.add_argument_group("soft-core policy")
    group.add_argument(
        "--max-softcore-atoms",
        type=int,
        default=12,
        metavar="N",
        help="Reject an edge whose repaired soft-core exceeds N heavy atoms (default: %(default)s).",
    )
    group.add_argument(
        "--max-softcore-fraction",
        type=float,
        default=0.6,
        metavar="F",
        help="Reject when the soft-core exceeds this fraction of a molecule (default: %(default)s).",
    )
    group.add_argument(
        "--min-core-atoms",
        type=int,
        default=4,
        metavar="N",
        help="Reject when fewer than N heavy atoms remain in the common core (default: %(default)s).",
    )
    group.add_argument(
        "--min-mcs-fraction",
        type=float,
        default=0.35,
        metavar="F",
        help="Reject before repair when the core covers less than F of the smaller ligand (default: %(default)s).",
    )
    group.add_argument(
        "--core-rmsd-threshold",
        type=float,
        default=2.0,
        metavar="A",
        help="Reject when the mapped core's in-place RMSD exceeds this (default: %(default)s).",
    )
    group.add_argument(
        "--ring-policy",
        choices=("ring_system", "none"),
        default="ring_system",
        help="'ring_system' never leaves a ring half soft-core (default: %(default)s).",
    )
    group.add_argument(
        "--charge-change-policy",
        choices=("allow", "penalize", "reject"),
        default="penalize",
        help="How to treat a net charge change across an edge (default: %(default)s).",
    )


def add_network_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the network-selection flags."""
    group = parser.add_argument_group("network selection")
    add_compat_argument(group)  # type: ignore[arg-type]
    group.add_argument("--planner", default="mst", help="Planner plugin (default: %(default)s).")
    group.add_argument(
        "--pair-strategy",
        choices=("all_unordered_pairs", "all_pairs", "star", "linear", "explicit"),
        default="all_unordered_pairs",
        help="How candidate pairs are enumerated (default: %(default)s).",
    )
    group.add_argument("--hub", metavar="LIGAND", help="Hub ligand for star networks.")
    group.add_argument("--n-edges", type=int, metavar="N", help="Cap on selected edges. Must allow a spanning network.")
    group.add_argument(
        "--edges-per-ligand",
        type=int,
        default=2,
        metavar="N",
        help="Target minimum degree per ligand (default: %(default)s).",
    )
    group.add_argument(
        "--min-cycle-coverage",
        type=float,
        default=1.0,
        metavar="F",
        help="Target fraction of ligands lying on a cycle (default: %(default)s).",
    )
    group.add_argument("--forced-edge", action="append", metavar="A~B", help="Require this edge, repeatable.")
    group.add_argument("--banned-edge", action="append", metavar="A~B", help="Forbid this edge, repeatable.")
    group.add_argument("--explicit-edge", action="append", metavar="A~B", help="Edge for the 'explicit' strategy.")
    group.add_argument(
        "--allow-disconnected", action="store_true", help="Permit a network that does not span every ligand."
    )
    group.add_argument(
        "--edge-direction",
        choices=("fewer_softcore_first", "lexicographic", "heavier_second"),
        default="fewer_softcore_first",
        help="How selected edges are oriented (default: %(default)s).",
    )
    group.add_argument(
        "--prefilter",
        choices=("none", "fingerprint"),
        default="none",
        help="Similarity prefilter applied before mapping (default: %(default)s).",
    )
    group.add_argument("--prefilter-k", type=int, default=8, metavar="K", help="Neighbours kept per ligand.")
    group.add_argument(
        "--prefilter-min-tanimoto", type=float, default=0.4, metavar="F", help="Prefilter similarity floor."
    )
    group.add_argument(
        "--selection-objective",
        choices=("uniform_redundancy", "connectivity_then_cycles"),
        default="uniform_redundancy",
        help=(
            "How redundancy is added after the spanning network is built (default: %(default)s). "
            "'connectivity_then_cycles' prioritizes getting ligands onto at least one cycle."
        ),
    )
    group.add_argument(
        "--max-cycle-size",
        type=int,
        metavar="N",
        help="When improving cycle coverage, prefer cycles of at most N ligands.",
    )
    group.add_argument(
        "--pair-evaluation",
        choices=("eager", "adaptive"),
        default="eager",
        help=(
            "Map all candidates up front, or expand fingerprint-ranked batches until network targets are met "
            "(default: %(default)s)."
        ),
    )
    group.add_argument(
        "--adaptive-initial-neighbors",
        type=int,
        default=3,
        metavar="K",
        help="Nearest neighbours mapped per ligand in the first adaptive batch (default: %(default)s).",
    )
    group.add_argument(
        "--adaptive-batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Pairs mapped per adaptive expansion (default: %(default)s).",
    )
    group.add_argument(
        "--cbfe",
        choices=("off", "bridge", "cycles", "all"),
        default="off",
        help=(
            "Use counterpoised (CBFE) edges, which need no atom mapping and so are available between any "
            "two ligands: 'bridge' only to join subnetworks no feasible RBFE edge can reach, 'cycles' also "
            "to put ligands on a cycle, 'all' for an entirely counterpoised network (skips mapping "
            "altogether). Default: %(default)s."
        ),
    )
    group.add_argument(
        "--cbfe-base-cost",
        type=float,
        default=8.0,
        metavar="COST",
        help="Fixed cost of a CBFE edge, on the scorer's scale (default: %(default)s).",
    )
    group.add_argument(
        "--cbfe-atom-weight",
        type=float,
        default=0.05,
        metavar="W",
        help="Added to the CBFE base cost per heavy atom, summed over both ligands (default: %(default)s).",
    )
    group.add_argument(
        "--intermediates",
        choices=("off", "bridge", "gaps"),
        default="off",
        help=(
            "Invent bridging ligands for pairs no mapping can relate: 'bridge' only for pairs whose "
            "endpoints fall in different components of the feasible pool, 'gaps' also for infeasible "
            "pairs inside a component. The invented molecules are posed against their parents and their "
            "sub-edges go through the same feasibility checks as any other edge; a proposal whose "
            "sub-edges do not survive is dropped whole. Default: %(default)s."
        ),
    )
    group.add_argument(
        "--intermediate-generator",
        default="pairmap",
        metavar="NAME",
        help="Intermediate generator plugin (default: %(default)s). Constructed only when --intermediates is on.",
    )
    group.add_argument(
        "--max-intermediates",
        type=int,
        metavar="N",
        help="Cap on how many ligands one run may invent in total (default: no cap beyond the edge budget).",
    )
    group.add_argument(
        "--max-intermediate-gaps",
        type=int,
        metavar="N",
        help="Offer at most N gaps to the generator, most similar first (default: every gap).",
    )
    group.add_argument(
        "--intermediates-per-gap",
        type=int,
        default=4,
        metavar="N",
        help="Cap on molecules proposed for one gap (default: %(default)s).",
    )
    group.add_argument(
        "--intermediate-seed",
        type=int,
        default=0xF00D,
        metavar="SEED",
        help="Base RDKit seed for posing invented molecules (default: %(default)s).",
    )
    group.add_argument(
        "--intermediate-pose-attempts",
        type=int,
        default=10,
        metavar="N",
        help="Embedding attempts spent posing one invented molecule (default: %(default)s).",
    )
    group.add_argument(
        "--intermediate-pose-rmsd-factor",
        type=float,
        default=0.5,
        metavar="F",
        help=(
            "Accept an invented pose only below F * --core-rmsd-threshold (default: %(default)s). The "
            "sub-edges are gated on the full threshold anyway; this refuses a pose that would only just "
            "scrape through."
        ),
    )
    group.add_argument(
        "--intermediate-min-link-score",
        type=float,
        default=0.2,
        metavar="S",
        help=(
            "Lowest link score a generator may consider worth proposing, on a (0, 1] similarity scale "
            "(default: %(default)s). PairMap's MIN_SCORE."
        ),
    )
    group.add_argument(
        "--intermediate-max-dist",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Longest source-to-target path, in links, a generator may propose (default: %(default)s). "
            "At least 2: a one-link path is the direct edge that was already rejected. PairMap's MAX_DIST."
        ),
    )
    group.add_argument(
        "--intermediate-max-cycle",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Largest cycle a generator may build to give a proposed link a second, independent route "
            "(default: %(default)s). PairMap's MAX_CYCLE."
        ),
    )
    group.add_argument(
        "--intermediate-max-subgraph-dist",
        type=int,
        default=4,
        metavar="N",
        help=(
            "How far from either parent, in links, a molecule may sit and still join the proposed "
            "subnetwork (default: %(default)s). PairMap's MAX_SUBGRAPH_DIST."
        ),
    )
    group.add_argument(
        "--intermediate-beta",
        type=float,
        default=0.1,
        metavar="B",
        help=(
            "Decay rate of the exponential link score, per heavy atom changed (default: %(default)s). "
            "The same constant LOMAP's similarity uses."
        ),
    )
    group.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show pair-mapping progress (default: enabled on an interactive terminal).",
    )
    group.add_argument(
        "--consistency",
        choices=("pairwise", "graph"),
        default="pairwise",
        help="'graph' intersects each ligand's core across all its edges (default: %(default)s).",
    )
    group.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Worker threads for pair mapping (default: %(default)s). Mapping is partly "
        "pure Python, so speedup is sublinear and there is nothing to gain from setting "
        "this above the CPU count. Each worker also holds its own MCS search, so peak "
        "memory scales with this: roughly 40 MB per second of --mcs-timeout per job.",
    )


def build_alignment_options(args: argparse.Namespace) -> AlignmentOptions | None:
    """Assemble :class:`AlignmentOptions`, or ``None`` when ``--align`` was not given.

    Returning ``None`` rather than an options object with alignment switched off keeps the
    "not requested" case out of the library entirely: nothing downstream has to test for a
    do-nothing method.
    """
    method = getattr(args, "align", None)
    if method is None:
        return None
    return AlignmentOptions(method=method, reference=args.align_reference, min_mcs_atoms=args.align_min_atoms)


def build_mapping_options(args: argparse.Namespace) -> MappingOptions:
    """Assemble :class:`MappingOptions` from parsed arguments."""
    return MappingOptions(
        timeout=args.mcs_timeout,
        match_selection=args.match_selection,
        distance_threshold=args.distance_threshold,
        core_pruning=CorePruningPolicy(),
    )


def build_network_options(args: argparse.Namespace) -> NetworkOptions:
    """Assemble :class:`NetworkOptions` from parsed arguments."""
    softcore = SoftcorePolicy(
        ring_policy=args.ring_policy,
        max_softcore_atoms=args.max_softcore_atoms,
        max_softcore_fraction=args.max_softcore_fraction,
        min_core_atoms=args.min_core_atoms,
        min_mcs_fraction=args.min_mcs_fraction,
        core_rmsd_threshold=args.core_rmsd_threshold,
        charge_change_policy=args.charge_change_policy,
    )
    intermediates = IntermediateOptions(
        mode=args.intermediates,
        generator=args.intermediate_generator,
        max_intermediates=args.max_intermediates,
        max_gaps=args.max_intermediate_gaps,
        max_molecules=args.intermediates_per_gap,
        seed=args.intermediate_seed,
        max_pose_attempts=args.intermediate_pose_attempts,
        pose_rmsd_factor=args.intermediate_pose_rmsd_factor,
        min_link_score=args.intermediate_min_link_score,
        max_dist=args.intermediate_max_dist,
        max_cycle=args.intermediate_max_cycle,
        max_subgraph_dist=args.intermediate_max_subgraph_dist,
        beta=args.intermediate_beta,
    )
    return NetworkOptions(
        pair_strategy=args.pair_strategy,
        hub=args.hub,
        explicit_pairs=tuple(args.explicit_edge or ()),
        n_edges=args.n_edges,
        edges_per_ligand=args.edges_per_ligand,
        min_cycle_coverage=args.min_cycle_coverage,
        forced_edges=tuple(args.forced_edge or ()),
        banned_edges=tuple(args.banned_edge or ()),
        require_connected=not args.allow_disconnected,
        edge_direction=args.edge_direction,
        prefilter=args.prefilter,
        prefilter_k=args.prefilter_k,
        prefilter_min_tanimoto=args.prefilter_min_tanimoto,
        selection_objective=args.selection_objective,
        max_cycle_size=args.max_cycle_size,
        pair_evaluation=args.pair_evaluation,
        adaptive_initial_neighbors=args.adaptive_initial_neighbors,
        adaptive_batch_size=args.adaptive_batch_size,
        show_progress=sys.stderr.isatty() if args.progress is None else args.progress,
        jobs=args.jobs,
        consistency=args.consistency,
        cbfe_mode=args.cbfe,
        cbfe_base_cost=args.cbfe_base_cost,
        cbfe_atom_weight=args.cbfe_atom_weight,
        softcore=softcore,
        intermediates=intermediates,
        compat=getattr(args, "compat", None),
    )
