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

from rbfenetmap.core.options import AlignmentOptions, CorePruningPolicy, MappingOptions, NetworkOptions, SoftcorePolicy

__all__ = (
    "add_ligand_arguments",
    "add_mapping_arguments",
    "add_network_arguments",
    "add_softcore_arguments",
    "build_alignment_options",
    "build_mapping_options",
    "build_network_options",
    "parse_key_values",
)


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
        "this above the CPU count.",
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
    )
