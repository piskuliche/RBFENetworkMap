"""CLI command handlers.

Each ``cmd_*`` is a thin adapter: parse arguments into options objects, call the library,
format the result. Anything a handler needs to decide is a decision the library should be
making, so that an embedding program gets the same behaviour without going through
argparse.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from rbfenetmap.cli._args import build_alignment_options, build_mapping_options, build_network_options, parse_key_values
from rbfenetmap.core.cost import network_cost_summary
from rbfenetmap.core.diagnostics import edge_budget_advice, summarize
from rbfenetmap.core.exceptions import NetworkPlanError, RBFENetworkMapError
from rbfenetmap.core.models import EdgeKind, Ligand, Network, RejectionReason, Transformation, parse_edge_key
from rbfenetmap.core.pipeline import build_candidate, build_network, evaluate_pairs
from rbfenetmap.io.loaders import load_ligands

__all__ = ("cmd_diagnose", "cmd_export", "cmd_inspect", "cmd_map", "cmd_plan", "cmd_plugins", "cmd_report", "cmd_score")

logger = logging.getLogger(__name__)


def _load(args: argparse.Namespace) -> dict[str, Ligand]:
    """Load ligands, aligning them first if asked, and index them by name.

    Notes
    -----
    Alignment happens here rather than inside
    :func:`~rbfenetmap.core.pipeline.build_network` because ``score`` and ``map`` never call
    that function -- they go straight to the pair generator and ``build_candidate``. Putting
    it there would leave ``rbfenet plan --align`` and ``rbfenet score --align`` working from
    different coordinates for the same inputs. This is the one seam all three commands share.
    """
    ligands = load_ligands(args.ligands, name_property=args.name_property)
    if len(ligands) < 2:
        raise RBFENetworkMapError(
            f"Loaded {len(ligands)} ligand(s) from {[str(p) for p in args.ligands]}; a network needs "
            "at least two. Check the file format and that the molecules carry 3D conformers."
        )

    alignment_options = build_alignment_options(args)
    if alignment_options is None:
        return {ligand.name: ligand for ligand in ligands}

    from rbfenetmap.core.align import align_ligands

    result = align_ligands(ligands, options=alignment_options, mapping_options=build_mapping_options(args))
    _report_alignment(result)
    if getattr(args, "write_aligned", None):
        _write_aligned(result, Path(args.write_aligned))
    return {ligand.name: ligand for ligand in result.ligands}


def _report_alignment(result) -> None:
    """Print the per-ligand alignment table and its caveats.

    Everything goes to stderr. ``score --format json`` writes a document to stdout that a
    caller may well be piping into something, and a table in the middle of it would be a
    parse error rather than a cosmetic problem.
    """
    print(
        f"Aligned {len(result.ligands)} ligand(s) into the frame of {result.reference} ({_alignment_method(result)}).\n",
        file=sys.stderr,
    )
    rows = [
        [
            record.name,
            record.reference or "-",
            str(record.n_fit_atoms),
            "-" if record.reference is None else f"{record.rmsd:.3f}",
            record.note or ("reference" if record.reference is None else ""),
        ]
        for record in result.records
    ]
    print(_format_table(rows, ["ligand", "reference", "fit atoms", "rmsd", "note"]), file=sys.stderr)

    if result.failures:
        print(
            f"\nWARNING: {len(result.failures)} ligand(s) could not be aligned and remain in their own frame:",
            file=sys.stderr,
        )
        for record in result.failures:
            print(f"  - {record.name}: {record.note}.", file=sys.stderr)
        print(
            "  Every edge touching them will be rejected for geometry. Supply them co-posed, or try --align o3a.",
            file=sys.stderr,
        )

    print(
        "\nAligned ligands are not constrained-embedded: each conformer was relaxed independently, "
        f"so some residual core RMSD is expected. The median post-alignment fit was {result.median_rmsd:.3f} A. "
        "If edges are rejected with core_geometry_mismatch, look at the actual core_rmsd values with "
        "`rbfenet score --show-rejected --explain` before raising --core-rmsd-threshold.",
        file=sys.stderr,
    )


def _alignment_method(result) -> str:
    """The method actually used, for the report header."""
    used = {record.method for record in result.records if record.method not in ("reference", "none")}
    return ", ".join(sorted(used)) if used else "none"


def _write_aligned(result, directory: Path) -> list[Path]:
    """Write each aligned ligand to its own SDF under *directory*.

    Notes
    -----
    One file per ligand rather than one multi-record file. PyMOL loads a multi-record SDF as
    a single object with N states, which is exactly the wrong shape for the question the user
    is asking -- whether the ligands overlay. ``load aligned/*.sdf`` gives N objects on
    screen at once.
    """
    from rdkit import Chem

    directory.mkdir(parents=True, exist_ok=True)
    records = {record.name: record for record in result.records}
    written: list[Path] = []
    for ligand in result.ligands:
        record = records[ligand.name]
        mol = Chem.Mol(ligand.mol)
        mol.SetProp("_Name", ligand.name)
        mol.SetProp("rbfenet_align_reference", str(record.reference or ligand.name))
        mol.SetProp("rbfenet_align_method", record.method)
        mol.SetProp("rbfenet_align_fit_atoms", str(record.n_fit_atoms))
        mol.SetProp("rbfenet_align_rmsd", f"{record.rmsd:.4f}")
        path = directory / f"{ligand.name}.sdf"
        with Chem.SDWriter(str(path)) as writer:
            writer.write(mol)
        written.append(path)
    print(f"\nWrote {len(written)} aligned structure(s) to {directory}", file=sys.stderr)
    return written


def _geometry_hint(args: argparse.Namespace, rejected: Sequence[Transformation], selected: int) -> str | None:
    """Advice to offer when the run looks like a frame problem rather than a pose one.

    Returns
    -------
    str or None
        ``None`` unless nothing was selected and the geometry gate accounts for most of the
        rejections -- the signature of a ligand set whose members were prepared apart from
        one another. A partial failure is a chemistry question and gets no hint.
    """
    if getattr(args, "align", None) is not None or selected or not rejected:
        return None
    geometry = sum(1 for c in rejected if RejectionReason.CORE_GEOMETRY_MISMATCH in c.score.rejections)
    if geometry * 2 <= len(rejected):
        return None
    return (
        f"Hint: {geometry} of {len(rejected)} candidate(s) were rejected for core_geometry_mismatch "
        "and nothing was selected. That is what a ligand set prepared in separate frames looks "
        "like -- each structure sitting wherever its own box put it. Re-run with --align to "
        "superpose them into a common frame first. If the frames really are common and only the "
        "conformers differ, raise --core-rmsd-threshold instead."
    )


def _hint_if_geometry_rejected(args: argparse.Namespace, rejected: Sequence[Transformation], selected: int) -> None:
    """Print :func:`_geometry_hint` to stderr, if there is one to give."""
    hint = _geometry_hint(args, rejected, selected)
    if hint:
        print(f"\n{hint}", file=sys.stderr)


def _make_scorer(args: argparse.Namespace):
    """Build the scorer, applying ``--weights`` / ``--weights-file``."""
    from rbfenetmap.plugins.scorers import create_scorer

    weights = parse_key_values(getattr(args, "weights", None))
    weights_file = getattr(args, "weights_file", None)
    if weights_file:
        loaded = json.loads(Path(weights_file).read_text())
        loaded.update(weights)
        weights = loaded
    if not weights:
        return create_scorer(args.scorer)
    # The keyword differs by scorer, and passing the wrong one should say so rather than
    # be silently swallowed by **kwargs.
    keyword = "parameters" if args.scorer == "lomaplike" else "weights"
    return create_scorer(args.scorer, **{keyword: weights})


def _format_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    """Render a fixed-width text table."""
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    lines = ["  ".join(str(h).ljust(w) for h, w in zip(headers, widths)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows]
    return "\n".join(lines)


def _run_exports(network: Network, names: Sequence[str], directory: Path, options: dict) -> list[Path]:
    """Run the named exporters into *directory*."""
    from rbfenetmap.plugins.exporters import create_exporter

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in names:
        exporter = create_exporter(name)
        written.extend(exporter.export(network, directory, **options))
    return written


def _cost_line(network: Network, units: str) -> str:
    """One line stating what the network costs, in the requested units.

    Both figures are always computed; only the phrasing changes. They answer different
    questions -- the scorer total says how hard the edges are, the GPU-hour figure says
    whether the run fits in an allocation -- so neither is a conversion of the other and
    reporting the wrong one silently is the failure worth avoiding.
    """
    totals = network_cost_summary(network)
    if units == "gpu_hours":
        return f"cost {totals['gpu_hours']:.1f} GPU-hours (about ${totals['price']:.2f})"
    return f"cost {totals['score']:.3f} (scorer units)"


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Report network-level metrics for an already-planned network.

    ``inspect`` is per-edge; this is per-network. The two are separate commands rather
    than flags on one because they answer questions at different scales, and a table that
    tried to do both would serve neither.
    """
    from rbfenetmap.io.networkio import load_network

    network = load_network(Path(args.network))
    report = summarize(
        network,
        seed=args.seed,
        failure_rate=args.failure_rate,
        n_repeats=args.repeats,
        max_cycle_length=args.max_cycle_length,
    )
    degrees, robustness, budget = report["degrees"], report["robustness"], report["budget"]
    totals = network_cost_summary(network)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "n_ligands": report["n_ligands"],
                    "n_edges": report["n_edges"],
                    "n_rbfe": report["n_rbfe"],
                    "n_cbfe": report["n_cbfe"],
                    "cost": report["cost"],
                    "efficiency": report["efficiency"],
                    "gpu_hours": totals["gpu_hours"],
                    "price": totals["price"],
                    "n_cycles": report["n_cycles"],
                    "max_cycle_length": report["max_cycle_length"],
                    "degree": {
                        "min": degrees.minimum,
                        "mean": degrees.mean,
                        "max": degrees.maximum,
                        "isolated": list(degrees.isolated),
                        "per_ligand": degrees.degrees,
                    },
                    "diameter": report["diameter"],
                    "robustness": {
                        "connected_fraction": robustness.connected_fraction,
                        "mean_ligands_retained": robustness.mean_ligands_retained,
                        "failure_rate": robustness.failure_rate,
                        "n_repeats": robustness.n_repeats,
                        "seed": robustness.seed,
                    },
                    "edge_budget": {
                        "recommended": budget.recommended,
                        "shortfall": budget.shortfall,
                        "message": budget.message,
                    },
                },
                indent=2,
            )
        )
        return 0

    rows = [
        ["ligands", str(report["n_ligands"])],
        ["edges", f"{report['n_edges']} ({report['n_rbfe']} RBFE, {report['n_cbfe']} CBFE)"],
        ["cost", _cost_line(network, args.cost_units).removeprefix("cost ")],
        ["efficiency", f"{report['efficiency']:.3f} per edge (scorer units)"],
        ["degree", f"min {degrees.minimum}, mean {degrees.mean:.2f}, max {degrees.maximum}"],
        ["isolated", ", ".join(degrees.isolated) or "none"],
        ["diameter", "disconnected" if report["diameter"] is None else str(report["diameter"])],
        ["cycles", f"{report['n_cycles']} of length <= {report['max_cycle_length']}"],
        [
            "robustness",
            f"{robustness.connected_fraction:.0%} of {robustness.n_repeats} trials stay connected at "
            f"{robustness.failure_rate:.0%} edge failure; {robustness.mean_ligands_retained:.1f} ligands "
            f"retained on average (seed {robustness.seed})",
        ],
    ]
    print(_format_table(rows, ["metric", "value"]))
    print(f"\n{budget.message}")
    if network.unmet_constraints:
        print("\nUnmet constraints:", file=sys.stderr)
        for constraint in network.unmet_constraints:
            print(f"  - {constraint}", file=sys.stderr)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Plan a network and write it out."""
    ligands = _load(args)
    network_options = build_network_options(args)
    mapping_options = build_mapping_options(args)
    scorer = _make_scorer(args)

    if args.validate_exporter:
        # Deliberately before the expensive mapping stage: a format constraint that is
        # knowable from the inputs should not cost a full planning run to discover.
        from rbfenetmap.plugins.exporters import create_exporter

        # Carrying the options matters: some format constraints depend on what was asked
        # for rather than on what was planned, and cbfe_mode is knowable right here.
        create_exporter(args.validate_exporter).validate(
            Network(ligands=ligands, edges=(), planner="preflight", options=network_options)
        )

    try:
        network = build_network(
            ligands,
            mapper=args.mapper,
            scorer=scorer,
            planner=args.planner,
            mapping_options=mapping_options,
            network_options=network_options,
        )
    except NetworkPlanError as exc:
        # A ligand set in mixed frames fails here rather than returning an empty network, so
        # the hint has to travel out through the refusal or it never reaches the user at all
        # -- which is exactly the case it exists for. Appending it to the message rather than
        # printing alongside keeps what-happened before what-to-do, and leaves `main` the
        # single place that decides between a one-line error and a traceback.
        hint = _geometry_hint(args, exc.rejected, selected=0)
        if hint is None:
            raise
        raise NetworkPlanError(f"{exc}\n\n{hint}", rejected=exc.rejected) from exc

    from rbfenetmap.io.networkio import dump_network

    out = Path(args.out)
    dump_network(network, out)

    summary = f"Planned {len(network.edges)} edge(s) over {len(network.ligands)} ligand(s) -> {out}"
    if network.cbfe_edges:
        summary += f"\n  {len(network.rbfe_edges)} RBFE, {len(network.cbfe_edges)} CBFE"
    summary += "\n  " + _cost_line(network, getattr(args, "cost_units", "score"))
    print(summary)
    # Advice, not a warning: with edges_per_ligand=2 the default network is always below
    # the n*ln(n) floor, so warnings.warn would fire on every run this package has ever
    # planned and be tuned out within a week. See core.diagnostics.
    print(f"  {edge_budget_advice(len(network.ligands), len(network.edges)).message}")
    rows = [
        [
            edge.key,
            edge.kind.value,
            f"{edge.score.total:.3f}",
            f"{edge.mapping.n_softcore_1}/{edge.mapping.n_softcore_2}",
            str(edge.mapping.n_common_core),
            "yes" if edge.repair.applied else "",
        ]
        for edge in sorted(network.edges, key=lambda e: e.score.total)
    ]
    print(_format_table(rows, ["edge", "kind", "cost", "soft-core", "core", "repaired"]))

    if network.unmet_constraints:
        print("\nUnmet constraints:", file=sys.stderr)
        for constraint in network.unmet_constraints:
            print(f"  - {constraint}", file=sys.stderr)

    if args.show_rejected and network.rejected:
        print(f"\n{len(network.rejected)} rejected candidate(s):")
        rejected_rows = [
            [c.key, ", ".join(r.value for r in c.score.rejections)]
            for c in sorted(network.rejected, key=lambda c: c.key)
        ]
        print(_format_table(rejected_rows, ["edge", "reason"]))

    _hint_if_geometry_rejected(args, network.rejected, selected=len(network.edges))

    if args.export:
        written = _run_exports(network, args.export, Path(args.export_dir), parse_key_values(args.exporter_opt))
        print(f"\nWrote {len(written)} export file(s) to {args.export_dir}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score candidate edges and print a ranked table, without selecting a network."""
    from rbfenetmap.core.pairs import generate_candidate_pairs
    from rbfenetmap.plugins.mappers import create_mapper

    ligands = _load(args)
    network_options = build_network_options(args)
    pairs, _ = generate_candidate_pairs(ligands, network_options)
    candidates = evaluate_pairs(
        ligands, pairs, create_mapper(args.mapper), _make_scorer(args), build_mapping_options(args), network_options
    )

    shown = [c for c in candidates if c.feasible or args.show_rejected]
    shown.sort(key=lambda c: (not c.feasible, c.score.total, c.key))
    if args.top:
        shown = shown[: args.top]

    _hint_if_geometry_rejected(args, [c for c in candidates if not c.feasible], selected=0)

    if args.format == "json":
        print(
            json.dumps(
                [
                    {
                        "edge": c.key,
                        "cost": c.score.total if c.feasible else None,
                        "feasible": c.feasible,
                        "rejections": [r.value for r in c.score.rejections],
                        "descriptors": dict(c.score.descriptors),
                        "contributions": dict(c.score.contributions),
                    }
                    for c in shown
                ],
                indent=2,
            )
        )
        return 0

    headers = ["edge", "cost", "soft-core", "core", "status"]
    terms = sorted({k for c in shown for k in c.score.contributions}) if args.explain else []
    headers += terms

    rows = []
    for candidate in shown:
        row = [
            candidate.key,
            f"{candidate.score.total:.3f}" if candidate.feasible else "inf",
            f"{candidate.mapping.n_softcore_1}/{candidate.mapping.n_softcore_2}",
            str(candidate.mapping.n_common_core),
            "ok" if candidate.feasible else ", ".join(r.value for r in candidate.score.rejections),
        ]
        row += [f"{candidate.score.contributions.get(t, 0.0):.3f}" for t in terms]
        rows.append(row)

    if args.format == "csv":
        print(",".join(headers))
        for row in rows:
            print(",".join(f'"{c}"' if "," in str(c) else str(c) for c in row))
    else:
        print(_format_table(rows, headers))
    return 0


def cmd_map(args: argparse.Namespace) -> int:
    """Compute and report mappings for specific pairs, without planning."""
    from rbfenetmap.plugins.mappers import create_mapper

    ligands = _load(args)
    network_options = build_network_options(args)
    mapper = create_mapper(args.mapper)
    scorer = _make_scorer(args)

    if args.pair:
        pairs = [parse_edge_key(spec) for spec in args.pair]
    else:
        from rbfenetmap.core.pairs import generate_candidate_pairs

        pairs, _ = generate_candidate_pairs(ligands, network_options)

    results: list[Transformation] = []
    for source, target in pairs:
        for name in (source, target):
            if name not in ligands:
                raise RBFENetworkMapError(f"Unknown ligand {name!r}. Loaded: {sorted(ligands)}.")
        results.append(
            build_candidate(
                ligands[source], ligands[target], mapper, scorer, build_mapping_options(args), network_options
            )
        )

    rows = [
        [
            r.key,
            str(r.mapping.n_common_core),
            f"{r.mapping.n_softcore_1}/{r.mapping.n_softcore_2}",
            f"{r.repair.n_fragments_before} -> {r.repair.n_fragments_after}",
            "ok" if r.feasible else ", ".join(x.value for x in r.score.rejections),
        ]
        for r in results
    ]
    print(_format_table(rows, ["edge", "core", "soft-core", "regions", "status"]))

    if args.draw:
        from rbfenetmap.viz.depict import render_edge_svg

        directory = Path(args.draw)
        directory.mkdir(parents=True, exist_ok=True)
        for result in results:
            source_svg, target_svg = render_edge_svg(result, ligands, show_indices=True)
            (directory / f"{result.key}_src.svg").write_text(source_svg)
            (directory / f"{result.key}_dst.svg").write_text(target_svg)
        print(f"\nWrote {2 * len(results)} SVG file(s) to {directory}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export an already-planned network."""
    from rbfenetmap.io.networkio import load_network

    network = load_network(Path(args.network))
    written = _run_exports(network, args.exporter, Path(args.dest), parse_key_values(args.exporter_opt))
    for path in written:
        print(path)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Render a self-contained HTML report of a planned network."""
    from rbfenetmap.io.networkio import load_network
    from rbfenetmap.viz.gallery import render_report

    network = load_network(Path(args.network))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(network, title=args.title, show_indices=args.show_indices))
    print(f"Wrote {out}")
    return 0


def cmd_plugins(args: argparse.Namespace) -> int:
    """List registered plugins and their availability."""
    from rbfenetmap.plugins.exporters import BUILTIN_EXPORTERS
    from rbfenetmap.plugins.mappers import BUILTIN_MAPPERS
    from rbfenetmap.plugins.planners import BUILTIN_PLANNERS
    from rbfenetmap.plugins.scorers import BUILTIN_SCORERS

    tables = {
        "mapper": BUILTIN_MAPPERS,
        "scorer": BUILTIN_SCORERS,
        "planner": BUILTIN_PLANNERS,
        "exporter": BUILTIN_EXPORTERS,
    }
    kinds = [args.kind] if args.kind else list(tables)

    for kind in kinds:
        rows = []
        for name, spec in sorted(tables[kind].items()):
            if not spec.available and not args.all:
                continue
            status = "ok" if spec.available else f"needs {', '.join(spec.missing_requirements)}"
            rows.append([name, status, spec.description])
        if rows:
            print(f"\n{kind}s")
            print(_format_table(rows, ["name", "status", "description"]))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Show everything known about one edge of a planned network."""
    from rbfenetmap.io.networkio import load_network

    network = load_network(Path(args.network))
    source, target = parse_edge_key(args.edge)
    wanted = tuple(sorted((source, target)))

    edge = next((e for e in (*network.edges, *network.candidates) if e.unordered_key == wanted), None)
    if edge is None:
        known = sorted({e.key for e in network.edges})
        raise RBFENetworkMapError(f"Edge {args.edge!r} is not in {args.network}. Selected edges: {known}.")

    selected = any(e.unordered_key == wanted for e in network.edges)
    print(f"edge          {edge.key}")
    print(f"kind          {edge.kind.value}")
    print(f"selected      {'yes' if selected else 'no'}")
    print(f"mapper        {edge.mapping.method}")
    print(f"common core   {edge.mapping.n_common_core} pair(s)")
    print(f"soft-core     {edge.mapping.n_softcore_1} / {edge.mapping.n_softcore_2} atom(s)")
    print(f"regions       {edge.repair.n_fragments_before} -> {edge.repair.n_fragments_after}")
    print(f"cost          {edge.score.total if edge.feasible else 'inf (rejected)'}")
    if edge.score.rejections:
        print(f"rejections    {', '.join(r.value for r in edge.score.rejections)}")

    if args.show_repair_trace and edge.repair.trace:
        print("\nrepair trace")
        for line in edge.repair.trace:
            print(f"  {line}")

    if args.show_descriptors and edge.score.descriptors:
        print("\ndescriptors")
        for key, value in sorted(edge.score.descriptors.items()):
            print(f"  {key:28s} {value:.4f}")
        if edge.score.contributions:
            print("\ncost contributions")
            for key, value in sorted(edge.score.contributions.items(), key=lambda kv: -kv[1]):
                print(f"  {key:28s} {value:.4f}")

    if args.show_masks:
        if edge.kind is EdgeKind.CBFE:
            # Printing masks here would print a mask that describes a different, and
            # runnable, calculation. See AmberExporter for the same refusal.
            print("\namber masks    not applicable: a CBFE edge decouples both ligands and has no atom mapping.")
        else:
            from rbfenetmap.io.amber_masks import build_amber_masks

            masks = build_amber_masks(network.ligands[edge.source], network.ligands[edge.target], edge.mapping)
            print("\namber masks")
            for key, value in masks.as_dict().items():
                print(f"  {key:10s} {value or '(empty)'}")

    if args.draw:
        from rbfenetmap.viz.depict import render_edge_svg

        source_svg, target_svg = render_edge_svg(edge, network.ligands, show_indices=True)
        out = Path(args.draw)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "<!doctype html><meta charset='utf-8'><div style='display:flex;gap:1rem;flex-wrap:wrap'>"
            f"{source_svg}{target_svg}</div>"
        )
        print(f"\nWrote {out}")
    return 0
