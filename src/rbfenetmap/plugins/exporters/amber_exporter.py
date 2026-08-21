"""Amber / amberstudio exporter.

Writes an ``edges.dat`` list plus one ``atommap_<src>~<dst>.runconfig`` YAML per edge, in
the layout ``amberstudio``'s ``BuildEdges`` produces and ``guimapper`` edits. That file
format is the interoperability contract between this package and the existing tooling:
plan a network here, hand-edit any edge in guimapper, run it in amberstudio.

Mixed RBFE/CBFE networks are written into ``rbfe/`` and ``cbfe/`` subdirectories, because
``BuildEdges`` takes its ``alchemical_mode`` per *invocation* rather than per edge: a
network containing both kinds is two ``BuildEdges`` runs, and the export mirrors that
rather than producing a single directory neither run can consume. Each subdirectory also
carries an ``edges.txt`` in amberstudio's own ``<src>~<dst>`` form. A CBFE edge needs
nothing beyond that line -- amberstudio synthesizes its masks from the edge name, since
there is no mapping to convey -- so ``cbfe/`` contains only the edge list.

A network that is entirely RBFE keeps the flat, historical layout, so existing callers see
no change.

When ``design_total_ns`` is set, each runconfig also carries a ``sample_allocation`` block:
the A-optimal share of the total simulation budget for that edge, and a lambda-window count
scaled to it. See :meth:`AmberExporter._sample_allocation` for why that computation lives
here rather than in the planner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Sequence

from rbfenetmap.core.exceptions import ExporterError
from rbfenetmap.core.meta.exporters import AbstractExporter
from rbfenetmap.core.models import EDGE_SEPARATOR, EdgeKind, Network, Transformation
from rbfenetmap.io.amber_masks import DEFAULT_RESIDUE_NAMES, build_amber_masks

__all__ = ("AmberExporter",)


class AmberExporter(AbstractExporter):
    """Write amberstudio-compatible edge and atom-map files.

    Requires ``pyyaml``.
    """

    name: ClassVar[str] = "amber"
    default_suffix: ClassVar[str] = ".runconfig"

    def validate(self, network: Network) -> None:
        """Check every selected edge can produce valid Amber masks.

        Called early by ``rbfenet plan --validate-exporter amber``. Without it, an atom
        name collision only surfaces after the whole mapping and planning run has
        completed -- which for a large series is many minutes of work discarded over a
        problem that was knowable from the inputs alone.

        Raises
        ------
        rbfenetmap.core.exceptions.ExporterError
            Reporting every offending edge at once, not just the first.
        """
        problems: list[str] = []
        for edge in network.edges:
            if edge.kind is EdgeKind.CBFE:
                continue  # no mapping, therefore no masks to check
            try:
                build_amber_masks(network.ligands[edge.source], network.ligands[edge.target], edge.mapping)
            except ExporterError as exc:
                problems.append(f"{edge.key}: {exc}")
        if problems:
            raise ExporterError(
                f"{len(problems)} edge(s) cannot produce valid Amber masks:\n  " + "\n  ".join(problems)
            )

    def export(self, network: Network, destination: Path, **options: Any) -> tuple[Path, ...]:
        """Write the edge lists and one runconfig per RBFE edge into *destination*.

        Parameters
        ----------
        network : Network
        destination : pathlib.Path
            Directory, created if absent. A network with counterpoised edges is written
            into ``rbfe/`` and ``cbfe/`` subdirectories of it; an all-RBFE network is
            written flat, as before.
        **options
            ``residue_names`` -- the two residue names, default ``("SRC", "DST")``.
            ``aggregate`` -- also write a single ``atommaps.runconfig`` keyed by edge,
            which guimapper can open as a multi-edge document.

        Returns
        -------
        tuple[pathlib.Path, ...]
        """
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ExporterError(
                "The 'amber' exporter requires PyYAML. Install it with `pip install rbfe-network-map[amber]`."
            ) from exc

        residue_names = tuple(options.get("residue_names", DEFAULT_RESIDUE_NAMES))
        aggregate = bool(options.get("aggregate", True))

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        rbfe_edges = network.rbfe_edges
        cbfe_edges = network.cbfe_edges
        split = bool(cbfe_edges)
        written: list[Path] = []

        rbfe_dir = destination / "rbfe" if split else destination
        rbfe_dir.mkdir(parents=True, exist_ok=True)
        written += self._write_edge_lists(rbfe_dir, rbfe_edges)

        allocation = self._sample_allocation(network)

        payloads: dict[str, dict[str, Any]] = {}
        for edge in rbfe_edges:
            payload = self._edge_payload(network, edge, residue_names)  # type: ignore[arg-type]
            if edge.unordered_key in allocation:
                payload["sample_allocation"] = allocation[edge.unordered_key]
            payloads[edge.key] = payload
            path = rbfe_dir / f"atommap_{edge.key}{self.default_suffix}"
            path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
            written.append(path)

        if aggregate:
            path = rbfe_dir / f"atommaps{self.default_suffix}"
            path.write_text(yaml.safe_dump(payloads, sort_keys=False, default_flow_style=False))
            written.append(path)

        if split:
            # An edge list and nothing else. amberstudio's CBFE mode never loads the
            # ligands and builds its masks from the residue roles alone, so a runconfig
            # written here would carry no information and be overwritten regardless.
            cbfe_dir = destination / "cbfe"
            cbfe_dir.mkdir(parents=True, exist_ok=True)
            written += self._write_edge_lists(cbfe_dir, cbfe_edges)

        return tuple(written)

    @staticmethod
    def _sample_allocation(network: Network) -> dict[tuple[str, str], dict[str, float | int]]:
        """Return the per-edge lambda-window and nanosecond budget, or ``{}`` if unrequested.

        Parameters
        ----------
        network : Network

        Returns
        -------
        dict[tuple[str, str], dict[str, float | int]]
            Keyed by unordered endpoint pair. Each value carries ``simulation_ns``, the
            A-optimal share of ``design_total_ns``; ``lambda_windows``, that share mapped
            onto the requested window range; and ``predicted_sigma_kcal``, the cost the
            allocation was derived from, so a reader can see what it was based on.

        Notes
        -----
        The runconfig is the natural home for this. It is already written once per edge,
        it is already the interoperability contract with amberstudio and guimapper, and a
        second file keyed by edge would have to be kept in step with it by hand.

        Computed here rather than in the planner deliberately: an allocation is a statement
        about *how to run* the network, not about which edges it contains, and computing it
        at export time means a network read back from JSON months later can still be
        allocated -- against a different budget, without replanning.

        Returns ``{}`` rather than a uniform split when ``design_total_ns`` is unset. A
        uniform split is a real decision about how to spend machine time, and writing one
        into every runconfig by default would silently override whatever the user's own
        protocol said.

        The allocation reads :attr:`EdgeScore.total
        <rbfenetmap.core.models.EdgeScore.total>` as a standard deviation in kcal/mol,
        which is true of the ``variance`` scorer and of no other. Under a different scorer
        the numbers are still internally consistent -- an edge the scorer disliked gets
        more time -- but they are not variances, and the twofold variance reduction the
        method promises is not on offer.
        """
        options = network.options
        if options is None or options.design_total_ns is None or not network.edges:
            return {}

        from rbfenetmap.core.design import allocate_effort

        nodes = sorted(network.ligands)
        pairs = [edge.unordered_key for edge in network.edges]
        sigmas = [max(float(edge.score.total), 1e-9) for edge in network.edges]
        try:
            effort = allocate_effort(nodes, pairs, sigmas, total=options.design_total_ns)
        except ValueError:
            # A disconnected network has an unbounded criterion, so there is no optimal
            # allocation to compute. Exporting is still perfectly valid -- the user asked
            # for --allow-disconnected somewhere upstream -- so omit the block rather than
            # failing an export over an optional annotation.
            return {}

        values = list(effort.values())
        low, high = min(values), max(values)
        span = high - low
        windows_low, windows_high = options.design_lambda_min, options.design_lambda_max
        allocation: dict[tuple[str, str], dict[str, float | int]] = {}
        for pair, sigma in zip(pairs, sigmas):
            share = effort[pair]
            fraction = (share - low) / span if span > 0 else 0.0
            allocation[pair] = {
                "lambda_windows": int(round(windows_low + fraction * (windows_high - windows_low))),
                "simulation_ns": round(float(share), 4),
                "predicted_sigma_kcal": round(float(sigma), 4),
            }
        return allocation

    @staticmethod
    def _write_edge_lists(directory: Path, edges: Sequence[Transformation]) -> list[Path]:
        """Write both edge-list spellings into *directory*.

        ``edges.txt`` is what amberstudio reads: one ``<src>~<dst>`` per line, discovered
        by prefix and suffix. ``edges.dat`` is the space-separated form this exporter has
        always written, kept because downstream scripts parse it positionally and dropping
        it would break them for no gain.
        """
        written: list[Path] = []

        txt = directory / "edges.txt"
        txt.write_text("".join(f"{e.source}{EDGE_SEPARATOR}{e.target}\n" for e in edges))
        written.append(txt)

        dat = directory / "edges.dat"
        dat.write_text("".join(f"{e.source} {e.target}\n" for e in edges))
        written.append(dat)
        return written

    @staticmethod
    def _edge_payload(network: Network, edge: Transformation, residue_names: tuple[str, str]) -> dict[str, Any]:
        """Build the runconfig mapping for one edge.

        ``atommapindices`` and ``atommapnames`` are both written. The indices are what a
        program should read; the names are what a human reviewing a diff can actually
        check, and are what guimapper displays.
        """
        source = network.ligands[edge.source]
        target = network.ligands[edge.target]
        masks = build_amber_masks(source, target, edge.mapping, residue_names=residue_names)

        names_1 = source.atom_names
        names_2 = target.atom_names
        return {
            "atommapindices": [[int(a), int(b)] for a, b in zip(edge.mapping.cc1, edge.mapping.cc2)],
            "atommapnames": [[names_1[a], names_2[b]] for a, b in zip(edge.mapping.cc1, edge.mapping.cc2)],
            **masks.as_dict(),
            "softcore1": [names_1[i] for i in edge.mapping.sc1],
            "softcore2": [names_2[i] for i in edge.mapping.sc2],
            "mapping_method": edge.mapping.method,
            "softcore_repaired": bool(edge.repair.applied),
        }
