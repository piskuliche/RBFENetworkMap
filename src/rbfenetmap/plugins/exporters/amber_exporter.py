"""Amber / amberstudio exporter.

Writes an ``edges.dat`` list plus one ``atommap_<src>~<dst>.runconfig`` YAML per edge, in
the layout ``amberstudio``'s ``BuildEdges`` produces and ``guimapper`` edits. That file
format is the interoperability contract between this package and the existing tooling:
plan a network here, hand-edit any edge in guimapper, run it in amberstudio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from rbfenetmap.core.exceptions import ExporterError
from rbfenetmap.core.meta.exporters import AbstractExporter
from rbfenetmap.core.models import Network, Transformation
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
            try:
                build_amber_masks(network.ligands[edge.source], network.ligands[edge.target], edge.mapping)
            except ExporterError as exc:
                problems.append(f"{edge.key}: {exc}")
        if problems:
            raise ExporterError(
                f"{len(problems)} edge(s) cannot produce valid Amber masks:\n  " + "\n  ".join(problems)
            )

    def export(self, network: Network, destination: Path, **options: Any) -> tuple[Path, ...]:
        """Write ``edges.dat`` and one runconfig per edge into *destination*.

        Parameters
        ----------
        network : Network
        destination : pathlib.Path
            Directory, created if absent.
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
        written: list[Path] = []

        edges_path = destination / "edges.dat"
        edges_path.write_text("\n".join(f"{e.source} {e.target}" for e in network.edges) + "\n")
        written.append(edges_path)

        payloads: dict[str, dict[str, Any]] = {}
        for edge in network.edges:
            payload = self._edge_payload(network, edge, residue_names)  # type: ignore[arg-type]
            payloads[edge.key] = payload
            path = destination / f"atommap_{edge.key}{self.default_suffix}"
            path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
            written.append(path)

        if aggregate:
            path = destination / f"atommaps{self.default_suffix}"
            path.write_text(yaml.safe_dump(payloads, sort_keys=False, default_flow_style=False))
            written.append(path)

        return tuple(written)

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
