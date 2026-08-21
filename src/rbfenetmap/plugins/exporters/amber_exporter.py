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

Structures are written too, into ``ligands/``, and that is not a convenience
-----------------------------------------------------------------------------

``edges.dat`` names residues, and ``BuildEdges`` needs a parameterised topology for every
one of them. Before intermediate generation existed, every name in that file was a molecule
the user had supplied and could find on their own disk. It is not any more: an invented
ligand exists only inside the planned network, nobody has ever seen it, and an
``edges.dat`` naming one with no structure beside it is a setup that fails deep inside
someone else's tooling with an error about a missing residue.

So every ligand is written as ``ligands/<name>.sdf`` -- the real ones as well, because an
invariant that holds for the whole file ("every name in ``edges.dat`` has a structure in
``ligands/``") is one a script can check, while "every name except the ones you already
had" is not. Invented ligands are additionally listed in ``intermediates.txt`` with their
parents and the generator that proposed them, so a setup script can tell which residues
need parameterising before anything can run.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, ClassVar, Sequence

from rbfenetmap.core.exceptions import ExporterError
from rbfenetmap.core.meta.exporters import AbstractExporter
from rbfenetmap.core.models import EDGE_SEPARATOR, EdgeKind, Ligand, Network, Transformation
from rbfenetmap.io.amber_masks import DEFAULT_RESIDUE_NAMES, build_amber_masks

__all__ = ("AmberExporter",)

#: Subdirectory holding one SDF per ligand.
LIGAND_DIRECTORY = "ligands"

#: Manifest of the ligands this package invented: ``<name> <parent> <parent> <generator>``.
INTERMEDIATE_MANIFEST = "intermediates.txt"


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

        Also **warns** about invented ligands, or about generation merely being enabled
        when the pre-flight network has none yet. It is a warning rather than a refusal
        because an invented ligand is a correct result that carries an obligation: every
        one of them is a residue somebody has to parameterise, and the moment to learn
        that is before the run, not when ``BuildEdges`` fails on a residue nobody has ever
        seen.

        Raises
        ------
        rbfenetmap.core.exceptions.ExporterError
            Reporting every offending edge at once, not just the first.
        """
        synthetic = network.synthetic_ligands
        if synthetic:
            warnings.warn(
                f"{len(synthetic)} of {len(network.ligands)} ligand(s) were invented by this package and "
                f"need parameterising before the edges can run: {', '.join(item.name for item in synthetic)}. "
                f"Their structures are written to {LIGAND_DIRECTORY}/ and listed in {INTERMEDIATE_MANIFEST}.",
                UserWarning,
                stacklevel=2,
            )
        elif network.options is not None and network.options.generates_intermediates:
            # The pre-flight call happens before planning, so there are no synthetic
            # ligands to count yet -- but the fact that there may be some is knowable right
            # now, and that is the point of a pre-flight check.
            warnings.warn(
                "Intermediate generation is enabled, so the exported edge list may name residues that were "
                "not in the input. Their structures will be written to "
                f"{LIGAND_DIRECTORY}/ and listed in {INTERMEDIATE_MANIFEST}; each still needs parameterising.",
                UserWarning,
                stacklevel=2,
            )

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
            ``write_ligands`` -- write ``ligands/<name>.sdf`` for every ligand and, when
            any were invented, ``intermediates.txt``. Default ``True``. Turning it off is
            for a caller who is regenerating only the edge files over an export directory
            whose structures are already correct; it is **not** a way to shrink an export
            that names an invented ligand, which would produce a directory nobody can run.

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

        # Before the edge lists, so a partial export never leaves a file naming a residue
        # whose structure has not been written yet.
        if bool(options.get("write_ligands", True)):
            written += self._write_structures(network, destination)

        rbfe_dir = destination / "rbfe" if split else destination
        rbfe_dir.mkdir(parents=True, exist_ok=True)
        written += self._write_edge_lists(rbfe_dir, rbfe_edges)

        payloads: dict[str, dict[str, Any]] = {}
        for edge in rbfe_edges:
            payload = self._edge_payload(network, edge, residue_names)  # type: ignore[arg-type]
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
    def _write_structures(network: Network, destination: Path) -> list[Path]:
        """Write one SDF per ligand, plus the manifest of the invented ones.

        One file per ligand rather than one multi-record SDF, following the
        ``--write-aligned`` precedent in :mod:`rbfenetmap.cli.commands`: the consumer here
        is a setup script looking up a residue by name, and a name is a filename.

        The manifest is written only when there is something to put in it, so an all-real
        export gains no file that says "nothing happened". Its columns are
        ``<name> <parent> <parent> <generator>``, whitespace-separated like ``edges.dat``,
        because the script that reads one already parses the other positionally.
        """
        from rdkit import Chem

        directory = destination / LIGAND_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for name, ligand in network.ligands.items():
            path = directory / f"{name}.sdf"
            mol = Chem.Mol(ligand.mol)
            mol.SetProp("_Name", name)
            if ligand.provenance is not None:
                # Carried on the structure as well as in the manifest: an SDF that leaves
                # the export directory has to be able to say what it is on its own.
                mol.SetProp("rbfenet_synthetic", "1")
                mol.SetProp("rbfenet_parents", " ".join(ligand.provenance.parents))
                mol.SetProp("rbfenet_generator", ligand.provenance.generator)
                mol.SetProp("rbfenet_pose_rmsd", f"{ligand.provenance.pose_rmsd:.3f}")
            writer = Chem.SDWriter(str(path))
            writer.write(mol)
            writer.close()
            written.append(path)

        synthetic: Sequence[Ligand] = network.synthetic_ligands
        if synthetic:
            manifest = destination / INTERMEDIATE_MANIFEST
            manifest.write_text(
                "".join(
                    f"{ligand.name} {' '.join(ligand.provenance.parents)} {ligand.provenance.generator}\n"
                    for ligand in synthetic
                )
            )
            written.append(manifest)
        return written

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
