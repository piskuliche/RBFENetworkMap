#!/usr/bin/env python
"""Recover a network from ligands that were prepared in separate frames.

Run ``python examples/data/make_conformers.py`` first to build the ligand file.

This script manufactures the situation deliberately: it takes the co-posed example series
and pushes each member into a frame of its own, which is what a set prepared individually
for ABFE runs and converted from separate Amber topologies looks like. The conformers stay
untouched -- only the frames disagree -- so the failure that follows is purely a frame
problem, and alignment is enough to fix it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from rbfenetmap.core.align import align_ligands
from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.models import Ligand
from rbfenetmap.core.pipeline import build_network
from rbfenetmap.io.loaders import load_ligands

DATA = Path(__file__).resolve().parent / "data" / "benzamides.sdf"


def scatter(ligand: Ligand, seed: int) -> Ligand:
    """Return *ligand* moved rigidly into a frame of its own.

    A proper rotation only. Slipping a reflection in here would produce a set that no
    rotation could ever bring back together, and the failure would look like a broken
    aligner rather than a broken example.
    """
    rng = np.random.default_rng(seed)
    rotation, upper = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation = rotation * np.sign(np.diag(upper))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = rng.normal(size=3) * 40.0

    mol = Chem.Mol(ligand.mol)
    rdMolTransforms.TransformConformer(mol.GetConformer(), matrix)
    return replace(ligand, mol=mol)


def main() -> int:
    """Show the failure, then fix it by aligning."""
    scattered = [scatter(ligand, seed=index + 1) for index, ligand in enumerate(load_ligands([DATA]))]

    print("Planning over ligands in mixed frames:")
    try:
        build_network(scattered, mapper="mcss-e2")
    except NetworkPlanError as exc:
        first = str(exc).splitlines()[0]
        print(f"  refused -- {first}")
        print("  Every candidate failed the geometry gate, which is the correct answer here.\n")
    else:  # pragma: no cover - the whole point of the example is that this does not happen
        print("  planned unexpectedly; the scatter did not take\n")

    result = align_ligands(scattered)
    print(f"Aligned onto {result.reference}, median fit RMSD {result.median_rmsd:.3f} A:")
    for record in result.records:
        parent = record.reference or "-"
        print(f"  {record.name:9s} onto {parent:9s} {record.n_fit_atoms:3d} atoms  rmsd {record.rmsd:.3f}")

    network = build_network(result.ligands, mapper="mcss-e2")
    network.validate()
    print(f"\nPlanned {len(network.edges)} edge(s) over {len(network.ligands)} ligands:")
    for edge in sorted(network.edges, key=lambda e: e.score.total):
        print(f"  {edge.key:26s} cost {edge.score.total:6.3f}  core_rmsd {edge.score.descriptors['core_rmsd']:.3f}")

    print(
        "\nNote the core_rmsd values are not zero even though the conformers are identical "
        "here:\nalignment is fitted on the common substructure, not on every atom. On real "
        "separately\nprepared structures they will be larger still, because those conformers "
        "genuinely differ."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
