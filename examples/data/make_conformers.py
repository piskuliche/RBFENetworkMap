#!/usr/bin/env python
"""Build a co-posed 3D SDF from ``benzamides.smi``.

No binary files are checked into the repository. This script regenerates
``benzamides.sdf`` deterministically from the SMILES list next to it, so the example
data is reproducible and reviewable as text.

The important part is that every ligand is embedded **against a shared scaffold** with
:func:`rdkit.Chem.AllChem.ConstrainedEmbed`, not independently. Ligands for an RBFE
network are assumed to be posed in a common binding-site frame, and the package's core
RMSD descriptor measures in-place deviation precisely to detect mappings that violate
that. A set of independently embedded conformers has no common frame, so every candidate
edge would be rejected for geometry -- correctly, but uselessly.

Run it with::

    python examples/data/make_conformers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

#: Substructure every ligand in the series shares; the frame the poses are built in.
SCAFFOLD_SMILES = "c1ccccc1C(=O)N"

#: Fixed so that regenerating the file twice gives identical coordinates.
RANDOM_SEED = 0xF00D


def build_scaffold() -> Chem.Mol:
    """Return the embedded, minimised reference scaffold."""
    scaffold = Chem.AddHs(Chem.MolFromSmiles(SCAFFOLD_SMILES))
    if AllChem.EmbedMolecule(scaffold, randomSeed=RANDOM_SEED) != 0:
        raise RuntimeError(f"Could not embed the scaffold {SCAFFOLD_SMILES!r}.")
    AllChem.MMFFOptimizeMolecule(scaffold)
    return Chem.RemoveHs(scaffold)


def build_ligand(smiles: str, name: str, scaffold: Chem.Mol) -> Chem.Mol:
    """Embed *smiles* with its scaffold atoms pinned to the reference pose."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES {smiles!r} for {name!r}.")
    mol = Chem.AddHs(mol)
    if not mol.HasSubstructMatch(scaffold):
        raise ValueError(f"{name!r} does not contain the scaffold {SCAFFOLD_SMILES!r}.")
    AllChem.ConstrainedEmbed(mol, scaffold, randomseed=RANDOM_SEED)
    mol.SetProp("_Name", name)
    return mol


def main(argv: list[str] | None = None) -> int:
    """Write ``benzamides.sdf`` beside this script."""
    del argv
    here = Path(__file__).resolve().parent
    smiles_path = here / "benzamides.smi"
    sdf_path = here / "benzamides.sdf"

    scaffold = build_scaffold()
    written = 0
    with Chem.SDWriter(str(sdf_path)) as writer:
        for line in smiles_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smiles, _, name = line.partition(" ")
            writer.write(build_ligand(smiles, name.strip(), scaffold))
            written += 1

    print(f"Wrote {written} co-posed ligands to {sdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
