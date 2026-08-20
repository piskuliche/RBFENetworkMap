#!/usr/bin/env python
"""Build the co-posed 3D SDF example sets from their SMILES lists.

No binary files are checked into the repository. This script regenerates each ``.sdf``
deterministically from the SMILES list next to it, so the example data is reproducible and
reviewable as text.

Two sets, and the difference between them is the point:

``benzamides``
    One scaffold, nine substituents. Every pair maps, so the candidate pool is connected
    and core clustering returns a single cluster.

``scaffolds``
    Three scaffolds sharing only their amide anchor. Within a scaffold the shared core is
    9, 9, or 8 heavy atoms on ligands of 8 to 10; across scaffolds it is 4. That gap is
    what ``--cluster-min-core-atoms`` has to land in, which makes this the set to try
    ``--core-clusters`` on.

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

#: ``stem -> substructure every ligand in that set shares``; the frame its poses are built
#: in. The three-scaffold set can only be anchored on the amide, because that is the only
#: thing its members have in common -- which is exactly why it clusters into three.
SCAFFOLDS = {"benzamides": "c1ccccc1C(=O)N", "scaffolds": "NC=O"}

#: Fixed so that regenerating the file twice gives identical coordinates.
RANDOM_SEED = 0xF00D


def build_scaffold(scaffold_smiles: str) -> Chem.Mol:
    """Return the embedded, minimised reference scaffold."""
    scaffold = Chem.AddHs(Chem.MolFromSmiles(scaffold_smiles))
    if AllChem.EmbedMolecule(scaffold, randomSeed=RANDOM_SEED) != 0:
        raise RuntimeError(f"Could not embed the scaffold {scaffold_smiles!r}.")
    AllChem.MMFFOptimizeMolecule(scaffold)
    return Chem.RemoveHs(scaffold)


def build_ligand(smiles: str, name: str, scaffold: Chem.Mol) -> Chem.Mol:
    """Embed *smiles* with its scaffold atoms pinned to the reference pose."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES {smiles!r} for {name!r}.")
    mol = Chem.AddHs(mol)
    if not mol.HasSubstructMatch(scaffold):
        raise ValueError(f"{name!r} does not contain the scaffold {Chem.MolToSmiles(scaffold)!r}.")
    AllChem.ConstrainedEmbed(mol, scaffold, randomseed=RANDOM_SEED)
    mol.SetProp("_Name", name)
    return mol


def build_set(stem: str, here: Path) -> int:
    """Write ``<stem>.sdf`` from ``<stem>.smi``; return how many ligands were written."""
    smiles_path = here / f"{stem}.smi"
    sdf_path = here / f"{stem}.sdf"

    scaffold = build_scaffold(SCAFFOLDS[stem])
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
    return written


def main(argv: list[str] | None = None) -> int:
    """Write every example SDF beside this script."""
    del argv
    here = Path(__file__).resolve().parent
    for stem in SCAFFOLDS:
        build_set(stem, here)
    return 0


if __name__ == "__main__":
    sys.exit(main())
