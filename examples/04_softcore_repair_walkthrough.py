#!/usr/bin/env python
"""Walk through the soft-core repair on a pair that starts fragmented.

CF3 to ethyl is the clearest small demonstration: three fluorines vanish and an ethyl
group appears, so both sides begin with several disconnected soft-core regions.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from rbfenetmap.core.models import Ligand
from rbfenetmap.core.options import MappingOptions, SoftcorePolicy
from rbfenetmap.core.softcore import detect_fragments, repair_softcore_connectivity, RepairContext
from rbfenetmap.plugins.mappers import create_mapper

SCAFFOLD = "c1ccccc1C(=O)N"
PAIR = {"CF3": "FC(F)(F)c1ccccc1C(=O)N", "Et": "CCc1ccccc1C(=O)N"}


def co_posed() -> dict[str, Ligand]:
    """Build the two ligands constrained onto a shared scaffold."""
    scaffold = Chem.AddHs(Chem.MolFromSmiles(SCAFFOLD))
    AllChem.EmbedMolecule(scaffold, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(scaffold)
    scaffold = Chem.RemoveHs(scaffold)

    ligands = {}
    for name, smiles in PAIR.items():
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.ConstrainedEmbed(mol, scaffold, randomseed=0xF00D)
        ligands[name] = Ligand.from_mol(mol, name)
    return ligands


def main() -> int:
    """Show the mapping before and after repair."""
    ligands = co_posed()
    source, target = ligands["CF3"], ligands["Et"]

    mapping = create_mapper("mcss-e2").map_pair(source, target, MappingOptions())
    context = RepairContext.build(source, target, mapping, SoftcorePolicy())

    print("Before repair")
    print(f"  common core : {mapping.n_common_core} pairs")
    print(f"  soft-core   : {mapping.n_softcore_1} / {mapping.n_softcore_2} atoms")
    print(
        f"  regions     : {len(detect_fragments(context.graph_1, mapping.sc1))} / "
        f"{len(detect_fragments(context.graph_2, mapping.sc2))}"
    )

    repaired, repair = repair_softcore_connectivity(source, target, mapping)

    print("\nRepair trace")
    for line in repair.trace:
        print(f"  {line}")

    print("\nAfter repair")
    print(f"  common core : {repaired.n_common_core} pairs")
    print(f"  soft-core   : {repaired.n_softcore_1} / {repaired.n_softcore_2} atoms")
    print(f"  regions     : {repair.n_fragments_after}")
    print(f"  demoted     : {repair.demoted_1} / {repair.demoted_2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
