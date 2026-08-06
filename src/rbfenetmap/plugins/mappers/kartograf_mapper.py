"""Adapter for the external `kartograf` atom mapper.

Kept behind an optional dependency. The import happens inside :meth:`map_pair`, not at
module scope, so ``rbfenet plugins`` can list this mapper -- and report exactly which
modules are missing -- in an environment where kartograf is not installed.
"""

from __future__ import annotations

from typing import ClassVar

from rbfenetmap.core.coreprune import prune_core
from rbfenetmap.core.exceptions import MappingError
from rbfenetmap.core.meta.mappers import AbstractMapper
from rbfenetmap.core.models import AtomMapping, Ligand
from rbfenetmap.core.options import MappingOptions

__all__ = ("KartografMapper",)


class KartografMapper(AbstractMapper):
    """Map two ligands using `kartograf`'s geometry-based mapper.

    Requires ``kartograf`` and ``gufe``.
    """

    name: ClassVar[str] = "kartograf"

    def map_pair(self, source: Ligand, target: Ligand, options: MappingOptions) -> AtomMapping:
        """Return kartograf's correspondence between *source* and *target*.

        Raises
        ------
        rbfenetmap.core.exceptions.MappingError
            If kartograf is unavailable or produces no mapping.
        """
        try:
            from gufe import SmallMoleculeComponent
            from kartograf.atom_aligner import align_mol_shape
            from kartograf.atom_mapper import KartografAtomMapper
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MappingError(
                "The 'kartograf' mapper requires the kartograf and gufe packages. "
                "Install them with `pip install rbfe-network-map[kartograf]`."
            ) from exc

        component_1 = SmallMoleculeComponent.from_rdkit(source.mol)
        component_2 = SmallMoleculeComponent.from_rdkit(target.mol)
        aligned_2 = align_mol_shape(component_2, ref_mol=component_1)

        mapper = KartografAtomMapper(atom_max_distance=options.distance_threshold)
        try:
            result = next(iter(mapper.suggest_mappings(component_1, aligned_2)))
        except StopIteration as exc:
            raise MappingError(f"{source.name}~{target.name}: kartograf suggested no mapping.") from exc

        core = {int(a): int(b) for a, b in result.componentA_to_componentB.items()}
        if not core:
            raise MappingError(f"{source.name}~{target.name}: kartograf returned an empty mapping.")

        mapping = AtomMapping.from_core_pairs(
            core, n_atoms_1=source.n_atoms, n_atoms_2=target.n_atoms, method=self.name
        )
        return prune_core(source, target, mapping, options.core_pruning)
