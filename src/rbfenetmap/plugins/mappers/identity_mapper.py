"""Index-identity mapper.

Pairs atom *i* with atom *i* for as far as both molecules run. Two uses: exercising the
downstream pipeline in tests without invoking a real mapping algorithm, and consuming
inputs that already carry a correspondence by construction -- ligands written out by a
tool that guarantees a shared atom ordering.
"""

from __future__ import annotations

from typing import ClassVar

from rbfenetmap.core.coreprune import prune_core
from rbfenetmap.core.meta.mappers import AbstractMapper
from rbfenetmap.core.models import AtomMapping, Ligand
from rbfenetmap.core.options import MappingOptions

__all__ = ("IdentityMapper",)


class IdentityMapper(AbstractMapper):
    """Map atom *i* to atom *i* over the shared index prefix.

    Notes
    -----
    Correct only when the two molecules genuinely share an atom ordering. Nothing here
    verifies that -- it cannot be verified from indices alone -- so this mapper trusts
    the caller. Core pruning still runs, which means an identity mapping across
    mismatched element ordering will at least be cut back rather than passed through
    wholesale.
    """

    name: ClassVar[str] = "identity"

    def map_pair(self, source: Ligand, target: Ligand, options: MappingOptions) -> AtomMapping:
        """Return the index-identity correspondence."""
        shared = min(source.n_atoms, target.n_atoms)
        mapping = AtomMapping.from_core_pairs(
            {i: i for i in range(shared)}, n_atoms_1=source.n_atoms, n_atoms_2=target.n_atoms, method=self.name
        )
        return prune_core(source, target, mapping, options.core_pruning)
