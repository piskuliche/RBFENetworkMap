"""The mapper contract: propose an atom correspondence between two ligands."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from rbfenetmap.core.models import AtomMapping, Ligand
from rbfenetmap.core.options import MappingOptions

__all__ = ("AbstractMapper",)


class AbstractMapper(ABC):
    """Produce a common-core / soft-core partition for one candidate pair.

    A mapper is responsible only for the correspondence. It does **not** need to produce
    a connected soft-core region: repairing fragmentation is the job of
    :func:`rbfenetmap.core.softcore.repair_softcore_connectivity`, which runs afterwards
    on every mapper's output. Trying to enforce connectivity inside a mapper duplicates
    that logic and makes mappers harder to write and compare.

    Attributes
    ----------
    name : str
        The registered plugin name, used in diagnostics and recorded on the mapping.
    """

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def map_pair(self, source: Ligand, target: Ligand, options: MappingOptions) -> AtomMapping:
        """Return the atom correspondence between *source* and *target*.

        Parameters
        ----------
        source, target : Ligand
            The two ligands, each with explicit hydrogens and one 3D conformer.
        options : MappingOptions
            Search settings and the pre-repair core-pruning policy.

        Returns
        -------
        AtomMapping
            A validated mapping. Construction enforces the contract, so an
            implementation that builds one via
            :meth:`~rbfenetmap.core.models.AtomMapping.from_core_pairs` cannot return
            something malformed.

        Raises
        ------
        rbfenetmap.core.exceptions.MappingError
            If no correspondence can be produced at all. The caller converts this into a
            :attr:`~rbfenetmap.core.models.RejectionReason.MAPPER_FAILED` rejection
            rather than letting it abort the whole run -- one impossible pair among
            hundreds should not stop the planning.
        """

    def supports_pair(self, source: Ligand, target: Ligand) -> bool:
        """Whether this mapper can handle the pair at all.

        Cheap pre-check, called before :meth:`map_pair`. The default accepts everything.
        """
        return True

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r})"
