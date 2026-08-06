"""The exporter contract: serialize a planned network for a downstream program.

This is the seam the package hangs its "hooks to other programs" on. An exporter is how
a network reaches Amber, a workflow engine, a viewer, or anything else, without any of
those systems' concerns leaking back into the core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from rbfenetmap.core.models import Network

__all__ = ("AbstractExporter",)


class AbstractExporter(ABC):
    """Write a planned network out in some downstream format.

    Attributes
    ----------
    name : str
        The registered plugin name.
    default_suffix : str
        Extension used when *destination* names a file with none.
    """

    name: ClassVar[str] = "abstract"
    default_suffix: ClassVar[str] = ""

    @abstractmethod
    def export(self, network: Network, destination: Path, **options: Any) -> tuple[Path, ...]:
        """Write *network* to *destination*.

        Parameters
        ----------
        network : Network
            The planned network.
        destination : pathlib.Path
            A file or a directory, depending on the exporter. Exporters that emit one
            file per edge take a directory.
        **options
            Exporter-specific settings, passed through from ``--exporter-opt``.

        Returns
        -------
        tuple[pathlib.Path, ...]
            Every path written, so a caller can report or clean up.

        Raises
        ------
        rbfenetmap.core.exceptions.ExporterError
            If the network cannot be represented in the target format.
        """

    def validate(self, network: Network) -> None:
        """Check *network* can be exported, without writing anything.

        Format-specific constraints that the core does not enforce belong here -- the
        Amber exporter's atom-name uniqueness requirement being the motivating case.

        Called early by ``rbfenet plan --validate-exporter`` so a constraint that would
        only surface at export time is caught before the expensive mapping work runs,
        rather than after.

        Raises
        ------
        rbfenetmap.core.exceptions.ExporterError
            If the network violates a format constraint.
        """
        return None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r})"
