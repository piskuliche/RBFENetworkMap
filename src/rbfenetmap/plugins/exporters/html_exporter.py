"""Self-contained HTML report exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from rbfenetmap.core.meta.exporters import AbstractExporter
from rbfenetmap.core.models import Network

__all__ = ("HTMLGalleryExporter",)


class HTMLGalleryExporter(AbstractExporter):
    """Write a single self-contained HTML report of the network."""

    name: ClassVar[str] = "html"
    default_suffix: ClassVar[str] = ".html"

    def export(self, network: Network, destination: Path, **options: Any) -> tuple[Path, ...]:
        """Write the report.

        Parameters
        ----------
        network : Network
        destination : pathlib.Path
            A file, or a directory in which ``network.html`` is written.
        **options
            ``title`` and ``show_indices``.
        """
        from rbfenetmap.viz.gallery import render_report

        destination = Path(destination)
        path = destination / f"network{self.default_suffix}" if destination.is_dir() else destination
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_report(
                network,
                title=str(options.get("title", "RBFE network")),
                show_indices=bool(options.get("show_indices", False)),
            )
        )
        return (path,)
