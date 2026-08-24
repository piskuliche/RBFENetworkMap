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
            ``title``, ``show_indices``, ``reject_depictions``, and
            ``max_reject_depictions``.
        """
        from rbfenetmap.viz.gallery import DEFAULT_MAX_REJECT_DEPICTIONS, render_report

        destination = Path(destination)
        path = destination / f"network{self.default_suffix}" if destination.is_dir() else destination
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_report(
                network,
                title=str(options.get("title", "RBFE network")),
                show_indices=bool(options.get("show_indices", False)),
                reject_depictions=bool(options.get("reject_depictions", True)),
                # --exporter-opt parses every number as a float, so an int() is required
                # rather than merely tidy: a float slice bound raises.
                max_reject_depictions=int(float(options.get("max_reject_depictions", DEFAULT_MAX_REJECT_DEPICTIONS))),
            )
        )
        return (path,)
