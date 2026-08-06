#!/usr/bin/env python
"""Export a planned network for amberstudio and guimapper."""

from __future__ import annotations

from pathlib import Path

from rbfenetmap.core.options import NetworkOptions
from rbfenetmap.core.pipeline import build_network
from rbfenetmap.io.loaders import load_ligands
from rbfenetmap.plugins.exporters import create_exporter

DATA = Path(__file__).resolve().parent / "data" / "benzamides.sdf"


def main() -> int:
    """Plan, validate for Amber, then export."""
    network = build_network(load_ligands([DATA]), network_options=NetworkOptions(edges_per_ligand=2))

    exporter = create_exporter("amber")
    # Checks atom-name uniqueness before writing anything: an Amber soft-core mask selects
    # by name, so a name shared with a common-core atom would silently widen the mask.
    exporter.validate(network)

    written = exporter.export(network, Path("amber_out"))
    print(f"Wrote {len(written)} file(s):")
    for path in written[:5]:
        print(f"  {path}")
    print("\nThese `atommap_*.runconfig` files open directly in guimapper for hand editing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
