#!/usr/bin/env python
"""Register a third-party scorer and use it.

The point of the scorer contract: a scorer sees a ``Mapping[str, float]`` and nothing
else. No RDKit, no mapping object, no molecules. That is what makes a custom scorer this
short, and testable against a hand-written dictionary.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Mapping, Sequence

from rbfenetmap.core.meta.scorers import AbstractScorer
from rbfenetmap.core.models import EdgeScore, RejectionReason
from rbfenetmap.core.options import NetworkOptions
from rbfenetmap.core.pipeline import build_network
from rbfenetmap.io.loaders import load_ligands

DATA = Path(__file__).resolve().parent / "data" / "benzamides.sdf"


class HeavyAtomScorer(AbstractScorer):
    """Cost is the change in heavy-atom count, ignoring everything else."""

    name: ClassVar[str] = "heavy-atom-delta"

    def score_edge(self, descriptors: Mapping[str, float], *, rejections: Sequence[RejectionReason]) -> EdgeScore:
        """Return the heavy-atom delta as the cost."""
        if rejections:
            return EdgeScore.rejected(*rejections, scorer=self.name)
        total = float(descriptors["heavy_atom_delta"])
        return EdgeScore(
            total=total,
            feasible=True,
            descriptors=dict(descriptors),
            contributions={"heavy_atom_delta": total},
            scorer=self.name,
        )


def main() -> int:
    """Plan a network with the custom scorer."""
    # A scorer instance can be passed straight in; registering a PluginSpec is only
    # needed to make it selectable by name from the CLI.
    network = build_network(
        load_ligands([DATA]), scorer=HeavyAtomScorer(), network_options=NetworkOptions(edges_per_ligand=2)
    )
    for edge in sorted(network.edges, key=lambda e: e.score.total):
        print(f"{edge.key:26s} {edge.score.total:5.1f} heavy-atom change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
