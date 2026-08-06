"""Trivial baseline scorer: cost equals the larger soft-core.

Deliberately the simplest defensible scoring rule. Its purpose is twofold: it is the
honest baseline any richer scorer should be shown to beat, and because its costs are
whole numbers that a reader can compute by eye, it makes planner tests verifiable by
hand -- a minimum spanning tree over integer weights has an obvious right answer.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar, Mapping, Sequence

from rbfenetmap.core.meta.scorers import AbstractScorer
from rbfenetmap.core.models import EdgeScore, RejectionReason

__all__ = ("SoftcoreSizeScorer",)


class SoftcoreSizeScorer(AbstractScorer):
    """Cost is the heavy-atom count of the larger soft-core region."""

    name: ClassVar[str] = "softcore-size"

    def score_edge(self, descriptors: Mapping[str, float], *, rejections: Sequence[RejectionReason]) -> EdgeScore:
        """Return the larger soft-core size as the cost."""
        if rejections:
            return EdgeScore.rejected(*rejections, scorer=self.name, descriptors=MappingProxyType(dict(descriptors)))
        total = float(descriptors.get("n_softcore_max_heavy", 0.0))
        return EdgeScore(
            total=total,
            feasible=True,
            descriptors=MappingProxyType(dict(descriptors)),
            contributions=MappingProxyType({"softcore_atoms": total}),
            rejections=(),
            scorer=self.name,
        )
