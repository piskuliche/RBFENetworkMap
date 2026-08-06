"""The scorer contract: reduce edge descriptors to a scalar cost."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Mapping, Sequence

from rbfenetmap.core.models import EdgeScore, RejectionReason

__all__ = ("AbstractScorer",)


class AbstractScorer(ABC):
    """Turn precomputed edge descriptors into a cost. Lower is better.

    Notes
    -----
    A scorer receives a plain ``Mapping[str, float]`` and nothing else -- no molecules,
    no mapping object, no RDKit. Descriptors are computed once, centrally, by
    :func:`rbfenetmap.core.descriptors.compute_descriptors`.

    That narrow interface buys three things. Re-scoring a network under different weights
    costs nothing, because no mapping has to be recomputed. A scorer can be tested
    against hand-written dictionaries, with no chemistry in the test at all. And a
    third-party scorer cannot accidentally reach past its inputs and reintroduce a
    dependency on how the mapping was produced.

    A scorer must not invent rejections. Feasibility is decided upstream by the mapper
    and the repair; *rejections* is passed in so the scorer can propagate it into the
    returned :class:`~rbfenetmap.core.models.EdgeScore`, not so it can add to it. A
    scorer that wants to express "this edge is terrible" returns a large finite cost --
    which leaves the planner free to use it anyway if the alternative is a disconnected
    network.
    """

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def score_edge(self, descriptors: Mapping[str, float], *, rejections: Sequence[RejectionReason]) -> EdgeScore:
        """Return the cost of an edge described by *descriptors*.

        Parameters
        ----------
        descriptors : Mapping[str, float]
            Raw descriptor values from
            :func:`rbfenetmap.core.descriptors.compute_descriptors`.
        rejections : Sequence[RejectionReason]
            Structural rejections already determined upstream. Non-empty means the
            implementation must return
            ``EdgeScore.rejected(*rejections, scorer=self.name)``.

        Returns
        -------
        EdgeScore
        """

    def describe_weights(self) -> Mapping[str, float]:
        """Return the scorer's tunable weights, for display by ``rbfenet score``."""
        return {}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r})"
