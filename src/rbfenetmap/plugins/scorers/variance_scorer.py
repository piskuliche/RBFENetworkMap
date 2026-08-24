"""Predicted per-edge standard deviation, in kcal/mol.

Every other scorer in this package returns a cost on an invented scale: the linear
scorer's totals are weighted, normalised descriptor units, and the only meaningful thing
to do with two of them is compare them. This one returns a physical quantity -- the
standard deviation the edge's free energy estimate is *predicted* to have -- which is what
makes statistical design possible at all. An optimal-design planner needs ``sigma_ij``,
not a ranking; the Fisher information of the network is built from ``1 / sigma_ij ** 2``
and nothing else.

The functional form is equation 19 of the NetBFE paper: a floor, a term in the transforming
(soft-core) heavy-atom count, and a smaller term in the total heavy-atom count.

.. math::

   s_{ij} = w_0 + w_1 \\sqrt{\\max(h_{ij}, h_{ji})} + w_2 \\sqrt{\\max(H_{ij}, H_{ji})}

with :math:`w = (1.0, 1.0, 0.5)`. Both counts are already computed centrally --
``n_softcore_max_heavy`` is :math:`\\max(h_{ij}, h_{ji})` by construction, and
:math:`\\max(H_{ij}, H_{ji})` is the larger of ``n_heavy_1`` and ``n_heavy_2`` -- so this
scorer needs no descriptor of its own and, like every scorer here, never sees a molecule.

Why square roots
----------------
Sampling error in an alchemical free energy grows roughly with the square root of the
number of degrees of freedom being decoupled, not linearly with it: doubling the soft-core
does not double the noise. The floor ``w_0`` is the irreducible part -- an edge that
transforms nothing at all still carries one run's worth of statistical error -- which also
keeps ``1 / sigma ** 2`` finite for a hypothetical zero-atom transformation, and so keeps
the Fisher information matrix finite.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import ClassVar, Mapping, Sequence

from rbfenetmap.core.meta.scorers import AbstractScorer
from rbfenetmap.core.models import EdgeScore, RejectionReason

__all__ = ("DEFAULT_VARIANCE_WEIGHTS", "VarianceScorer")

#: ``w0``, ``w1``, ``w2`` of NetBFE eq. 19, in kcal/mol. A mapping rather than a tuple so
#: ``rbfenet score`` can display the terms by name and a user can override one of them
#: without restating the others.
DEFAULT_VARIANCE_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {"intercept": 1.0, "softcore_heavy": 1.0, "total_heavy": 0.5}
)


class VarianceScorer(AbstractScorer):
    """Predict an edge's free energy standard deviation, in kcal/mol.

    Parameters
    ----------
    weights : Mapping[str, float], optional
        Overrides merged onto :data:`DEFAULT_VARIANCE_WEIGHTS`. Keys are ``intercept``,
        ``softcore_heavy``, and ``total_heavy``.

    Raises
    ------
    ValueError
        If *weights* names a term that does not exist, or if any weight is negative.
        Unknown terms are refused for the reason
        :class:`~rbfenetmap.plugins.scorers.linear_scorer.LinearScorer` refuses them: a
        typo that silently leaves the defaults in place is the worst failure mode a tuning
        knob can have. Negative weights are refused because they can drive the predicted
        standard deviation to zero or below, and ``1 / sigma ** 2`` then diverges or
        changes sign -- one such edge would make the whole Fisher matrix meaningless.

    Notes
    -----
    Pair this with ``--design`` for statistical edge selection, and with
    ``--design-total-ns`` for sample allocation. Both read
    :attr:`~rbfenetmap.core.models.EdgeScore.total` as a standard deviation in kcal/mol;
    under any other scorer they still run, but on a scale with no physical meaning.
    """

    name: ClassVar[str] = "variance"

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        """Merge *weights* onto the defaults, rejecting unknown or negative terms."""
        merged = dict(DEFAULT_VARIANCE_WEIGHTS)
        if weights:
            unknown = sorted(set(weights) - set(DEFAULT_VARIANCE_WEIGHTS))
            if unknown:
                raise ValueError(
                    f"Unknown variance term(s) {unknown}. Available terms: {sorted(DEFAULT_VARIANCE_WEIGHTS)}."
                )
            merged.update({k: float(v) for k, v in weights.items()})
        negative = sorted(k for k, v in merged.items() if v < 0)
        if negative:
            raise ValueError(
                f"Variance weight(s) {negative} are negative. A predicted standard deviation must stay "
                "positive: the Fisher information built from it is 1 / sigma ** 2."
            )
        self._weights = MappingProxyType(merged)

    def describe_weights(self) -> Mapping[str, float]:
        """Return the effective weights."""
        return self._weights

    def score_edge(self, descriptors: Mapping[str, float], *, rejections: Sequence[RejectionReason]) -> EdgeScore:
        """Return the predicted standard deviation, in kcal/mol.

        Parameters
        ----------
        descriptors : Mapping[str, float]
            Needs ``n_softcore_max_heavy`` and both of ``n_heavy_1`` / ``n_heavy_2``.
            Missing keys read as zero, degrading to the intercept rather than raising --
            the same tolerance every other scorer here shows.
        rejections : Sequence[RejectionReason]

        Returns
        -------
        EdgeScore
        """
        if rejections:
            return EdgeScore.rejected(*rejections, scorer=self.name, descriptors=MappingProxyType(dict(descriptors)))

        softcore_heavy = max(float(descriptors.get("n_softcore_max_heavy", 0.0)), 0.0)
        total_heavy = max(float(descriptors.get("n_heavy_1", 0.0)), float(descriptors.get("n_heavy_2", 0.0)), 0.0)
        contributions = {
            "intercept": self._weights["intercept"],
            "softcore_heavy": self._weights["softcore_heavy"] * math.sqrt(softcore_heavy),
            "total_heavy": self._weights["total_heavy"] * math.sqrt(total_heavy),
        }
        return EdgeScore(
            total=float(sum(contributions.values())),
            feasible=True,
            descriptors=MappingProxyType(dict(descriptors)),
            contributions=MappingProxyType(contributions),
            rejections=(),
            scorer=self.name,
        )
