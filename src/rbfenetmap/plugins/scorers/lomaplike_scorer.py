"""Multiplicative similarity scorer in the spirit of LOMAP.

Scores an edge as a product of independent penalty factors in ``(0, 1]``, then converts
that similarity to a cost with ``-log``. Multiplicative composition behaves differently
from the weighted sum in :mod:`rbfenetmap.plugins.scorers.linear_scorer`: any single
factor near zero drags the whole similarity to zero regardless of how good the rest is.
That is the right shape when the penalties are *independent reasons the edge will not
converge*, rather than competing preferences to be balanced.

Implemented from the published form; LOMAP itself is not a dependency.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import ClassVar, Mapping, Sequence

from rbfenetmap.core.meta.scorers import AbstractScorer
from rbfenetmap.core.models import EdgeScore, RejectionReason

__all__ = ("DEFAULT_LOMAP_PARAMETERS", "LomapLikeScorer")

#: Tunable factors. ``beta`` sets how fast similarity decays with soft-core size; the
#: ``*_penalty`` values are the multiplier applied per unit of the corresponding change.
DEFAULT_LOMAP_PARAMETERS: Mapping[str, float] = MappingProxyType(
    {"beta": 0.10, "charge_penalty": 0.10, "ring_penalty": 0.40, "ring_atom_penalty": 0.90, "rmsd_penalty": 0.70}
)

#: Similarity floor, so a hopeless-but-feasible edge yields a large finite cost rather
#: than an infinity that would be indistinguishable from a structural rejection.
_MIN_SIMILARITY = 1e-9


class LomapLikeScorer(AbstractScorer):
    """Score an edge by a product of penalty factors.

    Parameters
    ----------
    parameters : Mapping[str, float], optional
        Overrides merged onto :data:`DEFAULT_LOMAP_PARAMETERS`.

    Raises
    ------
    ValueError
        If *parameters* names an unknown key.
    """

    name: ClassVar[str] = "lomaplike"

    def __init__(self, parameters: Mapping[str, float] | None = None) -> None:
        """Merge *parameters* onto the defaults, rejecting unknown keys."""
        merged = dict(DEFAULT_LOMAP_PARAMETERS)
        if parameters:
            unknown = sorted(set(parameters) - set(DEFAULT_LOMAP_PARAMETERS))
            if unknown:
                raise ValueError(f"Unknown parameter(s) {unknown}. Available: {sorted(DEFAULT_LOMAP_PARAMETERS)}.")
            merged.update({k: float(v) for k, v in parameters.items()})
        self._parameters = MappingProxyType(merged)

    def describe_weights(self) -> Mapping[str, float]:
        """Return the effective parameters."""
        return self._parameters

    def score_edge(self, descriptors: Mapping[str, float], *, rejections: Sequence[RejectionReason]) -> EdgeScore:
        """Return ``-log(similarity)`` as the cost."""
        if rejections:
            return EdgeScore.rejected(*rejections, scorer=self.name, descriptors=MappingProxyType(dict(descriptors)))

        parameters = self._parameters
        factors = {
            "softcore": math.exp(-parameters["beta"] * float(descriptors.get("n_softcore_max_heavy", 0.0))),
            "charge": parameters["charge_penalty"] ** float(descriptors.get("charge_delta", 0.0)),
            "ring": parameters["ring_penalty"] ** float(descriptors.get("ring_delta", 0.0)),
            "ring_atoms": parameters["ring_atom_penalty"] ** float(descriptors.get("n_ring_atoms_in_softcore", 0.0)),
            "geometry": parameters["rmsd_penalty"] ** float(descriptors.get("core_rmsd", 0.0)),
        }
        similarity = max(math.prod(factors.values()), _MIN_SIMILARITY)

        # Report each factor's own -log contribution, so the terms sum to the total and
        # `rbfenet score --explain` reads the same way for both scorers.
        contributions = {name: -math.log(max(value, _MIN_SIMILARITY)) for name, value in factors.items()}
        total = -math.log(similarity)
        # Clamping the similarity can make the parts sum to more than the whole; scale
        # them back so the reported breakdown stays honest.
        part_sum = sum(contributions.values())
        if part_sum > 0:
            contributions = {k: v * total / part_sum for k, v in contributions.items()}

        return EdgeScore(
            total=float(total),
            feasible=True,
            descriptors=MappingProxyType(dict(descriptors)),
            contributions=MappingProxyType(contributions),
            rejections=(),
            scorer=self.name,
        )
