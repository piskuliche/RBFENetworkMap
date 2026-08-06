"""Weighted-sum edge scorer.

Each descriptor is normalised so that a value of ``1.0`` means roughly "one typical unit
of badness", then multiplied by a user-tunable weight and clipped. The normalisation is
what makes the weights interpretable: a weight of ``4.0`` on ``charge_delta`` against
``1.0`` on ``softcore_atoms`` says a unit charge change is about as costly as four
soft-core-sized problems, which is a statement a chemist can argue with.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar, Mapping, Sequence

from rbfenetmap.core.meta.scorers import AbstractScorer
from rbfenetmap.core.models import EdgeScore, RejectionReason

__all__ = ("DEFAULT_SCORE_WEIGHTS", "TERM_DEFINITIONS", "LinearScorer")

#: ``term -> (descriptor, divisor, cap)``. The divisor sets the scale on which one unit
#: of the descriptor equals 1.0; the cap keeps a single pathological descriptor from
#: dominating a sum that is meant to balance several concerns.
TERM_DEFINITIONS: Mapping[str, tuple[str, float, float]] = MappingProxyType(
    {
        "softcore_atoms": ("n_softcore_max_heavy", 8.0, 8.0),
        "softcore_asymmetry": ("softcore_asymmetry", 8.0, 4.0),
        "heavy_atom_delta": ("heavy_atom_delta", 8.0, 4.0),
        "charge_delta": ("charge_delta", 1.0, 2.0),
        "ring_delta": ("ring_delta", 1.0, 3.0),
        "ring_atoms_in_softcore": ("n_ring_atoms_in_softcore", 6.0, 4.0),
        "mcs_deficit": ("mcs_fraction", 1.0, 1.0),
        "core_rmsd": ("core_rmsd", 1.0, 3.0),
        "rotatable_delta": ("rotatable_delta", 3.0, 3.0),
        "repair_cost": ("n_demoted_atoms", 6.0, 4.0),
        "logp_delta": ("logp_delta", 2.0, 3.0),
    }
)

#: Default weights. Overridable per run via ``--weights`` or ``--weights-file``.
DEFAULT_SCORE_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "softcore_atoms": 1.00,
        "softcore_asymmetry": 0.25,
        "heavy_atom_delta": 0.25,
        "charge_delta": 4.00,
        "ring_delta": 1.00,
        "ring_atoms_in_softcore": 0.50,
        "mcs_deficit": 2.00,
        "core_rmsd": 1.00,
        "rotatable_delta": 0.20,
        "repair_cost": 0.75,
        "logp_delta": 0.10,
    }
)


class LinearScorer(AbstractScorer):
    """Score an edge as a weighted sum of normalised descriptors.

    Parameters
    ----------
    weights : Mapping[str, float], optional
        Overrides merged onto :data:`DEFAULT_SCORE_WEIGHTS`.

    Raises
    ------
    ValueError
        If *weights* names a term that does not exist. Silently ignoring an unknown key
        would let a typo in ``--weights softcore_atom=2`` look like it took effect while
        the run used the default, which is the worst possible failure mode for a tuning
        knob.
    """

    name: ClassVar[str] = "linear"

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        """Merge *weights* onto the defaults, rejecting unknown terms."""
        merged = dict(DEFAULT_SCORE_WEIGHTS)
        if weights:
            unknown = sorted(set(weights) - set(TERM_DEFINITIONS))
            if unknown:
                raise ValueError(f"Unknown scoring term(s) {unknown}. Available terms: {sorted(TERM_DEFINITIONS)}.")
            merged.update({k: float(v) for k, v in weights.items()})
        self._weights = MappingProxyType(merged)

    def describe_weights(self) -> Mapping[str, float]:
        """Return the effective weights."""
        return self._weights

    def score_edge(self, descriptors: Mapping[str, float], *, rejections: Sequence[RejectionReason]) -> EdgeScore:
        """Return the weighted cost, or an infeasible score if *rejections* is non-empty."""
        if rejections:
            return EdgeScore.rejected(*rejections, scorer=self.name, descriptors=MappingProxyType(dict(descriptors)))

        contributions: dict[str, float] = {}
        for term, weight in self._weights.items():
            if not weight:
                continue
            descriptor, divisor, cap = TERM_DEFINITIONS[term]
            raw = float(descriptors.get(descriptor, 0.0))
            # `mcs_deficit` is the one inverted term: a high mcs_fraction is good, so the
            # cost is what the core fails to cover.
            value = (1.0 - raw) if term == "mcs_deficit" else (raw / divisor)
            contributions[term] = weight * min(max(value, 0.0), cap)

        return EdgeScore(
            total=float(sum(contributions.values())),
            feasible=True,
            descriptors=MappingProxyType(dict(descriptors)),
            contributions=MappingProxyType(contributions),
            rejections=(),
            scorer=self.name,
        )
