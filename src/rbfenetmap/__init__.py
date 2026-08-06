"""Plan Relative Binding Free Energy (RBFE) perturbation networks.

The package turns a series of RDKit molecules into a scored, tunable network of
alchemical transformations. Each transformation carries a common-core / soft-core
partition that satisfies the constraint enforced throughout this package: **a
transformation has at most one connected soft-core region per side**.

The pipeline is four stages, each pluggable (see :mod:`rbfenetmap.core.pluginregistry`):

1. *map* -- a :class:`~rbfenetmap.core.meta.mappers.AbstractMapper` proposes an atom
   correspondence for a candidate pair (MCS-based, geometry-based, or external).
2. *repair* -- :func:`rbfenetmap.core.softcore.repair_softcore_connectivity` demotes
   bridging common-core atoms until each side has a single connected soft-core region,
   or rejects the pair outright.
3. *score* -- descriptors are computed centrally and an
   :class:`~rbfenetmap.core.meta.scorers.AbstractScorer` reduces them to a cost.
4. *plan* -- an :class:`~rbfenetmap.core.meta.planners.AbstractNetworkPlanner` selects
   the final edge set subject to the user's connectivity and redundancy knobs.

Nothing in the core imports ParmEd or any Amber machinery; Amber-specific output lives
behind the ``amber`` exporter plugin.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rbfe-network-map")
except PackageNotFoundError:  # pragma: no cover - only hit when running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ("__version__",)
