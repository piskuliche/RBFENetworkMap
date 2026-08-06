"""Abstract base classes defining the four plugin contracts.

Implementations live under :mod:`rbfenetmap.plugins`; only the contracts live here. The
split matters because the ABCs must be importable with no optional dependency present --
a third party writing a mapper against this package should not have to install kartograf
to see the interface.
"""

from __future__ import annotations

from rbfenetmap.core.meta.exporters import AbstractExporter
from rbfenetmap.core.meta.mappers import AbstractMapper
from rbfenetmap.core.meta.planners import AbstractNetworkPlanner
from rbfenetmap.core.meta.scorers import AbstractScorer

__all__ = ("AbstractExporter", "AbstractMapper", "AbstractNetworkPlanner", "AbstractScorer")
