"""Core data model, graph utilities, and the soft-core repair algorithm.

Nothing in this subpackage imports a plugin backend. The dependency direction is
strictly ``plugins -> core``, so the data model and the repair can be exercised without
any optional dependency installed.
"""

from __future__ import annotations

from rbfenetmap.core.models import (
    AtomMapping,
    EdgeScore,
    IntermediateRecord,
    Ligand,
    LigandProvenance,
    Network,
    RejectionReason,
    SoftcoreRepair,
    Transformation,
)
from rbfenetmap.core.options import CorePruningPolicy, MappingOptions, NetworkOptions, SoftcorePolicy

__all__ = (
    "AtomMapping",
    "CorePruningPolicy",
    "EdgeScore",
    "IntermediateRecord",
    "Ligand",
    "LigandProvenance",
    "MappingOptions",
    "Network",
    "NetworkOptions",
    "RejectionReason",
    "SoftcorePolicy",
    "SoftcoreRepair",
    "Transformation",
)
