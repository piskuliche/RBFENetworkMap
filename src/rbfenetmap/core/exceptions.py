"""Exception hierarchy for :mod:`rbfenetmap`.

Every error the package raises deliberately derives from :class:`RBFENetworkMapError`,
so a caller embedding this in a larger workflow can catch one type and know the failure
came from network planning rather than from RDKit, NumPy, or its own code.

Note the distinction this hierarchy encodes, which matters throughout the package: a
*rejected edge* is not an error. An edge whose soft-core cannot be repaired within the
configured budget is a normal, expected outcome recorded as a
:class:`~rbfenetmap.core.models.RejectionReason` on the transformation's score. These
exceptions are for situations where the caller asked for something impossible or
inconsistent -- a forced edge that cannot exist, a mapping that violates its own
invariants, a plugin that is not installed.
"""

from __future__ import annotations

from typing import Any, Sequence


class RBFENetworkMapError(Exception):
    """Base class for every error raised by :mod:`rbfenetmap`."""


class MappingError(RBFENetworkMapError):
    """An atom mapping could not be produced, or violates the mapping contract."""


class RepairError(RBFENetworkMapError):
    """The soft-core repair could not run.

    Raised for malformed input to the repair (for example a mapping whose indices do
    not match the molecules). A repair that runs correctly and concludes the edge is
    infeasible returns a :class:`~rbfenetmap.core.models.RejectionReason` instead --
    that is an answer, not an error.
    """


class NetworkPlanError(RBFENetworkMapError):
    """The requested network cannot be planned.

    Raised when user constraints are unsatisfiable rather than merely tight: a forced
    edge that is infeasible, ``n_edges`` too small to span the ligands, or a candidate
    pool that is disconnected while ``require_connected`` is set.

    Parameters
    ----------
    message : str
        The actionable description, which is what a bare ``str()`` of the error yields.
    rejected : Sequence, optional
        The infeasible candidates the planner had to work around, when it has them.

    Notes
    -----
    The payload exists because the *pattern* of rejections often diagnoses the run better
    than any single one of them does. A caller that wants to say "these all failed the same
    way, and here is what that usually means" should not have to re-derive the pool or, worse,
    parse it back out of the message it just formatted.
    """

    def __init__(self, message: str, rejected: "Sequence[Any]" = ()) -> None:
        super().__init__(message)
        self.rejected = tuple(rejected)


class ExporterError(RBFENetworkMapError):
    """A network could not be serialized for a downstream program."""


class PluginError(RBFENetworkMapError):
    """A plugin is unknown, duplicated, or its backend is not importable."""
