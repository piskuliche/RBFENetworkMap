"""The planner contract: select the final edge set from scored candidates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Mapping, Sequence

from rbfenetmap.core.models import Ligand, Network, Transformation
from rbfenetmap.core.options import NetworkOptions

__all__ = ("AbstractNetworkPlanner",)


class AbstractNetworkPlanner(ABC):
    """Choose which candidate transformations make up the network.

    Notes
    -----
    A planner selects; it does not judge feasibility. Candidates arrive already scored,
    with infeasible ones marked. An implementation must filter on
    :attr:`~rbfenetmap.core.models.Transformation.feasible` and must still place every
    candidate -- feasible or not -- on :attr:`~rbfenetmap.core.models.Network.candidates`.

    Retaining the infeasible ones is what makes a disconnected result explicable. When
    the planner has to report that two groups of ligands cannot be joined, the rejected
    candidates that span the gap, and their reasons, are the actionable part of the
    message; without them the user gets "disconnected" and no idea what to loosen.
    """

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def plan(
        self, ligands: Mapping[str, Ligand], candidates: Sequence[Transformation], options: NetworkOptions
    ) -> Network:
        """Select edges and return the planned network.

        Parameters
        ----------
        ligands : Mapping[str, Ligand]
            Every vertex, including any the planner ends up unable to connect.
        candidates : Sequence[Transformation]
            Scored candidates, feasible and infeasible.
        options : NetworkOptions
            The user's selection knobs.

        Returns
        -------
        Network

        Raises
        ------
        rbfenetmap.core.exceptions.NetworkPlanError
            If the constraints are unsatisfiable: a forced edge that is infeasible, an
            ``n_edges`` too small to span, or a disconnected candidate pool while
            ``require_connected`` is set. Constraints that are merely tight -- an
            ``edges_per_ligand`` the pool cannot support -- are recorded on
            :attr:`~rbfenetmap.core.models.Network.unmet_constraints` instead.
        """

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r})"
