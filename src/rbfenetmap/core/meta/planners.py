"""The planner contract: select the final edge set from scored candidates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Mapping, Sequence

from rbfenetmap.core.exceptions import NetworkPlanError
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

    #: Whether this planner knows how to place counterpoised (CBFE) edges. Both of the
    #: modes that need planner cooperation -- ``bridge`` and ``cycles`` -- are expressed as
    #: decisions about *where* an edge goes, which only a planner that reasons about
    #: components and cycles can make. ``all`` needs nothing from the planner, because the
    #: pipeline hands it a pool that is already entirely CBFE.
    supports_cbfe: ClassVar[bool] = False

    def check_cbfe_support(self, options: NetworkOptions) -> None:
        """Raise if *options* asks for CBFE placement this planner cannot do.

        Parameters
        ----------
        options : NetworkOptions

        Raises
        ------
        rbfenetmap.core.exceptions.NetworkPlanError

        Notes
        -----
        Called rather than silently ignored. A user who passes ``--cbfe bridge`` and a
        planner that cannot honour it would otherwise get a disconnected network, or a
        connectivity error, with nothing to connect either outcome to the flag they set --
        the same failure mode the scorers refuse for an unknown weight name.
        """
        if self.supports_cbfe or not options.cbfe_bridges_components:
            return
        raise NetworkPlanError(
            f"Planner {self.name!r} cannot place CBFE edges, but cbfe_mode={options.cbfe_mode!r} requires it. "
            "Use the 'mst' planner, or cbfe_mode='all' (which needs no planner support), or 'off'."
        )

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
